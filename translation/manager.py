import json
import math
import re
from typing import Any, Iterable

import sentencepiece as spm
import spacy
import torch
import torch.nn as nn
from sacremoses import MosesDetokenizer, MosesPunctNormalizer, MosesTokenizer
from subword_nmt.apply_bpe import BPE
from torch import Tensor

from translation.decoder import triu_mask
from translation.model import Model

Optimizer = torch.optim.Optimizer
LRScheduler = torch.optim.lr_scheduler.LRScheduler
Token = tuple[str, str, int]


class Vocab:
    def __init__(self):
        self.num_to_word = ['<UNK>', '<BOS>', '<EOS>', '<PAD>']
        self.word_to_num = {x: i for i, x in enumerate(self.num_to_word)}

        self.UNK = self.word_to_num['<UNK>']
        self.BOS = self.word_to_num['<BOS>']
        self.EOS = self.word_to_num['<EOS>']
        self.PAD = self.word_to_num['<PAD>']

    def add(self, word: str):
        if word not in self.word_to_num:
            self.word_to_num[word] = self.size()
            self.num_to_word.append(word)

    def numberize(self, words: list[str]) -> list[int]:
        return [self.word_to_num[word] if word in self.word_to_num else self.UNK for word in words]

    def denumberize(self, nums: list[int]) -> list[str]:
        try:
            start = nums.index(self.BOS) + 1
        except ValueError:
            start = 0
        try:
            end = nums.index(self.EOS)
        except ValueError:
            end = len(nums)
        return [self.num_to_word[num] for num in nums[start:end]]

    def size(self) -> int:
        return len(self.num_to_word)


class Batch:
    def __init__(
        self,
        src_nums: Tensor,
        tgt_nums: Tensor,
        ignore_index: int,
        device: str = 'cpu',
        dict_data: list | None = None,
    ):
        self._src_nums = src_nums
        self._tgt_nums = tgt_nums
        self._dict_data = dict_data
        self.ignore_index = ignore_index
        self.device = device

    @property
    def src_nums(self) -> Tensor:
        return self._src_nums.to(self.device)

    @property
    def tgt_nums(self) -> Tensor:
        return self._tgt_nums.to(self.device)

    @property
    def src_mask(self) -> Tensor:
        return (self.src_nums != self.ignore_index).unsqueeze(-2)

    @property
    def tgt_mask(self) -> Tensor:
        return triu_mask(self.tgt_nums.size(-1), device=self.device)

    @staticmethod
    def dict_mask_from_data(dict_data: list, mask_size: torch.Size, device: str):
        dict_mask = torch.zeros(mask_size, device=device).repeat((2, 1, mask_size[-1], 1))
        for i, (src_spans, tgt_spans) in enumerate(dict_data):
            for (a, b), spans in zip(src_spans, tgt_spans):
                for c, d in spans:
                    # headwords attend to their definitions
                    dict_mask[0, i, :, c:d] = 1.0
                    dict_mask[0, i, a:b, c:d] = 0.0
                    dict_mask[0, i, c:d, c:d] = 0.0
                    # definitions attend to themselves
                    dict_mask[1, i, c:d, :] = 1.0
                    dict_mask[1, i, c:d, c:d] = 0.0
        return dict_mask

    @property
    def dict_mask(self):
        if self._dict_data is None:
            return None
        mask_size = self.src_nums.unsqueeze(-2).size()
        return self.dict_mask_from_data(self._dict_data, mask_size, self.device)

    def length(self) -> int:
        return int((self.tgt_nums[:, 1:] != self.ignore_index).sum())

    def size(self) -> int:
        return self._src_nums.size(0)


class Tokenizer:
    def __init__(self, src_lang: str, tgt_lang: str | None = None, sw_model: Any | None = None):
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.normalizer = MosesPunctNormalizer(src_lang)
        self.tokenizer = MosesTokenizer(src_lang)
        lang = tgt_lang if tgt_lang else src_lang
        self.detokenizer = MosesDetokenizer(lang)
        self.sw_model = sw_model

    def tokenize(self, text: str) -> list[str]:
        text = self.normalizer.normalize(text)
        tokens = self.tokenizer.tokenize(text, escape=False)
        if self.sw_model is None:
            return tokens
        if isinstance(self.sw_model, BPE):
            return self.sw_model.process_line(' '.join(tokens)).split()
        return self.sw_model.encode_as_pieces(' '.join(tokens))

    def detokenize(self, tokens: list[str]) -> str:
        if self.sw_model:
            if isinstance(self.sw_model, BPE):
                text = re.sub('(@@ )|(@@ ?$)', '', ' '.join(tokens))
            else:
                # text = ''.join(tokens).replace('▁', ' ').strip()
                text = self.sw_model.decode(tokens)
        return self.detokenizer.detokenize(text.split())


class Lemmatizer:
    def __init__(self, model: str, sw_model: Any):
        match model.split('_')[0]:
            case 'en':
                self.nlp = spacy.load(
                    model,
                    enable=[
                        'tok2vec',
                        'tagger',
                        'attribute_ruler',
                        'lemmatizer',
                    ],
                )
            case 'de':
                self.nlp = spacy.load(
                    model,
                    enable=[
                        'tok2vec',
                        'tagger',
                        'morphologizer',
                        'lemmatizer',
                    ],
                )
            case _:
                self.nlp = spacy.load(model)

        def whitespace_tokenizer(nlp: spacy.language.Language) -> spacy.tokenizer.Tokenizer:
            return spacy.tokenizer.Tokenizer(nlp.vocab, token_match=lambda text: text)

        self.nlp.tokenizer = whitespace_tokenizer(self.nlp)
        self.sw_model = sw_model

    @staticmethod
    def subword_mapping(texts: list[list[str]], sw_model: Any):
        is_bpe = isinstance(sw_model, BPE)
        for text in texts:
            words, spans = '', []
            for j, subword in enumerate(text):
                if is_bpe:
                    if subword.endswith('@@'):
                        words += subword.removesuffix('@@')
                    else:
                        words += subword + ' '
                        spans.append(j + 2)
                else:
                    if subword.startswith('▁'):
                        words += ' ' + subword.removeprefix('▁')
                        if j > 0:
                            spans.append(j + 1)
                    else:
                        words += subword
            if not is_bpe:
                spans.append(len(text) + 2)
            yield words.strip(), spans

    def lemmatize(self, texts: list[list[str]]):
        _texts = list(self.subword_mapping(texts, self.sw_model))
        for doc, spans in self.nlp.pipe(_texts, as_tuples=True):
            yield [token.lemma_ for token in doc], spans


class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_leaf = False


class Trie:
    def __init__(self, phrases: Iterable[str]):
        self.root = TrieNode()
        for phrase in phrases:
            self.insert(phrase)

    def insert(self, phrase: str):
        node = self.root
        for word in phrase.split():
            if word not in node.children:
                node.children[word] = TrieNode()
            node = node.children[word]
        node.is_leaf = True

    def search_longest_match(self, tokens: list[Token], start: int = 0) -> list[Token]:
        longest_match = []
        i, node = start, self.root
        while i < len(tokens) and tokens[i][0] in node.children:
            node = node.children[tokens[i][0]]
            if node.is_leaf:
                longest_match = tokens[start : i + 1]
            i += 1
        return longest_match

    def greedy_match(self, tokens: list[Token]) -> list[Token]:
        i, matched_tokens = 0, []
        while i < len(tokens):
            longest_match = self.search_longest_match(tokens, i)
            if longest_match:
                word, lemma, ends = list(zip(*longest_match))
                matched_tokens.append((' '.join(word), ' '.join(lemma), ends[-1]))
                i += len(longest_match)
            else:
                matched_tokens.append(tokens[i])
                i += 1
        return matched_tokens


class Manager:
    embed_dim: int
    ff_dim: int
    num_heads: int
    dropout: float
    num_layers: int
    max_epochs: int
    lr: float
    patience: int
    decay_factor: float
    min_lr: float
    max_patience: int
    label_smoothing: float
    clip_grad: float
    batch_size: int
    max_length: int
    beam_size: int
    threshold: int
    max_append: int
    dpe_embed: bool

    def __init__(
        self,
        config: dict,
        device: str,
        src_lang: str,
        tgt_lang: str,
        model_file: str,
        sw_vocab_file: str,
        sw_model_file: str,
        dict_file: str | None = None,
        freq_file: str | None = None,
    ):
        self.config = config
        self.device = device
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self._model_name = model_file

        for option, value in config.items():
            self.__setattr__(option, value)
        self.dpe_embed = bool(config['dpe_embed']) if 'dpe_embed' in config else False

        with open(sw_vocab_file) as sw_vocab_f:
            self.vocab = Vocab()
            for line in sw_vocab_f.readlines():
                self.vocab.add(line.split()[0])
        with open(sw_model_file) as sw_model_f:
            header = sw_model_f.readline()
        if header.startswith('#version'):
            with open(sw_model_file) as sw_model_f:
                self.sw_model = BPE(sw_model_f)
        else:
            self.sw_model = spm.SentencePieceProcessor(sw_model_file)

        self.model = Model(
            self.vocab.size(),
            self.embed_dim,
            self.ff_dim,
            self.num_heads,
            self.dropout,
            self.num_layers,
        ).to(device)

        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

        self.model.apply(init_weights)

        self.dict: dict[str, list[str]] = {}
        if dict_file:
            with open(dict_file) as dict_f:
                self.dict = json.load(dict_f)
            self.trie = Trie(self.dict)
            self.dict_coverage = 0.0

        self.freq: dict[str, int] = {}
        if freq_file:
            with open(freq_file) as freq_f:
                for line in freq_f:
                    word, freq = line.split()
                    self.freq[word] = int(freq)

    def save_model(
        self, train_state: tuple[int, float], optimizer: Optimizer, scheduler: LRScheduler
    ):  # train_state: (Final Epoch, Best Loss)
        torch.save(
            {
                'config': self.config,
                'src_lang': self.src_lang,
                'tgt_lang': self.tgt_lang,
                'optimizer': optimizer.state_dict,
                'scheduler': scheduler.state_dict,
                'state_dict': self.model.state_dict(),
                'train_state': train_state,
            },
            self._model_name,
        )

    def append_defs(
        self,
        src_words: list[str],
        lem_spans: list[tuple[str, int]],
        conf_list: list[tuple[str, float]] | None = None,
    ):
        src_spans, tgt_spans = [], []  # type: ignore[var-annotated]
        delimiter = '@@' if isinstance(self.sw_model, BPE) else '▁'
        accum = 'sum' if 'accum' not in self.config else self.config['accum']

        tokens, src_start = [], 1
        for lemma, src_end in lem_spans:
            tokens.append(
                (
                    ''.join(
                        (
                            subword.removesuffix(delimiter)
                            if delimiter == '@@'
                            else subword.removeprefix(delimiter)
                        )
                        for subword in src_words[src_start:src_end]
                    ),
                    lemma,
                    src_end,
                )
            )
            src_start = src_end
        matched_tokens = self.trie.greedy_match(tokens)

        i, src_start = 0, 1
        for word, lemma, src_end in matched_tokens:
            j = i + len(word.split())
            headword = None
            if conf_list is None:
                if word not in self.freq or self.freq[word] <= self.threshold:
                    headword = word if word in self.dict else lemma if lemma in self.dict else None
            else:
                words, scores = list(zip(*conf_list[i:j]))
                _word = ' '.join(words)
                if _word.strip() == '':  # during training
                    _word = word
                match accum:
                    case 'sum':
                        confidence = sum(scores)
                    case 'avg':
                        confidence = sum(scores) / len(scores)
                    case 'max':
                        confidence = max(scores)
                assert _word == word
                if confidence >= self.threshold:
                    headword = word if word in self.dict else lemma if lemma in self.dict else None

            if headword is not None:
                definitions = self.dict[headword][: self.max_append]
                tgt_start, spans = len(src_words), []
                for definition in definitions:
                    tgt_end = tgt_start + len(definition.split())
                    spans.append((tgt_start, tgt_end))
                    tgt_start = tgt_end
                if tgt_end > self.max_length:
                    break
                src_spans.append((src_start, src_end))
                tgt_spans.append(spans)
                for definition in definitions:
                    src_words.extend(definition.split())

            src_start = src_end
            i = j

        # for (a, b), spans in zip(src_spans, tgt_spans):
        #     print(' '.join(src_words[a:b]))
        #     for c, d in spans:
        #         print('  ', ' '.join(src_words[c:d]))

        return src_spans, tgt_spans

    def batch_data(self, data: list) -> tuple[list[Batch], list[int]]:
        batched_data = []

        sorted_indices = list(range(len(data)))
        combined_data = list(zip(data, sorted_indices))
        combined_data.sort(key=lambda x: (len(x[0][0]), len(x[0][1])), reverse=True)
        data, sorted_indices = list(zip(*combined_data))  # type: ignore[assignment]

        i = batch_size = 0
        while (i := i + batch_size) < len(data):
            src_len, tgt_len = len(data[i][0]), len(data[i][1])

            while True:
                seq_len = math.ceil(max(src_len, tgt_len) / 8) * 8
                batch_size = max(self.batch_size // (seq_len * 8) * 8, 1)

                src_batch, tgt_batch, src_spans, tgt_spans = zip(*data[i : (i + batch_size)])
                src_len = math.ceil(max(len(src_words) for src_words in src_batch) / 8) * 8
                tgt_len = math.ceil(max(len(tgt_words) for tgt_words in tgt_batch) / 8) * 8

                if batch_size * max(src_len, tgt_len) <= self.batch_size:
                    dict_data = list(zip(src_spans, tgt_spans))
                    break
            assert batch_size > 0

            src_nums = torch.stack(
                [
                    nn.functional.pad(
                        torch.tensor(self.vocab.numberize(src_words)),
                        (0, src_len - len(src_words)),
                        value=self.vocab.PAD,
                    )
                    for src_words in src_batch
                ]
            )
            tgt_nums = torch.stack(
                [
                    nn.functional.pad(
                        torch.tensor(self.vocab.numberize(tgt_words)),
                        (0, tgt_len - len(tgt_words)),
                        value=self.vocab.PAD,
                    )
                    for tgt_words in tgt_batch
                ]
            )

            batched_data.append(Batch(src_nums, tgt_nums, self.vocab.PAD, self.device, dict_data))

        return batched_data, sorted_indices  # type: ignore[return-value]

    def load_data(
        self,
        data_file: str,
        lem_file: str | None = None,
        conf_lists: list[list[float]] | None = None,
    ) -> tuple[list[Batch], list[int]]:
        data, lem_data = [], []
        if lem_file:
            with open(lem_file) as lem_f:
                for line in lem_f.readlines():
                    words, spans = line.split('\t')
                    list_of_words = words.split()
                    list_of_spans = list(map(int, spans.split()))
                    lem_data.append(list(zip(list_of_words, list_of_spans)))

        count = total = 0
        with open(data_file) as data_f:
            for i, line in enumerate(data_f.readlines()):
                src_line, tgt_line = line.split('\t')
                src_words = ['<BOS>'] + src_line.split() + ['<EOS>']
                tgt_words = ['<BOS>'] + tgt_line.split() + ['<EOS>']
                src_spans, tgt_spans = [], []
                if lem_data and self.dict:
                    lem_spans, conf_list = lem_data[i], None
                    if conf_lists is not None:
                        assert len(conf_lists[i][1:]) >= len(lem_spans)
                        conf_list = list(zip([''] * len(lem_spans), conf_lists[i][1:]))
                        # assert len(src_words) - 2 == len(self.vocab.denumberize(conf_lists[i].tolist()))
                    src_spans, tgt_spans = self.append_defs(src_words, lem_spans, conf_list)
                    if any(src_spans):
                        count += 1
                data.append((src_words, tgt_words, src_spans, tgt_spans))
                total += 1
            # print(f'{(count / total * 100):.2f}')
            if lem_data and self.dict:
                self.dict_coverage = count / total

        return self.batch_data(data)
