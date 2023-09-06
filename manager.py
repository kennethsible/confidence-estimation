import json
import math
import re
from io import StringIO

import spacy
import torch
import torch.nn as nn
from sacremoses import MosesDetokenizer, MosesTokenizer
from subword_nmt.apply_bpe import BPE
from torch import Tensor

from decoder import triu_mask
from model import Model

nlp = spacy.load('de_core_news_sm', enable=['tok2vec', 'tagger', 'lemmatizer'])


def lemmatize(texts):
    for text, (doc, context) in zip(texts, nlp.pipe(texts, as_tuples=True)):
        if text[0].split() == [token.text for token in doc]:
            yield [token.lemma_ for token in doc], context
        else:
            yield text[0].split(), context


class Vocab:
    def __init__(self, words: list[str] | None = None):
        self.num_to_word = ['<UNK>', '<BOS>', '<EOS>', '<PAD>']
        self.word_to_num = {x: i for i, x in enumerate(self.num_to_word)}

        self.UNK = self.word_to_num['<UNK>']
        self.BOS = self.word_to_num['<BOS>']
        self.EOS = self.word_to_num['<EOS>']
        self.PAD = self.word_to_num['<PAD>']

        if words is not None:
            for line in words:
                self.add(line.split()[0])

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
        self, src_nums: Tensor, tgt_nums: Tensor, ignore_index: int, device: str, dict_data=None
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
        return triu_mask(self.tgt_nums[:, :-1].size(-1), device=self.device)

    @staticmethod
    def dict_mask_from_data(dict_data, mask_size, device):
        dict_mask = torch.zeros(mask_size, device=device).repeat((2, 1, mask_size[-1], 1))
        for i, (lemmas, senses) in enumerate(dict_data):
            if len(senses) > 0:
                c_first = senses[0][0]
            for (a, b), (c, d) in zip(lemmas, senses):
                # only lemmas can attend to their senses
                dict_mask[0, i, :c_first, c:d] = 1.0
                dict_mask[0, i, a:b, c:d] = 0.0
                # senses can only attend to themselves
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
    def __init__(self, bpe: BPE, src_lang: str, tgt_lang: str | None = None):
        self.bpe = bpe
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.tokenizer = MosesTokenizer(src_lang)
        lang = tgt_lang if tgt_lang else src_lang
        self.detokenizer = MosesDetokenizer(lang)

    def tokenize(self, text: str) -> str:
        tokens = self.tokenizer.tokenize(text)
        return self.bpe.process_line(' '.join(tokens))

    def detokenize(self, tokens: list[str]) -> str:
        text = self.detokenizer.detokenize(tokens)
        return re.sub('(@@ )|(@@ ?$)', '', text)


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
    label_smoothing: float
    clip_grad: float
    batch_size: int
    max_length: int
    beam_size: int
    threshold: int
    learnable: int
    max_senses: int
    append_dict: int
    exp_position: str
    word_dropout: float

    def __init__(
        self,
        src_lang: str,
        tgt_lang: str,
        config: dict,
        device: str,
        model_file: str,
        vocab_file: str | list[str],
        codes_file: str | list[str],
        dict_file,
        freq_file,
        data_file: str | None = None,
        test_file: str | None = None,
    ):
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.config = config
        self.device = device
        self._model_name = model_file
        self._vocab_list = vocab_file
        self._codes_list = codes_file

        for option, value in config.items():
            self.__setattr__(option, value)

        if isinstance(self._vocab_list, str):
            with open(self._vocab_list) as file:
                self._vocab_list = list(file.readlines())
        self.vocab = Vocab(self._vocab_list)

        if isinstance(self._codes_list, str):
            with open(self._codes_list) as file:
                self._codes_list = list(file.readlines())
        self.bpe = BPE(StringIO(''.join(self._codes_list)))

        self.model = Model(
            self.vocab.size(),
            self.embed_dim,
            self.ff_dim,
            self.num_heads,
            self.dropout,
            self.num_layers,
            self.exp_position,
            self.learnable,
        ).to(device)

        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

        self.model.apply(init_weights)

        self.dict = None
        self.freq = None
        self.data_cache = None

        if dict_file:
            with open(dict_file) as file:
                self.dict = json.load(file)

        if freq_file:
            self.freq = {}
            with open(freq_file) as file:
                for line in file:
                    word, freq = line.split()
                    self.freq[word] = int(freq)

        self.data: list[Batch] | None = None
        self.test: list[Batch] | None = None

    def save_model(self):
        torch.save(
            {
                'state_dict': self.model.state_dict(),
                'src_lang': self.src_lang,
                'tgt_lang': self.tgt_lang,
                'vocab_list': self._vocab_list,
                'codes_list': self._codes_list,
                'model_config': self.config,
            },
            self._model_name,
        )

    def append_senses(self, src_words, src_spans):
        lemmas, senses = [], []
        for word, (lemma_start, lemma_end) in zip(*src_spans):
            if word not in self.dict:
                continue
            if word not in self.freq or self.freq[word] <= self.threshold:
                sense_start = len(src_words)
                sense = self.dict[word][: self.max_senses]
                sense = [sb for w in sense for sb in w.split()]
                sense_end = sense_start + len(sense)
                if len(src_words) + len(sense) > self.max_length:
                    break

                lemmas.append((lemma_start + 1, lemma_end + 1))
                senses.append((sense_start, sense_end))

                # if random.random() <= self.word_dropout:
                #     for j in range(lemma_start, lemma_end):
                #         src_words[j] = '<UNK>'
                src_words.extend(sense)

        # for (a, b), (c, d) in zip(lemmas, senses):
        #     print(src_words[a:b], src_words[c:d])

        return lemmas, senses

    def lemmatize_data(self, data_file):
        words_spans = []
        with open(data_file) as file:
            for line in file.readlines():
                src_line, tgt_line = line.split('\t')
                if not src_line or not tgt_line:
                    continue

                src_words = src_line.split()
                if self.max_length and len(src_words) > self.max_length - 2:
                    src_words = src_words[: self.max_length - 2]

                i, words, spans = 0, [''], []
                for j, subword in enumerate(src_words):
                    if subword.endswith('@@'):
                        words[-1] += subword.rstrip('@@')
                    else:
                        words[-1] += subword
                        words.append('')
                        spans.append((i, j + 1))
                        i = j + 1

                words_spans.append((' '.join(words), spans))

        return list(lemmatize(words_spans))

    def batch_data(self, data) -> list[Batch]:
        batched_data = []

        data.sort(key=lambda x: (len(x[0]), len(x[1])), reverse=True)

        i = batch_size = 0
        while (i := i + batch_size) < len(data):
            src_len = len(data[i][0])
            tgt_len = len(data[i][1])

            while True:
                batch_size = self.batch_size // (max(src_len, tgt_len) * 8) * 8

                src_batch, tgt_batch, lemmas, senses = zip(*data[i : (i + batch_size)])
                max_src_len = max(len(src_words) for src_words in src_batch)
                max_tgt_len = max(len(tgt_words) for tgt_words in tgt_batch)

                if src_len >= max_src_len and tgt_len >= max_tgt_len:
                    break
                src_len, tgt_len = max_src_len, max_tgt_len

            max_src_len = math.ceil(max_src_len / 8) * 8
            max_tgt_len = math.ceil(max_tgt_len / 8) * 8

            src_nums = torch.stack(
                [
                    nn.functional.pad(
                        torch.tensor(self.vocab.numberize(src_words)),
                        (0, max_src_len - len(src_words)),
                        value=self.vocab.PAD,
                    )
                    for src_words in src_batch
                ]
            )
            tgt_nums = torch.stack(
                [
                    nn.functional.pad(
                        torch.tensor(self.vocab.numberize(tgt_words)),
                        (0, max_tgt_len - len(tgt_words)),
                        value=self.vocab.PAD,
                    )
                    for tgt_words in tgt_batch
                ]
            )

            batched_data.append(
                Batch(
                    src_nums,
                    tgt_nums,
                    self.vocab.PAD,
                    self.device,
                    zip(lemmas, senses) if self.dict else None,
                )
            )

        return batched_data

    def load_data(self, data_file, dict_file=None, tokenizer=None, use_cache=True):
        if self.dict:
            if self.data_cache and use_cache:
                src_spans = self.data_cache
            else:
                src_spans = self.lemmatize_data(data_file)
                self.data_cache = src_spans

        i, data = 0, []
        with open(data_file) as file:
            for line in file.readlines():
                src_line, tgt_line = line.split('\t')
                if not src_line or not tgt_line:
                    continue

                src_words = ['<BOS>'] + src_line.split() + ['<EOS>']
                tgt_words = ['<BOS>'] + tgt_line.split() + ['<EOS>']
                if self.max_length:
                    if len(src_words) > self.max_length:
                        src_words = src_words[: self.max_length]
                    if len(tgt_words) > self.max_length:
                        tgt_words = tgt_words[: self.max_length]

                if self.dict:
                    lemmas, senses = self.append_senses(src_words, src_spans[i])
                    data.append((src_words, tgt_words, lemmas, senses))
                else:
                    data.append((src_words, tgt_words, None, None))
                i += 1

        if dict_file and tokenizer:
            with open(dict_file) as file:
                for lemma, sense in json.load(file).items():
                    src_words = tokenizer.tokenize(''.join(lemma)).split()
                    tgt_words = sense[: self.max_senses]
                    tgt_words = [sb for w in sense for sb in w.split()]

                    src_words = ['<BOS>'] + src_line.split() + ['<EOS>']
                    tgt_words = ['<BOS>'] + tgt_line.split() + ['<EOS>']

                    data.append((src_words, tgt_words, [], []))

        return self.batch_data(data)
