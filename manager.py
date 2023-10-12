import json
import math
import random
import re

# from difflib import SequenceMatcher
from io import StringIO

import spacy
import torch
import torch.nn as nn
from sacremoses import MosesDetokenizer, MosesPunctNormalizer, MosesTokenizer
from subword_nmt.apply_bpe import BPE
from torch import Tensor

from decoder import triu_mask
from model import Model

# def similarity(a: str, b: str) -> float:
#     return SequenceMatcher(None, a, b).ratio()


def noisify(text: str) -> str:
    i, chars = random.choice(range(len(text))), list(text)
    if i == 0 and chars[0].isupper():
        chars[i] = random.choice([chr(i) for i in range(65, 91)] + list('ÄÖÜ'))
    else:
        alphabet = [chr(i) for i in range(97, 123)] + list('äöüß')
        match random.randint(1, 3 if i + 1 == len(chars) else 4):
            case 1:  # Insert
                chars.insert(i, random.choice(alphabet))
            case 2:  # Delete
                chars.pop(i)
            case 3:  # Replace
                chars[i] = random.choice(alphabet)
            case 4:  # Swap
                chars[i], chars[i + 1] = chars[i + 1], chars[i]
    return ''.join(chars)


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
            for (a, b), sense_spans in zip(lemmas, senses):
                for c, d in sense_spans:
                    # only lemmas can attend to their senses
                    dict_mask[0, i, :, c:d] = 1.0
                    dict_mask[0, i, a:b, c:d] = 0.0
                    dict_mask[0, i, c:d, c:d] = 0.0
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
        self.normalizer = MosesPunctNormalizer()
        self.tokenizer = MosesTokenizer(src_lang)
        lang = tgt_lang if tgt_lang else src_lang
        self.detokenizer = MosesDetokenizer(lang)

    def tokenize(self, text: str) -> str:
        text = self.normalizer.normalize(text)
        tokens = self.tokenizer.tokenize(text)
        return self.bpe.process_line(' '.join(tokens))

    def detokenize(self, tokens: list[str]) -> str:
        text = self.detokenizer.detokenize(tokens)
        return re.sub('(@@ )|(@@ ?$)', '', text)


class Lemmatizer:
    def __init__(self, model: str):
        self.nlp = spacy.load(model, enable=['tok2vec', 'tagger', 'lemmatizer'])

    @staticmethod
    def subword_mapping(texts):
        for text in texts:
            words, spans = '', []
            for j, subword in enumerate(text):
                if subword.endswith('@@'):
                    words += subword.rstrip('@@')
                else:
                    words += subword + ' '
                    spans.append(j + 2)
            yield words.rstrip(), spans

    def lemmatize(self, texts):
        texts = list(self.subword_mapping(texts))
        docs = self.nlp.pipe(texts, as_tuples=True)
        for (words, spans), (doc, _) in zip(texts, docs):
            if words.split() == [token.text for token in doc]:
                yield [token.lemma_ for token in doc], spans
            else:
                yield words.split(), spans


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
    lemmatize: int
    append_dict: int
    word_dropout: float
    exp_function: str
    noise_level: float
    threshold: int
    max_senses: int

    def __init__(
        self,
        src_lang: str,
        tgt_lang: str,
        config: dict,
        device: str,
        model_file: str,
        vocab_file: str | list[str],
        codes_file: str | list[str],
        dict_file=None,
        freq_file=None,
        lem_data_file=None,
        lem_test_file=None,
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
            self.exp_function if dict_file else None,
        ).to(device)

        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

        self.model.apply(init_weights)

        self.dict = None
        if dict_file:
            with open(dict_file) as file:
                self.dict = json.load(file)
        self.dict_file = dict_file

        self.freq = None
        if freq_file:
            self.freq = {}
            with open(freq_file) as file:
                for line in file:
                    word, freq = line.split()
                    self.freq[word] = int(freq)
        self.freq_file = freq_file

        self.lem_data = None
        if lem_data_file:
            self.lem_data = []
            with open(lem_data_file) as file:
                for line in file:
                    words, spans = line.split('\t')
                    self.lem_data.append([words.split(), list(map(int, spans.split()))])
        self.lem_data_file = lem_data_file

        self.lem_test = None
        if lem_test_file:
            self.lem_test = []
            with open(lem_test_file) as file:
                for line in file:
                    words, spans = line.split('\t')
                    self.lem_test.append([words.split(), list(map(int, spans.split()))])
        self.lem_test_file = lem_data_file

        self.data: list[Batch] | None = None
        self.data_file = data_file

        self.test: list[Batch] | None = None
        self.test_file = test_file

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

    def _attach_senses(self, src_words, src_spans, tokenizer):
        lemma_start, common_words = 1, []
        for i, (lemma, lemma_end) in enumerate(zip(*src_spans)):
            headword = word = ''
            for j in range(lemma_start, lemma_end):
                word += src_words[j].rstrip('@@')

            if word in self.dict:
                if word in self.freq and self.freq[word] > self.threshold:
                    headword = word
            elif self.lemmatize and lemma in self.dict:
                if lemma in self.freq and self.freq[lemma] > self.threshold:
                    headword = lemma

            if headword:
                lemma_span = (lemma_start, lemma_end)
                common_words.append((i, word, headword, lemma_span))

            lemma_start = lemma_end

        if not common_words:
            return None, None
        k, word, headword, lemma_span = random.choice(common_words)
        word = tokenizer.tokenize(noisify(word)).split()
        lemma_start, lemma_end = lemma_span
        shift = len(word) - (lemma_end - lemma_start)
        if len(src_words) + shift > self.max_length:
            return None, None
        src_words[lemma_start:lemma_end] = word
        lemma_end = lemma_start + len(word)

        sense_start = len(src_words)
        defs = self.dict[headword][: self.max_senses]
        defs = [sb for w in defs for sb in w.split()]
        sense_end = sense_start + len(defs)
        if sense_end > self.max_length:
            return None, None

        lemma_span = (lemma_start, lemma_end)
        sense_span = (sense_start, sense_end)
        src_words.extend(defs)

        for j, lemma_end in enumerate(src_spans[1][k:]):
            src_spans[1][j + k] = lemma_end + shift

        ##= Unit Test =###
        # lemma_start = 1
        # for lemma, lemma_end in zip(*src_spans):
        #     print(lemma, src_words[lemma_start:lemma_end])
        #     lemma_start = lemma_end
        ##################

        return lemma_span, sense_span

    def attach_senses(self, src_words, src_spans, tokenizer=None):
        lemmas, senses = [], []
        if tokenizer and random.random() <= self.noise_level:
            raise NotImplementedError('noise-level out-of-date')
        #     src_spans = copy.deepcopy(src_spans)
        #     lemma_span, sense_span = self._attach_senses(src_words, src_spans, tokenizer)
        #     if lemma_span and sense_span:
        #         lemmas.append(lemma_span)
        #         senses.append(sense_span)

        lemma_start = 1
        for lemma, lemma_end in zip(*src_spans):
            word = ''
            for i in range(lemma_start, lemma_end):
                word += src_words[i].rstrip('@@')

            headword = (
                word
                if word in self.dict
                and (word not in self.freq or self.freq[word] <= self.threshold)
                else (
                    lemma
                    if self.lemmatize
                    and lemma in self.dict
                    and (lemma not in self.freq or self.freq[lemma] <= self.threshold)
                    else ''
                )
            )

            if len(headword) > 0:
                defs = self.dict[headword][: self.max_senses]
                sense_start, sense_spans = len(src_words), []
                for w in defs:
                    sense_end = sense_start + len(w.split())
                    sense_spans.append((sense_start, sense_end))
                    sense_start = sense_end
                if sense_end <= self.max_length:
                    for w in defs:
                        src_words.extend(w.split())
                    lemmas.append((lemma_start, lemma_end))
                    senses.append(sense_spans)

                    ##= Unit Test =###
                    # word = ''
                    # for j in range(lemma_start, lemma_end):
                    #     word += src_words[j].rstrip('@@')
                    # assert similarity(word, lemma) >= 0.5
                    ##################

            lemma_start = lemma_end

        return lemmas, senses

    def append_dict_data(self, data_file, tokenizer):
        data = []
        with open(data_file) as file:
            for headword, senses in json.load(file).items():
                src_words = tokenizer.tokenize(''.join(headword)).split()
                src_words = ['<BOS>'] + src_words + ['<EOS>']
                for sense in senses[: self.max_senses]:
                    src_words += sense.split()
                    tgt_words = ['<BOS>'] + sense.split() + ['<EOS>']
                    if len(src_words) <= self.max_length and len(tgt_words) <= self.max_length:
                        data.append((src_words, tgt_words, [], []))
        return data

    def batch_data(self, data) -> list[Batch]:
        batched_data = []

        data.sort(key=lambda x: (len(x[0]), len(x[1])), reverse=True)

        i = batch_size = 0
        while (i := i + batch_size) < len(data):
            src_len = len(data[i][0])
            tgt_len = len(data[i][1])

            while True:
                batch_size = min(self.batch_size // (max(src_len, tgt_len) * 8) * 8, 1000)

                if self.dict:
                    src_batch, tgt_batch, lemmas, senses = zip(*data[i : (i + batch_size)])
                else:
                    src_batch, tgt_batch = zip(*data[i : (i + batch_size)])
                dict_data = list(zip(lemmas, senses)) if self.dict else None

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
                    dict_data,
                )
            )

        return batched_data

    def load_data(self, data_file, src_spans=None, append_data=None, tokenizer=None):
        data = []
        with open(data_file) as file:
            for i, line in enumerate(file.readlines()):
                src_line, tgt_line = line.split('\t')
                src_words = ['<BOS>'] + src_line.split() + ['<EOS>']
                tgt_words = ['<BOS>'] + tgt_line.split() + ['<EOS>']

                if self.dict:
                    lemmas, senses = self.attach_senses(src_words, src_spans[i], tokenizer)
                    data.append((src_words, tgt_words, lemmas, senses))
                else:
                    data.append((src_words, tgt_words))

        if append_data:
            data.extend(append_data)

        return self.batch_data(data)
