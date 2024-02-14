import json
import math
import re
from collections import Counter
from io import StringIO

import nltk
import spacy
import torch
import torch.nn as nn
from sacremoses import MosesDetokenizer, MosesPunctNormalizer, MosesTokenizer
from subword_nmt.apply_bpe import BPE
from torch import Tensor
from tqdm import tqdm

from translation.decoder import triu_mask
from translation.model import Model


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
        for i, (src_spans, tgt_spans) in enumerate(dict_data):
            for (a, b), spans in zip(src_spans, tgt_spans):
                for c, d in spans:
                    # only headwords can attend to their definitions
                    dict_mask[0, i, :, c:d] = 1.0
                    dict_mask[0, i, a:b, c:d] = 0.0
                    dict_mask[0, i, c:d, c:d] = 0.0
                    # definitions can only attend to themselves
                    dict_mask[1, i, c:d, :] = 1.0
                    dict_mask[1, i, c:d, a:b] = 0.0
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
    def __init__(self, codes: BPE, src_lang: str, tgt_lang: str | None = None):
        self.codes = codes
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.normalizer = MosesPunctNormalizer(src_lang)
        self.tokenizer = MosesTokenizer(src_lang)
        lang = tgt_lang if tgt_lang else src_lang
        self.detokenizer = MosesDetokenizer(lang)

    def tokenize(self, text: str, dropout: int = 0) -> str:
        text = self.normalizer.normalize(text)
        tokens = self.tokenizer.tokenize(text, escape=False)
        return self.codes.process_line(' '.join(tokens), dropout)

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
                    words += subword.replace('@@', '')
                else:
                    words += subword + ' '
                    spans.append(j + 2)
            yield words.rstrip(), spans

    def lemmatize(self, texts):
        _texts = list(self.subword_mapping(texts))
        docs = self.nlp.pipe(_texts, as_tuples=True)
        for (words, spans), (doc, _) in zip(_texts, docs):
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
    max_patience: int
    label_smoothing: float
    clip_grad: float
    batch_size: int
    max_length: int
    len_ratio: int
    beam_size: int
    threshold: int
    max_append: int
    dpe_embed: int

    def __init__(
        self,
        config: dict,
        device: str,
        src_lang: str,
        tgt_lang: str,
        model_file: str,
        vocab_file: str | list[str],
        codes_file: str | list[str],
        dict_file: str | None = None,
        freq_file: str | None = None,
        samples_file: str | Counter[str] | None = None,
        bigrams_file: str | Counter[str] | None = None,
    ):
        self.config = config
        self.device = device
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self._model_name = model_file
        self._vocab_list = vocab_file
        self._codes_list = codes_file
        self._samples_counter = samples_file
        self._bigrams_counter = bigrams_file

        for option, value in config.items():
            self.__setattr__(option, value)

        if isinstance(self._vocab_list, str):
            with open(self._vocab_list) as vocab_f:
                self._vocab_list = list(vocab_f.readlines())
        self.vocab = Vocab(self._vocab_list)

        if isinstance(self._codes_list, str):
            with open(self._codes_list) as codes_f:
                self._codes_list = list(codes_f.readlines())
        self.codes = BPE(StringIO(''.join(self._codes_list)))

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

        self.freq: dict[str, int] = {}
        if freq_file:
            with open(freq_file) as freq_f:
                for line in freq_f:
                    word, freq = line.split()
                    self.freq[word] = int(freq)

        if isinstance(self._samples_counter, str):
            with open(self._samples_counter) as samples_f:
                samples = [
                    sample
                    for samples in samples_f.readlines()
                    for sample in samples.split('\t')[0].split()
                ]
                self._samples_counter = Counter(samples)
                self._bigrams_counter = Counter(nltk.bigrams(samples))
        self.unigram = nltk.FreqDist(self._samples_counter)
        self.bigram = nltk.FreqDist(self._bigrams_counter)

        self.tokenizer = Tokenizer(self.codes, src_lang, tgt_lang)
        self.lemmatizer = Lemmatizer('de_core_news_sm')

    def save_model(self):
        torch.save(
            {
                'config': self.config,
                'src_lang': self.src_lang,
                'tgt_lang': self.tgt_lang,
                'vocab_list': self._vocab_list,
                'codes_list': self._codes_list,
                'samples_counter': self._samples_counter,
                'bigrams_counter': self._bigrams_counter,
                'state_dict': self.model.state_dict(),
            },
            self._model_name,
        )

    def append_defs(self, src_words: list[str], lem_data):
        src_spans, tgt_spans = [], []

        headwords = []
        src_start = 1
        # src_start, total_shift = 1, 0
        for lemma, src_end in zip(*lem_data):
            # src_end += total_shift

            headword = ''
            if 'pmi_threshold' in self.config:
                pmi_threshold: int = self.config['pmi_threshold']

                word = src_words[src_start].replace('@@', '')
                word_pmi = 0.0
                for i in range(src_start + 1, src_end):
                    word += src_words[i].replace('@@', '')
                    try:
                        word_pmi += (
                            math.log(self.bigram.freq((src_words[i - 1], src_words[i])))
                            - math.log(self.unigram.freq(src_words[i - 1]))
                            - math.log(self.unigram.freq(src_words[i]))
                        )
                    except ValueError:
                        pass
                if src_start + 1 == src_end:
                    try:
                        word_pmi = math.log(self.unigram.freq(word))
                    except ValueError:
                        pass
                if word in self.dict:
                    if word_pmi >= pmi_threshold:
                        headword = word
                elif lemma in self.dict and 'lemmatize' in self.config and self.config['lemmatize']:
                    src_lemmas = self.tokenizer.tokenize(lemma).split()  # ! BOTTLENECK
                    lemma_pmi = 0.0
                    for i in range(1, len(src_lemmas)):
                        try:
                            lemma_pmi += (
                                math.log(self.bigram.freq((src_lemmas[i - 1], src_lemmas[i])))
                                - math.log(self.unigram.freq(src_lemmas[i - 1]))
                                - math.log(self.unigram.freq(src_lemmas[i]))
                            )
                        except ValueError:
                            pass
                    if len(src_lemmas) == 1:
                        try:
                            lemma_pmi = math.log(self.unigram.freq(lemma))
                        except ValueError:
                            pass
                    if lemma_pmi >= pmi_threshold:
                        headword = lemma

                #     headword = (
                #         word
                #         if word in self.dict and word_pmi >= pmi_threshold
                #         else lemma if lemma in self.dict and lemma_pmi >= pmi_threshold else ''
                #     )
                # else:
                #     headword = word if word in self.dict and word_pmi >= pmi_threshold else ''

                # print(
                #     (
                #         ' '.join([src_words[i] for i in range(src_start, src_end)]),
                #         word,
                #         int(word_pmi),
                #     ),
                #     '\t',
                #     (lemma, int(lemma_pmi)),
                # )  # ! DEBUG
            else:
                word = ''
                for i in range(src_start, src_end):
                    word += src_words[i].rstrip('@')

                # headword = (
                #     word
                #     if word in self.dict
                #     and (word not in self.freq or self.freq[word] <= self.threshold)
                #     else (
                #         lemma
                #         if lemma in self.dict
                #         and (lemma not in self.freq or self.freq[lemma] <= self.threshold)
                #         else ''
                #     )
                # )

                if word in self.dict:
                    if word not in self.freq or self.freq[word] <= self.threshold:
                        headword = word
                elif lemma in self.dict:
                    if lemma not in self.freq or self.freq[lemma] <= self.threshold:
                        headword = lemma

            if len(headword) > 0:
                # subwords = self.tokenizer.tokenize(word).split()
                # shift = len(subwords) - (src_end - src_start)
                # if len(src_words) + shift > self.max_length:
                #     src_start = src_end
                #     continue
                headwords.append(headword)
                # total_shift += shift
                # src_words[src_start:src_end] = subwords
                # src_end = src_start + len(subwords)
                src_spans.append((src_start, src_end))

            src_start = src_end

        for i, headword in enumerate(headwords):
            definitions = self.dict[headword][: self.max_append]
            tgt_start, spans = len(src_words), []
            for definition in definitions:
                tgt_end = tgt_start + len(definition.split())
                spans.append((tgt_start, tgt_end))
                tgt_start = tgt_end
            if tgt_end > self.max_length:
                src_spans = src_spans[:i]
                break
            for definition in definitions:
                src_words.extend(definition.split())
            tgt_spans.append(spans)

        # exit()  # ! DEBUG
        return src_spans, tgt_spans

    def batch_data(self, data) -> list[Batch]:
        batched_data = []

        data.sort(key=lambda x: (len(x[0]), len(x[1])), reverse=True)

        i = batch_size = 0
        while (i := i + batch_size) < len(data):
            src_len = len(data[i][0])
            tgt_len = len(data[i][1])

            while True:
                batch_size = min(self.batch_size // (max(src_len, tgt_len) * 8) * 8, 1000)
                src_batch, tgt_batch, src_spans, tgt_spans = zip(*data[i : (i + batch_size)])
                max_src_len = max(len(src_words) for src_words in src_batch)
                max_tgt_len = max(len(tgt_words) for tgt_words in tgt_batch)

                if src_len >= max_src_len and tgt_len >= max_tgt_len:
                    dict_data = list(zip(src_spans, tgt_spans))
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

            batched_data.append(Batch(src_nums, tgt_nums, self.vocab.PAD, self.device, dict_data))

        return batched_data

    def load_data(
        self, data_file: str, lem_file: str | None = None, dict_file: str | None = None
    ) -> list[Batch]:
        lem_data = []
        if lem_file:
            with open(lem_file) as lem_f:
                for line in lem_f.readlines():
                    words, spans = line.split('\t')
                    lem_data.append([words.split(), list(map(int, spans.split()))])

        data = []
        # count = total = 0
        with open(data_file) as data_f:
            for i, line in enumerate(tqdm(data_f.readlines())):
                src_line, tgt_line = line.split('\t')
                src_words = ['<BOS>'] + src_line.split() + ['<EOS>']
                tgt_words = ['<BOS>'] + tgt_line.split() + ['<EOS>']
                src_spans, tgt_spans = [], []
                if lem_data and self.dict and self.freq:
                    src_spans, tgt_spans = self.append_defs(src_words, lem_data[i])
                    # if any(src_spans):
                    #     count += 1
                data.append((src_words, tgt_words, src_spans, tgt_spans))
                # total += 1
            # print(f'{(count / total * 100):.2f}')
            # exit()  # ! DEBUG

        if dict_file:
            with open(dict_file) as dict_f:
                for headword, definitions in json.load(dict_f).items():
                    src_words = self.tokenizer.tokenize(''.join(headword)).split()
                    src_words = ['<BOS>'] + src_words + ['<EOS>']
                    for definition in definitions[: self.max_append]:
                        src_words += definition.split()
                        tgt_words = ['<BOS>'] + definition.split() + ['<EOS>']
                        if (
                            1 <= len(src_words) <= self.max_length
                            and 1 <= len(tgt_words) <= self.max_length
                            and len(src_words) / len(tgt_words) <= self.len_ratio
                            and len(tgt_words) / len(src_words) <= self.len_ratio
                        ):
                            data.append((src_words, tgt_words, [], []))

        return self.batch_data(data)
