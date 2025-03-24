import pickle

import numpy as np
import torch
from scipy.spatial.distance import cosine
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from translation.manager import Manager, Tokenizer


class KNNModel:
    def __init__(self, manager: Manager, freq_file: str, *, n_neighbors: int):
        self.nn_model: NearestNeighbors
        with open(freq_file) as freq_f:
            self.freq: dict[str, float] = {}
            for line in freq_f.readlines():
                word, freq = line.split()
                self.freq[word] = float(freq)
            self.nn_vocab: list[str] = list(self.freq.keys())
        self.n_neighbors = n_neighbors
        self.tokenizer = Tokenizer(manager.src_lang, sw_model=manager.sw_model)
        self.nmt_model = manager.model
        self.nmt_vocab = manager.vocab
        self.nmt_model.eval()

    def save(self, model_file: str):
        with open(model_file, 'wb') as model_f:
            pickle.dump(self.nn_model, model_f)

    def load(self, model_file: str):
        with open(model_file, 'rb') as model_f:
            self.nn_model = pickle.load(model_f)

    @staticmethod
    def cosine_metric(u: np.ndarray, v: np.ndarray, a: float, b: float) -> float:
        return cosine(u[:-1], v[:-1]) + a / (v[-1] + b)

    def fit(self, a: float = 10.0, b: float = 1e-6):
        word_embs = []
        for word in tqdm(self.nn_vocab):
            subword_nums = self.nmt_vocab.numberize(self.tokenizer.tokenize(word))
            subword_embs, _ = self.nmt_model.encode(torch.tensor(subword_nums).unsqueeze(0))
            word_emb = subword_embs.squeeze(0).mean(dim=0).detach().numpy()
            word_embs.append(np.append(word_emb, self.freq[word]))

        self.nn_model = NearestNeighbors(
            n_neighbors=self.n_neighbors + 1,
            algorithm='ball_tree',
            metric=self.cosine_metric,
            metric_params={'a': a, 'b': b},
        )
        self.nn_model.fit(np.array(word_embs))

    def kneighbors(self, word: str) -> list[str]:
        subword_nums = self.nmt_vocab.numberize(self.tokenizer.tokenize(word))
        subword_embs, _ = self.nmt_model.encode(torch.tensor(subword_nums).unsqueeze(0))
        word_emb = subword_embs.squeeze(0).mean(dim=0).detach().numpy()
        word_emb = np.append(word_emb, self.freq.get(word, 0.0))[None, ...]
        indices = self.nn_model.kneighbors(word_emb, return_distance=False)[0]
        return [self.nn_vocab[i] for i in indices if self.nn_vocab[i] != word][: self.n_neighbors]
