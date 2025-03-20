import pickle

import numpy as np
import numpy.typing as npt
from scipy.spatial.distance import cosine
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from translation.manager import Manager, Tokenizer


class KNNModel:
    def __init__(self, manager: Manager, freq_file: str, *, n_neighbors: int):
        self.manager = manager
        self.n_neighbors = n_neighbors

        self.freq = {}
        with open(freq_file) as freq_f:
            for line in freq_f.readlines():
                word, freq = line.split()
                self.freq[word] = float(freq)
        self.tokenizer = Tokenizer(manager.src_lang, sw_model=manager.sw_model)

    def _metric(self, u: npt.NDArray, v: npt.NDArray, a: int = 1, b: int = 1) -> npt.NDArray:
        # return np.dot(u[:-1], v[:-1])

        # return minkowski(u[:-1], v[:-1])

        # if u[-1:] == v[-1:]:
        #     return a * minkowski(u[:-1], v[:-1])
        # return a * minkowski(u[:-1], v[:-1]) + b / abs(u[-1:] - v[-1:])

        # return cosine(u[:-1], v[:-1])

        if u[-1:] == v[-1:]:
            return a * cosine(u[:-1], v[:-1]) + b
        return a * cosine(u[:-1], v[:-1]) + b / abs(u[-1:] - v[-1:])

    def fit(self):
        model, vocab = self.manager.model, self.manager.vocab
        self.nn_model = NearestNeighbors(n_neighbors=self.n_neighbors, metric=self._metric)

        embs = []
        for word in tqdm(self.freq, bar_format='Fitting Model {desc}{percentage:3.0f}%|{bar:10}'):
            subwords = self.tokenizer.tokenize(word)
            A = model.out_embed(vocab.numberize(subwords)).detach().numpy()
            # A[np.linalg.norm(A, axis=1).argmax()]
            embs.append(np.append(np.mean(A, axis=0), self.freq[word]))
        self.nn_model.fit(np.array(embs))

    def kneighbors(self, word: str) -> list[str]:
        model, vocab = self.manager.model, self.manager.vocab
        subwords = self.tokenizer.tokenize(word)
        A = model.out_embed(vocab.numberize(subwords)).detach().numpy()
        # if word not in self.freq:
        #     emb = np.append(np.mean(A, axis=0), 1e-9)
        emb = np.append(np.mean(A, axis=0), self.freq.get(word, 0))
        indices = self.nn_model.kneighbors(emb[None, ...], return_distance=False)
        return [list(self.freq.keys())[index] for index in indices[0]]

    def save(self, model_file: str):
        with open(model_file, 'wb') as model_f:
            pickle.dump(self.nn_model, model_f)

    def load(self, model_file: str):
        with open(model_file, 'rb') as model_f:
            self.nn_model = pickle.load(model_f)
