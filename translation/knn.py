import pickle

import numpy as np
from scipy.spatial.distance import cosine
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from translation.manager import Manager, Tokenizer


class KNNModel:
    def __init__(self, manager: Manager, freq_file: str, *, n_neighbors: int):
        self.manager = manager
        self.tokenizer = Tokenizer(manager.src_lang, sw_model=manager.sw_model)
        with open(freq_file) as freq_f:
            self.freq = {}
            for line in freq_f.readlines():
                word, freq = line.split()
                self.freq[word] = float(freq)
            self.vocab = list(self.freq.keys())
        self.n_neighbors = n_neighbors

    def save(self, model_file: str):
        with open(model_file, 'wb') as model_f:
            pickle.dump(self.nbrs, model_f)

    def load(self, model_file: str):
        with open(model_file, 'rb') as model_f:
            self.nbrs = pickle.load(model_f)

    @staticmethod
    def cosine_similarity(u: np.ndarray, v: np.ndarray, a: int, b: int) -> float:
        # return cosine(u[:-1], v[:-1]) / (1 + np.log(1 + v[-1]))
        return cosine(u[:-1], v[:-1]) + a / (v[-1] + b)

    def fit(self):
        model, vocab = self.manager.model, self.manager.vocab
        self.nbrs = NearestNeighbors(
            n_neighbors=self.n_neighbors + 1,
            algorithm='ball_tree',
            metric=self.cosine_similarity,
            metric_params={'a': 10.0, 'b': 1e-6},
        )
        word_embeds = []
        for word in tqdm(self.vocab, bar_format='Fitting Model {desc}{percentage:3.0f}%|{bar:10}'):
            subwords = self.tokenizer.tokenize(word)
            subword_embeds = model.out_embed(vocab.numberize(subwords)).detach().numpy()
            word_embeds.append(np.append(np.mean(subword_embeds, axis=0), self.freq[word]))
        self.nbrs.fit(np.array(word_embeds))

    def kneighbors(self, word: str) -> list[str]:
        model, vocab = self.manager.model, self.manager.vocab
        tokens = self.tokenizer.tokenize(word)
        subword_embeds = model.out_embed(vocab.numberize(tokens)).detach().numpy()
        word_embed = np.append(np.mean(subword_embeds, axis=0), self.freq.get(word, 0.0))
        indices = self.nbrs.kneighbors(word_embed[None, ...], return_distance=False)[0]
        return [self.vocab[index] for index in indices if self.vocab[index] != word][
            : self.n_neighbors
        ]
