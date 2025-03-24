import faiss
import numpy as np
import torch
from tqdm import tqdm

from translation.manager import Manager, Tokenizer


class KNNModel:
    def __init__(self, manager: Manager, freq_file: str, *, n_neighbors: int):
        self.faiss_index: faiss.IndexFlat
        with open(freq_file) as freq_f:
            self.freq: dict[str, float] = {}
            for line in freq_f.readlines():
                word, freq = line.split()
                self.freq[word] = float(freq)
        self.faiss_vocab: list[str] = list(self.freq.keys())
        self.n_neighbors = n_neighbors

        self.tokenizer = Tokenizer(manager.src_lang, sw_model=manager.sw_model)
        self.nmt_model = manager.model
        self.nmt_model.eval()
        self.nmt_vocab = manager.vocab

    def save(self, model_file: str):
        faiss.write_index(self.faiss_index, model_file)

    def load(self, model_file: str):
        self.faiss_index = faiss.read_index(model_file)

    def _numberize(self, word: str) -> torch.Tensor:
        return torch.tensor(self.nmt_vocab.numberize(self.tokenizer.tokenize(word)))

    def build_index(self):
        word_embs = []
        for word in tqdm(self.faiss_vocab):
            subword_nums = self._numberize(word).unsqueeze(0)
            with torch.no_grad():
                subword_embs, _ = self.nmt_model.encode(subword_nums)
            word_embs.append(subword_embs.mean(dim=1).detach().numpy())

        emb_dim = word_embs[-1].shape[-1]
        self.faiss_index = faiss.IndexFlatIP(emb_dim)
        self.faiss_index.add(np.vstack(word_embs))

    def search(self, word: str) -> list[str]:
        subword_nums = self._numberize(word).unsqueeze(0)
        with torch.no_grad():
            subword_embs, _ = self.nmt_model.encode(subword_nums)
        word_emb = subword_embs.mean(dim=1).detach().numpy()

        _, indices = self.faiss_index.search(word_emb, self.n_neighbors + 50)
        neighbors = [self.faiss_vocab[index] for index in indices[0]]
        neighbors_filtered = [nbr for nbr in neighbors if nbr != word and self.freq[nbr] > 15]

        if len(neighbors_filtered) < self.n_neighbors:
            remaining = [nbr for nbr in neighbors if nbr not in neighbors_filtered]
            neighbors_filtered.extend(remaining[: self.n_neighbors - len(neighbors_filtered)])
        return neighbors_filtered[: self.n_neighbors]
