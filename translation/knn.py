import faiss
import numpy as np
import torch
from tqdm import tqdm

from translation.manager import Manager, Tokenizer


class KNNModel:
    def __init__(self, manager: Manager, vocab_file: str):
        self.faiss_index: faiss.IndexFlat
        self.faiss_vocab: list[str] = []
        with open(vocab_file) as vocab_f:
            for line in vocab_f.readlines():
                word, _ = line.split()
                self.faiss_vocab.append(word)

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
            faiss.normalize_L2(word_embs[-1])  # cosine similarity

        emb_dim = word_embs[-1].shape[-1]
        self.faiss_index = faiss.IndexFlatIP(emb_dim)
        self.faiss_index.add(np.vstack(word_embs))

    def search(
        self, word: str, *, n_neighbors: int, restrict_vocab: int | None = None
    ) -> list[str]:
        if restrict_vocab and restrict_vocab < self.faiss_index.ntotal:
            faiss_index = faiss.IndexFlatIP(self.faiss_index.d)
            faiss_index.add(
                np.vstack([self.faiss_index.reconstruct(i) for i in range(restrict_vocab)])
            )
        else:
            faiss_index = self.faiss_index

        subword_nums = self._numberize(word).unsqueeze(0)
        with torch.no_grad():
            subword_embs, _ = self.nmt_model.encode(subword_nums)
        word_emb = subword_embs.mean(dim=1).detach().numpy()
        faiss.normalize_L2(word_emb)  # cosine similarity

        _, indices = faiss_index.search(word_emb, n_neighbors * 10)
        neighbors = [
            self.faiss_vocab[index]
            for index in indices[0]
            if self.faiss_vocab[index].lower() != word.lower()
        ]
        return neighbors[:n_neighbors]
