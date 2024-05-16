import math

import faiss
import torch

from translation.decoder import greedy_search
from translation.manager import Batch, Manager


class ExactIndex:

    def __init__(self, vectors, labels):
        self.dimension = vectors.shape[1]
        self.vectors = vectors.astype('float32')
        self.labels = labels

    def build(self):
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(self.vectors)

    def query(self, vectors, k=10):
        _, indices = self.index.search(vectors, k)
        return [self.labels[i] for i in indices[0]]


def millify(x: int) -> str:
    y, abbrs = float(x), ['', 'K', 'M', 'B', 'T']
    index = max(0, min(len(abbrs) - 1, int(math.floor(0 if y == 0 else math.log10(abs(y)) / 3))))
    return '{:.0f}{}'.format(y / 10 ** (3 * index), abbrs[index])


def translate(string: str, manager: Manager, *, conf: bool = False):
    model, vocab, device = manager.model, manager.vocab, manager.device
    tokenizer, lemmatizer = manager.tokenizer, manager.lemmatizer
    src_words = ['<BOS>'] + tokenizer.tokenize(string).split() + ['<EOS>']

    model.eval()
    if manager.dict and manager.freq:
        lem_data = next(lemmatizer.lemmatize([src_words[1:-1]]))
        src_spans, tgt_spans = manager.append_defs(src_words, lem_data)
        src_nums = torch.tensor(vocab.numberize(src_words), device=device)
        dict_data = list(zip([src_spans], [tgt_spans]))
        if manager.dpe_embed:
            src_encs, src_embs = model.encode(
                src_nums.unsqueeze(0), dict_mask=None, dict_data=dict_data
            )
        else:
            mask_size = src_nums.unsqueeze(-2).size()
            dict_mask = Batch.dict_mask_from_data(dict_data, mask_size, device)
            src_encs, src_embs = model.encode(
                src_nums.unsqueeze(0), dict_mask=dict_mask, dict_data=None
            )
    else:
        src_nums = torch.tensor(vocab.numberize(src_words), device=device)
        src_encs, src_embs = model.encode(src_nums.unsqueeze(0))
    out_nums, out_prob = greedy_search(manager, src_encs, max_length=manager.max_length * 2)
    if not conf:
        print(tokenizer.detokenize(vocab.denumberize(out_nums.tolist())))
    else:
        print('HYP:', tokenizer.detokenize(vocab.denumberize(out_nums.tolist())), '\n')
        print('Conf.\tFreq.\tWord')
        print('=====\t=====\t=====')
        word, score = '', 0.0
        for subword, gradient in zip(
            src_words, torch.autograd.grad(out_prob, src_embs)[0].squeeze(0).abs().sum(dim=1)
        ):
            word += subword.rstrip('@')
            score += gradient.item()  # TODO average? maximum?
            if not subword.endswith('@@'):
                frequency = manager.freq[word] if word in manager.freq else 0
                print(f'{score:0.2f}\t{millify(frequency).ljust(4)}\t{word}')
                word, score = '', 0.0
        print()


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dict', metavar='FILE_PATH', help='bilingual dictionary')
    parser.add_argument('--freq', metavar='FILE_PATH', help='frequency statistics')
    parser.add_argument('--model', metavar='FILE_PATH', required=True, help='translation model')
    parser.add_argument('--conf', action='store_true', help='estimate confidence')
    parser.add_argument('--input', metavar='FILE_PATH', help='detokenized input')
    args, unknown = parser.parse_known_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_state = torch.load(args.model, map_location=device)

    config = model_state['config']
    for i, arg in enumerate(unknown):
        if arg[:2] == '--' and len(unknown) > i:
            option, value = arg[2:].replace('-', '_'), unknown[i + 1]
            try:
                config[option] = (int if value.isdigit() else float)(value)
            except ValueError:
                config[option] = value

    manager = Manager(
        config,
        device,
        model_state['src_lang'],
        model_state['tgt_lang'],
        args.model,
        model_state['vocab_list'],
        model_state['codes_list'],
        args.dict,
        args.freq,
        model_state['samples_counter'],
        model_state['bigrams_counter'],
    )
    manager.model.load_state_dict(model_state['state_dict'])

    if device == 'cuda' and torch.cuda.get_device_capability()[0] >= 8:
        torch.set_float32_matmul_precision('high')

    with open(args.input) as data_f:
        for string in data_f.readlines():
            if not args.conf:
                with torch.no_grad():
                    translate(string, manager, conf=args.conf)
            else:
                translate(string, manager, conf=args.conf)


if __name__ == '__main__':
    main()
