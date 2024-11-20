import json

import torch

from translation.decoder import beam_search
from translation.manager import Batch, Manager


def translate(string: str, manager: Manager, *, confidence: bool = False) -> str | tuple[str, list]:
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
    out_nums, out_prob = beam_search(manager, src_encs, max_length=manager.max_length * 2)

    if confidence:
        order, accum = 1, 'sum'
        if 'order' in manager.config:
            order = manager.config['order']
        if 'accum' in manager.config:
            accum = manager.config['accum']
        gradient = torch.autograd.grad(out_prob, src_embs)[0]
        partials = gradient.squeeze(0).norm(p=order, dim=1)
        word, scores, conf_list = '', [], []
        for subword, partial in zip(src_words, partials):
            word += subword.rstrip('@')
            scores.append(partial.item())
            if not subword.endswith('@@'):
                match accum:
                    case 'sum':
                        score = sum(scores)
                    case 'avg':
                        score = sum(scores) / len(scores)
                    case 'max':
                        score = max(scores)
                conf_list.append((word, score))
                word, scores = '', []
        return tokenizer.detokenize(vocab.denumberize(out_nums.tolist())), conf_list

    out_nums, _ = beam_search(manager, src_encs, max_length=manager.max_length * 2)
    return tokenizer.detokenize(vocab.denumberize(out_nums.tolist()))


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dict', metavar='FILE_PATH', help='bilingual dictionary')
    parser.add_argument('--freq', metavar='FILE_PATH', help='frequency statistics')
    parser.add_argument('--conf', metavar='FILE_PATH', help='confidence scores')
    parser.add_argument('--model', metavar='FILE_PATH', required=True, help='translation model')
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

    if args.conf:
        json_list = []
        with open(args.input) as data_f:
            for string in data_f.readlines():
                output, conf_list = translate(string, manager, confidence=True)
                json_list.append(conf_list)
                print(output)
        with open(args.conf, 'w') as json_f:
            json.dump(json_list, json_f, indent=4)
    else:
        with open(args.input) as data_f:
            for string in data_f.readlines():
                with torch.no_grad():
                    print(translate(string, manager))


if __name__ == '__main__':
    main()
