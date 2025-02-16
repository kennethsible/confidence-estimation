import json
import math
from itertools import chain

import spacy
import torch
from torch import Tensor

from translation.decoder import beam_search, greedy_search, triu_mask
from translation.manager import Batch, Lemmatizer, Manager, Tokenizer


def conf_gradient(
    manager: Manager, src_words: list[str], out_prob: Tensor, src_embs: Tensor
) -> list[tuple[str, float]]:
    order, accum = 1, 'sum'
    if 'order' in manager.config:
        order = manager.config['order']
    if 'accum' in manager.config:
        accum = manager.config['accum']

    gradient = torch.autograd.grad(out_prob, src_embs)[0]
    partials = gradient.squeeze(0).norm(p=order, dim=1)

    word, scores, conf_list = '', [], []
    for subword, partial in zip(src_words, partials):
        word += subword.removesuffix('@@')
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

    return conf_list


def conf_attention(
    manager: Manager, src_words: list[str], out_probs: Tensor
) -> list[tuple[str, float]]:
    accum = 'sum'
    if 'accum' in manager.config:
        accum = manager.config['accum']

    layers = manager.model.decoder.layers
    scores = sum(layer.crss_attn.scores.sum(dim=1) for layer in layers)
    posteriors = out_probs[1:-1].abs() @ scores[0, 1:]

    # alignments: dict[str, list[str]] = {}
    # posteriors = torch.zeros(len(src_words), device=manager.device)
    # for i, (subword, out_prob) in enumerate(zip(out_words, out_probs[1:-1])):
    #     index = scores[0, i, 1:-1].argmax().item() + 1
    #     src_word_aligned = src_words[index]
    #     if src_word_aligned not in alignments:
    #         alignments[src_word_aligned] = []
    #     posteriors[index] += abs(out_prob.item())
    #     alignments[src_word_aligned].append(subword)
    # print(json.dumps(alignments, index=4))

    word, scores, conf_list = '', [], []
    for subword, posterior in zip(src_words, posteriors):
        word += subword.removesuffix('@@')
        scores.append(posterior.item())
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

    return conf_list


def translate(string: str, manager: Manager, *, conf_method: str | None = None) -> tuple[str, list]:
    model, vocab, device = manager.model, manager.vocab, manager.device
    beam_size, max_length = manager.beam_size, math.floor(manager.max_length * 1.3)
    tokenizer = Tokenizer(manager.src_lang, manager.tgt_lang, manager.sw_model)
    src_words, conf_list = ['<BOS>'] + tokenizer.tokenize(string) + ['<EOS>'], []

    model.eval()
    if conf_method is not None:
        match conf_method:
            case 'gradient':
                src_nums = torch.tensor(vocab.numberize(src_words), device=device)
                src_encs, src_embs = model.encode(src_nums.unsqueeze(0))
                out_nums, out_prob = greedy_search(manager, src_encs, max_length)
                conf_list = conf_gradient(manager, src_words, out_prob, src_embs)
                del src_nums, src_embs, src_encs, out_prob
            case 'attention':
                with torch.no_grad():
                    src_nums = torch.tensor(vocab.numberize(src_words), device=device)
                    src_encs, src_embs = model.encode(src_nums.unsqueeze(0))
                    out_nums, out_probs = beam_search(
                        manager, src_encs, beam_size, max_length, cumulative=False
                    )
                    tgt_mask = triu_mask(len(out_nums) - 1, device=device)
                    model.decode(
                        src_encs, out_nums[:-1].unsqueeze(0), tgt_mask=tgt_mask, store_attn=True
                    )
                    conf_list = conf_attention(manager, src_words, out_probs)
                del src_nums, src_embs, src_encs, out_probs
            case _:
                raise NotImplementedError(conf_method)

    with torch.no_grad():
        if manager.dict and manager.freq and manager.spacy_model:
            lemmatizer = Lemmatizer(manager.spacy_model, manager.sw_model)
            lem_data = next(lemmatizer.lemmatize([src_words[1:-1]]))
            _conf_list = None if conf_method is None else conf_list[1:-1]
            if 'span_mode' in manager.config and manager.config['span_mode'] == 2:
                src_spans, tgt_spans = manager.append_defs_2(
                    src_words, list(zip(*lem_data)), _conf_list
                )  # supports space-separated words and phrases
            else:
                src_spans, tgt_spans = manager.append_defs_1(
                    src_words, list(zip(*lem_data)), _conf_list
                )
            src_nums = torch.tensor(vocab.numberize(src_words), device=device)
            dict_data = list(zip([src_spans], [tgt_spans]))
            if manager.dpe_embed:
                src_encs, _ = model.encode(
                    src_nums.unsqueeze(0), dict_mask=None, dict_data=dict_data
                )
            else:
                mask_size = src_nums.unsqueeze(-2).size()
                dict_mask = Batch.dict_mask_from_data(dict_data, mask_size, device)
                src_encs, _ = model.encode(
                    src_nums.unsqueeze(0), dict_mask=dict_mask, dict_data=None
                )
            out_nums, _ = beam_search(manager, src_encs, beam_size, max_length)
        elif conf_method is None:
            src_nums = torch.tensor(vocab.numberize(src_words), device=device)
            src_encs, _ = model.encode(src_nums.unsqueeze(0))
            out_nums, _ = beam_search(manager, src_encs, beam_size, max_length)

    return tokenizer.detokenize(vocab.denumberize(out_nums.tolist())), conf_list


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dict', metavar='FILE_PATH', help='bilingual dictionary')
    parser.add_argument('--freq', metavar='FILE_PATH', help='frequency statistics')
    parser.add_argument(
        '--conf', nargs=2, metavar=('CONF_TYPE', 'FILE_PATH'), help='confidence scores'
    )
    parser.add_argument('--spacy-model', metavar='FILE_PATH', help='spaCy model')
    parser.add_argument('--sw-vocab', metavar='FILE_PATH', required=True, help='subword vocab')
    parser.add_argument('--sw-model', metavar='FILE_PATH', required=True, help='subword model')
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
        args.sw_vocab,
        args.sw_model,
        args.dict,
        args.freq,
        args.spacy_model,
    )
    manager.model.load_state_dict(model_state['state_dict'])

    if device == 'cuda' and torch.cuda.get_device_capability()[0] >= 8:
        torch.set_float32_matmul_precision('high')

    if args.conf:
        json_list = []
        with open(args.input) as data_f:
            for string in data_f.readlines():
                if args.spacy_model:
                    sents = spacy.load(args.spacy_model)(string).sents
                    outputs, conf_lists = zip(
                        *(translate(sent, manager, conf_method=args.conf[0]) for sent in sents)
                    )
                    output, conf_list = ' '.join(outputs), list(chain.from_iterable(conf_lists))
                else:
                    output, conf_list = translate(string, manager, conf_method=args.conf[0])
                json_list.append(conf_list)
                print(output)
        with open(args.conf[1], 'w') as json_f:
            json.dump(json_list, json_f)
    else:
        with open(args.input) as data_f:
            for string in data_f.readlines():
                if args.spacy_model:
                    sents = spacy.load(args.spacy_model)(string).sents
                    output = ' '.join(translate(sent, manager)[0] for sent in sents)
                else:
                    output, _ = translate(string, manager)
                print(output)


if __name__ == '__main__':
    main()
