import json
import math

import torch
from torch import Tensor

from translation.decoder import beam_search, triu_mask
from translation.manager import Batch, Lemmatizer, Manager, Tokenizer

sent_align: dict[int, list[int]] = {}


def conf_grad(
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


def conf_attn(manager: Manager, src_words: list[str], out_probs: Tensor) -> list[tuple[str, float]]:
    accum = 'sum'
    if 'accum' in manager.config:
        accum = manager.config['accum']

    layers = manager.model.decoder.layers
    scores = sum(layer.crss_attn.scores.sum(dim=1) for layer in layers)
    posteriors = out_probs[:-1].abs() @ scores[0, 1:]

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


def conf_giza(
    manager: Manager, src_words: list[str], out_probs: Tensor, out_words: list[str]
) -> list[tuple[str, float]]:

    words, src_spans = next(Lemmatizer.subword_mapping([src_words], manager.sw_model))
    _, out_spans = next(Lemmatizer.subword_mapping([out_words], manager.sw_model))

    out_probs = out_probs[:-1].abs()
    posteriors = [0.0] * len(src_spans)
    i = 0
    for tgt_i in sorted(sent_align):
        # edge case (j = -1) where final subwords are incomplete
        j = -1 if tgt_i >= len(out_spans) else out_spans[tgt_i] - 1
        for src_i in sent_align[tgt_i]:
            # if src:tgt is one-to-many or many-to-many, divide probability equally
            posteriors[src_i + 1] += out_probs[i:j].sum().item() / len(sent_align[tgt_i])
        i = j
    posteriors[0], posteriors[-1] = out_probs[0].abs().item(), out_probs[-1].abs().item()

    conf_list = []
    for word, score in zip(words.split(), posteriors):
        conf_list.append((word, score))

    return conf_list


def compute_probs(manager: Manager, src_encs: Tensor, out_nums: Tensor) -> Tensor:
    model, device = manager.model, manager.device
    tgt_mask = triu_mask(len(out_nums) - 1, device=device)
    tgt_encs = model.decode(src_encs, out_nums[:-1].unsqueeze(0), tgt_mask=tgt_mask)
    logits = model.out_embed(tgt_encs, inverse=True).squeeze(0)
    probs, indices = logits.softmax(dim=-1), out_nums[1:].view(-1, 1)
    return torch.gather(probs, dim=-1, index=indices).squeeze(-1)


def translate(
    string: str, manager: Manager, *, spacy_model: str | None = None, conf_type: str | None = None
) -> tuple[str, list]:
    model, vocab, device = manager.model, manager.vocab, manager.device
    beam_size, max_length = manager.beam_size, math.floor(manager.max_length * 1.3)
    tokenizer = Tokenizer(manager.src_lang, manager.tgt_lang, manager.sw_model)
    src_words, conf_list = ['<BOS>'] + tokenizer.tokenize(string) + ['<EOS>'], []

    model.eval()
    if conf_type is not None:
        match conf_type:
            case 'grad':
                src_nums = torch.tensor(vocab.numberize(src_words), device=device)
                src_encs, src_embs = model.encode(src_nums.unsqueeze(0))
                with torch.no_grad():
                    out_nums = beam_search(manager, src_encs, beam_size, max_length)
                out_prob = compute_probs(manager, src_encs, out_nums).sum(dim=-1)
                conf_list = conf_grad(manager, src_words, out_prob, src_embs)
                del src_nums, src_embs, src_encs, out_prob
            case 'attn':
                with torch.no_grad():
                    src_nums = torch.tensor(vocab.numberize(src_words), device=device)
                    src_encs, src_embs = model.encode(src_nums.unsqueeze(0))
                    out_nums = beam_search(manager, src_encs, beam_size, max_length)
                    out_probs = compute_probs(manager, src_encs, out_nums)
                    tgt_mask = triu_mask(len(out_nums) - 1, device=device)
                    model.decode(
                        src_encs, out_nums[:-1].unsqueeze(0), tgt_mask=tgt_mask, store_attn=True
                    )
                    conf_list = conf_attn(manager, src_words, out_probs)
                del src_nums, src_embs, src_encs, out_probs
            case 'giza':
                with torch.no_grad():
                    src_nums = torch.tensor(vocab.numberize(src_words), device=device)
                    src_encs, src_embs = model.encode(src_nums.unsqueeze(0))
                    out_nums = beam_search(manager, src_encs, beam_size, max_length)
                    out_probs = compute_probs(manager, src_encs, out_nums)
                    out_words = vocab.denumberize(out_nums.tolist())
                    conf_list = conf_giza(manager, src_words, out_probs, out_words)
                del src_nums, src_embs, src_encs, out_probs
            case _:
                raise NotImplementedError(conf_type)

    with torch.no_grad():
        if manager.dict and spacy_model:
            lemmatizer = Lemmatizer(spacy_model, manager.sw_model)
            lem_data = list(zip(*next(lemmatizer.lemmatize([src_words[1:-1]]))))
            if conf_type is None:
                src_spans, tgt_spans = manager.append_defs(src_words, lem_data)
            else:
                src_spans, tgt_spans = manager.append_defs(src_words, lem_data, conf_list[1:-1])
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
            out_nums = beam_search(manager, src_encs, beam_size, max_length)
        elif conf_type is None:
            src_nums = torch.tensor(vocab.numberize(src_words), device=device)
            src_encs, _ = model.encode(src_nums.unsqueeze(0))
            out_nums = beam_search(manager, src_encs, beam_size, max_length)

    return tokenizer.detokenize(vocab.denumberize(out_nums.tolist())), conf_list


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dict', metavar='FILE_PATH', help='bilingual dictionary')
    parser.add_argument('--freq', metavar='FILE_PATH', help='frequency statistics')
    parser.add_argument(
        '--conf', nargs=2, metavar=('CONF_TYPE', 'FILE_PATH'), help='confidence estimation'
    )
    parser.add_argument('--align', metavar='FILE_PATH', help='phrase table format')
    parser.add_argument('--spacy-model', metavar='FILE_PATH', help='spaCy model')
    parser.add_argument('--sw-vocab', metavar='FILE_PATH', required=True, help='subword vocab')
    parser.add_argument('--sw-model', metavar='FILE_PATH', required=True, help='subword model')
    parser.add_argument('--model', metavar='FILE_PATH', required=True, help='translation model')
    parser.add_argument('--input', metavar='FILE_PATH', help='detokenized input')
    args, unknown = parser.parse_known_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_state = torch.load(args.model, weights_only=False, map_location=device)

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
    )
    manager.model.load_state_dict(model_state['state_dict'])

    # if device == 'cuda' and torch.cuda.get_device_capability()[0] >= 8:
    #     torch.set_float32_matmul_precision('high')

    if args.align:
        sent_aligns = []
        with open(args.align) as align_f:
            for line in align_f.readlines():
                sent_aligns.append({})  # target -> source
                for alignment in line.split():
                    src_i, tgt_i = alignment.split(':')[0].split('-')
                    sent_aligns[-1].setdefault(int(tgt_i), []).append(int(src_i))
        global sent_align

    if args.conf:
        json_list = []
        with open(args.input) as data_f:
            for i, string in enumerate(data_f.readlines()):
                if args.align:
                    sent_align = sent_aligns[i]
                output, conf_list = translate(
                    string, manager, spacy_model=args.spacy_model, conf_type=args.conf[0]
                )
                json_list.append(conf_list)
                print(output)
        with open(args.conf[1], 'w') as json_f:
            json.dump(json_list, json_f)
    else:
        with open(args.input) as data_f:
            for string in data_f.readlines():
                output, _ = translate(string, manager, spacy_model=args.spacy_model)
                print(output)


if __name__ == '__main__':
    main()
