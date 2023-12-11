import random

import torch

from decoder import beam_search
from manager import Batch, Lemmatizer, Manager, Tokenizer


def translate_file(data_file: str, manager: Manager, tokenizer: Tokenizer) -> list[str]:
    with open(data_file) as file:
        return [translate_string(line, manager, tokenizer) for line in file]


def translate_string(string: str, manager: Manager, tokenizer: Tokenizer) -> str:
    model, vocab, device = manager.model, manager.vocab, manager.device
    src_words = tokenizer.tokenize(string).split()
    if manager.dict:
        lemmatizer = Lemmatizer('de_core_news_sm')
        src_spans = list(lemmatizer.lemmatize([src_words]))[0]
    src_words = ['<BOS>'] + src_words + ['<EOS>']
    if manager.dict:
        lemmas, senses = manager.attach_senses(src_words, src_spans, tokenizer)
    else:
        dict_mask = None

    model.eval()
    with torch.no_grad():
        src_nums, src_mask = torch.tensor(vocab.numberize(src_words)), None
        mask_size = src_nums.unsqueeze(-2).size()
        if manager.dict:
            dict_mask = Batch.dict_mask_from_data(zip([lemmas], [senses]), mask_size, device)
        src_encs = model.encode(src_nums.unsqueeze(0).to(device), src_mask, dict_mask)
        out_nums = beam_search(manager, src_encs, src_mask, manager.beam_size)

    return tokenizer.detokenize(vocab.denumberize(out_nums.tolist()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', metavar='FILE', required=True, help='model file (.pt)')
    parser.add_argument('--dict', metavar='FILE', required=False, help='dictionary data')
    parser.add_argument('--freq', metavar='FILE', required=False, help='frequency data')
    parser.add_argument('--seed', type=int, help='random seed')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--string', metavar='STRING', help='input string')
    group.add_argument('--file', metavar='FILE', help='input file')
    args, unknown = parser.parse_known_args()

    if args.seed:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_dict = torch.load(args.model, map_location=device)
    src_lang, tgt_lang = model_dict['src_lang'], model_dict['tgt_lang']
    vocab_list, codes_list = model_dict['vocab_list'], model_dict['codes_list']

    config = model_dict['model_config']
    for i, arg in enumerate(unknown):
        if arg[:2] == '--' and len(unknown) > i:
            option, value = arg[2:].replace('-', '_'), unknown[i + 1]
            try:
                config[option] = (int if value.isdigit() else float)(value)
            except ValueError:
                config[option] = value

    manager = Manager(
        src_lang,
        tgt_lang,
        config,
        device,
        args.model,
        vocab_list,
        codes_list,
        args.dict,
        args.freq,
        data_file=None,
        test_file=None,
    )
    manager.model.load_state_dict(model_dict['state_dict'])
    tokenizer = Tokenizer(manager.bpe, src_lang, tgt_lang)

    if device == 'cuda' and torch.cuda.get_device_capability()[0] >= 8:
        torch.set_float32_matmul_precision('high')

    if args.file:
        data_file = open(args.file)
        print(*translate_file(data_file, manager, tokenizer), sep='\n')
        data_file.close()
    elif args.string:
        print(translate_string(args.string, manager, tokenizer))


if __name__ == '__main__':
    import argparse

    main()
