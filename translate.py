from manager import Manager, Batch
from decode import beam_decode
import torch

def translate_file(data_file, manager):
    return [translate_string(line, manager) for line in data_file]

def translate_string(string, manager):
    model, vocab, device = manager.model, manager.vocab, manager.device
    src_words = ['<BOS>'] + manager.tokenize(string).split() + ['<EOS>']
    lemmas, senses = manager.append_senses(src_words)

    model.eval()
    with torch.no_grad():
        src_nums, src_mask = vocab.numberize(*src_words).unsqueeze(0), None
        batch = Batch(src_nums, dict_data=zip([lemmas], [senses]), device=device)
        src_encs = model.encode(batch.src_nums, src_mask, batch.dict_mask)
        out_nums = beam_decode(manager, src_encs, src_mask, manager.beam_size)

    return manager.detokenize(vocab.denumberize(*out_nums))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', metavar='FILE', required=True, help='load model')
    parser.add_argument('--dict', metavar='FILE', required=True, help='dictionary data')
    parser.add_argument('--freq', metavar='FILE', required=True, help='frequency data')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--file', metavar='FILE', help='file input')
    group.add_argument('--string', metavar='STRING', help='string input')
    args, unknown = parser.parse_known_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_dict = torch.load(args.model, map_location=device)
    src_lang, tgt_lang = model_dict['src_lang'], model_dict['tgt_lang']
    vocab_file, codes_file = model_dict['vocab_file'], model_dict['codes_file']

    config = model_dict['config']
    for i, arg in enumerate(unknown):
        if arg[:2] == '--' and len(unknown) > i:
            option, value = arg[2:], unknown[i + 1]
            config[option] = (int if value.isdigit() else float)(value)

    with open(args.dict) as dict_file, open(args.freq) as freq_file:
        manager = Manager(src_lang, tgt_lang, vocab_file, codes_file, args.model,
            config, device, dict_file, freq_file, data_file=None, test_file=None)
    manager.model.load_state_dict(model_dict['state_dict'])

    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        torch.set_float32_matmul_precision('high')
    # if torch.__version__ >= '2.0':
    #     manager.model = torch.compile(manager.model)

    if args.file:
        data_file = open(args.file)
        print(*translate_file(data_file, manager), sep='\n')
        data_file.close()
    elif args.string:
        print(translate_string(args.string, manager))

if __name__ == '__main__':
    import argparse
    main()
