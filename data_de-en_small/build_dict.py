import argparse
import json
import os
import re

from subword_nmt.apply_bpe import BPE
from tqdm import tqdm

from translation.manager import Tokenizer

filters = [
    r'\([^\)]+\)',
    r'\[[^\]]+\]',
    r'\{[^\}]+\}',
    r'<[^>]+>',
]

abbrs = [
    r'etw\.',
    r'jdm\.',
    r'jdn\.',
    r'jds\.',
    r'sth\.',
    r'sb\.',
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True, help='data directory')
    args = parser.parse_args()

    url = 'https://ftp.tu-chemnitz.de/pub/Local/urz/ding/de-en-devel/de-en.txt.gz'
    os.system(f'wget -nc -q -P {args.data_dir} {url} --show-progress')
    os.system(f'gzip -d {args.data_dir}/de-en.txt.gz')
    os.system(
        f'cat {args.data_dir}/train/train.norm.tok.de | subword-nmt get-vocab > {args.data_dir}/de-en.freq'
    )
    os.system(
        f'cat {args.data_dir}/train/train.norm.tok.en | subword-nmt get-vocab > {args.data_dir}/en-de.freq'
    )

    with open(f'{args.data_dir}/en-de.model') as sw_model_f:
        tokenizer_de = Tokenizer('de', sw_model=BPE(sw_model_f))
        tokenizer_en = Tokenizer('en', sw_model=BPE(sw_model_f))

    deen_dict, ende_dict = {}, {}
    with open(f'{args.data_dir}/de-en.txt') as dict_file:
        for line in tqdm(dict_file.readlines()):
            if line.startswith('#') or '::' not in line:
                continue
            line = re.sub('|'.join(filters), '', line)
            de, en = re.sub(r'\s+', ' ', line).split('::')
            for words, defns in zip(de.split('|'), en.split('|')):
                for headword in words.split(';'):
                    headword = re.split('|'.join(abbrs), headword)[0]
                    headword = re.sub(r'^sich ', '', headword).strip()
                    for definition in defns.split(';'):
                        definition = re.split('|'.join(abbrs), definition)[0]
                        definition = re.sub(r'^to ', '', definition).strip()
                        if not (headword and definition) or headword == definition:
                            continue
                        _headword, _definition = definition, headword

                        definition = ' '.join(tokenizer_en.tokenize(definition))
                        if headword not in deen_dict:
                            deen_dict[headword] = []
                        if definition not in deen_dict[headword]:
                            deen_dict[headword].append(definition)

                        _definition = ' '.join(tokenizer_de.tokenize(_definition))
                        if _headword not in ende_dict:
                            ende_dict[_headword] = []
                        if _definition not in ende_dict[_headword]:
                            ende_dict[_headword].append(_definition)

    with open(f'{args.data_dir}/de-en.dict', 'w') as dict_file:
        deen_dict = dict(sorted(deen_dict.items(), key=lambda x: x[0]))
        json.dump(deen_dict, dict_file, indent=4, ensure_ascii=False)
    print(len(deen_dict), f'{args.data_dir}/de-en.dict')

    with open(f'{args.data_dir}/en-de.dict', 'w') as dict_file:
        ende_dict = dict(sorted(ende_dict.items(), key=lambda x: x[0]))
        json.dump(ende_dict, dict_file, indent=4, ensure_ascii=False)
    print(len(ende_dict), f'{args.data_dir}/en-de.dict')


if __name__ == '__main__':
    main()
