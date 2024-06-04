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
]

preps = [
    'in',
    'neben',
    'hinter',
    'vor',
    'an',
    'auf',
    'über',
    'unter',
    'zwischen',
    'aus',
    'außer',
    'bei',
    'mit',
    'nach',
    'seit',
    'von',
    'zu',
    'bis',
    'durch',
    'für',
    'ohne',
    'gegen',
    'um',
    'statt',
    'trotz',
    'während',
    'wegen',
]

for prep in preps:
    for abbr1 in abbrs:
        for abbr2 in abbrs:
            filters.append(f'{prep} {abbr1}/{abbr2}')
for prep in preps:
    for abbr in abbrs:
        filters.append(f'{prep} {abbr}')
for abbr1 in abbrs:
    for abbr2 in abbrs:
        filters.append(f'{abbr1}/{abbr2}')
filters.extend(abbrs)


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

    with open(f'{args.data_dir}/de-en.model') as sw_model_f:
        tokenizer = Tokenizer('en', sw_model=BPE(sw_model_f))

    deen_dict = {}
    with open(f'{args.data_dir}/de-en.txt') as dict_file:
        for line in tqdm(dict_file.readlines()):
            if line.startswith('#'):
                continue
            line = re.sub('|'.join(filters), '', line)
            line = re.sub(r'\s+', ' ', line)
            if '::' not in line:
                continue
            de, en = line.split('::')
            for words, defns in zip(de.split('|'), en.split('|')):
                for headword in words.split(';'):
                    headword = headword.strip()
                    if not headword.replace('-', '').isalpha():
                        continue
                    headword = re.sub(r'^sich ', '', headword)
                    for definition in defns.split(';'):
                        definition = definition.strip()
                        if headword == definition:
                            continue
                        definition = ' '.join(tokenizer.tokenize(definition))
                        if not definition:
                            continue
                        if headword not in deen_dict:
                            deen_dict[headword] = []
                        if definition not in deen_dict[headword]:
                            deen_dict[headword].append(definition)

    with open(f'{args.data_dir}/de-en.dict', 'w') as dict_file:
        deen_dict = dict(sorted(deen_dict.items(), key=lambda x: x[0]))
        json.dump(deen_dict, dict_file, indent=4, ensure_ascii=False)
    print(len(deen_dict), f'{args.data_dir}/de-en.dict')


if __name__ == '__main__':
    main()
