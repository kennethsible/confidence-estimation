import json
import re

from sacremoses import MosesPunctNormalizer
from subword_nmt.apply_bpe import BPE
from tqdm import tqdm

from manager import Tokenizer
from preprocess import download

filters = [
    r'\([^)]+\)',
    r'\[[^\]]+\]',
    r'\{[^}]+\}',
    r'sich',
]

preps = [
    'in',
    'neben',
    'hinter',
    'vor',
    'an',
    'auf',
    'neben',
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

abbrs = [
    'etw.',
    'jdm.',
    'jdn.',
    'jds.',
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
    download('https://ftp.tu-chemnitz.de/pub/Local/urz/ding/de-en-devel/de-en.txt.gz', 'dict.deen')

    mpn = MosesPunctNormalizer()
    with open('data/codes.ende') as codes_file:
        tokenizer = Tokenizer(BPE(codes_file), 'en')

    deen_dict = {}
    with open('data/dict.deen') as dict_file:
        for line in tqdm(dict_file.readlines()):
            if line.startswith('#'):
                continue
            line = re.sub('|'.join(filters), '', line)
            line = re.sub(r'\s+', ' ', line)
            if '::' in line:
                de, en = line.split('::')
            for words, defns in zip(de.split('|'), en.split('|')):
                for headword in words.split(';'):
                    headword = mpn.normalize(headword.strip())
                    if not headword.isalpha() or len(headword.split()) > 1:
                        continue
                    for defn in defns.split(';'):
                        if headword == defn.strip():
                            continue
                        defn = tokenizer.tokenize(defn.strip())
                        if headword not in deen_dict:
                            deen_dict[headword] = []
                        if defn not in deen_dict[headword]:
                            deen_dict[headword].append(defn)

    with open('data/dict.json', 'w') as dict_file:
        deen_dict = dict(sorted(deen_dict.items(), key=lambda x: x[0]))
        json.dump(deen_dict, dict_file, indent=4)
    print(len(deen_dict), 'data/dict.json')


if __name__ == "__main__":
    main()
