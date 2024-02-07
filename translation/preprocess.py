import os
import re

from tqdm import tqdm

from .manager import Lemmatizer

DATA_DIR = 'data'
TRAIN_DIR = f'{DATA_DIR}/training'
VAL_DIR = f'{DATA_DIR}/validation'
TEST_DIR = f'{DATA_DIR}/testing'

lemmatizer = Lemmatizer('de_core_news_sm')


def lemmatize(src_file: str, lem_file: str):
    src_words = []
    with open(src_file) as infile:
        for line in infile.readlines():
            src_line, tgt_line = line.split('\t')
            src_words.append(src_line.split())

    data = lemmatizer.lemmatize(src_words)
    with open(lem_file, 'w') as outfile:
        for words, spans in tqdm(data, total=len(src_words)):
            outfile.write(f"{' '.join(words)}\t{' '.join(map(str, spans))}\n")


def clean(data_file: str, max_length: int, len_ratio: float):
    with open(data_file) as infile:
        unique_lines = list(dict.fromkeys(infile.readlines()))

    with open(data_file, 'w') as outfile:
        for line in tqdm(unique_lines):
            src_line, tgt_line = line.split('\t')
            src_words = src_line.split()
            tgt_words = tgt_line.split()
            if (
                1 <= len(src_words) <= max_length
                and 1 <= len(tgt_words) <= max_length
                and len(src_words) / len(tgt_words) <= len_ratio
            ):
                outfile.write(line)


def sgml_to_text(sgml_file: str, text_file: str):
    with open(sgml_file) as infile, open(text_file, 'w') as outfile:
        for line in infile:
            text = re.split(r'(<[^>]+>)', line.strip())[1:-1]
            if len(text) != 3:
                continue
            tag, sentence, _ = text
            if tag[1:-1].split(' ')[0] == 'seg':
                outfile.write(sentence + '\n')


def download(url: str, outfile: str):
    os.system(f'wget -q -O {outfile} {url} --show-progress')
    if url.split('.')[-1] == 'gz':
        os.system(f'gzip -cd {outfile} > data/{outfile} && rm {outfile}')
    else:
        os.system(f'tar -xzf {outfile} -C data && rm {outfile}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang', nargs=2, required=True, help='source/target language')
    parser.add_argument('--merge-ops', metavar='FILE', required=True, help='merge operations (BPE)')
    parser.add_argument('--max-length', metavar='FILE', required=True, help='maximum string length')
    parser.add_argument('--len-ratio', metavar='FILE', required=True, help='maximum length ratio')
    args = parser.parse_args()

    src_lang, tgt_lang = args.lang

    print('[1/12] Downloading WMT17 Training Data...')
    os.system(f'mkdir -p {DATA_DIR} {TRAIN_DIR}')
    download(
        f'http://www.statmt.org/europarl/v7/{src_lang}-{tgt_lang}.tgz',
        f'{DATA_DIR}/europarl-v7.tgz',
    )
    path = 'data'
    for lang in args.lang:
        os.system(
            f'cat "{DATA_DIR}/europarl-v7.{src_lang}-{tgt_lang}.{lang}" \
                >> "{TRAIN_DIR}/{path}.{lang}"'
        )
    os.system(f'wc -l "{TRAIN_DIR}/{path}.{src_lang}"')
    os.system(f'find {DATA_DIR} -maxdepth 1 -type f -delete')

    print('\n[2/12] Normalizing Training Data...')
    for lang in args.lang:
        os.system(
            f'sacremoses -j 4 normalize < "{TRAIN_DIR}/{path}.{lang}" \
                > "{TRAIN_DIR}/{path}.norm.{lang}"'
        )
    path += '.norm'

    print('\n[3/12] Tokenizing Training Data...')
    for lang in args.lang:
        os.system(
            f'sacremoses -l {lang} -j 4 tokenize -x < "{TRAIN_DIR}/{path}.{lang}" \
                > "{TRAIN_DIR}/{path}.tok.{lang}"'
        )
        os.system(
            f'cat "{TRAIN_DIR}/{path}.tok.{lang}" | subword-nmt get-vocab \
                > "{DATA_DIR}/freq.{lang}"'
        )

    print('\n[4/12] Downloading WMT17 Validation Data...')
    download(
        'http://data.statmt.org/wmt17/translation-task/dev.tgz',
        f'{DATA_DIR}/dev.tgz',
    )
    os.system(f'mv {DATA_DIR}/dev {VAL_DIR}')
    path = 'data'
    sgml_to_text(
        f'{VAL_DIR}/newstest2016-{src_lang}{tgt_lang}-src.{src_lang}.sgm',
        f'{VAL_DIR}/{path}.{src_lang}',
    )
    sgml_to_text(
        f'{VAL_DIR}/newstest2016-{src_lang}{tgt_lang}-ref.{tgt_lang}.sgm',
        f'{VAL_DIR}/{path}.{tgt_lang}',
    )
    os.system(f'wc -l "{VAL_DIR}/data.{src_lang}"')
    os.system(
        f'find {VAL_DIR} -type f ! -name "*{path}.{src_lang}" \
            -and ! -name "*{path}.{tgt_lang}" -delete'
    )

    print('\n[5/12] Normalizing Validation Data...')
    for lang in args.lang:
        os.system(
            f'sacremoses -j 4 normalize < "{VAL_DIR}/{path}.{lang}" \
                > "{VAL_DIR}/{path}.norm.{lang}"'
        )
    path += '.norm'

    print('\n[6/12] Tokenizing Validation Data...')
    for lang in args.lang:
        os.system(
            f'sacremoses -l {lang} -j 4 tokenize -x < "{VAL_DIR}/{path}.{lang}" \
                > "{VAL_DIR}/{path}.tok.{lang}"'
        )

    print('\n[7/12] Downloading WMT17 Testing Data...')
    download(
        'http://data.statmt.org/wmt17/translation-task/test.tgz',
        f'{DATA_DIR}/test.tgz',
    )
    os.system(f'mv {DATA_DIR}/test {TEST_DIR}')
    path = 'data'
    sgml_to_text(
        f'{TEST_DIR}/newstest2017-{src_lang}{tgt_lang}-src.{src_lang}.sgm',
        f'{TEST_DIR}/{path}.{src_lang}',
    )
    sgml_to_text(
        f'{TEST_DIR}/newstest2017-{src_lang}{tgt_lang}-ref.{tgt_lang}.sgm',
        f'{TEST_DIR}/{path}.{tgt_lang}',
    )
    os.system(f'wc -l "{TEST_DIR}/data.{src_lang}"')
    os.system(
        f'find {TEST_DIR} -type f ! -name "*{path}.{src_lang}" \
            -and ! -name "*{path}.{tgt_lang}" -delete'
    )

    print('\n[8/12] Normalizing Testing Data...')
    for lang in args.lang:
        os.system(
            f'sacremoses -j 4 normalize < "{TEST_DIR}/{path}.{lang}" \
                > "{TEST_DIR}/{path}.norm.{lang}"'
        )
    path += '.norm'

    print('\n[9/12] Tokenizing Testing Data...')
    for lang in args.lang:
        os.system(
            f'sacremoses -l {lang} -j 4 tokenize -x < "{TEST_DIR}/{path}.{lang}" \
                > "{TEST_DIR}/{path}.tok.{lang}"'
        )
    path += '.tok'

    print('\n[10/12] Learning and Applying BPE...')
    os.system(
        f'cat "{TRAIN_DIR}/{path}.{src_lang}" "{TRAIN_DIR}/{path}.{tgt_lang}" \
            | subword-nmt learn-bpe -s {args.merge_ops} -o "{DATA_DIR}/codes.{src_lang}{tgt_lang}"'
    )
    os.system(f'cp "{DATA_DIR}/codes.{src_lang}{tgt_lang}" "{DATA_DIR}/codes.{tgt_lang}{src_lang}"')
    for DIR in (TRAIN_DIR, VAL_DIR, TEST_DIR):
        for lang in args.lang:
            os.system(
                f'subword-nmt apply-bpe -c "{DATA_DIR}/codes.{src_lang}{tgt_lang}" \
                    < "{DIR}/{path}.{lang}" > "{DIR}/{path}.bpe.{lang}"'
            )
    path += '.bpe'
    os.system(
        f'cat "{TRAIN_DIR}/{path}.{src_lang}" "{TRAIN_DIR}/{path}.{tgt_lang}" \
            | subword-nmt get-vocab > "{DATA_DIR}/vocab.{src_lang}{tgt_lang}"'
    )
    os.system(f'cp "{DATA_DIR}/vocab.{src_lang}{tgt_lang}" "{DATA_DIR}/vocab.{tgt_lang}{src_lang}"')
    os.system(f'wc -l "{DATA_DIR}/vocab.{src_lang}{tgt_lang}"')

    print('\n[11/12] Cleaning Training Data...')
    for DIR in (TRAIN_DIR, VAL_DIR, TEST_DIR):
        os.system(
            f'paste "{DIR}/{path}.{src_lang}" "{DIR}/{path}.{tgt_lang}" \
                > "{DIR}/{path}.{src_lang}{tgt_lang}"'
        )
        os.system(
            f'paste "{DIR}/{path}.{tgt_lang}" "{DIR}/{path}.{src_lang}" \
                > "{DIR}/{path}.{tgt_lang}{src_lang}"'
        )
    clean(f'{TRAIN_DIR}/{path}.{src_lang}{tgt_lang}', int(args.max_length), float(args.len_ratio))
    clean(f'{TRAIN_DIR}/{path}.{tgt_lang}{src_lang}', int(args.max_length), float(args.len_ratio))
    os.system(f'wc -l "{TRAIN_DIR}/{path}.{src_lang}{tgt_lang}"')

    print('\n[12/12] Lemmatizing Source Data...')
    for DIR in (TRAIN_DIR, VAL_DIR, TEST_DIR):
        lemmatize(f'{DIR}/{path}.{src_lang}{tgt_lang}', f'{DIR}/data.norm.tok.lem.{src_lang}')

    print('\nDone.')


if __name__ == "__main__":
    import argparse

    main()
