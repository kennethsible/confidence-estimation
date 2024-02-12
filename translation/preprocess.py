import os

from tqdm import tqdm

from translation.manager import Lemmatizer

lemmatizer = Lemmatizer('de_core_news_sm')


def lemmatize(src_file: str, lem_file: str):
    src_words = []
    with open(src_file) as infile:
        for line in infile.readlines():
            src_line = line.split('\t')[0]
            src_words.append(src_line.split())

    data = lemmatizer.lemmatize(src_words)
    with open(lem_file, 'w') as outfile:
        for words, spans in tqdm(data, total=len(src_words)):
            outfile.write(f"{' '.join(words)}\t{' '.join(map(str, spans))}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang-pair', required=True, help='source-target language pair')
    parser.add_argument('--merge-ops', required=True, help='merge operations (subword-nmt)')
    parser.add_argument('--val-set', required=True, help='validation set (sacrebleu)')
    parser.add_argument('--test-set', required=True, help='test set (sacrebleu)')
    parser.add_argument('--data-dir', required=True, help='output directory')
    args = parser.parse_args()

    src_lang, tgt_lang = args.lang_pair.split('-')
    data_dir = args.data_dir
    os.system(f'mkdir -p {data_dir}')

    print('\n[1/11] Downloading Training Corpus...')
    train_path = f'{data_dir}/europarl-v10'
    os.system(f'mkdir -p {train_path}')
    train_path += '/europarl-v10'
    if not os.path.isfile(f'{train_path}.src'):
        url = f'https://www.statmt.org/europarl/v10/training/europarl-v10.{src_lang}-{tgt_lang}.tsv.gz'
        os.system(f'wget -q -P {data_dir}/europarl-v10 {url} --show-progress')
        os.system(f'gzip -d {train_path}.{src_lang}-{tgt_lang}.tsv.gz')
        with open(f'{train_path}.src', 'w') as src_file, open(f'{train_path}.ref', 'w') as tgt_file:
            for line in open(f'{train_path}.{src_lang}-{tgt_lang}.tsv'):
                src_line, tgt_line, *_ = line.split('\t')
                if not src_line or not tgt_line:
                    continue
                src_file.write(src_line + '\n')
                tgt_file.write(tgt_line + '\n')
        os.system(f'rm {train_path}.{src_lang}-{tgt_lang}.tsv')
    os.system(f'wc -l {train_path}.src')

    print('\n[2/11] Normalizing Training Corpus...')
    os.system(
        f'sacremoses -l {src_lang} -j 4 normalize < {train_path}.src \
            > {train_path}.norm.src'
    )
    os.system(
        f'sacremoses -l {tgt_lang} -j 4 normalize < {train_path}.ref \
            > {train_path}.norm.ref'
    )
    root_path = 'norm'

    print('\n[3/11] Tokenizing Training Corpus...')
    os.system(
        f'sacremoses -l {src_lang} -j 4 tokenize -x < {train_path}.{root_path}.src \
            > {train_path}.{root_path}.tok.src'
    )
    os.system(
        f'cat {train_path}.{root_path}.tok.src | subword-nmt get-vocab \
            > {data_dir}/{src_lang}-freq.tsv'
    )
    os.system(
        f'sacremoses -l {tgt_lang} -j 4 tokenize -x < {train_path}.{root_path}.ref \
            > {train_path}.{root_path}.tok.ref'
    )
    root_path += '.tok'

    print('\n[4/11] Downloading Validation Set...')
    val_path = f'{data_dir}/{args.val_set}'
    os.system(f'mkdir -p {val_path}')
    os.system(f'sacrebleu --language-pair {src_lang}-{tgt_lang} --download {args.val_set}')
    val_path += f'/{args.val_set}'
    os.system(
        f'cp ~/.sacrebleu/{args.val_set}/{args.val_set}.{src_lang}-{tgt_lang}.src {val_path}.src'
    )
    os.system(f'wc -l {val_path}.src')

    print('\n[5/11] Normalizing Validation Set...')
    os.system(
        f'sacremoses -l {src_lang} -j 4 normalize < {val_path}.src \
            > {val_path}.norm.src'
    )
    root_path = 'norm'

    print('\n[6/11] Tokenizing Validation Set...')
    os.system(
        f'sacremoses -l {src_lang} -j 4 tokenize -x < {val_path}.{root_path}.src \
            > {val_path}.{root_path}.tok.src'
    )

    print('\n[7/11] Downloading Test Set...')
    test_path = f'{data_dir}/{args.test_set}'
    os.system(f'mkdir -p {test_path}')
    os.system(f'sacrebleu --language-pair {src_lang}-{tgt_lang} --download {args.test_set}')
    test_path += f'/{args.test_set}'
    os.system(
        f'cp ~/.sacrebleu/{args.test_set}/{args.test_set}.{src_lang}-{tgt_lang}.src {test_path}.src'
    )
    os.system(f'wc -l {test_path}.src')

    print('\n[8/11] Normalizing Test Set...')
    os.system(
        f'sacremoses -l {src_lang} -j 4 normalize < {test_path}.src \
            > {test_path}.norm.src'
    )
    root_path = 'norm'

    print('\n[9/11] Tokenizing Test Set...')
    os.system(
        f'sacremoses -l {src_lang} -j 4 tokenize -x < {test_path}.{root_path}.src \
            > {test_path}.{root_path}.tok.src'
    )

    print('\n[10/11] Learning and Applying BPE...')
    root_path += '.tok'
    os.system(
        f'cat {train_path}.{root_path}.src {train_path}.{root_path}.ref \
            | subword-nmt learn-bpe -s {args.merge_ops} -o {data_dir}/codes.tsv'
    )
    for path in (train_path, val_path, test_path):
        os.system(
            f'subword-nmt apply-bpe -c {data_dir}/codes.tsv \
                < {path}.{root_path}.src > {path}.{root_path}.bpe.src'
        )
    os.system(
        f'subword-nmt apply-bpe -c {data_dir}/codes.tsv \
            < {train_path}.{root_path}.ref > {train_path}.{root_path}.bpe.ref'
    )
    root_path += '.bpe'
    os.system(
        f'cat {train_path}.{root_path}.src {train_path}.{root_path}.ref \
            | subword-nmt get-vocab > {data_dir}/vocab.tsv'
    )
    os.system(
        f'paste {train_path}.{root_path}.src {train_path}.{root_path}.ref \
            > {train_path}.{root_path}.src-ref'
    )
    os.system(f'wc -l {data_dir}/vocab.tsv')

    print('\n[11/11] Lemmatizing Training Source...')
    lemmatize(f'{train_path}.{root_path}.src-ref', f'{path}.lem.src')
    for path in (val_path, test_path):
        lemmatize(f'{path}.{root_path}.src', f'{path}.lem.src')

    print('\nDone.')


if __name__ == '__main__':
    import argparse

    main()
