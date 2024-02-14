import os

import tomllib
from tqdm import tqdm

from translation.manager import Lemmatizer

lemmatizer = Lemmatizer('de_core_news_sm')


def lemmatize(src_file: str, lem_file: str):
    src_words = []
    with open(src_file) as src_f:
        for line in src_f.readlines():
            src_line = line.split('\t')[0]
            src_words.append(src_line.split())

    data = lemmatizer.lemmatize(src_words)
    with open(lem_file, 'w') as lem_f:
        for words, spans in tqdm(data, total=len(src_words)):
            lem_f.write(f"{' '.join(words)}\t{' '.join(map(str, spans))}\n")


def apply_filter(data_file: str, max_length: int, len_ratio: int):
    data = []
    with open(data_file) as data_f:
        for line in tqdm(data_f.readlines()):
            src_line, tgt_line = line.split('\t')
            src_words, tgt_words = src_line.split(), tgt_line.split()
            if (
                1 <= len(src_words) <= max_length
                and 1 <= len(tgt_words) <= max_length
                and len(src_words) / len(tgt_words) <= len_ratio
                and len(tgt_words) / len(src_words) <= len_ratio
            ):
                data.append(src_line + '\t' + tgt_line)
    with open(data_file, 'w') as data_f:
        data_f.writelines(data)


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

    with open('translation/config.toml', 'rb') as config_f:
        config = tomllib.load(config_f)
    max_length, len_ratio = config['max_length'], config['len_ratio']

    print('\n[1/11] Downloading Training Corpus...')
    train_path = f'{data_dir}/europarl-v10'
    os.system(f'mkdir -p {train_path}')
    train_path += '/europarl-v10'
    if not os.path.isfile(f'{train_path}.{src_lang}-{tgt_lang}.tsv'):
        url = f'https://www.statmt.org/europarl/v10/training/europarl-v10.{src_lang}-{tgt_lang}.tsv.gz'
        os.system(f'wget -q -P {data_dir}/europarl-v10 {url} --show-progress')
        os.system(f'gzip -d {train_path}.{src_lang}-{tgt_lang}.tsv.gz')
        with open(f'{train_path}.src', 'w') as src_f, open(f'{train_path}.ref', 'w') as tgt_f:
            with open(f'{train_path}.{src_lang}-{tgt_lang}.tsv') as tsv_f:
                for line in tsv_f.readlines():
                    src_line, tgt_line, *_ = line.split('\t')
                    if len(src_line) > 0 and len(tgt_line) > 0:
                        src_f.write(src_line + '\n')
                        tgt_f.write(tgt_line + '\n')
    os.system(f'wc -l {train_path}.src')

    print('\n[2/11] Normalizing Training Corpus...')
    os.system(f'sacremoses -l {src_lang} -j 4 normalize < {train_path}.src > {train_path}.norm.src')
    os.system(f'sacremoses -l {tgt_lang} -j 4 normalize < {train_path}.ref > {train_path}.norm.ref')
    root_path = 'norm'

    print('\n[3/11] Tokenizing Training Corpus...')
    os.system(
        f'sacremoses -l {src_lang} -j 4 tokenize -x < {train_path}.{root_path}.src > {train_path}.{root_path}.tok.src'
    )
    os.system(
        f'sacremoses -l {tgt_lang} -j 4 tokenize -x < {train_path}.{root_path}.ref > {train_path}.{root_path}.tok.ref'
    )
    os.system(
        f'cat {train_path}.{root_path}.tok.src | subword-nmt get-vocab > {data_dir}/{src_lang}-freq.tsv'
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
    os.system(
        f'cp ~/.sacrebleu/{args.val_set}/{args.val_set}.{src_lang}-{tgt_lang}.ref {val_path}.ref'
    )
    os.system(f'wc -l {val_path}.src')

    print('\n[5/11] Normalizing Validation Set...')
    os.system(f'sacremoses -l {src_lang} -j 4 normalize < {val_path}.src > {val_path}.norm.src')
    os.system(f'sacremoses -l {tgt_lang} -j 4 normalize < {val_path}.ref > {val_path}.norm.ref')
    root_path = 'norm'

    print('\n[6/11] Tokenizing Validation Set...')
    os.system(
        f'sacremoses -l {src_lang} -j 4 tokenize -x < {val_path}.{root_path}.src > {val_path}.{root_path}.tok.src'
    )
    os.system(
        f'sacremoses -l {tgt_lang} -j 4 tokenize -x < {val_path}.{root_path}.ref > {val_path}.{root_path}.tok.ref'
    )
    root_path += '.tok'

    print('\n[7/11] Downloading Test Set...')
    test_path = f'{data_dir}/{args.test_set}'
    os.system(f'mkdir -p {test_path}')
    os.system(f'sacrebleu --language-pair {src_lang}-{tgt_lang} --download {args.test_set}')
    test_path += f'/{args.test_set}'
    os.system(
        f'cp ~/.sacrebleu/{args.test_set}/{args.test_set}.{src_lang}-{tgt_lang}.src {test_path}.src'
    )
    os.system(f'wc -l {test_path}.src')

    print('\n[8/11] Extracting Biomedical Set...')
    med_path = f'{data_dir}/medline'
    os.system(f'mkdir -p {med_path}')
    ref_to_id = {}
    with open(f'{med_path}/{src_lang}2{tgt_lang}_mapping.txt') as med_f:
        for line in med_f.readlines():
            doc_ref, doc_id = line.split('\t')
            ref_to_id[doc_ref] = doc_id.strip()
    src_data = {}
    with open(f'{med_path}/medline_{src_lang}2{tgt_lang}_{src_lang}.txt') as med_f:
        for line in med_f.readlines():
            doc_id, sent_id, sentence = line.split('\t')
            if doc_id not in src_data:
                src_data[doc_id] = {}
            src_data[doc_id][sent_id] = sentence.rstrip()
    tgt_data = {}
    with open(f'{med_path}/medline_{src_lang}2{tgt_lang}_{tgt_lang}.txt') as med_f:
        for line in med_f.readlines():
            doc_id, sent_id, sentence = line.split('\t')
            if doc_id not in tgt_data:
                tgt_data[doc_id] = {}
            tgt_data[doc_id][sent_id] = sentence.rstrip()
    with open(f'{med_path}/{src_lang}2{tgt_lang}_align_validation.tsv') as med_f:
        med_path += '/medline'
        with open(f'{med_path}.{src_lang}', 'w') as outfile:
            for line in med_f.readlines():
                status, doc_ref, src_sent_id, _ = line.split('\t')
                if status != 'OK':
                    continue
                if ',' in src_sent_id:
                    for sent_id in src_sent_id.split(','):
                        outfile.write(src_data[ref_to_id[doc_ref]][sent_id] + ' ')
                else:
                    outfile.write(src_data[ref_to_id[doc_ref]][src_sent_id])
                outfile.write('\n')
        med_f.seek(0)
        with open(f'{med_path}.{tgt_lang}', 'w') as outfile:
            for line in med_f.readlines():
                status, doc_ref, _, tgt_sent_id = line.split('\t')
                if status != 'OK':
                    continue
                if ',' in tgt_sent_id:
                    for sent_id in tgt_sent_id.split(','):
                        outfile.write(tgt_data[ref_to_id[doc_ref]][sent_id.rstrip()] + ' ')
                else:
                    outfile.write(tgt_data[ref_to_id[doc_ref]][tgt_sent_id.rstrip()])
                outfile.write('\n')
    os.system(f'wc -l "{med_path}.{src_lang}"')

    print('\n[9/11] Learning and Applying BPE...')
    os.system(
        f'cat {train_path}.{root_path}.src {train_path}.{root_path}.ref | subword-nmt learn-bpe -s {args.merge_ops} -o {data_dir}/codes.tsv'
    )
    for path in (train_path, val_path):
        os.system(
            f'subword-nmt apply-bpe -c {data_dir}/codes.tsv < {path}.{root_path}.src > {path}.{root_path}.bpe.src'
        )
        os.system(
            f'subword-nmt apply-bpe -c {data_dir}/codes.tsv < {path}.{root_path}.ref > {path}.{root_path}.bpe.ref'
        )
    os.system(
        f'subword-nmt apply-bpe -c {data_dir}/codes.tsv < {test_path}.{root_path}.src > {test_path}.{root_path}.bpe.src'
    )
    root_path += '.bpe'
    os.system(
        f'cat {train_path}.{root_path}.src {train_path}.{root_path}.ref | subword-nmt get-vocab > {data_dir}/vocab.tsv'
    )
    for path in (train_path, val_path):
        os.system(
            f'paste {path}.{root_path}.src {path}.{root_path}.ref > {path}.{root_path}.src-ref'
        )
    os.system(f'wc -l {data_dir}/vocab.tsv')

    print('\n[10/11] Filtering Training Data...')
    apply_filter(f'{train_path}.{root_path}.src-ref', max_length, len_ratio)
    os.system(f'wc -l {train_path}.{root_path}.src-ref')

    print('\n[11/11] Lemmatizing Source Data...')
    for path in (train_path, val_path):
        lemmatize(f'{path}.{root_path}.src-ref', f'{path}.lem.src')

    print('\nDone.')


if __name__ == '__main__':
    import argparse

    main()
