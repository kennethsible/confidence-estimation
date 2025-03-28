import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True, help='data directory')
    args = parser.parse_args()

    europarl_url = 'https://www.statmt.org/europarl/v10/training/europarl-v10.de-en.tsv.gz'
    os.system(f'mkdir -p {args.data_dir}/train {args.data_dir}/val {args.data_dir}/test')
    os.system(f'wget -q -P {args.data_dir}/train {europarl_url} --show-progress')
    os.system(f'gzip -d {args.data_dir}/train/europarl-v10.de-en.tsv.gz')
    with (
        open(f'{args.data_dir}/train/europarl-v10.de-en.tsv') as tsv_f,
        open(f'{args.data_dir}/train/train.de', 'w') as src_f,
        open(f'{args.data_dir}/train/train.en', 'w') as tgt_f,
    ):
        for line in tsv_f.readlines():
            src_line, tgt_line, *_ = line.split('\t')
            src_f.write(src_line.rstrip() + '\n')
            tgt_f.write(tgt_line.rstrip() + '\n')
    os.system(f'rm {args.data_dir}/train/europarl-v10.de-en.tsv')

    os.system('sacrebleu --language-pair en-de --download wmt19')
    os.system(f'cp ~/.sacrebleu/wmt19/wmt19.en-de.src {args.data_dir}/val/val.en')
    os.system(f'cp ~/.sacrebleu/wmt19/wmt19.en-de.ref {args.data_dir}/val/val.de')

    os.system('sacrebleu --language-pair en-de --download wmt22')
    os.system(f'cp ~/.sacrebleu/wmt22/wmt22.en-de.src {args.data_dir}/test/test.en')
    os.system(f'cp ~/.sacrebleu/wmt22/wmt22.en-de.ref-A {args.data_dir}/test/test1.de')
    os.system(f'cp ~/.sacrebleu/wmt22/wmt22.en-de.ref-B {args.data_dir}/test/test2.de')

    os.system('poetry run python -m spacy download de_core_news_sm')
    os.system('poetry run python -m spacy download en_core_web_sm')


if __name__ == '__main__':
    main()
