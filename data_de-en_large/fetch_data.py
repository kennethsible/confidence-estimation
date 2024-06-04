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
    with open(f'{args.data_dir}/train/europarl-v10.de-en.tsv') as tsv_f, open(
        f'{args.data_dir}/train/train.de', 'w'
    ) as src_f, open(f'{args.data_dir}/train/train.en', 'w') as tgt_f:
        for line in tsv_f.readlines():
            src_line, tgt_line, *_ = line.split('\t')
            src_f.write(src_line.rstrip() + '\n')
            tgt_f.write(tgt_line.rstrip() + '\n')
    os.system(f'rm {args.data_dir}/train/europarl-v10.de-en.tsv')

    os.system('sacrebleu --language-pair de-en --download wmt19')
    os.system(f'cp ~/.sacrebleu/wmt19/wmt19.de-en.src {args.data_dir}/val/val.de')
    os.system(f'cp ~/.sacrebleu/wmt19/wmt19.de-en.ref {args.data_dir}/val/val.en')

    os.system('sacrebleu --language-pair de-en --download wmt22')
    os.system(f'cp ~/.sacrebleu/wmt22/wmt22.de-en.src {args.data_dir}/test/test.de')


if __name__ == '__main__':
    main()
