import json
import os
from itertools import product

queues = [
    'gpu@@nlp-a10',
    'gpu@@nlp-gpu',
    'gpu@@csecri',
    'gpu@@crc_gpu',
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang-pair', required=True, help='source-target language pair')
    parser.add_argument(
        '--train-data', metavar='FILE_PATH', required=True, help='parallel training'
    )
    parser.add_argument(
        '--val-data', metavar='FILE_PATH', required=True, help='parallel validation'
    )
    parser.add_argument('--test-data', metavar='FILE_PATH', required=True, help='detokenized test')
    parser.add_argument('--test-set', required=True, help='test set (sacrebleu)')
    parser.add_argument('--lem-train', metavar='FILE_PATH', help='lemmatized training')
    parser.add_argument('--lem-val', metavar='FILE_PATH', help='lemmatized validation')
    parser.add_argument('--dict', metavar='FILE_PATH', help='bilingual dictionary')
    parser.add_argument('--freq', metavar='FILE_PATH', help='frequency statistics')
    parser.add_argument('--vocab', metavar='FILE_PATH', required=True, help='shared vocabulary')
    parser.add_argument('--codes', metavar='FILE_PATH', required=True, help='subword-nmt codes')
    parser.add_argument('--model', metavar='FILE_PATH', required=True, help='translation model')
    parser.add_argument('--seed', type=int, help='random seed')
    parser.add_argument('--start', type=int, default=1, help='starting index')
    parser.add_argument('--conda', help='conda environment')
    parser.add_argument('--email', help='email address')
    args = parser.parse_args()

    param_array = []
    with open('param_array.json') as json_file:
        for option, values in json.load(json_file).items():
            param_array.append([(option, value) for value in values])
    qf_submit = 'qf submit --queue ' + ' --queue '.join(queues)
    email = f'-M {args.email} -m abe' if args.email else ''

    for i, params in enumerate(product(*param_array), start=args.start):
        os.system(f'mkdir -p {args.model}')
        job_name = f"{args.model}_{str(i).rjust(3, '0')}"
        with open(f'{args.model}/{job_name}.sh', 'w') as job_file:
            job_file.write('#!/bin/bash\n\n')
            job_file.write(f'touch {args.model}/{job_name}.log\n')
            job_file.write(f'fsync -d 30 {args.model}/{job_name}.log &\n\n')
            job_file.write(f'conda activate {args.conda}\n\n')

            job_file.write('python translation/main.py  \\\n')
            job_file.write(f"  --lang-pair {args.lang_pair} \\\n")
            job_file.write(f'  --train-data {args.train_data} \\\n')
            job_file.write(f'  --val-data {args.val_data} \\\n')
            if args.lem_data:
                job_file.write(f'  --lem-train {args.lem_train} \\\n')
            if args.lem_test:
                job_file.write(f'  --lem-val {args.lem_val} \\\n')
            if args.dict:
                job_file.write(f'  --dict {args.dict} \\\n')
            if args.freq:
                job_file.write(f'  --freq {args.freq} \\\n')
            job_file.write(f'  --vocab {args.vocab} \\\n')
            job_file.write(f'  --codes {args.codes} \\\n')
            job_file.write(f'  --model {args.model}/{job_name}.pt \\\n')
            if args.seed:
                job_file.write(f'  --seed {args.seed} \\\n')
            for option, value in params:
                job_file.write(f'  --{option} {value} \\\n')
            job_file.write(f'  > {args.model}/{job_name}.log \\\n\n')

            job_file.write('python translation/translate.py  \\\n')
            if args.dict:
                job_file.write(f'  --dict {args.dict} \\\n')
            if args.freq:
                job_file.write(f'  --freq {args.freq} \\\n')
            job_file.write(f'  --model {args.model}/{job_name}.pt \\\n')
            job_file.write(f'  --input {args.test_data} \\\n')
            job_file.write(f'  > {args.model}/{args.test_set}.hyp \\\n\n')

            job_file.write(
                f'sacrebleu -t {args.test_Set} -l {args.lang_pair} -i {args.model}/{args.test_set}.hyp  \\\n'
            )
            job_file.write(f'  >> {args.model}/{job_name}.log \\\n\n')

        os.system(
            f"{qf_submit} --name {job_name} --deferred -- {email} -l gpu_card=1 {args.model}/{job_name}.sh"
        )
    os.system('qf check')


if __name__ == "__main__":
    import argparse

    main()
