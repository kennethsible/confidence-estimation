from itertools import product
from time import sleep
import json, os

#QF_CMD = "qf submit --queue 'gpu@@csecri' --queue 'gpu@@nlp-gpu'"
QF_CMD = "qf submit --queue 'gpu@@nlp-gpu'"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang', nargs=2, metavar=('SRC', 'TGT'), required=True, help='language pair')
    parser.add_argument('--data', metavar='FILE', required=True, help='training data')
    parser.add_argument('--test', metavar='FILE', required=True, help='validation data')
    parser.add_argument('--dict', metavar='FILE', required=True, help='dictionary data')
    parser.add_argument('--freq', metavar='FILE', required=True, help='frequency data')
    parser.add_argument('--vocab', metavar='FILE', required=True, help='shared vocab')
    parser.add_argument('--codes', metavar='FILE', required=True, help='shared codes')
    parser.add_argument('--model', metavar='FILE', required=True, help='save model')
    args = parser.parse_args()

    param_array = []
    with open('param_array.json') as json_file:
        for option, values in json.load(json_file).items():
            param_array.append([(option, value) for value in values])

    for i, params in enumerate(product(*param_array), start=1):
        os.system(f'mkdir -p {args.model}')
        job_name = f"{args.model}_{str(i).rjust(3, '0')}"
        with open(f'{args.model}/{job_name}.sh', 'w') as job_file:
            job_file.write(f'#!/bin/bash\n\n')
            job_file.write(f'conda activate pytorch\n\n')
            job_file.write(f"python main.py --lang {' '.join(args.lang)} \\\n")
            job_file.write(f'  --data {args.data} \\\n')
            job_file.write(f'  --test {args.test} \\\n')
            job_file.write(f'  --dict {args.dict} \\\n')
            job_file.write(f'  --freq {args.freq} \\\n')
            job_file.write(f'  --vocab {args.vocab} \\\n')
            job_file.write(f'  --codes {args.codes} \\\n')
            job_file.write(f'  --model {args.model}/{job_name} \\\n')
            for option, value in params:
                job_file.write(f'  --{option} {value} \\\n')
        os.system(f"{QF_CMD} --name {job_name} --deferred -- -l gpu_card=1 {args.model}/{job_name}.sh")
        sleep(1)
    os.system('qf check')

if __name__ == "__main__":
    import argparse
    main()
