import argparse
import json
import sys
from collections import Counter

import matplotlib.pyplot as plt

# from scipy.optimize import curve_fit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('infile', nargs='?', type=argparse.FileType('r'), default=sys.stdin)
    parser.add_argument('--output-dir', required=True, help='output directory')
    args = parser.parse_args()

    scale = 100 if args.output_dir.split('_')[-1] in ('2', 'inf') else 1
    counter = Counter(round(conf * scale) for sent in json.load(args.infile) for _, conf in sent)
    plt.hist(list(counter.keys())[:-1], bins=1000, weights=list(counter.values())[:-1])
    plt.xlim((0, 50))
    plt.xlabel('Confidence')
    plt.ylabel('# of Words')
    plt.title('Confidence Histogram')
    plt.savefig(f'{args.output_dir}/histogram.png', dpi=300)


if __name__ == '__main__':
    main()
