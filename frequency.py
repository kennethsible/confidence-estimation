import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('infile', nargs='?', type=argparse.FileType('r'), default=sys.stdin)
    parser.add_argument('outfile', nargs='?', type=argparse.FileType('w'), default=sys.stdout)
    args = parser.parse_args()

    with open('freq.tsv') as freq_f:
        freq_dict = {}
        for line in freq_f.readlines():
            word, freq = line.split()
            freq_dict[word] = int(freq)

    json_list = []
    for conf_list in json.load(args.infile):
        freq_list = []
        for word, _ in conf_list:
            frequency = freq_dict[word] if word in freq_dict else 0
            freq_list.append([word, frequency])
        json_list.append(freq_list)

    json.dump(json_list, args.outfile, indent=4)


if __name__ == '__main__':
    main()
