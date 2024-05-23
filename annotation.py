import argparse
import json
import re
import sys

from openai import OpenAI
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode', help='annotate, clean, or score')
    annotate = subparsers.add_parser('annotate')
    annotate.add_argument('infile', nargs='?', type=argparse.FileType('r'), default=sys.stdin)
    annotate.add_argument('outfile', nargs='?', type=argparse.FileType('w'), default=sys.stdout)
    clean = subparsers.add_parser('clean')
    clean.add_argument('infile', nargs='?', type=argparse.FileType('r'), default=sys.stdin)
    clean.add_argument('outfile', nargs='?', type=argparse.FileType('w'), default=sys.stdout)
    score = subparsers.add_parser('score')
    score.add_argument('infile', nargs=2, type=argparse.FileType('r'), default=sys.stdin)
    score.add_argument('--threshold', type=int, required=True, help='confidence threshold')
    score.add_argument('--frequency', action='store_true', help='frequency-based confidence')
    args = parser.parse_args()

    client = OpenAI()
    src_lang, tgt_lang = 'German', 'English'

    if args.mode == 'annotate':
        for line in tqdm(args.infile.readlines()):
            src_sent, ref_sent, hyp_sent = line.rstrip().split('\t')
            # prompt = f'SRC is a {src_lang} sentence, REF is a reference {tgt_lang} translation of SRC, and HYP is a candidate {tgt_lang} translation of SRC. Identify all {tgt_lang} words in HYP that are mistranslated, along with their corresponding {src_lang} words in SRC. Format: "{tgt_lang} word" is a mistranslation of "{src_lang} word" ({tgt_lang} translation). SRC: {src_sent} REF: {ref_sent} HYP: {hyp_sent}'
            prompt = f'SRC is a {src_lang} sentence, REF is a reference {tgt_lang} translation of SRC, and HYP is a candidate {tgt_lang} translation of SRC. Identify all {tgt_lang} words in HYP that are mistranslated, along with their corresponding {src_lang} words in SRC. If a compound word contains spaces, choose the most relevant word to represent the compound. Format: "{tgt_lang} word" is a mistranslation of "{src_lang} word" ({tgt_lang} translation). SRC: {src_sent} REF: {ref_sent} HYP: {hyp_sent}'
            response = client.chat.completions.create(
                model='gpt-3.5-turbo',
                messages=[{'role': 'user', 'content': prompt}],
            )
            content = response.choices[0].message.content.replace('- ', '')
            args.outfile.write(', '.join(re.sub(r'[0-9][1-9]*\. ', '', content).split('\n')) + '\n')
    elif args.mode == 'clean':
        for line in tqdm(args.infile.readlines()):
            words = re.findall(r'.+? is a mistranslation of (.+?) \(', line.rstrip())
            args.outfile.write(', '.join(w.strip('\'"') for w in words) + '\n')
    elif args.mode == 'score':
        false_positive = false_negative = true_positive = true_negative = 0
        for line1, line2 in zip(args.infile[0].readlines(), json.load(args.infile[1])):
            words = line1.split(', ')
            # words = [x for y in line1.split(', ') for x in y.split()]
            for word, score in line2:
                # high confidence
                if (
                    args.frequency
                    and score > args.threshold
                    or not args.frequency
                    and score < args.threshold
                ):
                    if word in words:
                        # mistranslation
                        false_positive += 1
                    else:
                        # correct translation
                        true_positive += 1
                # low confidence
                else:
                    if word in words:
                        # mistranslation
                        true_negative += 1
                    else:
                        # correct translation
                        false_negative += 1
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)
        f1 = 2 * (precision * recall) / (precision + recall)
        print()
        print(f'True Positive:  {true_positive}')
        print(f'False Positive: {false_positive}')
        print(f'True Negative:  {true_negative}')
        print(f'False Negative: {false_negative}')
        print()
        print(f'Precision:      {precision:0.2f}')
        print(f'Recall:         {recall:0.2f}')
        print(f'F1:             {f1:0.2f}')
        print()


if __name__ == '__main__':
    main()
