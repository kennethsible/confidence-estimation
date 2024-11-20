import argparse
import re
import sys

from openai import OpenAI
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode', help='annotate or collate')
    annotate = subparsers.add_parser('annotate')
    annotate.add_argument('infile', nargs='?', type=argparse.FileType('r'), default=sys.stdin)
    annotate.add_argument('outfile', nargs='?', type=argparse.FileType('w'), default=sys.stdout)
    clean = subparsers.add_parser('collate')
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
            prompt = f'SRC is a {src_lang} sentence, REF is a reference {tgt_lang} translation of SRC, and HYP is a candidate {tgt_lang} translation of SRC. Identify all {tgt_lang} words in HYP that are mistranslated, along with their corresponding {src_lang} words in SRC. If a compound word contains spaces, choose the most relevant word to represent the compound. Format: "{tgt_lang} word" is a mistranslation of "{src_lang} word" ({tgt_lang} translation). SRC: {src_sent} REF: {ref_sent} HYP: {hyp_sent}'
            response = client.chat.completions.create(
                model='gpt-4o',  # gpt-3.5-turbo
                messages=[{'role': 'user', 'content': prompt}],
            )
            content = response.choices[0].message.content.replace('- ', '')
            args.outfile.write(', '.join(re.sub(r'[0-9][1-9]*\. ', '', content).split('\n')) + '\n')
    elif args.mode == 'collate':
        for line in tqdm(args.infile.readlines()):
            words = re.findall(r'.+? is a mistranslation of (.+?) \(', line.rstrip())
            args.outfile.write(', '.join(w.strip('\'"') for w in words) + '\n')


if __name__ == '__main__':
    main()
