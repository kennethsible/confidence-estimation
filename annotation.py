import argparse
import sys

from openai import OpenAI


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('infile', nargs='?', type=argparse.FileType('r'), default=sys.stdin)
    parser.add_argument('outfile', nargs='?', type=argparse.FileType('w'), default=sys.stdout)
    args = parser.parse_args()

    client = OpenAI()
    src_lang, tgt_lang = 'German', 'English'

    for line in args.infile.readlines():
        src_sent, ref_sent, hyp_sent = line.split('\t')
        prompt = f'SRC is a {src_lang} sentence, REF is a reference {tgt_lang} translation of SRC, and HYP is a candidate {tgt_lang} translation of SRC. In a comma-separated list, identify all {tgt_lang} words in HYP that are mistranslated, along with their corresponding {src_lang} words in SRC. Format: "{tgt_lang} word" is a mistranslation of "{src_lang} word" ({tgt_lang} translation). SRC: {src_sent}. REF: {ref_sent}. HYP: {hyp_sent}.'
        response = client.chat.completions.create(
            model='gpt-3.5-turbo',
            messages=[{'role': 'user', 'content': prompt}],
        )
        args.outfile.write(response.choices[0].message.content + '\n')


if __name__ == '__main__':
    main()
