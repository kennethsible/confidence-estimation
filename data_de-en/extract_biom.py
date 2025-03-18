import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True, help='data directory')
    args = parser.parse_args()

    src_lang, tgt_lang = 'de', 'en'
    data_dir = args.data_dir
    os.system(f'mkdir -p {data_dir}')

    biom_path = f'{data_dir}/biom'
    os.system(f'mkdir -p {biom_path}')
    med_data = []
    for year in range(20, 23):
        wmt_path = f'{biom_path}/wmt{year}'
        os.system(f'mkdir -p {wmt_path}')

        ref_to_id = {}
        with open(f'{wmt_path}/{src_lang}2{tgt_lang}_mapping.txt') as med_f:
            for line in med_f.readlines():
                doc_ref, doc_id = line.split('\t')
                ref_to_id[doc_ref] = doc_id.strip()

        src_data = {}
        with open(f'{wmt_path}/medline_{src_lang}2{tgt_lang}_{src_lang}.txt') as med_f:
            for line in med_f.readlines():
                doc_id, sent_id, sentence = line.split('\t')
                if doc_id not in src_data:
                    src_data[doc_id] = {}
                src_data[doc_id][sent_id] = sentence.rstrip()

        tgt_data = {}
        with open(f'{wmt_path}/medline_{src_lang}2{tgt_lang}_{tgt_lang}.txt') as med_f:
            for line in med_f.readlines():
                doc_id, sent_id, sentence = line.split('\t')
                if doc_id not in tgt_data:
                    tgt_data[doc_id] = {}
                tgt_data[doc_id][sent_id] = sentence.rstrip()

        with open(f'{wmt_path}/{src_lang}2{tgt_lang}_align_validation.tsv') as med_f:
            wmt_path += '/medline'
            with open(f'{wmt_path}.{src_lang}', 'w') as outfile:
                for line in med_f.readlines():
                    status, doc_ref, src_sent_id, _ = line.split('\t')
                    if status != 'OK':
                        continue
                    try:
                        if ',' in src_sent_id:
                            for sent_id in src_sent_id.split(','):
                                outfile.write(src_data[ref_to_id[doc_ref]][sent_id] + ' ')
                        else:
                            outfile.write(src_data[ref_to_id[doc_ref]][src_sent_id])
                        outfile.write('\n')
                    except KeyError:
                        pass

            med_f.seek(0)
            with open(f'{wmt_path}.{tgt_lang}', 'w') as outfile:
                for line in med_f.readlines():
                    status, doc_ref, _, tgt_sent_id = line.split('\t')
                    if status != 'OK':
                        continue
                    try:
                        if ',' in tgt_sent_id:
                            for sent_id in tgt_sent_id.split(','):
                                outfile.write(tgt_data[ref_to_id[doc_ref]][sent_id.rstrip()] + ' ')
                        else:
                            outfile.write(tgt_data[ref_to_id[doc_ref]][tgt_sent_id.rstrip()])
                        outfile.write('\n')
                    except KeyError:
                        pass

        with open(f'{data_dir}/biom/wmt{year}/medline.{src_lang}') as src_f, open(
            f'{data_dir}/biom/wmt{year}/medline.{tgt_lang}'
        ) as ref_f:
            for src_line, ref_line in zip(src_f.readlines(), ref_f.readlines()):
                if not src_line.strip().isupper() or not ref_line.strip().isupper():
                    med_data.append(f'{src_line}\t{ref_line}')

    with open(f'{data_dir}/biom/biom.{src_lang}', 'w') as src_f, open(
        f'{data_dir}/biom/biom.{tgt_lang}', 'w'
    ) as ref_f:
        for line in dict.fromkeys(med_data):
            src_line, ref_line = line.split('\t')
            src_f.write(src_line)
            ref_f.write(ref_line)
    os.system(f'wc -l {data_dir}/biom/biom.{src_lang}')
    os.system(f'wc -l {data_dir}/biom/biom.{tgt_lang}')


if __name__ == '__main__':
    main()
