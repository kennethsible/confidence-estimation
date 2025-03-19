#!/bin/bash

#$ -M ksible@nd.edu
#$ -m abe
#$ -q *@@nlp-a10
#$ -l gpu_card=1
#$ -N conf_avg_1

set -eo pipefail
$(poetry env activate)

poetry run python -m translation.translate \
	--sw-vocab data_de-en/en-de.vocab \
	--sw-model data_de-en/en-de.model \
	--model data_de-en/en-de.pt \
	--conf grad experiments/avg_1/wmt17.en-de.grad \
	--order 1 \
	--accum "avg" \
	--input data_annotation/wmt17.en-de.src \
	> experiments/avg_1/wmt17.en-de.hyp1
diff -q experiments/avg_1/wmt17.en-de.hyp1 data_annotation/wmt17.en-de.hyp >/dev/null || exit 1
poetry run python experiments/conf2freq.py < experiments/avg_1/wmt17.en-de.grad > experiments/avg_1/wmt17.en-de.freq
poetry run python score_model.py --data-dir data_annotation --output-dir experiments/avg_1 > experiments/avg_1/max_F1.txt

poetry run python -m translation.translate \
	--sw-vocab data_de-en/en-de.vocab \
	--sw-model data_de-en/en-de.model \
	--model data_de-en/en-de.pt \
	--conf attn experiments/avg_1/wmt17.en-de.attn \
	--order 1 \
	--accum "avg" \
	--input data_annotation/wmt17.en-de.src \
	> experiments/avg_1/wmt17.en-de.hyp2
diff -q experiments/avg_1/wmt17.en-de.hyp2 data_annotation/wmt17.en-de.hyp >/dev/null || exit 1
poetry run python score_model_conf.py --data-dir data_annotation --output-dir experiments/avg_1 --conf-type attn >> experiments/avg_1/max_F1.txt

poetry run python -m translation.translate \
	--sw-vocab data_de-en/en-de.vocab \
	--sw-model data_de-en/en-de.model \
	--model data_de-en/en-de.pt \
	--conf giza experiments/avg_1/wmt17.en-de.giza \
	--align data_mgiza/alignments.txt \
	--order 1 \
	--accum "avg" \
	--input data_annotation/wmt17.en-de.src \
	> experiments/avg_1/wmt17.en-de.hyp3
diff -q experiments/avg_1/wmt17.en-de.hyp3 data_annotation/wmt17.en-de.hyp >/dev/null || exit 1
poetry run python score_model_conf.py --data-dir data_annotation --output-dir experiments/avg_1 --conf-type giza >> experiments/avg_1/max_F1.txt
