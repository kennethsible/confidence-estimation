#!/bin/bash

#$ -M ksible@nd.edu
#$ -m abe
#$ -q *@@nlp-a10
#$ -l gpu_card=1
#$ -N conf_sum_1

set -eo pipefail
$(poetry env activate)

poetry run python -m translation.translate \
	--sw-vocab data_de-en/en-de.vocab \
	--sw-model data_de-en/en-de.model \
	--model data_de-en/en-de.pt \
	--conf grad experiments/sum_1/wmt17.en-de.grad \
	--order 1 \
	--accum "sum" \
	--input data_annotation/wmt17.en-de.src \
	> experiments/sum_1/wmt17.en-de.hyp1
poetry run python experiments/conf2freq.py < experiments/sum_1/wmt17.en-de.grad > experiments/sum_1/wmt17.en-de.freq
poetry run python score_model.py --data-dir data_annotation --output-dir experiments/sum_1 > experiments/sum_1/max_F1.txt

poetry run python -m translation.translate \
	--sw-vocab data_de-en/en-de.vocab \
	--sw-model data_de-en/en-de.model \
	--model data_de-en/en-de.pt \
	--conf attn experiments/sum_1/wmt17.en-de.attn \
	--order 1 \
	--accum "sum" \
	--input data_annotation/wmt17.en-de.src \
	> experiments/sum_1/wmt17.en-de.hyp2
poetry run python score_model_conf.py --data-dir data_annotation --output-dir experiments/sum_1 --conf-type attn >> experiments/sum_1/max_F1.txt

poetry run python -m translation.translate \
	--sw-vocab data_de-en/en-de.vocab \
	--sw-model data_de-en/en-de.model \
	--model data_de-en/en-de.pt \
	--conf giza experiments/sum_1/wmt17.en-de.giza \
	--align data_mgiza/alignments.txt \
	--order 1 \
	--accum "sum" \
	--input data_annotation/wmt17.en-de.src \
	> experiments/sum_1/wmt17.en-de.hyp3
poetry run python score_model_conf.py --data-dir data_annotation --output-dir experiments/sum_1 --conf-type giza >> experiments/sum_1/max_F1.txt
