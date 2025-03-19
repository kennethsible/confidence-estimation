#!/bin/bash

#$ -M ksible@nd.edu
#$ -m abe
#$ -q *@@nlp-a10
#$ -l gpu_card=1
#$ -N conf_avg_2

set -eo pipefail
$(poetry env activate)

poetry run python -m translation.translate \
	--sw-vocab data_de-en/en-de.vocab \
	--sw-model data_de-en/en-de.model \
	--model data_de-en/en-de.pt \
	--conf grad experiments/avg_2/wmt17.en-de.grad \
	--order 2 \
	--accum "avg" \
	--input data_annotation/wmt17.en-de.src \
	> experiments/avg_2/wmt17.en-de.hyp1
poetry run python experiments/conf2freq.py < experiments/avg_2/wmt17.en-de.grad > experiments/avg_2/wmt17.en-de.freq
poetry run python score_model.py --data-dir data_annotation --output-dir experiments/avg_2 > experiments/avg_2/max_F1.txt

poetry run python -m translation.translate \
	--sw-vocab data_de-en/en-de.vocab \
	--sw-model data_de-en/en-de.model \
	--model data_de-en/en-de.pt \
	--conf attn experiments/avg_2/wmt17.en-de.attn \
	--order 2 \
	--accum "avg" \
	--input data_annotation/wmt17.en-de.src \
	> experiments/avg_2/wmt17.en-de.hyp2
poetry run python score_model_conf.py --data-dir data_annotation --output-dir experiments/avg_2 --conf-type attn >> experiments/avg_2/max_F1.txt

poetry run python -m translation.translate \
	--sw-vocab data_de-en/en-de.vocab \
	--sw-model data_de-en/en-de.model \
	--model data_de-en/en-de.pt \
	--conf giza experiments/avg_2/wmt17.en-de.giza \
	--align data_mgiza/alignments.txt \
	--order 2 \
	--accum "avg" \
	--input data_annotation/wmt17.en-de.src \
	> experiments/avg_2/wmt17.en-de.hyp3
poetry run python score_model_conf.py --data-dir data_annotation --output-dir experiments/avg_2 --conf-type giza >> experiments/avg_2/max_F1.txt
