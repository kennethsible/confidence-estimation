#!/bin/bash

#$ -M ksible@nd.edu
#$ -m abe
#$ -q *@@nlp-a10
#$ -l gpu_card=1
#$ -N conf_max_inf

set -eo pipefail
$(poetry env activate)

poetry run python -m translation.translate \
	--sw-vocab data_de-en/en-de.vocab \
	--sw-model data_de-en/en-de.model \
	--model data_de-en/en-de.pt \
	--conf grad experiments/max_inf/wmt17.en-de.grad \
	--order "inf" \
	--accum "max" \
	--input data_annotation/wmt17.en-de.src \
	> experiments/max_inf/wmt17.en-de.hyp1
poetry run python experiments/conf2freq.py < experiments/max_inf/wmt17.en-de.grad > experiments/max_inf/wmt17.en-de.freq
poetry run python score_model.py --data-dir data_annotation --output-dir experiments/max_inf > experiments/max_inf/max_F1.txt

poetry run python -m translation.translate \
	--sw-vocab data_de-en/en-de.vocab \
	--sw-model data_de-en/en-de.model \
	--model data_de-en/en-de.pt \
	--conf attn experiments/max_inf/wmt17.en-de.attn \
	--order "inf" \
	--accum "max" \
	--input data_annotation/wmt17.en-de.src \
	> experiments/max_inf/wmt17.en-de.hyp2
poetry run python score_model_conf.py --data-dir data_annotation --output-dir experiments/max_inf --conf-type attn >> experiments/max_inf/max_F1.txt

poetry run python -m translation.translate \
	--sw-vocab data_de-en/en-de.vocab \
	--sw-model data_de-en/en-de.model \
	--model data_de-en/en-de.pt \
	--conf giza experiments/max_inf/wmt17.en-de.giza \
	--align data_mgiza/alignments.txt \
	--order "inf" \
	--accum "max" \
	--input data_annotation/wmt17.en-de.src \
	> experiments/max_inf/wmt17.en-de.hyp3
poetry run python score_model_conf.py --data-dir data_annotation --output-dir experiments/max_inf --conf-type giza >> experiments/max_inf/max_F1.txt
