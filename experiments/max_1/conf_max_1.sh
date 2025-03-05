#!/bin/bash

#$ -M ksible@nd.edu
#$ -m abe
#$ -q *@@nlp-a10
#$ -l gpu_card=1
#$ -N conf_max_1

$(poetry env activate)
export PYTHONPATH="${PYTHONPATH}:{pwd}"

# python translation/translate.py \
# 	--sw-vocab data_de-en_large/en-de.vocab \
# 	--sw-model data_de-en_large/en-de.model \
#     --model data_annotation/en-de_large_001.pt \
# 	--conf gradient experiments/max_1/wmt17.en-de.grad \
# 	--order 1 \
# 	--accum "max" \
# 	--input data_annotation/wmt17.en-de.src \
# 	> experiments/max_1/wmt17.en-de.hyp1

# python translation/translate.py \
# 	--sw-vocab data_de-en_large/en-de.vocab \
# 	--sw-model data_de-en_large/en-de.model \
#     --model data_annotation/en-de_large_001.pt \
# 	--conf attention experiments/max_1/wmt17.en-de.attn \
# 	--order 1 \
# 	--accum "max" \
# 	--input data_annotation/wmt17.en-de.src \
# 	> experiments/max_1/wmt17.en-de.hyp2

python translation/translate.py \
	--sw-vocab data_de-en_large/en-de.vocab \
	--sw-model data_de-en_large/en-de.model \
    --model data_annotation/en-de_large_001.pt \
	--conf mgiza experiments/max_1/wmt17.en-de.giza \
    --align experiments/alignments.txt \
	--order 1 \
	--accum "max" \
	--input data_annotation/wmt17.en-de.src \
	> experiments/max_1/wmt17.en-de.hyp3

# python experiments/conf2freq.py < experiments/max_1/wmt17.en-de.grad > experiments/max_1/wmt17.en-de.freq
# python score_model.py --data-dir data_annotation --output-dir experiments/max_1 --mode "pr_F1" > experiments/max_1/max_F1.txt
# python score_model.py --data-dir data_annotation --output-dir experiments/max_1 --mode "pr_curve"
# python score_model.py --data-dir data_annotation --output-dir experiments/max_1 --mode "roc_curve"
# python score_model_conf.py --data-dir data_annotation --output-dir experiments/max_1 --mode "pr_F1" > experiments/max_1/max_F1_conf.txt
# python score_model_conf.py --data-dir data_annotation --output-dir experiments/max_1 --mode "pr_curve"
# python score_model_conf.py --data-dir data_annotation --output-dir experiments/max_1 --mode "roc_curve"
