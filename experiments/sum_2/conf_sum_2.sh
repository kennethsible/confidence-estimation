#!/bin/bash

#$ -M ksible@nd.edu
#$ -m abe
#$ -q *@@nlp-a10
#$ -l gpu_card=1
#$ -N conf_sum_2

$(poetry env activate)
export PYTHONPATH="${PYTHONPATH}:{pwd}"

# python translation/translate.py \
# 	--sw-vocab data_de-en_large/en-de.vocab \
# 	--sw-model data_de-en_large/en-de.model \
#     --model data_annotation/en-de_large_001.pt \
# 	--conf gradient experiments/sum_2/wmt17.en-de.grad \
# 	--order 2 \
# 	--accum "sum" \
# 	--input data_annotation/wmt17.en-de.src \
# 	> experiments/sum_2/wmt17.en-de.hyp1

# python translation/translate.py \
# 	--sw-vocab data_de-en_large/en-de.vocab \
# 	--sw-model data_de-en_large/en-de.model \
#     --model data_annotation/en-de_large_001.pt \
# 	--conf attention experiments/sum_2/wmt17.en-de.attn \
# 	--order 2 \
# 	--accum "sum" \
# 	--input data_annotation/wmt17.en-de.src \
# 	> experiments/sum_2/wmt17.en-de.hyp2

# python experiments/conf2freq.py < experiments/sum_2/wmt17.en-de.grad > experiments/sum_2/wmt17.en-de.freq
# python score_model.py --data-dir data_annotation --output-dir experiments/sum_2 --mode "pr_F1" > experiments/sum_2/max_F1.txt
# python score_model.py --data-dir data_annotation --output-dir experiments/sum_2 --mode "pr_curve"
# python score_model.py --data-dir data_annotation --output-dir experiments/sum_2 --mode "roc_curve"
python score_model_conf.py --data-dir data_annotation --output-dir experiments/sum_2 --mode "pr_F1" > experiments/sum_2/max_F1_conf.txt
python score_model_conf.py --data-dir data_annotation --output-dir experiments/sum_2 --mode "pr_curve"
python score_model_conf.py --data-dir data_annotation --output-dir experiments/sum_2 --mode "roc_curve"
