#!/bin/bash

python score_model.py --data-dir data_annotation --output-dir experiments/avg_1 --mode pr_F1
python score_model.py --data-dir data_annotation --output-dir experiments/avg_1 --mode pr_curve
python score_model.py --data-dir data_annotation --output-dir experiments/avg_1 --mode roc_curve
python score_model.py --data-dir data_annotation --output-dir experiments/avg_2 --mode pr_F1
python score_model.py --data-dir data_annotation --output-dir experiments/avg_2 --mode pr_curve
python score_model.py --data-dir data_annotation --output-dir experiments/avg_2 --mode roc_curve
python score_model.py --data-dir data_annotation --output-dir experiments/avg_inf --mode pr_F1
python score_model.py --data-dir data_annotation --output-dir experiments/avg_inf --mode pr_curve
python score_model.py --data-dir data_annotation --output-dir experiments/avg_inf --mode roc_curve

python score_model.py --data-dir data_annotation --output-dir experiments/max_1 --mode pr_F1
python score_model.py --data-dir data_annotation --output-dir experiments/max_1 --mode pr_curve
python score_model.py --data-dir data_annotation --output-dir experiments/max_1 --mode roc_curve
python score_model.py --data-dir data_annotation --output-dir experiments/max_2 --mode pr_F1
python score_model.py --data-dir data_annotation --output-dir experiments/max_2 --mode pr_curve
python score_model.py --data-dir data_annotation --output-dir experiments/max_2 --mode roc_curve
python score_model.py --data-dir data_annotation --output-dir experiments/max_inf --mode pr_F1
python score_model.py --data-dir data_annotation --output-dir experiments/max_inf --mode pr_curve
python score_model.py --data-dir data_annotation --output-dir experiments/max_inf --mode roc_curve

python score_model.py --data-dir data_annotation --output-dir experiments/sum_1 --mode pr_F1
python score_model.py --data-dir data_annotation --output-dir experiments/sum_1 --mode pr_curve
python score_model.py --data-dir data_annotation --output-dir experiments/sum_1 --mode roc_curve
python score_model.py --data-dir data_annotation --output-dir experiments/sum_2 --mode pr_F1
python score_model.py --data-dir data_annotation --output-dir experiments/sum_2 --mode pr_curve
python score_model.py --data-dir data_annotation --output-dir experiments/sum_2 --mode roc_curve
python score_model.py --data-dir data_annotation --output-dir experiments/sum_inf --mode pr_F1
python score_model.py --data-dir data_annotation --output-dir experiments/sum_inf --mode pr_curve
python score_model.py --data-dir data_annotation --output-dir experiments/sum_inf --mode roc_curve
