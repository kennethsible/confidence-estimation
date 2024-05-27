#!/bin/bash

python score_model.py --data-dir data_gpt-4o --output-dir output/avg_1 --mode pr_F1
python score_model.py --data-dir data_gpt-4o --output-dir output/avg_1 --mode pr_curve
python score_model.py --data-dir data_gpt-4o --output-dir output/avg_1 --mode roc_curve
python score_model.py --data-dir data_gpt-4o --output-dir output/avg_2 --mode pr_F1
python score_model.py --data-dir data_gpt-4o --output-dir output/avg_2 --mode pr_curve
python score_model.py --data-dir data_gpt-4o --output-dir output/avg_2 --mode roc_curve
python score_model.py --data-dir data_gpt-4o --output-dir output/avg_inf --mode pr_F1
python score_model.py --data-dir data_gpt-4o --output-dir output/avg_inf --mode pr_curve
python score_model.py --data-dir data_gpt-4o --output-dir output/avg_inf --mode roc_curve

python score_model.py --data-dir data_gpt-4o --output-dir output/max_1 --mode pr_F1
python score_model.py --data-dir data_gpt-4o --output-dir output/max_1 --mode pr_curve
python score_model.py --data-dir data_gpt-4o --output-dir output/max_1 --mode roc_curve
python score_model.py --data-dir data_gpt-4o --output-dir output/max_2 --mode pr_F1
python score_model.py --data-dir data_gpt-4o --output-dir output/max_2 --mode pr_curve
python score_model.py --data-dir data_gpt-4o --output-dir output/max_2 --mode roc_curve
python score_model.py --data-dir data_gpt-4o --output-dir output/max_inf --mode pr_F1
python score_model.py --data-dir data_gpt-4o --output-dir output/max_inf --mode pr_curve
python score_model.py --data-dir data_gpt-4o --output-dir output/max_inf --mode roc_curve

python score_model.py --data-dir data_gpt-4o --output-dir output/sum_1 --mode pr_F1
python score_model.py --data-dir data_gpt-4o --output-dir output/sum_1 --mode pr_curve
python score_model.py --data-dir data_gpt-4o --output-dir output/sum_1 --mode roc_curve
python score_model.py --data-dir data_gpt-4o --output-dir output/sum_2 --mode pr_F1
python score_model.py --data-dir data_gpt-4o --output-dir output/sum_2 --mode pr_curve
python score_model.py --data-dir data_gpt-4o --output-dir output/sum_2 --mode roc_curve
python score_model.py --data-dir data_gpt-4o --output-dir output/sum_inf --mode pr_F1
python score_model.py --data-dir data_gpt-4o --output-dir output/sum_inf --mode pr_curve
python score_model.py --data-dir data_gpt-4o --output-dir output/sum_inf --mode roc_curve
