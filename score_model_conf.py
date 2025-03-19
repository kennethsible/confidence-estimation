import argparse
import json
import math

from matplotlib import pyplot as plt
from numpy import linspace
from scipy import integrate
from tqdm import tqdm


def format_auc(x: float, y: float):
    x_str, y_str = f'{x:.2E}', f'{y:.2E}'
    x_bstr, x_exp = x_str.split('E')
    y_bstr, y_exp = y_str.split('E')
    x_base, y_base = float(x_bstr), float(y_bstr)
    if int(x_exp) < int(y_exp):
        y_base *= (int(y_exp) - int(x_exp)) * 10
        return f'{x_base:.0f}E{x_exp}', f'{y_base:.0f}E{x_exp}'
    elif int(x_exp) > int(y_exp):
        x_base *= (int(x_exp) - int(y_exp)) * 10
        return f'{x_base:.0f}E{y_exp}', f'{y_base:.0f}E{y_exp}'
    return x_str, y_str


def score_model(
    gpt_file, conf_file, threshold: int, rescale: bool = False, frequency: bool = False
) -> tuple[tuple[float | None, float | None, float | None], tuple[int, int]]:
    false_positive = false_negative = true_positive = true_negative = 0
    for line1, line2 in zip(gpt_file.readlines(), json.load(conf_file)):
        words = line1.split(', ')  # [x for y in line1.split(', ') for x in y.split()]
        for word, score in line2:
            if rescale:
                score *= 100
            if (frequency and score < threshold) or (not frequency and score > threshold):
                if word in words:  # mistranslation
                    true_positive += 1
                else:
                    false_positive += 1
            else:  # high confidence
                if word in words:  # mistranslation
                    false_negative += 1
                else:
                    true_negative += 1
    # Accuracy -> balanced, every class is equally important
    # F1 ->       imbalanced, positive class is more important
    # ROC AUC ->  heavily imbalanced, every class is equally important
    # PR AUC ->   heavily imbalanced, positive class is more important
    precision = recall = F1 = None
    # Precision -> out of all the times the model predicted positive, how often was it actually correct?
    # Recall ->    out of all the actual positives, how many did the model correctly identify?
    if true_positive > 0 or false_positive > 0:
        # Class Imbalance: If your dataset has very few actual positives, at extreme
        #   thresholds, almost all predictions may be incorrect, reducing precision.
        # Solution: Precision-Recall (PR) Curve
        precision = true_positive / (true_positive + false_positive)
    if true_positive > 0 or false_negative > 0:
        recall = true_positive / (true_positive + false_negative)
    if precision and recall:
        F1 = 2 * (precision * recall) / (precision + recall)
    return (precision, recall, F1), (true_positive, false_positive)


def pr_curve(data_dir: str, output_dir: str, conf_type: str):
    grad_precision = []
    grad_recall = []
    with (
        open(f'{data_dir}/annotate_errors.txt') as gpt_f,
        open(f'{output_dir}/wmt17.en-de.grad') as grad_f,
    ):
        if output_dir.split('_')[-1] in ('2', 'inf'):
            N = max(math.floor(grad * 100) for sent in json.load(grad_f) for _, grad in sent)
            rescale = True
        else:
            N = max(math.floor(grad) for sent in json.load(grad_f) for _, grad in sent)
            rescale = False
        grad_f.seek(0)
        for threshold in tqdm(linspace(0, N + 1, 100), disable=True):
            (precision, recall, _), _ = score_model(
                gpt_f, grad_f, threshold=threshold, rescale=rescale
            )
            if precision and recall:
                grad_precision.append(precision)
                grad_recall.append(recall)
            gpt_f.seek(0)
            grad_f.seek(0)

    conf_precision = []
    conf_recall = []
    with (
        open(f'{data_dir}/annotate_errors.txt') as gpt_f,
        open(f'{output_dir}/wmt17.en-de.{conf_type}') as conf_f,
    ):
        if output_dir.split('_')[-1] in ('2', 'inf'):
            N = max(math.floor(attn * 100) for sent in json.load(conf_f) for _, attn in sent)
            rescale = True
        else:
            N = max(math.floor(attn) for sent in json.load(conf_f) for _, attn in sent)
            rescale = False
        conf_f.seek(0)
        for threshold in tqdm(linspace(0, N + 1, 1000)):
            (precision, recall, _), _ = score_model(
                gpt_f, conf_f, threshold=threshold, rescale=rescale
            )
            if precision and recall:
                conf_precision.append(precision)
                conf_recall.append(recall)
            gpt_f.seek(0)
            conf_f.seek(0)

    plt.figure()
    grad_auc = abs(integrate.trapezoid(grad_recall, grad_precision))
    conf_auc = abs(integrate.trapezoid(conf_recall, conf_precision))
    grad_label, conf_label = format_auc(grad_auc, conf_auc)
    plt.plot(grad_recall, grad_precision, label=f'Gradient, AUC = {grad_label}')
    match conf_type:
        case 'attn':
            conf_title = 'Attention'
        case 'giza':
            conf_title = 'MGIZA'
        case _:
            raise NotImplementedError(conf_type)
    plt.plot(conf_recall, conf_precision, 'r', label=f'{conf_title}, AUC = {conf_label}')
    plt.title('PR Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.savefig(f'{output_dir}/pr_curve_{conf_type}.png', dpi=300)


def roc_curve(data_dir: str, output_dir: str, conf_type: str):
    grad_tp = []
    grad_fp = []
    with (
        open(f'{data_dir}/annotate_errors.txt') as gpt_f,
        open(f'{output_dir}/wmt17.en-de.grad') as grad_f,
    ):
        if output_dir.split('_')[-1] in ('2', 'inf'):
            N = max(math.floor(grad * 100) for sent in json.load(grad_f) for _, grad in sent)
            rescale = True
        else:
            N = max(math.floor(grad) for sent in json.load(grad_f) for _, grad in sent)
            rescale = False
        grad_f.seek(0)
        for threshold in tqdm(linspace(0, N + 1, 1000)):
            _, (true_positive, false_positive) = score_model(
                gpt_f, grad_f, threshold=threshold, rescale=rescale
            )
            grad_tp.append(true_positive)
            grad_fp.append(false_positive)
            gpt_f.seek(0)
            grad_f.seek(0)

    conf_tp = []
    conf_fp = []
    with (
        open(f'{data_dir}/annotate_errors.txt') as gpt_f,
        open(f'{output_dir}/wmt17.en-de.{conf_type}') as conf_f,
    ):
        if output_dir.split('_')[-1] in ('2', 'inf'):
            N = max(math.floor(attn * 100) for sent in json.load(conf_f) for _, attn in sent)
            rescale = True
        else:
            N = max(math.floor(attn) for sent in json.load(conf_f) for _, attn in sent)
            rescale = False
        conf_f.seek(0)
        for threshold in tqdm(linspace(0, N + 1, 1000)):
            _, (true_positive, false_positive) = score_model(
                gpt_f, conf_f, threshold=threshold, rescale=rescale
            )
            conf_tp.append(true_positive)
            conf_fp.append(false_positive)
            gpt_f.seek(0)
            conf_f.seek(0)

    plt.figure()
    grad_auc = abs(integrate.trapezoid(grad_tp, grad_fp))
    conf_auc = abs(integrate.trapezoid(conf_tp, conf_fp))
    grad_label, conf_label = format_auc(grad_auc, conf_auc)
    plt.plot(grad_fp, grad_tp, label=f'Gradient, AUC = {grad_label}')
    match conf_type:
        case 'attn':
            conf_title = 'Attention'
        case 'giza':
            conf_title = 'MGIZA'
        case _:
            raise NotImplementedError(conf_type)
    plt.plot(conf_fp, conf_tp, 'r', label=f'{conf_title}, AUC = {conf_label}')
    plt.title('ROC Curve')
    plt.xlabel('False Positive')
    plt.ylabel('True Positive')
    plt.legend()
    plt.savefig(f'{output_dir}/roc_curve_{conf_type}.png', dpi=300)


def pr_F1(data_dir: str, output_dir: str, conf_type: str):
    grad_precision = []
    grad_recall = []
    grad_F1 = []
    grad_x = []
    with (
        open(f'{data_dir}/annotate_errors.txt') as gpt_f,
        open(f'{output_dir}/wmt17.en-de.grad') as grad_f,
    ):
        if output_dir.split('_')[-1] in ('2', 'inf'):
            N = max(math.floor(grad * 100) for sent in json.load(grad_f) for _, grad in sent)
            rescale = True
        else:
            N = max(math.floor(grad) for sent in json.load(grad_f) for _, grad in sent)
            rescale = False
        grad_f.seek(0)
        max_F1 = (0.0, 0)
        for threshold in tqdm(linspace(0, N + 1, 1000)):
            (precision, recall, F1), _ = score_model(
                gpt_f, grad_f, threshold=threshold, rescale=rescale
            )
            if F1 and F1 > max_F1[0]:
                max_F1 = (F1, threshold)
            grad_x.append(threshold)
            grad_precision.append(precision)
            grad_recall.append(recall)
            grad_F1.append(F1)
            gpt_f.seek(0)
            grad_f.seek(0)
        print(output_dir, max_F1)

        # print(f'Precision: {precision:0.2f}')
        # print(f'Recall:    {recall:0.2f}')
        # print(f'F1:        {F1:0.2f}')
        # return

    conf_precision = []
    conf_recall = []
    conf_F1 = []
    conf_x = []
    with (
        open(f'{data_dir}/annotate_errors.txt') as gpt_f,
        open(f'{output_dir}/wmt17.en-de.{conf_type}') as conf_f,
    ):
        if output_dir.split('_')[-1] in ('2', 'inf'):
            N = max(math.floor(attn * 100) for sent in json.load(conf_f) for _, attn in sent)
            rescale = True
        else:
            N = max(math.floor(attn) for sent in json.load(conf_f) for _, attn in sent)
            rescale = False
        conf_f.seek(0)
        max_F1 = (0.0, 0)
        for threshold in tqdm(linspace(0, N + 1, 1000)):
            (precision, recall, F1), _ = score_model(
                gpt_f, conf_f, threshold=threshold, rescale=rescale
            )
            if F1 and F1 > max_F1[0]:
                max_F1 = (F1, threshold)
            conf_x.append(threshold)
            conf_precision.append(precision)
            conf_recall.append(recall)
            conf_F1.append(F1)
            gpt_f.seek(0)
            conf_f.seek(0)
        print(output_dir, max_F1)

    fig, axs = plt.subplots(3, 2, figsize=(10, 10))

    axs[0, 0].set_title('Gradient')
    axs[0, 0].set_ylabel('Precision')
    axs[0, 0].plot(grad_x, grad_precision)
    match conf_type:
        case 'attn':
            conf_title = 'Attention'
        case 'giza':
            conf_title = 'MGIZA'
        case _:
            raise NotImplementedError(conf_type)
    axs[0, 1].set_title(conf_title)
    axs[0, 1].plot(conf_x, conf_precision, 'r')
    axs[1, 0].set_ylabel('Recall')
    axs[1, 0].plot(grad_x, grad_recall)
    axs[1, 1].set_xlabel('Threshold')
    axs[1, 1].plot(conf_x, conf_recall, 'r')
    axs[2, 0].set_ylabel('F1')
    axs[2, 0].set_xlabel('Threshold')
    axs[2, 0].plot(grad_x, grad_F1)
    axs[2, 1].set_xlabel('Threshold')
    axs[2, 1].plot(conf_x, conf_F1, 'r')

    fig.savefig(f'{output_dir}/pr_F1_{conf_type}.png', dpi=300)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True, help='data directory')
    parser.add_argument('--output-dir', required=True, help='output directory')
    parser.add_argument('--conf-type', required=True, help='confidence type')
    args = parser.parse_args()

    pr_curve(args.data_dir, args.output_dir, args.conf_type)
    roc_curve(args.data_dir, args.output_dir, args.conf_type)
    pr_F1(args.data_dir, args.output_dir, args.conf_type)


if __name__ == '__main__':
    main()
