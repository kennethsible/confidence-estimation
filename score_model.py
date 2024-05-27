import json
import math
from itertools import chain

from matplotlib import pyplot as plt
from scipy import integrate
from tqdm import tqdm


def format_auc(x: float, y: float):
    x_str, y_str = f'{x:.2E}', f'{y:.2E}'
    x_bstr, x_exp = x_str.split('E')
    y_bstr, y_exp = y_str.split('E')
    x_base, y_base = float(x_bstr), float(y_bstr)
    if int(x_exp) < int(y_exp):
        y_base *= (int(y_exp) - int(x_exp)) * 10
        return f'{x_base:.2f}E{x_exp}', f'{y_base:.2f}E{x_exp}'
    elif int(x_exp) > int(y_exp):
        x_base *= (int(x_exp) - int(y_exp)) * 10
        return f'{x_base:.2f}E{y_exp}', f'{y_base:.2f}E{y_exp}'
    return x_str, y_str


def score_model(
    gpt_file, conf_file, threshold: int, rescale: bool = False, frequency: bool = False
) -> tuple[tuple[float, float, float], tuple[int, int]]:
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
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    F1 = 2 * (precision * recall) / (precision + recall)
    return (precision, recall, F1), (true_positive, false_positive)


def pr_curve(data_dir: str, output_dir: str):
    conf_precision = []
    conf_recall = []
    with open(f'{data_dir}/mistranslated.txt') as gpt_f, open(
        f'{output_dir}/news-test2008.json'
    ) as conf_f:
        if output_dir.split('_')[-1] in ('2', 'inf'):
            N = max(math.floor(conf * 100) for sent in json.load(conf_f) for _, conf in sent)
            rescale = True
        else:
            N = max(math.floor(conf) for sent in json.load(conf_f) for _, conf in sent)
            rescale = False
        conf_f.seek(0)
        for threshold in tqdm(range(1, N + 1)):
            try:
                (precision, recall, _), _ = score_model(
                    gpt_f, conf_f, threshold=threshold, rescale=rescale
                )
            except ZeroDivisionError:
                gpt_f.seek(0)
                conf_f.seek(0)
                continue
            conf_precision.append(precision)
            conf_recall.append(recall)
            gpt_f.seek(0)
            conf_f.seek(0)
    conf_precision.append(0.0)
    conf_recall.append(0.0)

    freq_precision = [0.0]
    freq_recall = [0.0]
    with open(f'{data_dir}/mistranslated.txt') as gpt_f, open(
        f'{output_dir}/news-test2008.freq.json'
    ) as freq_f:
        N = max(math.floor(freq) for sent in json.load(freq_f) for _, freq in sent)
        freq_f.seek(0)
        for threshold in tqdm(list(chain(range(1, 200), range(200, N + 1, (N + 1 - 200) // 1000)))):
            (precision, recall, _), _ = score_model(
                gpt_f, freq_f, threshold=threshold, frequency=True
            )
            freq_precision.append(precision)
            freq_recall.append(recall)
            gpt_f.seek(0)
            freq_f.seek(0)

    plt.figure()
    conf_auc = abs(integrate.trapezoid(conf_recall, conf_precision))
    freq_auc = abs(integrate.trapezoid(freq_recall, freq_precision))
    conf_label, freq_label = format_auc(conf_auc, freq_auc)
    plt.plot(conf_recall, conf_precision, label=f'Gradient, AUC = {conf_label}')
    plt.plot(freq_recall, freq_precision, label=f'Frequency, AUC = {freq_label}')
    plt.title('PR Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.savefig(f'{output_dir}/pr_curve.png', dpi=300)


def roc_curve(data_dir: str, output_dir: str):
    conf_tp = []
    conf_fp = []
    with open(f'{data_dir}/mistranslated.txt') as gpt_f, open(
        f'{output_dir}/news-test2008.json'
    ) as conf_f:
        if output_dir.split('_')[-1] in ('2', 'inf'):
            N = max(math.floor(conf * 100) for sent in json.load(conf_f) for _, conf in sent)
            rescale = True
        else:
            N = max(math.floor(conf) for sent in json.load(conf_f) for _, conf in sent)
            rescale = False
        conf_f.seek(0)
        for threshold in tqdm(range(1, N + 1)):
            try:
                _, (true_positive, false_positive) = score_model(
                    gpt_f, conf_f, threshold=threshold, rescale=rescale
                )
            except ZeroDivisionError:
                gpt_f.seek(0)
                conf_f.seek(0)
                continue
            conf_tp.append(true_positive)
            conf_fp.append(false_positive)
            gpt_f.seek(0)
            conf_f.seek(0)
    conf_tp.append(0)
    conf_fp.append(0)

    freq_tp = [0]
    freq_fp = [0]
    with open(f'{data_dir}/mistranslated.txt') as gpt_f, open(
        f'{output_dir}/news-test2008.freq.json'
    ) as freq_f:
        N = max(math.floor(freq) for sent in json.load(freq_f) for _, freq in sent)
        freq_f.seek(0)
        for threshold in tqdm(list(chain(range(1, 200), range(200, N + 1, (N + 1 - 200) // 1000)))):
            _, (true_positive, false_positive) = score_model(
                gpt_f, freq_f, threshold=threshold, frequency=True
            )
            freq_tp.append(true_positive)
            freq_fp.append(false_positive)
            gpt_f.seek(0)
            freq_f.seek(0)

    plt.figure()
    conf_auc = abs(integrate.trapezoid(conf_tp, conf_fp))
    freq_auc = abs(integrate.trapezoid(freq_tp, freq_fp))
    conf_label, freq_label = format_auc(conf_auc, freq_auc)
    plt.plot(conf_fp, conf_tp, label=f'Gradient, AUC = {conf_label}')
    plt.plot(freq_fp, freq_tp, label=f'Frequency, AUC = {freq_label}')
    plt.title('ROC Curve')
    plt.xlabel('False Positive')
    plt.ylabel('True Positive')
    plt.legend()
    plt.savefig(f'{output_dir}/roc_curve.png', dpi=300)


def pr_F1(data_dir: str, output_dir: str):
    conf_precision = [0.0]
    conf_recall = [0.0]
    conf_F1 = [0.0]
    conf_x = [0]
    with open(f'{data_dir}/mistranslated.txt') as gpt_f, open(
        f'{output_dir}/news-test2008.json'
    ) as conf_f:
        if output_dir.split('_')[-1] in ('2', 'inf'):
            N = max(math.floor(conf * 100) for sent in json.load(conf_f) for _, conf in sent)
            rescale = True
        else:
            N = max(math.floor(conf) for sent in json.load(conf_f) for _, conf in sent)
            rescale = False
        conf_f.seek(0)
        max_F1 = (0.0, 0)
        for threshold in tqdm(range(1, N + 1)):
            try:
                (precision, recall, F1), _ = score_model(
                    gpt_f, conf_f, threshold=threshold, rescale=rescale
                )
            except ZeroDivisionError:
                gpt_f.seek(0)
                conf_f.seek(0)
                continue
            if F1 > max_F1[0]:
                max_F1 = (F1, threshold)
            conf_x.append(threshold)
            conf_precision.append(precision)
            conf_recall.append(recall)
            conf_F1.append(F1)
            gpt_f.seek(0)
            conf_f.seek(0)
        print(output_dir, max_F1)

        # print(f'Precision: {precision:0.2f}')
        # print(f'Recall:    {recall:0.2f}')
        # print(f'F1:        {F1:0.2f}')
        # return

    freq_precision = [0.0]
    freq_recall = [0.0]
    freq_F1 = [0.0]
    freq_x = [0]
    with open(f'{data_dir}/mistranslated.txt') as gpt_f, open(
        f'{output_dir}/news-test2008.freq.json'
    ) as freq_f:
        for threshold in tqdm(list(range(1, conf_x[-1] + 1))):
            (precision, recall, F1), _ = score_model(
                gpt_f, freq_f, threshold=threshold, frequency=True
            )
            freq_x.append(threshold)
            freq_precision.append(precision)
            freq_recall.append(recall)
            freq_F1.append(F1)
            gpt_f.seek(0)
            freq_f.seek(0)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 5))

    # plt.figure()
    ax1.set_xlabel('Threshold')
    ax1.set_title('Precision')
    ax1.plot(conf_x, conf_precision, label='Gradient')
    ax1.plot(freq_x, freq_precision, '--', label='Frequency')
    ax1.legend()
    # plt.savefig(f'{output_dir}/precision.png', dpi=300)

    # plt.figure()
    ax2.set_xlabel('Threshold')
    ax2.set_title('Recall')
    ax2.plot(conf_x, conf_recall, label='Gradient')
    ax2.plot(freq_x, freq_recall, '--', label='Frequency')
    ax2.legend()
    # plt.savefig(f'{output_dir}/recall.png', dpi=300)

    # plt.figure()
    ax3.set_xlabel('Threshold')
    ax3.set_title('F1')
    ax3.plot(conf_x, conf_F1, label='Gradient')
    ax3.plot(freq_x, freq_F1, '--', label='Frequency')
    ax3.legend()
    # plt.savefig(f'{output_dir}/F1.png', dpi=300)

    fig.savefig(f'{output_dir}/pr_F1.png', dpi=300)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True, help='data directory')
    parser.add_argument('--output-dir', required=True, help='output directory')
    parser.add_argument('--mode', required=True, help='pr_curve, roc_curve, or pr_F1')
    args = parser.parse_args()

    if args.mode == 'pr_curve':
        pr_curve(args.data_dir, args.output_dir)
    elif args.mode == 'roc_curve':
        roc_curve(args.data_dir, args.output_dir)
    elif args.mode == 'pr_F1':
        pr_F1(args.data_dir, args.output_dir)


if __name__ == '__main__':
    import argparse

    main()
