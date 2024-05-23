import json

from matplotlib import pyplot as plt
from scipy import integrate
from tqdm import tqdm


def score_model(
    gpt_file, conf_file, threshold: int, frequency: bool = False
) -> tuple[tuple[float, float, float], tuple[int, int]]:
    false_positive = false_negative = true_positive = true_negative = 0
    for line1, line2 in zip(gpt_file.readlines(), json.load(conf_file)):
        words = line1.split(', ')  # [x for y in line1.split(', ') for x in y.split()]
        for word, score in line2:
            # high confidence
            if (frequency and score >= threshold) or (not frequency and score <= threshold):
                if word in words:
                    # mistranslation
                    false_positive += 1
                else:
                    # correct translation
                    true_positive += 1
            # low confidence
            else:
                if word in words:
                    # mistranslation
                    true_negative += 1
                else:
                    # correct translation
                    false_negative += 1
    # Accuracy -> balanced, every class is equally important
    # F1 ->       imbalanced, positive class is more important
    # ROC AUC ->  heavily imbalanced, every class is equally important
    # PR AUC ->   heavily imbalanced, positive class is more important
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    F1 = 2 * (precision * recall) / (precision + recall)  # harmonic mean
    # F_beta = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall)
    return (precision, recall, F1), (true_positive, false_positive)


def pr_curve():
    conf_x = []
    conf_precision = []
    conf_recall = []
    with open('data_annotation/mistranslated.txt') as gpt_f, open(
        'data_annotation/news-test2008.json'
    ) as conf_f:
        threshold = 1
        while True:
            (precision, recall, _), _ = score_model(
                gpt_f, conf_f, threshold=threshold, frequency=False
            )
            if precision > 0.99:
                gpt_f.seek(0)
                conf_f.seek(0)
                threshold += 1
                continue
            if precision < 0.97:
                print(threshold)
                break
            conf_x.append(threshold)
            conf_precision.append(precision)
            conf_recall.append(recall)
            gpt_f.seek(0)
            conf_f.seek(0)
            threshold += 1

    freq_x = []
    freq_precision = []
    freq_recall = []
    with open('data_annotation/mistranslated.txt') as gpt_f, open(
        'data_annotation/news-test2008.freq.json'
    ) as freq_f:
        threshold = 0
        while True:
            (precision, recall, _), _ = score_model(
                gpt_f, freq_f, threshold=threshold, frequency=True
            )
            if precision > 0.99:
                print(threshold)
                break
            if precision < 0.97:
                gpt_f.seek(0)
                freq_f.seek(0)
                threshold += 1
                continue
            freq_x.append(threshold)
            freq_precision.append(precision)
            freq_recall.append(recall)
            gpt_f.seek(0)
            freq_f.seek(0)
            threshold += 1

    plt.figure()
    aoc = integrate.trapezoid(conf_recall, conf_precision)
    plt.plot(conf_precision, conf_recall, label=f'Gradient, AOC = {-aoc:.2E}')
    aoc = integrate.trapezoid(freq_recall, freq_precision)
    plt.plot(freq_precision, freq_recall, label=f'Frequency, AOC = {aoc:.2E}')
    plt.title('PR Curve')
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.legend()
    plt.savefig('figures/pr_curve.png', dpi=300)


def roc_curve():
    conf_tp = []
    conf_fp = []
    with open('data_annotation/mistranslated.txt') as gpt_f, open(
        'data_annotation/news-test2008.json'
    ) as conf_f:
        threshold = 1
        while True:
            _, (true_positive, false_positive) = score_model(
                gpt_f, conf_f, threshold=threshold, frequency=False
            )
            if false_positive <= 250:
                gpt_f.seek(0)
                conf_f.seek(0)
                threshold += 1
                continue
            if false_positive >= 1400:
                print(threshold)
                break
            conf_tp.append(true_positive)
            conf_fp.append(false_positive)
            gpt_f.seek(0)
            conf_f.seek(0)
            threshold += 1

    freq_tp = []
    freq_fp = []
    with open('data_annotation/mistranslated.txt') as gpt_f, open(
        'data_annotation/news-test2008.freq.json'
    ) as freq_f:
        threshold = 0
        while True:
            _, (true_positive, false_positive) = score_model(
                gpt_f, freq_f, threshold=threshold, frequency=True
            )
            if false_positive == 250:
                print(threshold)
                break
            if false_positive >= 1400:
                gpt_f.seek(0)
                freq_f.seek(0)
                threshold += 1
                continue
            freq_tp.append(true_positive)
            freq_fp.append(false_positive)
            gpt_f.seek(0)
            freq_f.seek(0)
            threshold += 1

    plt.figure()
    aoc = integrate.trapezoid(conf_tp, conf_fp)
    plt.plot(conf_fp, conf_tp, label=f'Gradient, AOC = {aoc:.2E}')
    aoc = integrate.trapezoid(freq_tp, freq_fp)
    plt.plot(freq_fp, freq_tp, label=f'Frequency, AOC = {-aoc:.2E}')
    plt.title('ROC Curve')
    plt.xlabel('False Positive')
    plt.ylabel('True Positive')
    plt.legend()
    plt.savefig('figures/roc_curve.png', dpi=300)


def pr_f1():
    conf_x = []
    conf_precision = []
    conf_recall = []
    conf_F1 = []
    with open('data_annotation/mistranslated.txt') as gpt_f, open(
        'data_annotation/news-test2008.json'
    ) as conf_f:
        for threshold in tqdm(range(1, 51)):
            # threshold = 20
            (precision, recall, F1), _ = score_model(
                gpt_f, conf_f, threshold=threshold, frequency=False
            )
            conf_x.append(threshold)
            conf_precision.append(precision)
            conf_recall.append(recall)
            conf_F1.append(F1)
            gpt_f.seek(0)
            conf_f.seek(0)

            # print(f'Precision: {precision:0.2f}')
            # print(f'Recall:    {recall:0.2f}')
            # print(f'F1:        {F1:0.2f}')
            # return

    freq_x = []
    freq_precision = []
    freq_recall = []
    freq_F1 = []
    with open('data_annotation/mistranslated.txt') as gpt_f, open(
        'data_annotation/news-test2008.freq.json'
    ) as freq_f:
        for threshold in tqdm(range(1, 51)):
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
    ax1.set_title('Precision')
    ax1.set_xlabel('Threshold')
    ax1.set_ylabel('Confidence')
    ax1.plot(conf_x, conf_precision, label='Gradient')
    ax1.plot(freq_x, freq_precision, '--', label='Frequency')
    ax1.legend()
    # plt.savefig('figures/precision.png', dpi=300)

    # plt.figure()
    ax2.set_title('Recall')
    ax2.set_xlabel('Threshold')
    # ax2.set_ylabel('Confidence')
    ax2.plot(conf_x, conf_recall, label='Gradient')
    ax2.plot(freq_x, freq_recall, '--', label='Frequency')
    ax2.legend()
    # plt.savefig('figures/recall.png', dpi=300)

    # plt.figure()
    ax3.set_title('F1')
    ax3.set_xlabel('Threshold')
    # ax3.set_ylabel('Confidence')
    ax3.plot(conf_x, conf_F1, label='Gradient')
    ax3.plot(freq_x, freq_F1, '--', label='Frequency')
    ax3.legend()
    # plt.savefig('figures/F1.png', dpi=300)

    fig.savefig('figures/confidence.png', dpi=300)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True, help='pr_curve, roc_curve, or pr_f1')
    args = parser.parse_args()

    if args.mode == 'pr_curve':
        pr_curve()
    elif args.mode == 'roc_curve':
        roc_curve()
    elif args.mode == 'pr_f1':
        pr_f1()


if __name__ == '__main__':
    import argparse

    main()
