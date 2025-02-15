import glob
import json
import os
import re
from argparse import ArgumentParser
from statistics import mean, stdev

import numpy as np
from scipy.stats import t


def calculate_t_score(sample1: list[float], sample2: list[float]) -> float:
    mean1, mean2 = np.mean(sample1), np.mean(sample2)
    std1, std2 = np.std(sample1, ddof=1), np.std(sample2, ddof=1)
    n1, n2 = len(sample1), len(sample2)
    t_score = (mean1 - mean2) / np.sqrt((std1**2 / n1) + (std2**2 / n2))
    return t_score


def calculate_degrees_of_freedom(sample1: list[float], sample2: list[float]) -> int:
    n1, n2 = len(sample1), len(sample2)
    df = n1 + n2 - 2  # two-sample t-test
    return df


def calculate_p_value(t_score: float, df: int) -> float:
    p_value = 2 * (1 - t.cdf(np.abs(t_score), df))
    return p_value


def interpret_p_value(p_value: float, alpha: float) -> bool:
    return bool(p_value < alpha)


def is_significant(sample1: list[float], sample2: list[float], alpha: float) -> tuple[float, bool]:
    t_score = calculate_t_score(sample1, sample2)
    df = calculate_degrees_of_freedom(sample1, sample2)
    p_value = calculate_p_value(t_score, df)
    return p_value, interpret_p_value(p_value, alpha)


def extract_scores(data_dirs: list[str], test_sets: list[str], alpha: float | None = None) -> str:
    results = {}  # type: ignore[var-annotated]
    statistics = {}  # type: ignore[var-annotated]
    for data_dir in data_dirs:
        results.setdefault(data_dir, {})
        statistics.setdefault(data_dir, {})
        for test_set in test_sets:
            statistics[data_dir].setdefault(test_set, {})
            for metric in ('BLEU', 'COMET'):
                scores = []
                for log_file in glob.glob(os.path.join(data_dir, '*.log')):
                    with open(log_file) as log_f:
                        log_data = log_f.read()
                    log_name = os.path.basename(log_file)
                    results[data_dir].setdefault(log_name, {})

                    match metric:
                        case 'BLEU':
                            pattern = re.compile(
                                rf'data_de-en_small/{test_set}/{test_set} \(BLEU\).*?BLEU.*?= ([0-9]+\.[0-9]+)',
                                re.DOTALL,
                            )
                        case 'COMET':
                            pattern = re.compile(
                                rf'data_de-en_small/{test_set}/{test_set} \(COMET\).*?score:\s*([0-9]+\.[0-9]+)',
                                re.DOTALL,
                            )

                    for match in pattern.finditer(log_data):
                        scores.append(float(match.group(1)))
                        results[data_dir][log_name].setdefault(test_set, {})[metric] = scores[-1]

                if alpha is not None:
                    statistics[data_dir][test_set].setdefault(metric, {})['mean'] = mean(scores)
                    statistics[data_dir][test_set].setdefault(metric, {})['stdev'] = stdev(scores)

    if alpha is not None:
        for data_dir in data_dirs:
            for test_set in test_sets:
                for metric in ('BLEU', 'COMET'):
                    sample1 = [
                        sample[test_set][metric] for sample in results[data_dirs[0]].values()
                    ]
                    sample2 = [
                        sample[test_set][metric] for sample in results[data_dirs[1]].values()
                    ]
                    p_value, significance = is_significant(sample1, sample2, alpha)
                    statistics[data_dir][test_set][metric]['p_value'] = p_value
                    statistics[data_dir][test_set][metric]['significance'] = significance
        return json.dumps(statistics, indent=4)

    return json.dumps(results, indent=4)


def main():
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(dest='subcommand')
    extract_parser = subparsers.add_parser('extract')
    extract_parser.add_argument('--data-dirs', nargs='+', metavar='DATA_DIR', required=True)
    extract_parser.add_argument('--test-sets', nargs='+', metavar='TEST_SET', required=True)
    compare_parser = subparsers.add_parser('compare')
    compare_parser.add_argument('--data-dirs', nargs=2, metavar='DATA_DIR', required=True)
    compare_parser.add_argument('--test-sets', nargs='+', metavar='TEST_SET', required=True)
    compare_parser.add_argument('--alpha', default=0.05)
    args = parser.parse_args()

    # if args.subcommand == 'compare' and len(args.data_dirs) < 2:  # TODO
    #     parser.error('at least two data_dirs are required for comparison.')
    if args.subcommand == 'compare' and args.data_dirs[0] == args.data_dirs[1]:
        parser.error('data_dirs must be different for comparison.')

    alpha = None if args.subcommand == 'extract' else args.alpha
    print(extract_scores(args.data_dirs, args.test_sets, alpha))


if __name__ == '__main__':
    main()
