import os
import csv
import json
import pandas as pd


def load_csv(path):
    results = []
    with open(path) as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            results.append(row)
    return results


def load_json(path):
    with open(path) as file:
        results = json.load(file)
    return results


def find_idx(results, key, value):
    for idx, res in enumerate(results):
        if res[key] == value:
            return idx
    raise RuntimeError


def parse_mathvista(path):
    results = load_csv(path)
    idx = find_idx(results, 'Task&Skill', 'Overall')
    return float(results[idx]['acc'])


def parse_mathvision(path):
    results = load_csv(path)
    idx = find_idx(results, 'Subject', 'Overall')
    return float(results[idx]['acc'])


def parse_mathverse(path):
    results = load_csv(path)
    idx = find_idx(results, 'split', 'Vision Only')
    return float(results[idx]['Overall'])


def parse_mmmu(path):
    results = load_csv(path)
    idx = find_idx(results, 'split', 'validation')
    return float(results[idx]['Overall']) * 100


def parse_dynamath(path):
    results = load_csv(path)
    idx = find_idx(results, 'Setting', 'Worst Case')
    return float(results[idx]['Overall']) * 100


def parse_wemath(path):
    results = load_csv(path)
    idx = find_idx(results, 'Model', '')
    return float(results[idx]['Score (Strict)'].strip('%'))


def parse_logicvista(path):
    results = load_csv(path)
    idx = find_idx(results, 'Task&Skill', 'Overall')
    return float(results[idx]['acc'])


def parse_mmstar(path):
    results = load_csv(path)
    assert len(results) == 1
    return float(results[0]['Overall']) * 100


def parse_mmvet(path):
    results = load_csv(path)
    idx = find_idx(results, 'Category', 'Overall')
    return float(results[idx]['acc'])


def parse_llavabench(path):
    results = load_csv(path)
    idx = find_idx(results, 'split', 'overall')
    return float(results[idx]['Relative Score (main)'])


def parse_ocrbench(path):
    results = load_json(path)
    return results['Final Score']


def parse_ai2d(path):
    results = load_csv(path)
    assert len(results) == 1
    return float(results[0]['Overall']) * 100


def parse_crpe(path):
    results = load_json(path)
    return float(results['total']) * 100


def parse_hallusionbench(path):
    results = load_csv(path)
    idx = find_idx(results, 'split', 'Overall')
    scores = [
        float(results[idx]['aAcc']),
        float(results[idx]['fAcc']),
        float(results[idx]['qAcc']),
    ]
    return sum(scores) / len(scores)


def parse_single_performance(outputs_path, dsname, suffix, parse_function):
    if parse_function is None:
        return '-'

    benchmark_results_path = None
    for filename in os.listdir(outputs_path):
        if filename.endswith(suffix):
            benchmark_results_path = os.path.join(outputs_path, filename)

    if benchmark_results_path is None:
        # print(f'[Warning] Fail to find results file of {dsname} from {outputs_path}')
        return '-'

    score = f'{parse_function(benchmark_results_path):.1f}'
    return score


def parse_performance(outputs_path, benchmark_list):
    benchmark_info = {
        'MMMU_VAL': ('MMMU_DEV_VAL_acc.csv', parse_mmmu),
        'MathVista_MINI': ('MathVista_MINI_gpt-4o-mini_score.csv', parse_mathvista),
        'MathVision_MINI': ('MathVision_MINI_gpt-4o-mini_score.csv', parse_mathvision),
        'MathVerse_MINI_Vision_Only': ('MathVerse_MINI_Vision_Only_gpt-4o-mini_score.csv', parse_mathverse),
        'DynaMath': ('DynaMath_gpt-4o-mini_score.csv', parse_dynamath),
        'WeMath': ('WeMath_gpt4o-mini_score.csv', parse_wemath),
        'WeMath_CoT': ('WeMath_COT_gpt4o-mini_score.csv', parse_wemath),
        'LogicVista': ('LogicVista_gpt4o-mini_score.csv', parse_logicvista),
        #
        'CRPE': ('CRPE_RELATION_score.json', parse_crpe),
        'HallusionBench': ('HallusionBench_score.csv', parse_hallusionbench),
        'MMStar': ('MMStar_acc.csv', parse_mmstar),
        'MMVet': ('MMVet_gpt-4-turbo_score.csv', parse_mmvet),
        'LLaVABench': ('LLaVABench_score.csv', parse_llavabench),
        'AI2D': ('AI2D_TEST_acc.csv', parse_ai2d),
        'OCRBench': ('OCRBench_score.json', parse_ocrbench),
    }
    save_dir = 'outputs_excel'

    score = {}
    for benchmark_name in benchmark_list:
        suffix, parse_function = benchmark_info[benchmark_name]
        score[benchmark_name] = [parse_single_performance(outputs_path, benchmark_name, suffix, parse_function)]
        print(f'{benchmark_name}', score[benchmark_name][0])

    os.makedirs(save_dir, exist_ok=True)
    df = pd.DataFrame(score)
    df.to_excel(os.path.join(save_dir, f'{os.path.basename(outputs_path)}.xlsx'), index=False)
