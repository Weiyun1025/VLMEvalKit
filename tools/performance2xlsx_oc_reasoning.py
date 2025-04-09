import os
from tools.performance2xlsx_utils import parse_performance


def main():
    outputs_dir = './outputs'
    model_list = [
        "InternVL2_5-8B-MMR1-v3-2-HF",
    ]
    benchmark_list = [
        'MMMU_VAL',
        'MathVista_MINI',
        'MathVision_MINI',
        'MathVerse_MINI_Vision_Only',
        'DynaMath',
        'WeMath',
        'LogicVista',
    ]
    for model in model_list:
        print(model)
        parse_performance(
            os.path.join(outputs_dir, model),
            benchmark_list,
        )
        print()


if __name__ == '__main__':
    main()
