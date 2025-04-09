import os
from tools.performance2xlsx_utils import parse_performance


def main():
    outputs_dir = './outputs'
    model_list = [
        'InternImage-Chat-8B',
    ]
    benchmark_list = [
        'MMStar',
        'MMMU_VAL',
        'MathVista_MINI',
        'HallusionBench',
        'AI2D',
        'OCRBench',
        'MMVet',
    ]
    for model in model_list:
        parse_performance(
            os.path.join(outputs_dir, model),
            benchmark_list,
        )


if __name__ == '__main__':
    main()
