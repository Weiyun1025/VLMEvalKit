#!/bin/bash

# PARTITION=${PARTITION:-"llm_s"}
PARTITION=${PARTITION:-"Intern5"}
GPUS=${GPUS:-64}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
GPUS_PER_TASK=${GPUS_PER_TASK:-8}
QUOTA_TYPE=${QUOTA_TYPE:-"reserved"}

LOG_DIR="logs"

general_academic_vqa="ScienceQA_TEST"
general_vqa="MME POPE MMMU_DEV_VAL CCBench SEEDBench_IMG HallusionBench MMStar RealWorldQA"
ocr_vqa="TextVQA_VAL ChartQA_TEST DocVQA_VAL AI2D_TEST InfoVQA_VAL OCRBench"

datasets="${general_academic_vqa} ${general_vqa} ${ocr_vqa}"
datasets="MMMU_DEV_VAL"

declare -a model_paths=( \
    'InternVL-Chat-V2-0-20B' \
    'InternVL-Chat-V2-0-72B' \
    'InternVL-Chat-V2-0-100B' \
)

mkdir -p $LOG_DIR

for ((i=0; i<${#model_paths[@]}; i++)); do
    model_path=${model_paths[i]}

    model_name="$(basename ${model_path})"
    echo "$(date) ${model_name}"

    srun -p ${PARTITION} \
        --gres=gpu:${GPUS_PER_NODE} \
        --ntasks=$((GPUS / GPUS_PER_TASK)) \
        --ntasks-per-node=$((GPUS_PER_NODE / GPUS_PER_TASK)) \
        --quotatype=${QUOTA_TYPE} \
        --job-name="eval_${model_name}" \
        -o "${LOG_DIR}/${model_name}.log" \
        -e "${LOG_DIR}/${model_name}.log" \
        --async \
        python -u run.py \
        --data ${datasets} \
        --model $model_path \
        --verbose \

done
