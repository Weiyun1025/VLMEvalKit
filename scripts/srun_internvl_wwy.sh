#!/bin/bash
set -x
# PARTITION=${PARTITION:-"Intern5"}
PARTITION=${PARTITION:-"llm_s"}
GPUS=${GPUS:-32}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
GPUS_PER_TASK=${GPUS_PER_TASK:-8}
QUOTA_TYPE=${QUOTA_TYPE:-"reserved"}

CHECKPOINT=${1}
#MMMU_DEV_VAL
# datasets="MMMU_DEV_VAL MathVista_MINI OCRBench HallusionBench MMStar RealWorldQA LLaVABench AI2D_TEST MME MMBench_TEST_EN_V11 MMBench_TEST_CN_V11"
# datasets="MME MMVet"
# datasets="MMMU_TEST"
datasets="MMMU_TEST"

LOG_DIR="./logs"

srun -p ${PARTITION} \
  --gres=gpu:${GPUS_PER_NODE} \
  --ntasks=$((GPUS / GPUS_PER_TASK)) \
  --ntasks-per-node=$((GPUS_PER_NODE / GPUS_PER_TASK)) \
  --quotatype=${QUOTA_TYPE} \
  --job-name="eval" \
  -o "${LOG_DIR}/temp.log" \
  -e "${LOG_DIR}/temp.log" \
  python -u run.py \
  --data ${datasets} \
  --model ${CHECKPOINT} \
  --verbose \
