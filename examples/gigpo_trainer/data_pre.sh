#!/usr/bin/env bash
set -euo pipefail
set -x

PROJECT_DIR=/mnt/shared-storage-user/evobox-share/songfan/project/verl-agent-master
OUTPUT_DIR=${PROJECT_DIR}/data_pre_sok

train_data_size=${1:-4}
val_data_size=${2:-4}
mode=${3:-text}

cd "${PROJECT_DIR}"

mkdir -p "${OUTPUT_DIR}"

python3 -m examples.data_preprocess.prepare \
    --mode "${mode}" \
    --train_data_size "${train_data_size}" \
    --val_data_size "${val_data_size}" \
    --local_dir "${OUTPUT_DIR}"

echo "Preprocessed data has been saved to:"
echo "${OUTPUT_DIR}"

ls -lh "${OUTPUT_DIR}"
