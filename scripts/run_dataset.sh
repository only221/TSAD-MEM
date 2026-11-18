#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 DATASET DATA_CSV train|test [extra run_label_evaluation.py options...]" >&2
  echo "Example: $0 MSL data/MSL.csv train --device cuda" >&2
  exit 2
fi

dataset="$1"
data_path="$2"
mode="$3"
shift 3
if [[ "$mode" != train && "$mode" != test ]]; then
  echo "Mode must be train or test: $mode" >&2
  exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "${script_dir}/.." && pwd)"
config="${CONFIG_PATH:-${project_dir}/configs/${dataset}.json}"
model_path="${MODEL_PATH:-${project_dir}/checkpoints/${dataset}.pth}"
save_path="${SAVE_PATH:-${project_dir}/results/${dataset}}"
device="${DEVICE:-cuda}"

exec "${PYTHON_BIN:-python}" "${script_dir}/run_label_evaluation.py" \
  --dataset "$dataset" \
  --data_path "$data_path" \
  --config "$config" \
  --mode "$mode" \
  --model_path "$model_path" \
  --save_path "$save_path" \
  --device "$device" \
  --swanlab false \
  "$@"
