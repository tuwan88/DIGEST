#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
MAIN_ROOT="$REPO_ROOT/src/PartialDiffGED_main_table_snapshot"
CONDA="${CONDA_BIN:-conda}"
CONDA_ENV="${CONDA_ENV:-partialged}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/artifacts/Overall_Performance/direct_data}"
DATASET="IMDB"
MODEL_NAME="PartialDiff_IMDB_fixed_pairs"
MODEL_PATH="${MODEL_PATH:-$REPO_ROOT/artifacts/model_save/partialdiff/IMDB_fixed_pairs}"
RESULT_PATH="${RESULT_PATH:-$REPO_ROOT/results/training/$DATASET}"
EPOCH_END="${EPOCH_END:-5}"

if [[ ! -d "$DATA_ROOT/json_data/$DATASET" ]]; then
  echo "Dataset not found: $DATA_ROOT/json_data/$DATASET" >&2
  echo "Set DATA_ROOT to the directory containing json_data/." >&2
  exit 1
fi

mkdir -p "$MODEL_PATH" "$RESULT_PATH"
cd "$MAIN_ROOT"

CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "$CONDA" run --no-capture-output -n "$CONDA_ENV" python main_dense.py \
  --abs-path "$REPO_ROOT/" \
  --data-path "$DATA_ROOT/" \
  --dataset "$DATASET" \
  --model-name "$MODEL_NAME" \
  --model-path "$MODEL_PATH/" \
  --result-path "$RESULT_PATH/" \
  --hidden-dim 32 32 24 24 \
  --dense-topk-enable \
  --dense-topk-start-layer 1 \
  --dense-topk-row 32 \
  --dense-topk-col 32 \
  --model-epoch-end "$EPOCH_END" \
  --save-every-epochs 1
