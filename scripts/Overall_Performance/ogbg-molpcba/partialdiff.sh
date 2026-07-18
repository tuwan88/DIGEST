set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
EVAL_ROOT="$REPO_ROOT/artifacts/Overall_Performance"
PARTIALDIFF_ROOT="$REPO_ROOT/src/PartialDiffGED_main_table_snapshot"
CONDA="${CONDA_BIN:-conda}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
DATASET="ogbg-molpcba"
MODEL_NAME="PartialDiff_ogbg-molpcba"
MODEL_PATH="$REPO_ROOT/artifacts/model_save/partialdiff/ogbg-molpcba/"
BATCH_SIZE=128
OUT_DIR="$REPO_ROOT/results/Overall_Performance/partialdiff/$DATASET"
mkdir -p "$OUT_DIR"
cd "$PARTIALDIFF_ROOT"
CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "$CONDA" run --no-capture-output -n partialged python main_dense.py \
  --abs-path "$REPO_ROOT/" \
  --data-path "$EVAL_ROOT/direct_data/" \
  --dataset "$DATASET" \
  --model-name "$MODEL_NAME" \
  --model-path "$MODEL_PATH" \
  --result-path "$OUT_DIR/" \
  --hidden-dim 32 32 24 24 \
  --gt-heads 4 \
      --reverse-decode-mode blockwise_autoregressive \
  --dense-topk-enable \
  --dense-topk-start-layer 1 \
  --dense-topk-row 8 \
  --dense-topk-col 8 \
  --constrained-greedy-mode row_top1_unique_n2 \
  --reverse-row-top1-repair-mode final_step \
  --model-train 0 \
  --model-epoch-start 10 \
  --model-epoch-end 10 \
  --experiment test \
  --testset test \
  --test-batch-size "$BATCH_SIZE" \
  --test-k 1 \
  --topk-approach parallel \
  --fixed-pair-root "$EVAL_ROOT/fixed_pairs" \
  --max-test-pairs "${MAX_TEST_PAIRS:-0}" \
  --profile-runtime-enable
