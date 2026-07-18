set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
EVAL_ROOT="$REPO_ROOT/artifacts/Overall_Performance"
PARTIALDIFF_ROOT="$REPO_ROOT/src/PartialDiffGED_main_table_snapshot"
CONDA="${CONDA_BIN:-conda}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
DATASET="ogbg-molhiv"
MODEL_NAME="PartialDiff_ogbg-molhiv"
MODEL_PATH="$REPO_ROOT/artifacts/model_save/partialdiff/ogbg-molhiv/"
RESULT_SUBDIR="${RESULT_SUBDIR:-partialdiff_app_bmao_refine}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
APP_BMAO_SEARCH_STATES="${APP_BMAO_SEARCH_STATES:-100}"
APP_BMAO_ANCHOR_RATIO="${APP_BMAO_ANCHOR_RATIO:-0.6}"
APP_BMAO_WORKERS="${APP_BMAO_WORKERS:-1}"
APP_BMAO_TIMEOUT_SECONDS="${APP_BMAO_TIMEOUT_SECONDS:-120}"
APP_BMAO_GED_BIN="${APP_BMAO_GED_BIN:-$REPO_ROOT/third_party/App-BMao-tlimited_main_table/ged}"
OUT_DIR="$REPO_ROOT/results/Overall_Performance/${RESULT_SUBDIR}/$DATASET"
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
  --app-bmao-postprocess-enable \
  --refine-mode best \
  --app-bmao-search-backend external_app_bmao \
  --app-bmao-search-states "$APP_BMAO_SEARCH_STATES" \
  --app-bmao-anchor-ratio "$APP_BMAO_ANCHOR_RATIO" \
  --app-bmao-workers "$APP_BMAO_WORKERS" \
  --app-bmao-timeout-seconds "$APP_BMAO_TIMEOUT_SECONDS" \
  --app-bmao-diffusion-ub-enable \
  --app-bmao-ged-bin "$APP_BMAO_GED_BIN" \
  --profile-runtime-enable \
  --app-bmao-profile-enable
