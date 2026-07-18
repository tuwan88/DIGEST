set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
EVAL_ROOT="$REPO_ROOT/artifacts/Overall_Performance"
PARTIALDIFF_ROOT="$REPO_ROOT/src/PartialDiffGED_main_table_snapshot"
CONDA="${CONDA_BIN:-conda}"
CUDA_DEVICE="${CUDA_DEVICE:-2}"
WORKERS="${WORKERS:-0}"
DATALOADER_WORKERS="${DATALOADER_WORKERS:-0}"
REFINE_RESULT_SUBDIR="${REFINE_RESULT_SUBDIR:-partialdiff_refine_gpu_cap50_bw8_lb0}"
V9_CANDIDATE_CAP="${V9_CANDIDATE_CAP:-100}"
V9_BEAM_WIDTH="${V9_BEAM_WIDTH:-8}"
V9_BRANCH_WIDTH="${V9_BRANCH_WIDTH:-8}"
V9_RERANK_POOL="${V9_RERANK_POOL:-32}"
V9_LB_TIEBREAK_WEIGHT="${V9_LB_TIEBREAK_WEIGHT:-0}"
APP_BMAO_SEARCH_STATES="${APP_BMAO_SEARCH_STATES:-200}"
APP_BMAO_ANCHOR_RATIO="${APP_BMAO_ANCHOR_RATIO:-0.0}"
DATASET="IMDB"
MODEL_NAME="PartialDiff_IMDB_fixed_pairs"
MODEL_PATH="$REPO_ROOT/artifacts/model_save/partialdiff/IMDB_fixed_pairs/"
BATCH_SIZE="${BATCH_SIZE:-512}"
OUT_DIR="$REPO_ROOT/results/Overall_Performance/${REFINE_RESULT_SUBDIR}/$DATASET"
THREAD_ENV=()
if [ "$WORKERS" != "0" ]; then
  THREAD_ENV=(OMP_NUM_THREADS="$WORKERS" MKL_NUM_THREADS="$WORKERS" OPENBLAS_NUM_THREADS="$WORKERS" NUMEXPR_NUM_THREADS="$WORKERS" TORCH_NUM_THREADS="$WORKERS")
fi
mkdir -p "$OUT_DIR"
cd "$PARTIALDIFF_ROOT"
env "${THREAD_ENV[@]}" CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "$CONDA" run --no-capture-output -n partialged python main_dense.py \
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
  --dense-topk-row 32 \
  --dense-topk-col 32 \
  --constrained-greedy-mode row_top1_unique_n2 \
  --reverse-row-top1-repair-mode final_step \
  --model-train 0 \
  --model-epoch-start 5 \
  --model-epoch-end 5 \
  --experiment test \
  --testset test \
  --test-batch-size "$BATCH_SIZE" \
  --num-workers "$DATALOADER_WORKERS" \
  --test-k 1 \
  --topk-approach parallel \
  --fixed-pair-root "$EVAL_ROOT/fixed_pairs" \
  --max-test-pairs "${MAX_TEST_PAIRS:-0}" \
  --app-bmao-postprocess-enable \
  --app-bmao-search-backend gpu_refine \
  --app-bmao-search-states "$APP_BMAO_SEARCH_STATES" \
  --app-bmao-anchor-ratio "$APP_BMAO_ANCHOR_RATIO" \
  --gpu-beam-width "$V9_BEAM_WIDTH" \
  --gpu-branch-width "$V9_BRANCH_WIDTH" \
  --gpu-candidate-cap "$V9_CANDIDATE_CAP" \
  --gpu-lb-type row_col_min \
  --gpu-lb-tiebreak-weight "$V9_LB_TIEBREAK_WEIGHT" \
  --gpu-rerank-pool "$V9_RERANK_POOL" \
  --profile-runtime-enable \
  --app-bmao-profile-enable


