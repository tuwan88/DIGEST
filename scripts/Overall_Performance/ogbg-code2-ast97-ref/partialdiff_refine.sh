set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
EVAL_ROOT="$REPO_ROOT/artifacts/Overall_Performance"
PARTIALDIFF_ROOT="$REPO_ROOT/src/PartialDiffGED_main_table_snapshot"
CONDA="${CONDA_BIN:-conda}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
REFINE_RESULT_SUBDIR="${REFINE_RESULT_SUBDIR:-partialdiff_refine_gpu_cap100_bw16_lb0}"
V9_CANDIDATE_CAP="${V9_CANDIDATE_CAP:-100}"
V9_BEAM_WIDTH="${V9_BEAM_WIDTH:-4}"
V9_BRANCH_WIDTH="${V9_BRANCH_WIDTH:-4}"
V9_RERANK_POOL="${V9_RERANK_POOL:-16}"
V9_LB_TIEBREAK_WEIGHT="${V9_LB_TIEBREAK_WEIGHT:-0}"
APP_BMAO_SEARCH_STATES="${APP_BMAO_SEARCH_STATES:-200}"
APP_BMAO_ANCHOR_RATIO="${APP_BMAO_ANCHOR_RATIO:-0.0}"
DATASET="ogbg-code2-ast97-ref"
MODEL_NAME="PartialDiff_code2_ast97"
MODEL_PATH="$REPO_ROOT/artifacts/model_save/partialdiff/code2_ast97/"
BATCH_SIZE=256
OUT_DIR="$REPO_ROOT/results/Overall_Performance/${REFINE_RESULT_SUBDIR}/$DATASET"
EXPECTED_RAW="$OUT_DIR/raw_pairs_PartialDiff_refine_${DATASET}_test_parallel_1_backend-gpu_refine_mode-best_anchor-0p0.jsonl"
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
  --dense-topk-row 16 \
  --dense-topk-col 16 \
  --constrained-greedy-mode row_top1_unique_n2 \
  --reverse-row-top1-repair-mode final_step \
  --model-train 0 \
  --model-epoch-start 21 \
  --model-epoch-end 21 \
  --experiment test \
  --testset test \
  --test-batch-size "$BATCH_SIZE" \
  --test-k 5 \
  --topk-approach parallel \
  --fixed-pair-root "$EVAL_ROOT/fixed_pairs" \
  --max-test-pairs "${MAX_TEST_PAIRS:-0}" \
  --app-bmao-postprocess-enable \
  --refine-mode best \
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
