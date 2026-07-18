#!/usr/bin/env bash
# Train or evaluate dense and disjoint PartialDiffGED variants.
# Expected data layout: <DATA_ROOT>/json_data/<DATASET>/{train,test}/
# Examples:
#   bash scripts/train_molhiv_dense_disjoint.sh train_dense
#   CUDA_VISIBLE_DEVICES=0 EPOCH_END=10 bash scripts/train_molhiv_dense_disjoint.sh all

set -euo pipefail

MODE="${1:-all}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MAIN_PY="${REPO_ROOT}/src/PartialDiffGED/main.py"

if [[ ! -f "${MAIN_PY}" ]]; then
  echo "[ERROR] Training entry point not found: ${MAIN_PY}" >&2
  exit 1
fi

if [[ -f "${REPO_ROOT}/scripts/activate_runtime.sh" ]]; then
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/scripts/activate_runtime.sh"
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET="${DATASET:-ogbg-molhiv}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/artifacts/training}"
EPOCH_START="${EPOCH_START:-0}"
EPOCH_END="${EPOCH_END:-10}"
TEST_EPOCH="${TEST_EPOCH:-${EPOCH_END}}"
BATCH_SIZE="${BATCH_SIZE:-128}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-4}"
TEST_K="${TEST_K:-100}"
NUM_WORKERS="${NUM_WORKERS:-0}"
CONSTRAINED_GREEDY_MODE="${CONSTRAINED_GREEDY_MODE:-global_n3}"

if [[ ! -d "${DATA_ROOT}/json_data/${DATASET}" ]]; then
  echo "[ERROR] Dataset not found: ${DATA_ROOT}/json_data/${DATASET}" >&2
  echo "Set DATA_ROOT to the directory containing json_data/." >&2
  exit 1
fi

DENSE_TAG="${DATASET}_lightgt_dense_e${EPOCH_END}"
DISJOINT_TAG="${DATASET}_lightgt_disjoint_e${EPOCH_END}"
OUTPUT_ROOT="${REPO_ROOT}/results/training"
DENSE_MODEL_PATH="${OUTPUT_ROOT}/models/${DENSE_TAG}"
DENSE_RESULT_PATH="${OUTPUT_ROOT}/metrics/${DENSE_TAG}"
DISJOINT_MODEL_PATH="${OUTPUT_ROOT}/models/${DISJOINT_TAG}"
DISJOINT_RESULT_PATH="${OUTPUT_ROOT}/metrics/${DISJOINT_TAG}"

mkdir -p "${DENSE_MODEL_PATH}" "${DENSE_RESULT_PATH}" "${DISJOINT_MODEL_PATH}" "${DISJOINT_RESULT_PATH}"

run_train_dense() {
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "${PYTHON_BIN}" "${MAIN_PY}" \
    --dataset "${DATASET}" --abs-path "${DATA_ROOT}/" \
    --denoise-network lightgt_dense --model-name "${DENSE_TAG}" --model-train 1 \
    --model-epoch-start "${EPOCH_START}" --model-epoch-end "${EPOCH_END}" \
    --save-every-epochs 1 --batch-size "${BATCH_SIZE}" --num-workers "${NUM_WORKERS}" \
    --disable-tqdm --model-path "${DENSE_MODEL_PATH}" --result-path "${DENSE_RESULT_PATH}"
}

run_train_disjoint() {
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "${PYTHON_BIN}" "${MAIN_PY}" \
    --dataset "${DATASET}" --abs-path "${DATA_ROOT}/" \
    --denoise-network lightgt_disjoint --model-name "${DISJOINT_TAG}" --model-train 1 \
    --model-epoch-start "${EPOCH_START}" --model-epoch-end "${EPOCH_END}" \
    --save-every-epochs 1 --batch-size "${BATCH_SIZE}" --num-workers "${NUM_WORKERS}" \
    --disable-tqdm --model-path "${DISJOINT_MODEL_PATH}" --result-path "${DISJOINT_RESULT_PATH}"
}

run_test_dense() {
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "${PYTHON_BIN}" "${MAIN_PY}" \
    --dataset "${DATASET}" --abs-path "${DATA_ROOT}/" \
    --denoise-network lightgt_dense --model-name "${DENSE_TAG}" --model-train 0 \
    --model-epoch-start "${TEST_EPOCH}" --model-epoch-end "${TEST_EPOCH}" \
    --test-k "${TEST_K}" --topk-approach parallel --test-batch-size "${TEST_BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" --disable-tqdm --inference-decoder constrained_greedy \
    --constrained-greedy-mode "${CONSTRAINED_GREEDY_MODE}" \
    --model-path "${DENSE_MODEL_PATH}" --result-path "${DENSE_RESULT_PATH}"
}

run_test_disjoint() {
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "${PYTHON_BIN}" "${MAIN_PY}" \
    --dataset "${DATASET}" --abs-path "${DATA_ROOT}/" \
    --denoise-network lightgt_disjoint --model-name "${DISJOINT_TAG}" --model-train 0 \
    --model-epoch-start "${TEST_EPOCH}" --model-epoch-end "${TEST_EPOCH}" \
    --test-k "${TEST_K}" --topk-approach parallel --test-batch-size "${TEST_BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" --disable-tqdm --inference-decoder constrained_greedy \
    --constrained-greedy-mode "${CONSTRAINED_GREEDY_MODE}" \
    --model-path "${DISJOINT_MODEL_PATH}" --result-path "${DISJOINT_RESULT_PATH}"
}

case "${MODE}" in
  train_dense) run_train_dense ;;
  train_disjoint) run_train_disjoint ;;
  test_dense) run_test_dense ;;
  test_disjoint) run_test_disjoint ;;
  train_all) run_train_dense; run_train_disjoint ;;
  test_all) run_test_dense; run_test_disjoint ;;
  all) run_train_dense; run_train_disjoint; run_test_dense; run_test_disjoint ;;
  *)
    echo "Usage: $0 {train_dense|train_disjoint|test_dense|test_disjoint|train_all|test_all|all}" >&2
    exit 2
    ;;
esac
