#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
SPLIT="${SPLIT:-test}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-${REPO_ROOT}/artifacts/Overall_Performance/app_bmao_benchmark}"
GED_BIN="${GED_BIN:-${SCRIPT_DIR}/ged}"
EVAL_SCRIPT="${EVAL_SCRIPT:-${SCRIPT_DIR}/eval_ogbg_large_dataset_pairs.py}"
WORKERS="${WORKERS:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-30}"
OUT_ROOT="${OUT_ROOT:-${SCRIPT_DIR}/eval_results_test_serial_k100_k1000_dense_ready}"

mkdir -p "${OUT_ROOT}"

preflight_check() {
  if [[ ! -x "${GED_BIN}" ]]; then
    echo "[fatal] ged binary is missing or not executable: ${GED_BIN}" >&2
    exit 1
  fi

  local check_output
  if ! check_output="$(${GED_BIN} -h 2>&1 >/dev/null)"; then
    echo "[fatal] ged binary failed preflight execution: ${GED_BIN}" >&2
    echo "${check_output}" >&2
    echo "[hint] Rebuild it on this machine:" >&2
    echo "  cd ${SCRIPT_DIR}" >&2
    echo "  make clean && make" >&2
    exit 1
  fi
}

run_one_dataset() {
  local dataset="$1"
  local k_value="$2"
  local out_dir="${OUT_ROOT}/k${k_value}_w${WORKERS}"
  local out_json="${out_dir}/${dataset}_${SPLIT}_summary.json"

  mkdir -p "${out_dir}"

  echo ""
  echo "==== Evaluating ${dataset} (${SPLIT}) | k=${k_value} ===="
  echo "Command: ${PYTHON_BIN} ${EVAL_SCRIPT} --dataset ${dataset} --split ${SPLIT} --benchmark-root ${BENCHMARK_ROOT} --ged-bin ${GED_BIN} -k ${k_value} --batch-size ${BATCH_SIZE} --workers ${WORKERS} --timeout-seconds ${TIMEOUT_SECONDS} --output-json ${out_json}"

  "${PYTHON_BIN}" "${EVAL_SCRIPT}"     --dataset "${dataset}"     --split "${SPLIT}"     --benchmark-root "${BENCHMARK_ROOT}"     --ged-bin "${GED_BIN}"     -k "${k_value}"     --batch-size "${BATCH_SIZE}"     --workers "${WORKERS}"     --timeout-seconds "${TIMEOUT_SECONDS}"     --output-json "${out_json}"
}

run_one_k() {
  local k_value="$1"

  echo "============================================================"
  echo "Running App-BMao-tlimited benchmark eval"
  echo "split=${SPLIT} k=${k_value} workers=${WORKERS} batch_size=${BATCH_SIZE}"
  echo "benchmark_root=${BENCHMARK_ROOT}"
  echo "ged_bin=${GED_BIN}"
  echo "eval_script=${EVAL_SCRIPT}"
  echo "out_dir=${OUT_ROOT}/k${k_value}_w${WORKERS}"
  echo "============================================================"

  run_one_dataset "ogbg-molhiv" "${k_value}"
  run_one_dataset "ogbg-molpcba" "${k_value}"
  run_one_dataset "ogbg-code2" "${k_value}"
}

preflight_check

run_one_k 100
run_one_k 1000

echo
 echo "All App-BMao-tlimited benchmark runs finished."
 echo "Results saved under: ${OUT_ROOT}"
