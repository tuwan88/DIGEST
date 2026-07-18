#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MODEL_DIR="${SCRIPT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-python}"
EVAL_SCRIPT="${MODEL_DIR}/eval_ogbg_benchmark_pairs.py"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-${REPO_ROOT}/artifacts/Overall_Performance/app_bmao_benchmark}"
GED_BIN="${GED_BIN:-${MODEL_DIR}/ged}"
SPLIT="${SPLIT:-test}"
K="${K:-100}"
BATCH_SIZE="${BATCH_SIZE:-64}"
WORKERS="${WORKERS:-4}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-30}"
MAX_PAIRS="${MAX_PAIRS:-}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/results/Overall_Performance/app_bmao}"

mkdir -p "${OUT_DIR}"

run_one() {
  local dataset="$1"
  local out_dir="${OUT_DIR}/${dataset}"
  mkdir -p "${out_dir}"
  local out_json="${out_dir}/${dataset}_${SPLIT}_summary.json"
  local raw_jsonl="${out_dir}/raw_pairs_AppBMao_${dataset}_${SPLIT}_astar_k${K}.jsonl"

  echo "==== Evaluating ${dataset} (${SPLIT}) ===="
  cmd=(
    "${PYTHON_BIN}" "${EVAL_SCRIPT}"
    --dataset "${dataset}"
    --split "${SPLIT}"
    --benchmark-root "${BENCHMARK_ROOT}"
    --ged-bin "${GED_BIN}"
    -k "${K}"
    --batch-size "${BATCH_SIZE}"
    --workers "${WORKERS}"
    --timeout-seconds "${TIMEOUT_SECONDS}"
    --output-json "${out_json}"
    --raw-output-jsonl "${raw_jsonl}"
  )

  if [[ -n "${MAX_PAIRS}" ]]; then
    cmd+=(--max-pairs "${MAX_PAIRS}")
  fi

  "${cmd[@]}"
}

run_one "ogbg-molhiv"
run_one "ogbg-molpcba"
run_one "ogbg-code2"

echo "All done. Summaries in: ${OUT_DIR}"
