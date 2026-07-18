set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
DATASET="ogbg-molhiv"
K=100
BENCHMARK_ROOT="$REPO_ROOT/artifacts/Overall_Performance/app_bmao_per_dataset_wall/k${K}/${DATASET}"
python3 "$REPO_ROOT/scripts/Overall_Performance/tools/app_bmao_k1000_min_merge.py" compute-app \
  --benchmark-root "$BENCHMARK_ROOT" \
  --app-bmao-bin "$REPO_ROOT/third_party/App-BMao-tlimited_main_table/ged" \
  --workers "${WORKERS:-56}" \
  --batch-size "${BATCH_SIZE:-10}" \
  --timeout-seconds "${TIMEOUT_SECONDS:-300}" \
  --k "$K" \
  --path-prefix-from "${PATH_PREFIX_FROM:-}" \
  --path-prefix-to "${PATH_PREFIX_TO:-}"
