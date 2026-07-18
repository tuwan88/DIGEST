#!/usr/bin/env python3
"""Check whether local artifacts needed for the main table are present."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_ROOT = REPO_ROOT / "artifacts/Overall_Performance"
MODEL_ROOT = REPO_ROOT / "artifacts/model_save/partialdiff"
APP_BMAO_ROOT = REPO_ROOT / "third_party/App-BMao-tlimited_main_table"

REQUIRED_PATHS = [
    EVAL_ROOT / "direct_data/json_data/ogbg-molhiv",
    EVAL_ROOT / "direct_data/json_data/ogbg-molpcba",
    EVAL_ROOT / "direct_data/json_data/ogbg-code2-ast97-ref",
    EVAL_ROOT / "direct_data/json_data/IMDB",
    EVAL_ROOT / "fixed_pairs/ogbg-molhiv/test",
    EVAL_ROOT / "fixed_pairs/ogbg-molpcba/test",
    EVAL_ROOT / "fixed_pairs/ogbg-code2-ast97-ref/test",
    EVAL_ROOT / "fixed_pairs/IMDB/test",
    EVAL_ROOT / "fixed_pairs/IMDB/synthetic",
    EVAL_ROOT / "app_bmao_per_dataset_wall/k100/ogbg-molhiv/pairs.jsonl",
    EVAL_ROOT / "app_bmao_per_dataset_wall/k100/ogbg-molpcba/pairs.jsonl",
    EVAL_ROOT / "app_bmao_per_dataset_wall/k100/ogbg-code2-ast97-ref/pairs.jsonl",
    EVAL_ROOT / "app_bmao_per_dataset_wall/k100/IMDB/pairs.jsonl",
    EVAL_ROOT / "graph_sources/no_same_formula_test_graphs/ogbg-molhiv/graphs",
    EVAL_ROOT / "graph_sources/no_same_formula_test_graphs/ogbg-molpcba/graphs",
    EVAL_ROOT / "graph_sources/no_same_formula_test_graphs_ast97/ogbg-code2-ast97-ref/graphs",
    MODEL_ROOT / "ogbg-molhiv/ogbg-molhiv_10_PartialDiff_ogbg-molhiv.pt",
    MODEL_ROOT / "ogbg-molpcba/ogbg-molpcba_10_PartialDiff_ogbg-molpcba.pt",
    MODEL_ROOT / "code2_ast97/ogbg-code2-ast97-ref_10_PartialDiff_code2_ast97.pt",
    MODEL_ROOT / "IMDB_fixed_pairs/IMDB_5_PartialDiff_IMDB_fixed_pairs.pt",
    APP_BMAO_ROOT / "ged",
]


def main() -> int:
    print(f"REPO_ROOT={REPO_ROOT}")
    missing = []
    for path in REQUIRED_PATHS:
        if path.exists():
            print(f"[OK]      {path}")
        else:
            print(f"[MISSING] {path}")
            missing.append(path)
    if missing:
        print(f"\nMissing {len(missing)} required paths.")
        return 1
    print("\nAll required main-table artifacts are present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
