import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


MODEL_DIR = Path(__file__).resolve().parent
REPO_ROOT = MODEL_DIR.parents[1]
DEFAULT_EVAL_SCRIPT = str(MODEL_DIR / "eval_ogbg_benchmark_pairs.py")
DEFAULT_BENCHMARK_ROOT = str(REPO_ROOT / "artifacts/Overall_Performance/app_bmao_benchmark")
DEFAULT_GED_BIN = str(MODEL_DIR / "ged")
DEFAULT_DATASETS = ["ogbg-molhiv", "ogbg-molpcba", "ogbg-code2"]
DEFAULT_OUT_DIR = str(REPO_ROOT / "results/Overall_Performance/app_bmao")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run App-BMao t-limited benchmark evaluation on multiple OGB benchmark datasets."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        choices=DEFAULT_DATASETS,
        help="Datasets to evaluate. Defaults to all three benchmark datasets.",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "val", "test"],
        help="Which benchmark pair split to evaluate.",
    )
    parser.add_argument(
        "--python-bin",
        default=sys.executable,
        help="Python interpreter used to launch the per-dataset evaluator.",
    )
    parser.add_argument(
        "--eval-script",
        default=DEFAULT_EVAL_SCRIPT,
        help="Path to eval_ogbg_benchmark_pairs.py.",
    )
    parser.add_argument(
        "--benchmark-root",
        default=DEFAULT_BENCHMARK_ROOT,
        help="Root directory containing benchmark datasets.",
    )
    parser.add_argument(
        "--ged-bin",
        default=DEFAULT_GED_BIN,
        help="Path to App-BMao t-limited ged binary.",
    )
    parser.add_argument("-k", "--k", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument(
        "--out-dir",
        default=DEFAULT_OUT_DIR,
        help="Directory where per-dataset summaries and aggregate summary are written.",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue with remaining datasets even if one dataset run fails.",
    )
    return parser.parse_args()


def ensure_inputs_exist(args):
    if not os.path.exists(args.eval_script):
        raise FileNotFoundError(f"eval script not found: {args.eval_script}")
    if not os.path.exists(args.ged_bin):
        raise FileNotFoundError(f"ged binary not found: {args.ged_bin}")
    if not os.path.isdir(args.benchmark_root):
        raise FileNotFoundError(f"benchmark root not found: {args.benchmark_root}")


def build_command(args, dataset, out_json):
    cmd = [
        args.python_bin,
        args.eval_script,
        "--dataset",
        dataset,
        "--split",
        args.split,
        "--benchmark-root",
        args.benchmark_root,
        "--ged-bin",
        args.ged_bin,
        "-k",
        str(args.k),
        "--batch-size",
        str(args.batch_size),
        "--workers",
        str(args.workers),
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--output-json",
        out_json,
    ]
    if args.max_pairs is not None:
        cmd.extend(["--max-pairs", str(args.max_pairs)])
    return cmd


def aggregate_successes(dataset_summaries):
    successful = [item["summary"] for item in dataset_summaries if item["status"] == "ok"]
    if not successful:
        return {
            "datasets_succeeded": 0,
            "pairs_total": 0,
            "pairs_scored": 0,
            "pairs_timeout": 0,
            "pairs_failed": 0,
            "avg_mae_across_datasets": None,
            "avg_solver_total_seconds": None,
            "avg_wall_time_seconds": None,
        }

    maes = [item["mae"] for item in successful if item["mae"] is not None]
    return {
        "datasets_succeeded": len(successful),
        "pairs_total": sum(item["pairs_total"] for item in successful),
        "pairs_scored": sum(item["pairs_scored"] for item in successful),
        "pairs_timeout": sum(item["pairs_timeout"] for item in successful),
        "pairs_failed": sum(item["pairs_failed"] for item in successful),
        "avg_mae_across_datasets": (sum(maes) / len(maes)) if maes else None,
        "avg_solver_total_seconds": sum(item["solver_total_seconds"] for item in successful)
        / len(successful),
        "avg_wall_time_seconds": sum(item["wall_time_seconds"] for item in successful)
        / len(successful),
    }


def main():
    args = parse_args()
    ensure_inputs_exist(args)
    os.makedirs(args.out_dir, exist_ok=True)

    run_started_at = time.time()
    dataset_results = []

    for dataset in args.datasets:
        dataset_out_dir = os.path.join(args.out_dir, dataset)
        os.makedirs(dataset_out_dir, exist_ok=True)
        out_json = os.path.join(dataset_out_dir, f"{dataset}_{args.split}_summary.json")
        cmd = build_command(args, dataset, out_json)

        print(f"\n==== Evaluating {dataset} ({args.split}) ====")
        print("Command:", " ".join(cmd), flush=True)

        started_at = time.time()
        proc = subprocess.run(
            cmd,
            cwd=MODEL_DIR,
            text=True,
            capture_output=True,
        )
        elapsed = time.time() - started_at

        if proc.stdout:
            print(proc.stdout, end="" if proc.stdout.endswith("\n") else "\n")
        if proc.stderr:
            print(proc.stderr, file=sys.stderr, end="" if proc.stderr.endswith("\n") else "\n")

        if proc.returncode != 0:
            result = {
                "dataset": dataset,
                "status": "failed",
                "returncode": proc.returncode,
                "wall_time_seconds": elapsed,
                "summary_json": out_json,
            }
            dataset_results.append(result)
            print(f"[error] {dataset} failed with return code {proc.returncode}", flush=True)
            if not args.keep_going:
                break
            continue

        if not os.path.exists(out_json):
            result = {
                "dataset": dataset,
                "status": "failed",
                "returncode": 0,
                "wall_time_seconds": elapsed,
                "summary_json": out_json,
                "error": "summary json was not produced",
            }
            dataset_results.append(result)
            print(f"[error] {dataset} finished without summary json: {out_json}", flush=True)
            if not args.keep_going:
                break
            continue

        with open(out_json, "r") as f:
            summary = json.load(f)

        dataset_results.append(
            {
                "dataset": dataset,
                "status": "ok",
                "returncode": 0,
                "wall_time_seconds": elapsed,
                "summary_json": out_json,
                "summary": summary,
            }
        )

    aggregate = {
        "datasets_requested": args.datasets,
        "split": args.split,
        "k": args.k,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "timeout_seconds": args.timeout_seconds,
        "max_pairs": args.max_pairs,
        "ged_bin": args.ged_bin,
        "benchmark_root": args.benchmark_root,
        "run_wall_time_seconds": time.time() - run_started_at,
        "datasets": dataset_results,
        "aggregate_success": aggregate_successes(dataset_results),
    }

    aggregate_json = os.path.join(args.out_dir, f"aggregate_{args.split}_summary.json")
    with open(aggregate_json, "w") as f:
        json.dump(aggregate, f, indent=2)

    print("\n==== Aggregate Summary ====")
    print(f"datasets_requested: {len(args.datasets)}")
    print(f"datasets_finished: {len(dataset_results)}")
    print(f"datasets_succeeded: {aggregate['aggregate_success']['datasets_succeeded']}")
    print(f"aggregate_json: {aggregate_json}")

    all_ok = (
        len(dataset_results) == len(args.datasets)
        and aggregate["aggregate_success"]["datasets_succeeded"] == len(args.datasets)
    )
    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
