import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent
REPO_ROOT = MODEL_DIR.parents[1]
DEFAULT_GED_BIN = str(MODEL_DIR / "ged")
DEFAULT_BENCHMARK_ROOT = str(REPO_ROOT / "artifacts/Overall_Performance/app_bmao_benchmark")
DEFAULT_APP_BMAO_K = 100
DEFAULT_BATCH_SIZE = 64
DEFAULT_WORKERS = min(8, os.cpu_count() or 1)
DEFAULT_TIMEOUT_SECONDS = 10.0

MATCHING_PAT = re.compile(r"^matching\s+(\d+)\s+\((.*?),\s(.*?)\):")
PAIR_PAT = re.compile(r"^\(pair_(\d+)\s+(.*?)\s+(.*?)\)\s+GED:\s*(-?\d+),\s+Time:\s*([\d,]+)")
TIME_PAT = re.compile(r"Total time:\s*([\d,]+)\s*\(microseconds\)")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate App-BMao on the local large_dataset pair-package format.")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["ogbg-molhiv", "ogbg-molpcba", "ogbg-code2"],
        help="Dataset name under benchmark root.",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "val", "test"],
        help="Which benchmark pair split to evaluate.",
    )
    parser.add_argument(
        "--benchmark-root",
        default=DEFAULT_BENCHMARK_ROOT,
        help="Root directory containing large_dataset benchmark packages.",
    )
    parser.add_argument(
        "--ged-bin",
        default=DEFAULT_GED_BIN,
        help="Path to App-BMao t-limited ged binary.",
    )
    parser.add_argument(
        "--paradigm",
        default="astar",
        choices=["astar", "dfs"],
        help="Search paradigm passed to ged.",
    )
    parser.add_argument("-k", "--k", type=int, default=DEFAULT_APP_BMAO_K)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--output-json", default=None, help="Optional path to dump evaluation summary json.")
    parser.add_argument(
        "--raw-output-jsonl",
        default=None,
        help="Optional path to write one JSON record per evaluated pair. Defaults next to --output-json.",
    )
    return parser.parse_args()


def load_rows(pair_file):
    rows = []
    with open(pair_file, "r") as f:
        for line in f:
            item = json.loads(line)
            if item.get("batch_status") != "ok" or item.get("status") != "success":
                continue
            rows.append(item)
    return rows


def chunked(items, chunk_size):
    for start in range(0, len(items), chunk_size):
        yield items[start : start + chunk_size]


def render_graph_entry(path, graph_id):
    lines = [f"t # {graph_id}\n"]
    with open(path, "r") as f:
        n, m, features = map(int, f.readline().split())
        labels = []
        for _ in range(n):
            x, y = map(int, f.readline().split())
            label = 1 if features == 1 else y
            labels.append((x, label))
        edges = set()
        for _ in range(m):
            x, y = map(int, f.readline().split())
            if x > y:
                x, y = y, x
            edges.add((x, y))
    labels.sort()
    for x, label in labels:
        lines.append(f"v {x} {label}\n")
    for x, y in sorted(edges):
        lines.append(f"e {x} {y} 1\n")
    return "".join(lines)


def materialize_graph_cache(data_dir, rows, cache_dir):
    cache_index = {}
    needed_graph_ids = set()
    for item in rows:
        needed_graph_ids.add(item["graph_1"])
        needed_graph_ids.add(item["graph_2"])

    for graph_id in sorted(needed_graph_ids):
        src_path = os.path.join(data_dir, graph_id)
        out_path = os.path.join(cache_dir, graph_id.replace("/", "__"))
        with open(out_path, "w") as f:
            f.write(render_graph_entry(src_path, graph_id))
        cache_index[graph_id] = out_path
    return cache_index


def parse_pair_output(output_text, expected_pairs, batch_rows):
    pred_by_pair_index = {}
    pending_ged = None
    solver_total_us = 0

    for raw_line in output_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if re.fullmatch(r"-?\d+", line):
            pending_ged = int(line)
            continue

        pair_match = PAIR_PAT.match(line)
        if pair_match:
            pair_index, _graph_id_1, _graph_id_2, pred_ged, pair_time_us = pair_match.groups()
            pred_by_pair_index[int(pair_index)] = int(pred_ged)
            solver_total_us += int(pair_time_us.replace(',', ''))
            pending_ged = None
            continue

        match = MATCHING_PAT.match(line)
        if match:
            if pending_ged is None:
                raise RuntimeError(f"Found matching line without GED value:\n{output_text}")
            pair_index, _graph_id_1, _graph_id_2 = match.groups()
            pred_by_pair_index[int(pair_index)] = pending_ged
            pending_ged = None

    if len(pred_by_pair_index) != expected_pairs:
        raise RuntimeError(
            f"Expected {expected_pairs} parsed pairs, got {len(pred_by_pair_index)}.\n{output_text}"
        )

    pred_by_pair = {}
    for pair_index, item in enumerate(batch_rows):
        pred_ged = pred_by_pair_index.get(pair_index)
        if pred_ged is None:
            raise RuntimeError(f"Missing parsed output for pair index {pair_index}.\n{output_text}")
        pred_by_pair[(item["graph_1"], item["graph_2"])] = pred_ged
    return pred_by_pair, solver_total_us


def run_batch(batch_rows, cache_index, cache_dir, batch_index, k, timeout_seconds, ged_bin, paradigm):
    query_batch_path = os.path.join(cache_dir, f"query_batch_{batch_index}.txt")
    database_batch_path = os.path.join(cache_dir, f"database_batch_{batch_index}.txt")

    with open(query_batch_path, "w") as qf, open(database_batch_path, "w") as df:
        for item in batch_rows:
            with open(cache_index[item["graph_1"]], "r") as f:
                qf.write(f.read())
            with open(cache_index[item["graph_2"]], "r") as f:
                df.write(f.read())

    try:
        proc = subprocess.run(
            [
                ged_bin,
                "-d",
                database_batch_path,
                "-q",
                query_batch_path,
                "-m",
                "pair",
                "-p",
                paradigm,
                "-l",
                "BMao",
                "-g",
                "-k",
                str(k),
                "-x",
            ],
            text=True,
            cwd=os.path.dirname(os.path.abspath(ged_bin)) or None,
            capture_output=True,
            check=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        return None, None, "timeout", None
    except subprocess.CalledProcessError as e:
        err_out = (e.stdout or "") + (e.stderr or "")
        return None, None, "error", err_out

    out = proc.stdout + proc.stderr
    pred_by_pair, parsed_solver_total_us = parse_pair_output(out, expected_pairs=len(batch_rows), batch_rows=batch_rows)
    m_time = TIME_PAT.search(out)
    solver_total_us = int(m_time.group(1).replace(",", "")) if m_time else parsed_solver_total_us
    return pred_by_pair, solver_total_us, "ok", None


def append_summary_json(output_json_path, summary):
    records = []
    if os.path.exists(output_json_path):
        with open(output_json_path, "r") as f:
            existing = json.load(f)
        if isinstance(existing, list):
            records = existing
        elif existing is None:
            records = []
        else:
            records = [existing]

    records.append(summary)
    output_dir = os.path.dirname(output_json_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_json_path, "w") as f:
        json.dump(records, f, indent=2)


def default_raw_output_path(output_json_path, dataset, split, paradigm, k):
    if not output_json_path:
        return None
    output_dir = os.path.dirname(output_json_path) or "."
    return os.path.join(output_dir, f"raw_pairs_AppBMao_{dataset}_{split}_{paradigm}_k{k}.jsonl")


def build_raw_record(args, item, pred_ged, pair_time_s, status, err_out=None):
    gt_ged = int(item["ged"])
    record = {
        "dataset": args.dataset,
        "split": args.split,
        "pair_id": item.get("pair_id", f"{item['graph_1']}_{item['graph_2']}"),
        "method": "App-BMao",
        "graph_id": item["graph_1"],
        "graph_id_2": item["graph_2"],
        "graph_1": item["graph_1"],
        "graph_2": item["graph_2"],
        "produced_ged": pred_ged,
        "reference_ged": gt_ged,
        "gt_ged": gt_ged,
        "solver_time": pair_time_s,
        "total_time": pair_time_s,
        "status": status,
        "timeout": status == "timeout",
        "comparison": "unknown",
        "better": False,
        "equal": False,
        "worse": False,
        "config_name": f"AppBMao_{args.dataset}_{args.split}_{args.paradigm}_k{args.k}",
        "budget_parameters": {
            "paradigm": args.paradigm,
            "k": args.k,
            "batch_size": args.batch_size,
            "workers": args.workers,
            "timeout_seconds": args.timeout_seconds,
        },
        "matching": None,
    }
    if pred_ged is not None:
        if pred_ged < gt_ged:
            record["comparison"] = "better"
            record["better"] = True
        elif pred_ged == gt_ged:
            record["comparison"] = "equal"
            record["equal"] = True
        else:
            record["comparison"] = "worse"
            record["worse"] = True
    if err_out:
        record["error"] = err_out[:1000]
    return record


def main():
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if args.k <= 0:
        raise ValueError("-k/--k must be positive")
    if args.timeout_seconds <= 0:
        raise ValueError("--timeout-seconds must be positive")

    data_dir = os.path.join(args.benchmark_root, args.dataset)
    pair_file = os.path.join(data_dir, f"{args.split}.jsonl")
    graph_map_file = os.path.join(data_dir, "graph_id_map.json")

    if not os.path.exists(args.ged_bin):
        raise FileNotFoundError(f"ged binary not found: {args.ged_bin}")
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"dataset directory not found: {data_dir}")
    if not os.path.exists(pair_file):
        raise FileNotFoundError(f"pair file not found: {pair_file}")
    if not os.path.exists(graph_map_file):
        raise FileNotFoundError(f"graph map file not found: {graph_map_file}")

    rows = load_rows(pair_file)
    if args.max_pairs is not None:
        rows = rows[: args.max_pairs]
    total_pairs = len(rows)
    if total_pairs == 0:
        raise RuntimeError(f"No usable rows found in: {pair_file}")

    num_pairs = 0
    scored_pairs = 0
    timeout_pairs = 0
    failed_pairs = 0
    sum_pred_ged = 0.0
    sum_gt_ged = 0.0
    abs_err_sum = 0.0
    pred_lt_gt_count = 0
    pred_eq_gt_count = 0
    pred_gt_gt_count = 0
    solver_total_us = 0
    wall_start = time.time()
    raw_output_path = args.raw_output_jsonl or default_raw_output_path(
        args.output_json, args.dataset, args.split, args.paradigm, args.k
    )
    raw_records = []

    with tempfile.TemporaryDirectory() as td:
        cache_index = materialize_graph_cache(data_dir, rows, td)
        batches = list(chunked(rows, args.batch_size))

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    run_batch,
                    batch_rows,
                    cache_index,
                    td,
                    batch_index,
                    args.k,
                    args.timeout_seconds,
                    args.ged_bin,
                    args.paradigm,
                ): batch_rows
                for batch_index, batch_rows in enumerate(batches)
            }

            for future in as_completed(futures):
                batch_rows = futures[future]
                pred_by_pair, batch_solver_total_us, batch_status, err_out = future.result()
                if batch_status == "timeout":
                    timeout_pairs += len(batch_rows)
                    num_pairs += len(batch_rows)
                    raw_records.extend(
                        build_raw_record(args, item, None, None, "timeout") for item in batch_rows
                    )
                    progress = num_pairs / total_pairs * 100
                    elapsed = time.time() - wall_start
                    eta = (elapsed / num_pairs) * (total_pairs - num_pairs) if num_pairs else 0.0
                    print(
                        f"Progress: {num_pairs}/{total_pairs} ({progress:.2f}%) | elapsed={elapsed:.1f}s | eta={eta:.1f}s | batch_timeout",
                        flush=True,
                    )
                    continue
                if batch_status == "error":
                    failed_pairs += len(batch_rows)
                    num_pairs += len(batch_rows)
                    raw_records.extend(
                        build_raw_record(args, item, None, None, "error", err_out)
                        for item in batch_rows
                    )
                    progress = num_pairs / total_pairs * 100
                    elapsed = time.time() - wall_start
                    eta = (elapsed / num_pairs) * (total_pairs - num_pairs) if num_pairs else 0.0
                    print(
                        f"Progress: {num_pairs}/{total_pairs} ({progress:.2f}%) | elapsed={elapsed:.1f}s | eta={eta:.1f}s | batch_failed",
                        flush=True,
                    )
                    if err_out:
                        print("[warn] batch failed output (truncated):")
                        print(err_out[:1000])
                    continue

                solver_total_us += batch_solver_total_us
                per_pair_solver_time_s = (batch_solver_total_us / 1e6 / len(batch_rows)) if batch_rows else 0.0
                for item in batch_rows:
                    g1 = item["graph_1"]
                    g2 = item["graph_2"]
                    gt_ged = int(item["ged"])
                    pred_ged = pred_by_pair.get((g1, g2))
                    if pred_ged is None:
                        raise RuntimeError(f"Failed to parse GED output for pair: {g1}, {g2}")

                    num_pairs += 1
                    scored_pairs += 1
                    sum_pred_ged += pred_ged
                    sum_gt_ged += gt_ged
                    abs_err_sum += abs(pred_ged - gt_ged)
                    if pred_ged < gt_ged:
                        pred_lt_gt_count += 1
                    elif pred_ged == gt_ged:
                        pred_eq_gt_count += 1
                    else:
                        pred_gt_gt_count += 1
                    raw_records.append(
                        build_raw_record(args, item, pred_ged, per_pair_solver_time_s, "success")
                    )

                progress = num_pairs / total_pairs * 100
                elapsed = time.time() - wall_start
                eta = (elapsed / num_pairs) * (total_pairs - num_pairs) if num_pairs else 0.0
                print(
                    f"Progress: {num_pairs}/{total_pairs} ({progress:.2f}%) | elapsed={elapsed:.1f}s | eta={eta:.1f}s",
                    flush=True,
                )

    wall_end = time.time()
    if scored_pairs == 0:
        avg_pred_ged = None
        avg_gt_ged = None
        mae = None
        pred_lt_gt_ratio = None
        pred_eq_gt_ratio = None
        pred_gt_gt_ratio = None
    else:
        avg_pred_ged = sum_pred_ged / scored_pairs
        avg_gt_ged = sum_gt_ged / scored_pairs
        mae = abs_err_sum / scored_pairs
        pred_lt_gt_ratio = pred_lt_gt_count / scored_pairs
        pred_eq_gt_ratio = pred_eq_gt_count / scored_pairs
        pred_gt_gt_ratio = pred_gt_gt_count / scored_pairs

    summary = {
        "dataset": args.dataset,
        "split": args.split,
        "pairs_total": total_pairs,
        "pairs_scored": scored_pairs,
        "pairs_timeout": timeout_pairs,
        "pairs_failed": failed_pairs,
        "avg_pred_ged": avg_pred_ged,
        "avg_gt_ged": avg_gt_ged,
        "mae": mae,
        "pred_lt_gt_count": pred_lt_gt_count,
        "pred_lt_gt_ratio": pred_lt_gt_ratio,
        "pred_eq_gt_count": pred_eq_gt_count,
        "pred_eq_gt_ratio": pred_eq_gt_ratio,
        "pred_gt_gt_count": pred_gt_gt_count,
        "pred_gt_gt_ratio": pred_gt_gt_ratio,
        "solver_total_seconds": solver_total_us / 1e6,
        "wall_time_seconds": wall_end - wall_start,
        "paradigm": args.paradigm,
        "k": args.k,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "timeout_seconds": args.timeout_seconds,
    }

    if raw_output_path:
        raw_output_dir = os.path.dirname(raw_output_path)
        if raw_output_dir:
            os.makedirs(raw_output_dir, exist_ok=True)
        with open(raw_output_path, "w") as f:
            for record in raw_records:
                f.write(json.dumps(record, sort_keys=True) + "\n")
        summary["raw_pair_log"] = raw_output_path

    print("\n=== Evaluation Summary ===")
    for key, value in summary.items():
        print(f"{key}: {value}")

    if args.output_json:
        append_summary_json(args.output_json, summary)
        print(f"Summary appended to: {args.output_json}")


if __name__ == "__main__":
    main()
