#!/usr/bin/env python3
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BENCHMARK_ROOT = REPO_ROOT / "artifacts/Overall_Performance/app_bmao_per_dataset_wall/k100"
DEFAULT_APP_BMAO_BIN = REPO_ROOT / "third_party/App-BMao-tlimited_main_table/ged"

GED_LINE_PAT = re.compile(r"^-?\d+$")
MATCHING_PAT = re.compile(r"^matching\s+(\d+)\s+\((.*?),\s(.*?)\):\s+(\{.*\})$")
TOTAL_TIME_PAT = re.compile(r"Total time:\s*([\d,]+)\s*\(microseconds\)")


def resolve_graph_path(graph_path, path_prefix_from=None, path_prefix_to=None):
    graph_path = str(graph_path)
    if path_prefix_from and path_prefix_to and graph_path.startswith(path_prefix_from):
        graph_path = str(path_prefix_to) + graph_path[len(path_prefix_from):]
    return Path(graph_path)


def render_graph(graph_path, graph_id, path_prefix_from=None, path_prefix_to=None):
    lines = [f"t # {graph_id}\n"]
    with resolve_graph_path(graph_path, path_prefix_from, path_prefix_to).open() as handle:
        n, m, feature_dim = map(int, handle.readline().split())
        labels = []
        for _ in range(n):
            node_id, label_value = map(int, handle.readline().split())
            label = 1 if feature_dim == 1 else label_value
            labels.append((node_id, label))
        edges = set()
        for _ in range(m):
            src, dst = map(int, handle.readline().split())
            if src > dst:
                src, dst = dst, src
            edges.add((src, dst))
    for node_id, label in sorted(labels):
        lines.append(f"v {node_id} {label}\n")
    for src, dst in sorted(edges):
        lines.append(f"e {src} {dst} 1\n")
    return "".join(lines)


def load_pairs(root, max_pairs=None):
    rows = []
    with (root / "pairs.jsonl").open() as handle:
        for line in handle:
            rows.append(json.loads(line))
            if max_pairs is not None and len(rows) >= max_pairs:
                break
    return rows


def load_done(path):
    done = set()
    if not path.exists():
        return done
    with path.open() as handle:
        for line in handle:
            if line.strip():
                done.add(json.loads(line)["global_pair_id"])
    return done


def chunked(items, size):
    for start in range(0, len(items), size):
        yield start, items[start:start + size]


def parse_app_output(output_text, batch_rows, solver_k):
    parsed = {}
    pending_ged = None
    for raw_line in output_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if GED_LINE_PAT.fullmatch(line):
            pending_ged = int(line)
            continue
        match = MATCHING_PAT.match(line)
        if match:
            local_idx = int(match.group(1))
            if pending_ged is None:
                raise RuntimeError(f"Matching line without preceding GED:\n{output_text}")
            parsed[local_idx] = {
                "app_bmao_ged": pending_ged,
                "matching": json.loads(match.group(4)),
            }
            pending_ged = None

    if len(parsed) != len(batch_rows):
        raise RuntimeError(f"Expected {len(batch_rows)} parsed pairs, got {len(parsed)}.\n{output_text}")

    total_time_s = None
    total_match = TOTAL_TIME_PAT.search(output_text)
    if total_match:
        total_time_s = int(total_match.group(1).replace(",", "")) / 1e6
    per_pair_time = total_time_s / len(batch_rows) if total_time_s is not None and batch_rows else None

    records = []
    for local_idx, row in enumerate(batch_rows):
        item = parsed[local_idx]
        records.append({
            **row,
            "solver": "App-BMao-tlimited",
            "solver_mode": "pair",
            "solver_paradigm": "astar",
            "solver_lower_bound": "BMao",
            "solver_k": solver_k,
            "app_bmao_ged": item["app_bmao_ged"],
            "ged": item["app_bmao_ged"],
            "matching": item["matching"],
            "solver_time_seconds": per_pair_time,
            "certified_exact": False,
            "exactness_note": f"App-BMao was run with -k {solver_k} as requested; result is k-limited.",
            "status": "ok",
        })
    return records


def run_batch(batch_index, batch_rows, app_bin, timeout_seconds, tmp_root, solver_k, path_prefix_from=None, path_prefix_to=None):
    batch_dir = Path(tmp_root) / f"app_batch_{batch_index:06d}"
    batch_dir.mkdir(parents=True, exist_ok=True)
    query_path = batch_dir / "query.txt"
    db_path = batch_dir / "database.txt"
    with query_path.open("w") as qf, db_path.open("w") as df:
        for row in batch_rows:
            qf.write(render_graph(row["query_clean_path"], row["query_ref"], path_prefix_from, path_prefix_to))
            df.write(render_graph(row["candidate_clean_path"], row["candidate_ref"], path_prefix_from, path_prefix_to))
    start = time.time()
    try:
        proc = subprocess.run(
            [
                str(app_bin),
                "-d", str(db_path),
                "-q", str(query_path),
                "-m", "pair",
                "-p", "astar",
                "-l", "BMao",
                "-g",
                "-x",
                "-k", str(solver_k),
            ],
            text=True,
            cwd=str(Path(app_bin).parent),
            capture_output=True,
            check=True,
            timeout=timeout_seconds,
        )
        records = parse_app_output(proc.stdout + proc.stderr, batch_rows, solver_k)
        return {
            "batch_index": batch_index,
            "status": "ok",
            "records": records,
            "wall_seconds": time.time() - start,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "batch_index": batch_index,
            "status": "timeout",
            "records": [],
            "wall_seconds": time.time() - start,
            "pair_ids": [row["global_pair_id"] for row in batch_rows],
            "error": (exc.stdout or "") + (exc.stderr or ""),
        }
    except Exception as exc:
        return {
            "batch_index": batch_index,
            "status": "error",
            "records": [],
            "wall_seconds": time.time() - start,
            "pair_ids": [row["global_pair_id"] for row in batch_rows],
            "error": str(exc),
        }


def compute_app(args):
    root = Path(args.benchmark_root)
    app_bin = Path(args.app_bmao_bin)
    if not app_bin.exists():
        raise FileNotFoundError(app_bin)
    pairs = load_pairs(root, args.max_pairs)
    results_path = root / f"app_bmao_k{args.k}_results.jsonl"
    failures_path = root / f"app_bmao_k{args.k}_failures.jsonl"
    done = load_done(results_path)
    pending = [row for row in pairs if row["global_pair_id"] not in done]
    batches = list(chunked(pending, args.batch_size))
    start = time.time()
    ok_records = 0
    failed = 0
    timeout = 0

    tmp_parent = root / "tmp_app_bmao_batches"
    tmp_parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=tmp_parent) as tmp_dir:
        with results_path.open("a") as out_handle, failures_path.open("a") as fail_handle:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = [
                    executor.submit(
                        run_batch,
                        batch_index,
                        batch_rows,
                        app_bin,
                        args.timeout_seconds,
                        tmp_dir,
                        args.k,
                        args.path_prefix_from,
                        args.path_prefix_to,
                    )
                    for batch_index, batch_rows in batches
                ]
                for future in as_completed(futures):
                    result = future.result()
                    if result["status"] == "ok":
                        for record in result["records"]:
                            out_handle.write(json.dumps(record, sort_keys=True) + "\n")
                            ok_records += 1
                        out_handle.flush()
                    else:
                        if result["status"] == "timeout":
                            timeout += 1
                        else:
                            failed += 1
                        fail_handle.write(json.dumps(result, sort_keys=True) + "\n")
                        fail_handle.flush()

    summary = {
        "benchmark_root": str(root),
        "app_bmao_bin": str(app_bin),
        "workers": args.workers,
        "batch_size": args.batch_size,
        "timeout_seconds": args.timeout_seconds,
        "max_pairs": args.max_pairs,
        "pairs_total_in_scope": len(pairs),
        "pairs_already_done_at_start": len(done),
        "pairs_attempted_this_run": len(pending),
        "pairs_succeeded_this_run": ok_records,
        "pairs_done_total": len(load_done(results_path)),
        "failed_batches_this_run": failed,
        "timed_out_batches_this_run": timeout,
        "wall_seconds_this_run": time.time() - start,
    }
    summary["solver_k"] = args.k
    (root / f"app_bmao_k{args.k}_run_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))


def load_jsonl_by_pair(path):
    out = {}
    with Path(path).open() as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                out[row["global_pair_id"]] = row
    return out


def merge_min(args):
    root = Path(args.benchmark_root)
    dfs = load_jsonl_by_pair(root / "ged_results.jsonl")
    app = load_jsonl_by_pair(root / f"app_bmao_k{args.app_k}_results.jsonl")
    if set(dfs) != set(app):
        missing_app = sorted(set(dfs) - set(app))[:10]
        missing_dfs = sorted(set(app) - set(dfs))[:10]
        raise RuntimeError(f"Pair id mismatch. missing_app={missing_app}, missing_dfs={missing_dfs}")

    out_path = root / f"min_ged_dfs_k{args.dfs_k}_app_k{args.app_k}_results.jsonl"
    counts = Counter()
    by_dataset = Counter()
    improved_by_dataset = Counter()
    equal_by_dataset = Counter()
    worse_by_dataset = Counter()

    with out_path.open("w") as handle:
        for pair_id in sorted(dfs):
            d = dfs[pair_id]
            a = app[pair_id]
            d_ged = int(d["ged"])
            a_ged = int(a["app_bmao_ged"])
            if a_ged < d_ged:
                chosen = a
                chosen_method = f"App-BMao-k{args.app_k}"
                improved_by_dataset[d["dataset"]] += 1
            else:
                chosen = d
                chosen_method = f"DFS-BMao-k{args.dfs_k}"
                if a_ged == d_ged:
                    equal_by_dataset[d["dataset"]] += 1
                else:
                    worse_by_dataset[d["dataset"]] += 1
            row = {
                **d,
                f"dfs_bmao_k{args.dfs_k}_ged": d_ged,
                f"app_bmao_k{args.app_k}_ged": a_ged,
                "ged": min(d_ged, a_ged),
                "gt_ged": min(d_ged, a_ged),
                "chosen_method": chosen_method,
                "chosen_solver": chosen.get("solver"),
                "chosen_matching": chosen.get("matching"),
                "matching": chosen.get("matching"),
                f"dfs_bmao_k{args.dfs_k}_matching": d.get("matching"),
                f"app_bmao_k{args.app_k}_matching": a.get("matching"),
                "certified_exact": False,
                "exactness_note": f"Minimum of k-limited DFS-BMao k={args.dfs_k} and k-limited App-BMao k={args.app_k}.",
            }
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            counts[chosen_method] += 1
            by_dataset[d["dataset"]] += 1

    summary = {
        "output": str(out_path),
        "pairs": len(dfs),
        "chosen_method_counts": dict(sorted(counts.items())),
        "pairs_by_dataset": dict(sorted(by_dataset.items())),
        "app_bmao_improved_over_dfs_by_dataset": dict(sorted(improved_by_dataset.items())),
        "app_bmao_equal_to_dfs_by_dataset": dict(sorted(equal_by_dataset.items())),
        "app_bmao_worse_than_dfs_by_dataset": dict(sorted(worse_by_dataset.items())),
    }
    (root / f"min_ged_dfs_k{args.dfs_k}_app_k{args.app_k}_merge_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))


def parse_args():
    parser = argparse.ArgumentParser(description="Run App-BMao k=1000 on fixed clean-rank pairs and merge with DFS-BMao k=5000 by smaller GED.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--benchmark-root", default=str(DEFAULT_BENCHMARK_ROOT))

    p_compute = sub.add_parser("compute-app", parents=[common])
    p_compute.add_argument("--app-bmao-bin", default=str(DEFAULT_APP_BMAO_BIN))
    p_compute.add_argument("--workers", type=int, default=min(56, os.cpu_count() or 1))
    p_compute.add_argument("--batch-size", type=int, default=10)
    p_compute.add_argument("--timeout-seconds", type=float, default=300.0)
    p_compute.add_argument("--max-pairs", type=int, default=None)
    p_compute.add_argument("--k", type=int, default=1000)
    p_compute.add_argument("--path-prefix-from", default=None)
    p_compute.add_argument("--path-prefix-to", default=None)

    p_merge = sub.add_parser("merge-min", parents=[common])
    p_merge.add_argument("--dfs-k", type=int, default=5000)
    p_merge.add_argument("--app-k", type=int, default=1000)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.cmd == "compute-app":
        compute_app(args)
    elif args.cmd == "merge-min":
        merge_min(args)


if __name__ == "__main__":
    main()
