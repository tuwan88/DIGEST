"""Minimal data-loading helpers for the standalone dense DiffGED runner."""

from concurrent.futures import ThreadPoolExecutor
import json
import os
import pickle
import re
from glob import glob
from os.path import basename, isdir, isfile, join

import networkx as nx

try:
    from texttable import Texttable
except ImportError:
    Texttable = None


DATASET_ALIASES = {
    "code2": ["code2", "ogbg-code2"],
    "ogbg-code2": ["ogbg-code2", "code2"],
}
CACHE_DIR_NAME = ".dense_min_cache"
CACHE_VERSION = 7
MAX_ONEHOT_LABEL_DIM = int(os.environ.get("PARTIALDIFF_MAX_ONEHOT_LABEL_DIM", "2048"))
SOLVER_PRINT_ORDER_BUG_DATASETS = {
    "code2",
    "ogbg-code2",
    "ogbg-molhiv",
    "ogbg-molpcba",
}


def resolve_load_workers(load_workers):
    if load_workers is None or int(load_workers) <= 0:
        return min(8, max(1, os.cpu_count() or 1))
    return max(1, int(load_workers))


def chunk_ranges(total, num_chunks):
    if total <= 0:
        return []
    num_chunks = max(1, min(num_chunks, total))
    base = total // num_chunks
    rem = total % num_chunks
    ranges = []
    start = 0
    for idx in range(num_chunks):
        size = base + (1 if idx < rem else 0)
        end = start + size
        if start < end:
            ranges.append((start, end))
        start = end
    return ranges


def tab_printer(args):
    rows = [["Parameter", "Value"]]
    for key in sorted(vars(args)):
        rows.append([key.replace("_", " ").capitalize(), getattr(args, key)])
    if Texttable is None:
        for key, value in rows:
            print(f"{key}: {value}")
        return
    table = Texttable()
    table.add_rows(rows)
    print(table.draw())


def sorted_nicely(values):
    def try_int(part):
        try:
            return int(part)
        except ValueError:
            return part

    def alphanum_key(text):
        return [try_int(part) for part in re.split(r"([0-9]+)", text)]

    return sorted(values, key=alphanum_key)


def get_file_paths(directory, file_format="json"):
    return sorted_nicely(glob(directory.rstrip("/") + f"/*.{file_format}"))


def ensure_cache_dir(dataset_root):
    cache_dir = join(dataset_root, CACHE_DIR_NAME)
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def cache_file_path(dataset_root, key):
    cache_dir = ensure_cache_dir(dataset_root)
    return join(cache_dir, f"{key}.pkl")


def load_pickle_cache(path):
    with open(path, "rb") as handle:
        payload = pickle.load(handle)
    if payload.get("cache_version") != CACHE_VERSION:
        return None
    return payload.get("value")


def save_pickle_cache(path, value):
    with open(path, "wb") as handle:
        pickle.dump({"cache_version": CACHE_VERSION, "value": value}, handle, protocol=pickle.HIGHEST_PROTOCOL)


def dataset_name_candidates(dataset_name):
    return DATASET_ALIASES.get(dataset_name, [dataset_name])


def dataset_has_solver_print_order_bug(dataset_name):
    return dataset_name in SOLVER_PRINT_ORDER_BUG_DATASETS


def resolve_dataset_root(data_location, dataset_name):
    base_dir = data_location.rstrip("/")
    search_roots = [
        join(base_dir, "json_data"),
        join(base_dir, "json_data_min_15k_testpairs"),
    ]
    for root in search_roots:
        for candidate in dataset_name_candidates(dataset_name):
            dataset_root = join(root, candidate)
            if isdir(dataset_root):
                return dataset_root, candidate
    tried = [join(root, candidate) for root in search_roots for candidate in dataset_name_candidates(dataset_name)]
    raise FileNotFoundError(f"Dataset root not found for {dataset_name}. Tried: {tried}")


def _load_graph_file(file_path, file_format):
    gid = int(basename(file_path).split(".")[0])
    if file_format == "gexf":
        graph = nx.read_gexf(file_path)
        graph.graph["gid"] = gid
        if not nx.is_connected(graph):
            raise RuntimeError(f"{gid} not connected")
        return graph

    graph = json.load(open(file_path, "r"))
    if file_format == "json":
        graph["gid"] = gid
    return graph


def iterate_get_graphs(directory, file_format, load_workers=None):
    if file_format not in {"gexf", "json", "onehot", "anchor"}:
        raise ValueError(f"Unsupported graph file format: {file_format}")

    dataset_root = directory.rsplit("/", 1)[0]
    split_name = basename(directory.rstrip("/"))
    cache_path = cache_file_path(dataset_root, f"{split_name}_{file_format}")
    if isfile(cache_path):
        cached = load_pickle_cache(cache_path)
        if cached is not None:
            return cached

    file_paths = get_file_paths(directory, file_format)
    if not file_paths:
        graphs = []
    else:
        worker_count = min(resolve_load_workers(load_workers), len(file_paths))
        if worker_count <= 1:
            graphs = [_load_graph_file(file_path, file_format) for file_path in file_paths]
        else:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                graphs = list(executor.map(lambda path: _load_graph_file(path, file_format), file_paths))
    save_pickle_cache(cache_path, graphs)
    return graphs


def load_all_graphs(data_location, dataset_name, load_workers=None):
    dataset_root, resolved_name = resolve_dataset_root(data_location, dataset_name)
    train_dir = join(dataset_root, "train")
    test_dir = join(dataset_root, "test")
    with ThreadPoolExecutor(max_workers=2) as executor:
        train_future = executor.submit(iterate_get_graphs, train_dir, "json", load_workers)
        test_future = executor.submit(iterate_get_graphs, test_dir, "json", load_workers)
        train_graphs = train_future.result()
        test_graphs = test_future.result()
    graphs = train_graphs + test_graphs
    train_total = len(train_graphs)
    test_num = len(graphs) - train_total

    split_count_path = join(dataset_root, "graph_split_counts.json")
    if isfile(split_count_path):
        split_counts = json.load(open(split_count_path, "r"))
        train_num = int(split_counts["train"])
        val_num = int(split_counts["val"])
        if train_num + val_num != train_total:
            raise ValueError(
                f"graph_split_counts.json mismatch for {resolved_name}: "
                f"train+val={train_num + val_num}, train_dir={train_total}"
            )
        if int(split_counts["test"]) != test_num:
            raise ValueError(
                f"graph_split_counts.json mismatch for {resolved_name}: "
                f"test={int(split_counts['test'])}, test_dir={test_num}"
            )
    else:
        val_num = test_num
        train_num = train_total - val_num
    return train_num, val_num, test_num, graphs


def load_labels(data_location, dataset_name, graphs=None, load_workers=None):
    dataset_root, resolved_name = resolve_dataset_root(data_location, dataset_name)
    global_labels = json.load(open(join(dataset_root, "labels.json"), "r"))

    train_onehot_paths = get_file_paths(join(dataset_root, "train"), "onehot")
    test_onehot_paths = get_file_paths(join(dataset_root, "test"), "onehot")
    if train_onehot_paths or test_onehot_paths:
        with ThreadPoolExecutor(max_workers=2) as executor:
            train_future = executor.submit(iterate_get_graphs, join(dataset_root, "train"), "onehot", load_workers)
            test_future = executor.submit(iterate_get_graphs, join(dataset_root, "test"), "onehot", load_workers)
            features = train_future.result() + test_future.result()
        print(f"Load one-hot label features (dim = {len(global_labels)}) of {resolved_name}.")
        return global_labels, features

    if graphs is None:
        graphs = load_all_graphs(data_location, dataset_name, load_workers=load_workers)[3]
    return build_features_from_graphs_with_workers(graphs, global_labels, resolved_name, load_workers)


def build_features_from_graphs_with_workers(graphs, global_labels, resolved_name, load_workers=None):
    label_to_index = {int(label): int(idx) for label, idx in global_labels.items()}
    feature_dim = len(global_labels)
    graph_count = len(graphs)
    worker_count = min(resolve_load_workers(load_workers), graph_count) if graph_count > 0 else 1

    if feature_dim > MAX_ONEHOT_LABEL_DIM:
        denom = float(max(feature_dim - 1, 1))

        def build_graph_feature(graph):
            return [[label_to_index[int(raw_label)] / denom] for raw_label in graph["labels"]]

        if worker_count <= 1 or graph_count <= 1:
            features = [build_graph_feature(graph) for graph in graphs]
        else:
            def build_chunk(start_end):
                start, end = start_end
                return [build_graph_feature(graph) for graph in graphs[start:end]]

            ranges = chunk_ranges(graph_count, worker_count)
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                chunk_results = list(executor.map(build_chunk, ranges))
            features = [feature for chunk in chunk_results for feature in chunk]

        print(
            f"Load compact label-id features (dim = 1) of {resolved_name} because "
            f"label vocab {feature_dim} exceeds one-hot limit {MAX_ONEHOT_LABEL_DIM}."
        )
        return global_labels, features

    def build_graph_feature(graph):
        node_features = []
        for raw_label in graph["labels"]:
            onehot = [0] * feature_dim
            onehot[label_to_index[int(raw_label)]] = 1
            node_features.append(onehot)
        return node_features

    if worker_count <= 1 or graph_count <= 1:
        features = [build_graph_feature(graph) for graph in graphs]
    else:
        def build_chunk(start_end):
            start, end = start_end
            return [build_graph_feature(graph) for graph in graphs[start:end]]

        ranges = chunk_ranges(graph_count, worker_count)
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            chunk_results = list(executor.map(build_chunk, ranges))
        features = [feature for chunk in chunk_results for feature in chunk]

    print(f"Load label-derived one-hot features (dim = {feature_dim}) of {resolved_name}.")
    return global_labels, features


def load_pair_manifest(data_location, dataset_name):
    dataset_root, _ = resolve_dataset_root(data_location, dataset_name)
    manifest_path = join(dataset_root, "pair_manifest.json")
    if not isfile(manifest_path):
        return None
    cache_path = cache_file_path(dataset_root, "pair_manifest")
    if isfile(cache_path):
        cached = load_pickle_cache(cache_path)
        if cached is not None:
            return cached
    manifest = json.load(open(manifest_path, "r"))
    save_pickle_cache(cache_path, manifest)
    return manifest


def _canonical_edge(src, dst):
    src = int(src)
    dst = int(dst)
    if src <= dst:
        return (src, dst)
    return (dst, src)


def _build_row_mapping_from_pairs(n1, matching_pairs):
    row_mapping = [None] * int(n1)
    seen_rows = set()
    seen_cols = set()
    for raw_row, raw_col in matching_pairs:
        row = int(raw_row)
        col = int(raw_col)
        if row in seen_rows:
            raise ValueError(f"Duplicate source-row match detected: row={row}")
        if col in seen_cols:
            raise ValueError(f"Duplicate target-col match detected: col={col}")
        seen_rows.add(row)
        seen_cols.add(col)
        row_mapping[row] = col
    return row_mapping


def _compute_dense_minimal_ged_from_row_mapping(left_graph, right_graph, row_mapping):
    mapping = {}
    reverse_mapping = {}
    for row, maybe_col in enumerate(row_mapping):
        if maybe_col is None:
            continue
        col = int(maybe_col)
        if row in mapping:
            raise ValueError(f"Duplicate source-row match detected: row={row}")
        if col in reverse_mapping:
            raise ValueError(f"Duplicate target-col match detected: col={col}")
        mapping[row] = col
        reverse_mapping[col] = row

    matched_count = len(mapping)
    left_labels = left_graph["labels"]
    right_labels = right_graph["labels"]
    node_sub_cost = sum(
        1 for row, col in mapping.items() if int(left_labels[row]) != int(right_labels[col])
    )
    node_insdel_cost = (int(left_graph["n"]) - matched_count) + (int(right_graph["n"]) - matched_count)

    left_edges = {
        _canonical_edge(src, dst)
        for src, dst in left_graph["graph"]
        if int(src) != int(dst)
    }
    right_edges = {
        _canonical_edge(src, dst)
        for src, dst in right_graph["graph"]
        if int(src) != int(dst)
    }

    edge_delete_cost = 0
    for src, dst in left_edges:
        mapped_src = mapping.get(src)
        mapped_dst = mapping.get(dst)
        if mapped_src is None or mapped_dst is None:
            edge_delete_cost += 1
            continue
        if _canonical_edge(mapped_src, mapped_dst) not in right_edges:
            edge_delete_cost += 1

    edge_insert_cost = 0
    for src, dst in right_edges:
        mapped_src = reverse_mapping.get(src)
        mapped_dst = reverse_mapping.get(dst)
        if mapped_src is None or mapped_dst is None:
            edge_insert_cost += 1
            continue
        if _canonical_edge(mapped_src, mapped_dst) not in left_edges:
            edge_insert_cost += 1

    total_ged = node_sub_cost + node_insdel_cost + edge_delete_cost + edge_insert_cost
    return (
        float(total_ged),
        float(node_sub_cost),
        float(node_insdel_cost),
        float(edge_delete_cost + edge_insert_cost),
    )


def _load_dense_minimal_ged_map_from_matching_payloads(dataset_root):
    payload_path = join(dataset_root, "pair_matching_payloads.jsonl")
    if not isfile(payload_path):
        return None

    split_count_path = join(dataset_root, "graph_split_counts.json")
    if not isfile(split_count_path):
        raise FileNotFoundError(
            f"graph_split_counts.json is required to load matching payloads for {dataset_root}"
        )
    split_counts = json.load(open(split_count_path, "r"))
    train_dir_count = int(split_counts["train"]) + int(split_counts["val"])

    graph_cache = {}

    def load_graph_by_gid(gid):
        gid = int(gid)
        cached = graph_cache.get(gid)
        if cached is not None:
            return cached
        split_name = "train" if gid < train_dir_count else "test"
        graph_path = join(dataset_root, split_name, f"{gid}.json")
        graph = json.load(open(graph_path, "r"))
        graph_cache[gid] = graph
        return graph

    parsed = {}
    with open(payload_path, "r", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            if payload.get("batch_status") != "ok":
                continue
            gid_1 = int(payload["new_query_gid"])
            gid_2 = int(payload["new_db_gid"])
            row_mapping = _build_row_mapping_from_pairs(
                n1=int(load_graph_by_gid(gid_1)["n"]),
                matching_pairs=payload["matching"]["query_to_db"],
            )
            ta_ged = _compute_dense_minimal_ged_from_row_mapping(
                left_graph=load_graph_by_gid(gid_1),
                right_graph=load_graph_by_gid(gid_2),
                row_mapping=row_mapping,
            )
            parsed[(gid_1, gid_2)] = (ta_ged, [row_mapping])
    return parsed


def _load_tagged_ged_map(path):
    tagged = json.load(open(path, "r"))
    parsed = {}
    for id_1, id_2, ged_value, ged_nc, ged_in, ged_ie, mappings in tagged:
        parsed[(id_1, id_2)] = ((ged_value, ged_nc, ged_in, ged_ie), mappings)
    return parsed


def _correct_solver_print_pairs_to_row_mapping(matching_pairs, n1=None, trust_source_rows=False):
    ordered_pairs = [[int(raw_row), int(raw_col)] for raw_row, raw_col in matching_pairs]
    if trust_source_rows:
        if n1 is None:
            raise ValueError("n1 is required when trusting source rows in matching pairs.")
        return _build_row_mapping_from_pairs(n1=n1, matching_pairs=ordered_pairs)
    if all(src == idx for idx, (src, _) in enumerate(ordered_pairs)):
        return [int(raw_col) for raw_row, raw_col in ordered_pairs]
    # Legacy DFS print bug: `(MO[i], BX[i])` with `BX` indexed by query node id.
    return [int(raw_col) for _raw_row, raw_col in matching_pairs]


def _load_tagged_ged_map_with_corrected_payload_mappings(dataset_root, path):
    payload_path = join(dataset_root, "pair_matching_payloads.jsonl")
    if not isfile(payload_path):
        return _load_tagged_ged_map(path)

    tagged = json.load(open(path, "r"))
    payload_rows = []
    with open(payload_path, "r", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            if payload.get("batch_status") not in {None, "ok"}:
                continue
            payload_rows.append(payload)

    if len(payload_rows) != len(tagged):
        raise ValueError(
            f"Tagged/payload pair-count mismatch for {dataset_root}: "
            f"TaGED={len(tagged)} payload_rows={len(payload_rows)}"
        )

    parsed = {}
    for tagged_row, payload in zip(tagged, payload_rows):
        id_1, id_2, ged_value, ged_nc, ged_in, ged_ie, _mappings = tagged_row
        payload_id_1 = int(payload["new_query_gid"])
        payload_id_2 = int(payload["new_db_gid"])
        if (int(id_1), int(id_2)) != (payload_id_1, payload_id_2):
            raise ValueError(
                f"Tagged/payload pair mismatch for {dataset_root}: "
                f"TaGED=({id_1}, {id_2}) payload=({payload_id_1}, {payload_id_2})"
            )
        matching_payload = payload["matching"]
        trust_source_rows = matching_payload.get("source") == "reference_graph_1_to_graph_2"
        corrected_row_mapping = _correct_solver_print_pairs_to_row_mapping(
            matching_payload["query_to_db"],
            n1=len(_mappings[0]) if _mappings else None,
            trust_source_rows=trust_source_rows,
        )
        parsed[(int(id_1), int(id_2))] = (
            (ged_value, ged_nc, ged_in, ged_ie),
            [corrected_row_mapping],
        )
    return parsed


def load_ged_map(data_location="", dataset_name="AIDS", file_name="TaGED.json"):
    dataset_root, resolved_name = resolve_dataset_root(data_location, dataset_name)
    path = join(dataset_root, file_name)
    if isfile(path):
        cache_path = cache_file_path(dataset_root, f"ged_dict_{basename(path)}")
        if isfile(cache_path):
            cached = load_pickle_cache(cache_path)
            if cached is not None:
                return cached
        if dataset_has_solver_print_order_bug(dataset_name) or dataset_has_solver_print_order_bug(resolved_name):
            parsed = _load_tagged_ged_map_with_corrected_payload_mappings(dataset_root, path)
        else:
            parsed = _load_tagged_ged_map(path)
        save_pickle_cache(cache_path, parsed)
        return parsed

    payload_path = join(dataset_root, "pair_matching_payloads.jsonl")
    cache_path = cache_file_path(dataset_root, f"ged_dict_matching_payloads_{basename(payload_path)}")
    if isfile(cache_path):
        cached = load_pickle_cache(cache_path)
        if cached is not None:
            return cached

    payload_parsed = _load_dense_minimal_ged_map_from_matching_payloads(dataset_root)
    if payload_parsed is None:
        raise FileNotFoundError(
            f"Could not find GED metadata for {dataset_name}: missing {path} and {payload_path}"
        )
    save_pickle_cache(cache_path, payload_parsed)
    return payload_parsed


def load_ged(ged_dict, data_location="", dataset_name="AIDS", file_name="TaGED.json"):
    ged_dict.update(load_ged_map(data_location, dataset_name, file_name))
