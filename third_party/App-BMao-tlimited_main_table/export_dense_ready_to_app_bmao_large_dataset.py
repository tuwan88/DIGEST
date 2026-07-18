import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_ROOT = REPO_ROOT / 'artifacts/Overall_Performance/direct_data/json_data'
OUTPUT_ROOT = REPO_ROOT / 'artifacts/Overall_Performance/app_bmao_benchmark'
DATASETS = ['ogbg-molhiv', 'ogbg-molpcba', 'ogbg-code2']


def load_graph_json(dataset_root: Path, new_gid: int):
    for split_dir in ('train', 'test'):
        path = dataset_root / split_dir / f'{new_gid}.json'
        if path.exists():
            with open(path, 'r') as f:
                return json.load(f)
    raise FileNotFoundError(f'graph json not found for gid={new_gid} under {dataset_root}')


def write_graph_text(graph_obj, out_path: Path):
    n = int(graph_obj['n'])
    edges = [tuple(map(int, e)) for e in graph_obj['graph']]
    labels = [int(x) for x in graph_obj['labels']]
    feature_dim = max(labels) + 1 if labels else 1
    if feature_dim <= 1:
        feature_dim = 1
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        f.write(f"{n} {len(edges)} {feature_dim}\n")
        for idx, label in enumerate(labels):
            f.write(f"{idx} {label}\n")
        for u, v in edges:
            f.write(f"{u} {v}\n")


def build_pair_to_mapping(dataset_root: Path):
    tag_rows = json.load(open(dataset_root / 'TaGED.json', 'r'))
    pair_to_mapping = {}
    for row in tag_rows:
        new_q_gid, new_db_gid, ged = int(row[0]), int(row[1]), int(row[2])
        row_mapping = row[6][0]
        query_to_db = [[int(i), int(col)] for i, col in enumerate(row_mapping) if col is not None and int(col) >= 0]
        pair_to_mapping[(new_q_gid, new_db_gid, ged)] = query_to_db
    return pair_to_mapping


def export_dataset(ds: str):
    dataset_root = INPUT_ROOT / ds
    out_root = OUTPUT_ROOT / ds
    out_root.mkdir(parents=True, exist_ok=True)

    manifest = json.load(open(dataset_root / 'pair_manifest.json', 'r'))
    pair_to_mapping = build_pair_to_mapping(dataset_root)

    orig_to_new = {}
    orig_to_graph = {}
    for item in manifest:
        for original_id, new_gid in [
            (item['query_graph_id'], int(item['new_query_gid'])),
            (item['db_graph_id'], int(item['new_db_gid'])),
        ]:
            prev = orig_to_new.get(original_id)
            if prev is not None and prev != new_gid:
                raise ValueError(f'inconsistent gid mapping for {original_id}: {prev} vs {new_gid}')
            orig_to_new[original_id] = new_gid

    graph_id_map = []
    for original_id, new_gid in sorted(orig_to_new.items()):
        graph_obj = load_graph_json(dataset_root, new_gid)
        out_path = out_root / original_id
        write_graph_text(graph_obj, out_path)
        orig_to_graph[original_id] = graph_obj
        graph_id_map.append({
            'original_graph_id': original_id,
            'new_gid': new_gid,
            'n': int(graph_obj['n']),
            'm': int(graph_obj['m']),
            'graph_label': graph_obj.get('graph_label'),
        })

    with open(out_root / 'graph_id_map.json', 'w') as f:
        json.dump(graph_id_map, f, indent=2)

    split_rows = {'train': [], 'val': [], 'test': []}
    for item in manifest:
        split = item['benchmark_split']
        graph_1 = item['graph_1']
        graph_2 = item['graph_2']
        q_id = item['query_graph_id']
        d_id = item['db_graph_id']
        new_q_gid = int(item['new_query_gid'])
        new_d_gid = int(item['new_db_gid'])
        ged = int(item['ged'])
        mapping_key = (new_q_gid, new_d_gid, ged)
        query_to_db = pair_to_mapping.get(mapping_key)
        if query_to_db is None:
            raise KeyError(f'missing TaGED mapping for {mapping_key} in {ds}')

        g1 = orig_to_graph[graph_1]
        g2 = orig_to_graph[graph_2]
        row = {
            'benchmark_split': split,
            'graph_1': graph_1,
            'graph_2': graph_2,
            'graph_1_num_nodes': int(g1['n']),
            'graph_2_num_nodes': int(g2['n']),
            'graph_1_num_edges': int(g1['m']),
            'graph_2_num_edges': int(g2['m']),
            'batch_status': 'ok',
            'status': 'success',
            'ged': ged,
            'normalized_ged': float(item['normalized_ged']),
            'matching': {
                'query_graph_id': q_id,
                'db_graph_id': d_id,
                'q_g_swapped': bool(item['q_g_swapped']),
                'query_to_db': query_to_db,
            },
        }
        split_rows[split].append(row)

    for split, rows in split_rows.items():
        with open(out_root / f'{split}.jsonl', 'w') as f:
            for row in rows:
                f.write(json.dumps(row) + '\n')

    summary = {
        'dataset': ds,
        'source_root': str(dataset_root),
        'num_graphs': len(orig_to_new),
        'split_pair_counts': {k: len(v) for k, v in split_rows.items()},
        'avg_ged': {
            split: (sum(row['ged'] for row in rows) / len(rows) if rows else None)
            for split, rows in split_rows.items()
        },
    }
    with open(out_root / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'exported {ds}:', summary)


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for ds in DATASETS:
        export_dataset(ds)
    print('done:', OUTPUT_ROOT)


if __name__ == '__main__':
    main()
