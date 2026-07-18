"""Adapters from external probability matrices to PartialDiff refine candidates.

The refine backends consume a common candidate-pair schema.  PartialDiff
produces that schema during diffusion inference, but any method with a
query-target probability or score matrix can be normalized into the same form.
"""

import json
import math
from pathlib import Path

import torch
from scipy.optimize import linear_sum_assignment


PROBABILITY_KEYS = (
    "probabilities",
    "probability_matrix",
    "prob_matrix",
    "final_probabilities",
    "probs",
)
SCORE_KEYS = ("scores", "score_matrix", "final_scores", "logits")
MATCHING_KEYS = ("matching", "final_matching", "final_matchings")
CANDIDATE_GED_KEYS = ("candidate_ged", "ged", "baseline_candidate_ged")


def _as_tensor(value, dtype=torch.float):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().to(dtype=dtype)
    return torch.tensor(value, dtype=dtype)


def _first_present(record, keys):
    for key in keys:
        if key in record and record[key] is not None:
            return key, record[key]
    return None, None


def _load_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_external_candidate_records(path):
    """Load external candidate records from .pt/.pth/.json/.jsonl files."""
    path = Path(path)
    if path.suffix in {".pt", ".pth"}:
        payload = torch.load(path, map_location="cpu")
        if isinstance(payload, dict):
            if "pairs" in payload:
                return payload["pairs"]
            for key in ("records", "candidates", "candidate_records", "data"):
                if key in payload:
                    return payload[key]
        if isinstance(payload, list):
            return payload
        raise ValueError(f"Unsupported torch candidate payload in {path}")
    if path.suffix == ".jsonl":
        return _load_jsonl(path)
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            for key in ("pairs", "records", "candidates", "candidate_records", "data"):
                if key in payload:
                    return payload[key]
        if isinstance(payload, list):
            return payload
        raise ValueError(f"Unsupported JSON candidate payload in {path}")
    raise ValueError(f"Unsupported candidate file extension: {path.suffix}")


def looks_like_refine_candidate(record):
    required = {"pair", "pair_gid", "n1", "n2", "candidate_ged", "final_probabilities", "final_matchings", "final_scores"}
    return isinstance(record, dict) and required.issubset(record.keys())


def _resolve_graph_pair(record, trainer):
    if "pair" in record:
        pair = _as_tensor(record["pair"], dtype=torch.long).view(-1)
        if pair.numel() >= 2:
            graph_1 = int(pair[0].item())
            graph_2 = int(pair[1].item())
            graph_1_gid = int(record.get("graph_1_gid", trainer.gid[graph_1]))
            graph_2_gid = int(record.get("graph_2_gid", trainer.gid[graph_2]))
            return graph_1, graph_2, graph_1_gid, graph_2_gid

    graph_1 = record.get("graph_1", record.get("query_graph", record.get("query_index")))
    graph_2 = record.get("graph_2", record.get("target_graph", record.get("db_index")))
    if graph_1 is not None and graph_2 is not None:
        graph_1 = int(graph_1)
        graph_2 = int(graph_2)
        graph_1_gid = int(record.get("graph_1_gid", trainer.gid[graph_1]))
        graph_2_gid = int(record.get("graph_2_gid", trainer.gid[graph_2]))
        return graph_1, graph_2, graph_1_gid, graph_2_gid

    if "pair_gid" in record:
        pair_gid = _as_tensor(record["pair_gid"], dtype=torch.long).view(-1)
        graph_1_gid = int(pair_gid[0].item())
        graph_2_gid = int(pair_gid[1].item())
    else:
        graph_1_gid = record.get("graph_1_gid", record.get("query_gid"))
        graph_2_gid = record.get("graph_2_gid", record.get("target_gid", record.get("db_gid")))
        if graph_1_gid is None or graph_2_gid is None:
            raise KeyError("Each external record needs pair/pair_gid or graph_1_gid/graph_2_gid.")
        graph_1_gid = int(graph_1_gid)
        graph_2_gid = int(graph_2_gid)

    try:
        graph_1 = int(trainer.gid_to_index[graph_1_gid])
        graph_2 = int(trainer.gid_to_index[graph_2_gid])
    except KeyError as exc:
        raise KeyError(f"Graph gid {exc.args[0]} is not present in trainer.gid_to_index") from exc
    return graph_1, graph_2, graph_1_gid, graph_2_gid


def _matrix_stack(record):
    prob_key, prob_value = _first_present(record, PROBABILITY_KEYS)
    score_key, score_value = _first_present(record, SCORE_KEYS)
    if prob_value is None and score_value is None:
        raise KeyError(f"External record needs one of probability keys {PROBABILITY_KEYS} or score keys {SCORE_KEYS}.")

    probabilities = None if prob_value is None else _as_tensor(prob_value, dtype=torch.float)
    scores = None if score_value is None else _as_tensor(score_value, dtype=torch.float)
    if probabilities is not None and probabilities.dim() == 2:
        probabilities = probabilities.unsqueeze(0)
    if scores is not None and scores.dim() == 2:
        scores = scores.unsqueeze(0)
    if probabilities is not None and probabilities.dim() != 3:
        raise ValueError(f"{prob_key} must have shape [n1,n2] or [K,n1,n2], got {tuple(probabilities.shape)}")
    if scores is not None and scores.dim() != 3:
        raise ValueError(f"{score_key} must have shape [n1,n2] or [K,n1,n2], got {tuple(scores.shape)}")

    if probabilities is None:
        probabilities = torch.sigmoid(scores)
    if scores is None:
        scores = torch.logit(probabilities.clamp(1e-6, 1 - 1e-6))

    if probabilities.shape != scores.shape:
        raise ValueError(f"Probability and score tensors must have the same shape, got {probabilities.shape} vs {scores.shape}")
    return probabilities, scores


def decode_probability_matrix(matrix, mode="hungarian"):
    """Decode a 2-D probability/score matrix into a one-to-one dense matching."""
    if matrix.dim() != 2:
        raise ValueError(f"decode_probability_matrix expects [n1,n2], got {tuple(matrix.shape)}")
    n1, n2 = int(matrix.shape[0]), int(matrix.shape[1])
    matching = torch.zeros((n1, n2), dtype=torch.float)
    if n1 == 0 or n2 == 0:
        return matching

    work = matrix.detach().cpu().float()
    valid = torch.isfinite(work)
    if not bool(valid.any()):
        return matching
    work = work.masked_fill(~valid, -1e12)

    if mode == "hungarian":
        rows, cols = linear_sum_assignment((-work).numpy())
        for row, col in zip(rows.tolist(), cols.tolist()):
            if math.isfinite(float(work[row, col].item())):
                matching[int(row), int(col)] = 1.0
        return matching

    if mode == "greedy":
        flat_order = torch.argsort(work.reshape(-1), descending=True)
        used_rows = set()
        used_cols = set()
        for flat_idx in flat_order.tolist():
            row = int(flat_idx // n2)
            col = int(flat_idx % n2)
            if row in used_rows or col in used_cols:
                continue
            if not math.isfinite(float(work[row, col].item())):
                continue
            matching[row, col] = 1.0
            used_rows.add(row)
            used_cols.add(col)
            if len(used_rows) >= min(n1, n2):
                break
        return matching

    raise ValueError(f"Unsupported decode mode: {mode}")


def _matching_stack(record, scores, decode_mode):
    _, matching_value = _first_present(record, MATCHING_KEYS)
    if matching_value is not None:
        matchings = _as_tensor(matching_value, dtype=torch.float)
        if matchings.dim() == 2:
            matchings = matchings.unsqueeze(0)
        if matchings.dim() != 3:
            raise ValueError(f"matching must have shape [n1,n2] or [K,n1,n2], got {tuple(matchings.shape)}")
        return matchings
    return torch.stack([decode_probability_matrix(scores[idx], mode=decode_mode) for idx in range(scores.shape[0])], dim=0)


def _candidate_ged_stack(record, trainer, graph_1, graph_2, matchings):
    _, candidate_value = _first_present(record, CANDIDATE_GED_KEYS)
    if candidate_value is not None:
        candidate_ged = _as_tensor(candidate_value, dtype=torch.float).view(-1)
        if candidate_ged.numel() == 1 and matchings.shape[0] > 1:
            candidate_ged = candidate_ged.repeat(matchings.shape[0])
        if candidate_ged.numel() != matchings.shape[0]:
            raise ValueError(f"candidate_ged length {candidate_ged.numel()} does not match K={matchings.shape[0]}")
        return candidate_ged

    batch = {
        "x1": trainer.features[graph_1].unsqueeze(0).repeat(matchings.shape[0], 1, 1),
        "x2": trainer.features[graph_2].unsqueeze(0).repeat(matchings.shape[0], 1, 1),
        "ged_adj1": trainer.ged_adj[graph_1].unsqueeze(0).repeat(matchings.shape[0], 1, 1),
        "ged_adj2": trainer.ged_adj[graph_2].unsqueeze(0).repeat(matchings.shape[0], 1, 1),
        "n1": torch.full((matchings.shape[0],), int(trainer.gn[graph_1]), dtype=torch.long),
        "n2": torch.full((matchings.shape[0],), int(trainer.gn[graph_2]), dtype=torch.long),
        "pair": torch.tensor([[graph_1, graph_2]], dtype=torch.long).repeat(matchings.shape[0], 1),
    }
    return trainer.dense_ged_from_clean_matchings_direct(batch, matchings).detach().cpu().float()


def _gt_ged(record, trainer, graph_1, graph_2):
    for key in ("gt_ged", "reference_ged", "true_ged"):
        if key in record and record[key] is not None:
            return float(record[key])
    metadata = trainer.get_pair_metadata(graph_1, graph_2)
    if metadata is not None and "ta_ged" in metadata:
        ta_ged = metadata["ta_ged"]
        if isinstance(ta_ged, (list, tuple)) and ta_ged:
            return float(ta_ged[0])
        return float(ta_ged)
    return -1.0


def normalize_external_candidate_record(record, trainer, decode_mode="hungarian"):
    """Normalize one external record into the refine candidate schema."""
    if looks_like_refine_candidate(record):
        return record

    graph_1, graph_2, graph_1_gid, graph_2_gid = _resolve_graph_pair(record, trainer)
    n1 = int(trainer.gn[graph_1])
    n2 = int(trainer.gn[graph_2])
    probabilities, scores = _matrix_stack(record)
    probabilities = probabilities[:, :n1, :n2].contiguous()
    scores = scores[:, :n1, :n2].contiguous()
    matchings = _matching_stack(record, scores, decode_mode=decode_mode)[:, :n1, :n2].contiguous()
    candidate_ged = _candidate_ged_stack(record, trainer, graph_1, graph_2, matchings)
    best_index = int(torch.argmin(candidate_ged).item()) if candidate_ged.numel() > 0 else 0

    return {
        "pair": torch.tensor([graph_1, graph_2], dtype=torch.long),
        "pair_gid": torch.tensor([graph_1_gid, graph_2_gid], dtype=torch.long),
        "n1": n1,
        "n2": n2,
        "gt_ged": _gt_ged(record, trainer, graph_1, graph_2),
        "candidate_ged": candidate_ged,
        "best_index": best_index,
        "final_probabilities": probabilities,
        "final_matchings": matchings,
        "final_scores": scores,
        "x1": trainer.features[graph_1].detach().cpu().clone(),
        "x2": trainer.features[graph_2].detach().cpu().clone(),
        "ged_adj1": trainer.ged_adj[graph_1].detach().cpu().clone(),
        "ged_adj2": trainer.ged_adj[graph_2].detach().cpu().clone(),
    }


def normalize_external_candidate_records(records, trainer, decode_mode="hungarian", max_pairs=0):
    normalized = []
    limit = int(max_pairs or 0)
    for idx, record in enumerate(records):
        if limit > 0 and idx >= limit:
            break
        normalized.append(normalize_external_candidate_record(record, trainer, decode_mode=decode_mode))
    return normalized
