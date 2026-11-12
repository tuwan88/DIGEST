"""Use the smaller graph as the source without losing the input direction."""
import torch


def smaller_first_item(item):
    """Orient an unpadded dense pair and its supervision together.

    Equal-size pairs retain their input direction. A repeated call is safe,
    including when an already oriented batch is trimmed and collated again.
    """
    result = dict(item)
    swapped = int(item["n1"]) > int(item["n2"])
    previous = bool(item.get("pair_swapped", False))
    result["pair_swapped"] = torch.tensor(previous ^ swapped, dtype=torch.bool)
    if not swapped:
        return result
    for left, right in (("x1", "x2"), ("adj1", "adj2"),
                        ("ged_adj1", "ged_adj2"), ("n1", "n2"),
                        ("left_degree", "right_degree"),
                        ("left_label", "right_label")):
        if left in item and right in item:
            result[left], result[right] = item[right], item[left]
    for key in ("pair", "pair_gid"):
        result[key] = item[key].flip(0)
    for key in ("gt_matching", "base_cost"):
        if key in item:
            result[key] = item[key].transpose(-2, -1).contiguous()
    return result


def smaller_first_fixed_spec(pair, sizes):
    """Orient fixed-pair indices and sparse reference coordinates on the CPU."""
    if sizes[int(pair[1])] <= sizes[int(pair[2])]:
        return pair, False
    oriented = list(pair)
    oriented[1], oriented[2] = pair[2], pair[1]
    if len(pair) > 4:
        oriented[4] = [(int(col), int(row)) for row, col in pair[4] or []]
    return tuple(oriented), True


def restore_raw_pair_direction(record, swapped):
    """Restore a public record to the original query/target direction."""
    record["input_pair_swapped"] = bool(swapped or record.get("input_pair_swapped", False))
    if not swapped:
        return record
    for left, right in (("graph_1", "graph_2"), ("graph_1_gid", "graph_2_gid"),
                        ("n1", "n2"), ("m1", "m2")):
        record[left], record[right] = record[right], record[left]
    record["pair_id"] = "{}_{}".format(record["graph_1_gid"], record["graph_2_gid"])
    if record.get("matching") is not None:
        record["matching"] = tuple(sorted((int(col), int(row)) for row, col in record["matching"]))
    return record


def restore_refine_result_direction(item):
    """Restore final result JSON, independently of compact/batched search I/O."""
    if item.get("output_original_direction", False):
        return item
    swapped = bool(item.get("pair_swapped", False))
    item["input_pair_swapped"] = swapped
    item["output_original_direction"] = True
    if not swapped:
        return item
    for left, right in (("graph_1", "graph_2"), ("graph_1_gid", "graph_2_gid"),
                        ("n1", "n2")):
        if left in item and right in item:
            item[left], item[right] = item[right], item[left]
    payload = item.get("best_matching_payload")
    if payload is not None:
        payload = dict(payload)
        pairs = payload["query_to_db"]
        # First undo any solver-local reversal, then undo the input reversal.
        if not payload.get("q_g_swapped", False):
            pairs = tuple(sorted((int(col), int(row)) for row, col in pairs))
        payload.update(query_to_db=pairs, q_g_swapped=False,
                       query_graph_id=str(item["graph_1_gid"]),
                       db_graph_id=str(item["graph_2_gid"]))
        item["best_matching_payload"] = payload
    if "anchor_targets" in item:
        anchors = {str(col): int(row) for row, col in item["anchor_targets"].items()}
        item["anchor_targets"] = anchors
        item["anchor_rows"] = sorted(int(row) for row in anchors)
    if "best_anchor_violations" in item:
        item["best_anchor_violations"] = [
            [int(col), int(row)] for row, col in item["best_anchor_violations"]]
    return item
