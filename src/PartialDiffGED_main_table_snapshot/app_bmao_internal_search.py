from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import time
import math
import os

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


try:
    from torch_linear_assignment import batch_linear_assignment
except ImportError:
    batch_linear_assignment = None

"""Shared runtime config helpers kept for the gpu_refine internal path."""


def resolve_v5_runtime_config(runner):
    return {}


def _resolve_gpu_refine_runtime_config_base(runner):
    cfg = resolve_v5_runtime_config(runner)
    args = runner.trainer.args
    cfg.update({
        'beam_width': max(1, int(getattr(args, 'v9_beam_width', runner.beam_width))),
        'branch_width': max(1, int(getattr(args, 'v9_branch_width', runner.branch_topk))),
        'candidate_cap': max(1, int(getattr(args, 'v9_candidate_cap', max(runner.branch_topk, 1)))),
        'lb_type': str(getattr(args, 'v9_lb_type', 'row_col_min')),
        'action_lb_slack': float(getattr(args, 'v9_action_lb_slack', -1.0)),
        'rerank_pool': max(1, int(getattr(args, 'v9_rerank_pool', 32))),
        'lb_tiebreak_weight': float(getattr(args, 'v9_lb_tiebreak_weight', 0.01)),
        'completion_mode': str(getattr(args, 'v9_completion_mode', os.environ.get('PARTIALDIFF_GPU_REFINE_COMPLETION_MODE', 'diffusion_score'))),
        'search_prefix_fraction': float(getattr(args, 'v9_search_prefix_fraction', os.environ.get('PARTIALDIFF_GPU_REFINE_SEARCH_PREFIX_FRACTION', 1.0))),
        'search_prefix_min_rows': max(0, int(getattr(args, 'v9_search_prefix_min_rows', os.environ.get('PARTIALDIFF_GPU_REFINE_SEARCH_PREFIX_MIN_ROWS', 1)))),
        'max_search_depth': max(0, int(getattr(args, 'v9_max_search_depth', 0))),
    })
    return cfg


def _build_batched_candidate_cols(score_batch, col_mask, candidate_cap):
    num_contexts, max_n1, max_n2 = score_batch.shape
    device = score_batch.device
    cols = torch.full((num_contexts, max_n1, candidate_cap), -1, dtype=torch.long, device=device)
    if num_contexts == 0 or max_n1 == 0 or max_n2 == 0:
        return cols

    topk_count = min(int(candidate_cap), int(max_n2))
    valid_scores = score_batch.masked_fill(~col_mask.view(num_contexts, 1, max_n2), float('-inf'))
    top_values, top_cols = torch.topk(valid_scores, k=topk_count, dim=2)
    top_cols = top_cols.masked_fill(~torch.isfinite(top_values), -1)
    cols[:, :, :topk_count] = top_cols
    return cols


def _matched_probabilities_from_dense_matching(final_matching, final_probabilities, device):
    work_device = final_probabilities.device
    final_matching = final_matching[: final_probabilities.shape[0], : final_probabilities.shape[1]].to(device=work_device)
    final_probabilities = final_probabilities.to(dtype=torch.float32)
    n1 = int(final_matching.shape[0])
    matched_probs = torch.full((n1,), float('-inf'), dtype=torch.float32, device=work_device)
    if n1 <= 0 or int(final_matching.shape[1]) <= 0:
        return matched_probs.to(device=device)

    matching = final_matching > 0.5
    has_match = matching.any(dim=1)
    rows = torch.arange(n1, device=work_device, dtype=torch.long)
    cols = matching.to(dtype=torch.long).argmax(dim=1)
    selected_probs = final_probabilities[rows, cols]
    return torch.where(has_match, selected_probs, matched_probs).to(device=device)


def _select_batched_anchor_targets(baseline_row_maps, matched_prob_batch, row_mask, n2, anchor_ratio):
    device = baseline_row_maps.device
    num_contexts, max_n1 = baseline_row_maps.shape
    if float(anchor_ratio) <= 0.0 or num_contexts == 0 or max_n1 == 0:
        return torch.full_like(baseline_row_maps, -2), torch.zeros_like(baseline_row_maps, dtype=torch.bool)

    valid = (baseline_row_maps >= 0) & row_mask & (baseline_row_maps < n2.view(-1, 1))
    matched_counts = valid.sum(dim=1)
    keep_counts = torch.ceil(matched_counts.to(dtype=torch.float32) * float(anchor_ratio)).to(dtype=torch.long)
    keep_counts = torch.minimum(keep_counts, matched_counts)
    matched_probs = matched_prob_batch.to(device=device, dtype=torch.float32).masked_fill(~valid, float('-inf'))

    order = torch.argsort(matched_probs, dim=1, descending=True)
    rank_values = torch.arange(max_n1, device=device, dtype=torch.long).view(1, -1).expand(num_contexts, -1)
    ranks = torch.empty_like(order)
    ranks.scatter_(1, order, rank_values)
    anchor_mask = valid & (ranks < keep_counts.view(-1, 1))
    anchor_targets = torch.where(anchor_mask, baseline_row_maps, torch.full_like(baseline_row_maps, -2))
    return anchor_targets, anchor_mask


def _filter_row_order_by_anchor(row_order, search_row_count, anchor_mask):
    if row_order.numel() == 0:
        return row_order, search_row_count
    device = row_order.device
    max_n1 = row_order.shape[1]
    order_pos = torch.arange(max_n1, device=device, dtype=torch.long).view(1, -1)
    order_valid = order_pos < search_row_count.view(-1, 1)
    safe_order = row_order.clamp(min=0, max=max(max_n1 - 1, 0))
    ordered_anchor = anchor_mask.gather(1, safe_order)
    keep_order = order_valid & (~ordered_anchor)
    sort_key = torch.where(keep_order, order_pos, order_pos + max_n1)
    perm = torch.argsort(sort_key, dim=1)
    filtered_order = row_order.gather(1, perm)
    filtered_count = keep_order.sum(dim=1).to(dtype=torch.long)
    return filtered_order, filtered_count


def _apply_search_prefix(search_row_count, prefix_fraction, prefix_min_rows):
    if float(prefix_fraction) >= 1.0:
        return search_row_count
    prefix = torch.ceil(search_row_count.to(dtype=torch.float32) * max(0.0, float(prefix_fraction))).to(dtype=torch.long)
    min_rows = torch.full_like(search_row_count, max(0, int(prefix_min_rows)))
    clipped = torch.minimum(search_row_count, torch.maximum(min_rows, prefix))
    return torch.where(search_row_count > 0, clipped, search_row_count)


def _compute_batched_anchor_initial_cost(anchor_targets, anchor_mask, n2, node_cost_batch, ged_adj1_batch, ged_adj2_batch):
    device = anchor_targets.device
    num_contexts, max_n1 = anchor_targets.shape
    if num_contexts == 0 or max_n1 == 0:
        return torch.zeros((num_contexts,), dtype=torch.float32, device=device)

    safe_cols = anchor_targets.clamp(min=0, max=max(int(ged_adj2_batch.shape[1]) - 1, 0))
    if ged_adj2_batch.shape[1] > 0:
        node_cost = node_cost_batch.gather(2, safe_cols.unsqueeze(-1)).squeeze(-1)
        node_cost = node_cost.masked_fill(~anchor_mask, 0.0).sum(dim=1)
    else:
        node_cost = torch.zeros((num_contexts,), dtype=torch.float32, device=device)

    if max_n1 <= 1 or ged_adj2_batch.shape[1] <= 0:
        return node_cost.to(dtype=torch.float32)

    pair_mask = anchor_mask.unsqueeze(1) & anchor_mask.unsqueeze(2)
    upper_mask = torch.triu(torch.ones((max_n1, max_n1), dtype=torch.bool, device=device), diagonal=1).view(1, max_n1, max_n1)
    pair_mask = pair_mask & upper_mask
    batch_idx = torch.arange(num_contexts, device=device, dtype=torch.long).view(-1, 1, 1)
    src_cols = safe_cols.unsqueeze(2).expand(-1, -1, max_n1)
    dst_cols = safe_cols.unsqueeze(1).expand(-1, max_n1, -1)
    target_edges = (ged_adj2_batch > 0.5)[batch_idx, src_cols, dst_cols]
    query_edges = ged_adj1_batch > 0.5
    edge_cost = ((query_edges ^ target_edges) & pair_mask).sum(dim=(1, 2)).to(dtype=torch.float32)
    return node_cost.to(dtype=torch.float32) + edge_cost


def _compute_anchor_action_cost_cache(anchor_targets, anchor_mask, ged_adj1_batch, ged_adj2_batch):
    device = anchor_targets.device
    num_contexts, max_n1 = anchor_targets.shape
    max_n2 = int(ged_adj2_batch.shape[1])
    anchor_delete_cost = torch.zeros((num_contexts, max_n1), dtype=torch.float32, device=device)
    anchor_edge_cost = torch.zeros((num_contexts, max_n1, max_n2), dtype=torch.float32, device=device)
    if num_contexts == 0 or max_n1 == 0:
        return anchor_edge_cost, anchor_delete_cost

    adj1 = ged_adj1_batch > 0.5
    adj2 = ged_adj2_batch > 0.5
    batch_idx = torch.arange(num_contexts, device=device, dtype=torch.long)
    for anchor_row in range(max_n1):
        valid = anchor_mask[:, anchor_row].to(dtype=torch.float32)
        q_edges = adj1[:, :, anchor_row]
        anchor_delete_cost.add_(q_edges.to(dtype=torch.float32) * valid.view(-1, 1))
        if max_n2 > 0:
            anchor_cols = anchor_targets[:, anchor_row].clamp(min=0, max=max_n2 - 1)
            g_edges = adj2[batch_idx, :, anchor_cols]
            diff = q_edges.unsqueeze(2) ^ g_edges.unsqueeze(1)
            anchor_edge_cost.add_(diff.to(dtype=torch.float32) * valid.view(-1, 1, 1))
    return anchor_edge_cost, anchor_delete_cost


def _score_compact_row_maps_exact(
    runner,
    pair_idx_all,
    row_maps_all,
    n1,
    n2,
    ged_adj2_batch,
    label1_batch,
    label2_batch,
    left_upper_batch,
    left_edge_count_batch,
    right_edge_count_batch,
):
    if pair_idx_all.numel() == 0:
        return torch.empty((0,), dtype=torch.float, device=row_maps_all.device)
    batch_n1 = n1.index_select(0, pair_idx_all)
    batch_n2 = n2.index_select(0, pair_idx_all)
    right_adj_batch = ged_adj2_batch.index_select(0, pair_idx_all)
    return runner._ged_from_compact_row_maps_precomputed(
        row_maps=row_maps_all,
        pair_idx=pair_idx_all,
        batch_n1=batch_n1,
        batch_n2=batch_n2,
        right_adj_batch=right_adj_batch,
        label1_batch=label1_batch,
        label2_batch=label2_batch,
        left_upper_batch=left_upper_batch,
        left_edge_count_batch=left_edge_count_batch,
        right_edge_count_batch=right_edge_count_batch,
    )


def _compute_gpu_refine_immediate_costs(
    runner,
    match_cols,
    current_rows,
    action_cols,
    valid_parent,
    node_cost_batch,
    ged_adj1_batch,
    ged_adj2_batch,
    pair_flat,
    anchor_mask_batch=None,
    anchor_edge_cost_batch=None,
    anchor_delete_cost_batch=None,
):
    device = match_cols.device
    num_contexts, beam_width, max_n1 = match_cols.shape
    max_n2 = ged_adj2_batch.shape[1]
    action_count = action_cols.shape[2]
    immediate_cost = torch.full((num_contexts, beam_width, action_count), float('inf'), dtype=torch.float32, device=device)
    if num_contexts == 0 or action_count == 0:
        return immediate_cost

    flat_size = num_contexts * beam_width
    current_rows_flat = current_rows.reshape(-1)
    matched_bool_flat = (match_cols >= 0).view(flat_size, max_n1)
    if anchor_mask_batch is not None:
        anchor_bool_flat = anchor_mask_batch.index_select(0, pair_flat)
        matched_bool_for_edges = matched_bool_flat & (~anchor_bool_flat)
    else:
        matched_bool_for_edges = matched_bool_flat
    matched_flat = matched_bool_for_edges.to(dtype=torch.float32)
    match_flat = match_cols.view(flat_size, max_n1)
    action_flat = action_cols.view(flat_size, action_count)

    if ged_adj1_batch.dtype == torch.bool or ged_adj2_batch.dtype == torch.bool:
        aq_cur_bool = ged_adj1_batch[pair_flat, current_rows_flat, :].to(dtype=torch.bool)
        delete_edge_cost = (aq_cur_bool & matched_bool_for_edges).sum(dim=1).to(dtype=torch.float32).view(num_contexts, beam_width)
    else:
        aq_cur = ged_adj1_batch[pair_flat, current_rows_flat, :].to(dtype=torch.float32)
        delete_edge_cost = (aq_cur * matched_flat).sum(dim=1).view(num_contexts, beam_width)
    if anchor_delete_cost_batch is not None:
        delete_edge_cost = delete_edge_cost + anchor_delete_cost_batch[pair_flat, current_rows_flat].view(num_contexts, beam_width)
    delete_mask = action_cols == max_n2
    immediate_cost = torch.where(
        delete_mask & valid_parent.unsqueeze(-1),
        delete_edge_cost.unsqueeze(-1) + float(runner.unmatched_cost),
        immediate_cost,
    )
    if max_n2 == 0:
        return immediate_cost

    action_safe = action_flat.clamp(min=0, max=max_n2 - 1)
    row_node_cost = node_cost_batch[pair_flat, current_rows_flat, :].to(dtype=torch.float32)
    node_selected = row_node_cost.gather(1, action_safe).view(num_contexts, beam_width, action_count)

    match_safe = match_flat.clamp(min=0, max=max_n2 - 1)
    if ged_adj1_batch.dtype == torch.bool or ged_adj2_batch.dtype == torch.bool:
        adj2_rows_bool = ged_adj2_batch[pair_flat[:, None], action_safe, :].to(dtype=torch.bool)
        mapped_ag_bool = torch.gather(
            adj2_rows_bool,
            2,
            match_safe.unsqueeze(1).expand(-1, action_count, -1),
        )
        edge_cost = (
            (mapped_ag_bool ^ aq_cur_bool.unsqueeze(1))
            & matched_bool_for_edges.unsqueeze(1)
        ).sum(dim=2).to(dtype=torch.float32).view(num_contexts, beam_width, action_count)
    else:
        adj2_rows = ged_adj2_batch[pair_flat[:, None], action_safe, :].to(dtype=torch.float32)
        mapped_ag = torch.gather(
            adj2_rows,
            2,
            match_safe.unsqueeze(1).expand(-1, action_count, -1),
        )
        edge_cost = (
            (mapped_ag - aq_cur.unsqueeze(1)).abs()
            * matched_flat.unsqueeze(1)
        ).sum(dim=2).view(num_contexts, beam_width, action_count)
    if anchor_edge_cost_batch is not None:
        anchor_edge = anchor_edge_cost_batch[pair_flat[:, None], current_rows_flat[:, None], action_safe]
        edge_cost = edge_cost + anchor_edge.view(num_contexts, beam_width, action_count)

    chosen_real = (action_cols >= 0) & (action_cols < max_n2)
    return torch.where(chosen_real, node_selected + edge_cost, immediate_cost)


def _compute_gpu_refine_row_col_lb(
    runner,
    pair_idx,
    row_maps,
    used_cols,
    current_rows,
    action_cols,
    child_g,
    node_cost_batch,
    row_mask,
    col_mask,
    max_n2,
):
    device = row_maps.device
    num_states, max_n1 = row_maps.shape
    action_count = action_cols.shape[1]
    if num_states == 0 or action_count == 0:
        return child_g

    unmatched_cost = float(runner.unmatched_cost)
    inf = float('inf')

    node_cost = node_cost_batch.index_select(0, pair_idx).to(dtype=torch.float32)
    available_cols = col_mask.index_select(0, pair_idx) & (~used_cols)
    available_cost = node_cost.masked_fill(~available_cols.unsqueeze(1), inf)

    row_free = row_mask.index_select(0, pair_idx) & (row_maps < -1)
    row_remaining = row_free.clone()
    row_remaining[torch.arange(num_states, device=device), current_rows] = False
    if os.environ.get('PARTIALDIFF_ROW_ONLY_LB', '0') == '1':
        if max_n2 > 0:
            row_min = available_cost.min(dim=2).values
        else:
            row_min = torch.full((num_states, max_n1), inf, dtype=torch.float32, device=device)
        row_min.clamp_max_(unmatched_cost)
        row_min.masked_fill_(~row_remaining, 0.0)
        future_lb = row_min.sum(dim=1, keepdim=True).expand(-1, action_count)
        return child_g + future_lb

    if max_n2 >= 2:
        row_two = torch.topk(available_cost, k=2, dim=2, largest=False).values
        row_min1 = row_two[..., 0]
        row_min2 = row_two[..., 1]
        row_argmin = available_cost.argmin(dim=2)
    elif max_n2 == 1:
        row_min1 = available_cost[..., 0]
        row_min2 = torch.full_like(row_min1, inf)
        row_argmin = torch.zeros((num_states, max_n1), dtype=torch.long, device=device)
    else:
        row_min1 = torch.full((num_states, max_n1), inf, dtype=torch.float32, device=device)
        row_min2 = torch.full_like(row_min1, inf)
        row_argmin = torch.zeros((num_states, max_n1), dtype=torch.long, device=device)
    row_min1.clamp_max_(unmatched_cost)
    row_min2.clamp_max_(unmatched_cost)
    row_min1.masked_fill_(~row_remaining, 0.0)
    row_min2.masked_fill_(~row_remaining, 0.0)

    chosen_real = (action_cols >= 0) & (action_cols < max_n2)
    row_base = row_min1.sum(dim=1, keepdim=True)
    if max_n2 > 0:
        row_delta = (row_min2 - row_min1).masked_fill(~row_remaining, 0.0)
        delta_by_col = torch.zeros((num_states, max_n2), dtype=torch.float32, device=device)
        delta_by_col.scatter_add_(
            1,
            row_argmin.clamp(min=0, max=max_n2 - 1),
            row_delta,
        )
        chosen_safe = action_cols.clamp(min=0, max=max_n2 - 1)
        chosen_delta = delta_by_col.gather(1, chosen_safe)
        row_future = row_base + torch.where(chosen_real, chosen_delta, torch.zeros_like(chosen_delta))
    else:
        row_future = row_base.expand(-1, action_count)

    available_cost.masked_fill_(~row_remaining.unsqueeze(2), inf)
    col_min = available_cost.min(dim=1).values
    col_min.clamp_max_(unmatched_cost)
    col_min.masked_fill_(~available_cols, 0.0)
    col_sum = col_min.sum(dim=1, keepdim=True).expand(-1, action_count)
    if max_n2 > 0:
        chosen_safe = action_cols.clamp(min=0, max=max_n2 - 1)
        chosen_col_cost = col_min.gather(1, chosen_safe)
        col_future = torch.where(chosen_real, col_sum - chosen_col_cost, col_sum)
    else:
        col_future = col_sum

    future_lb = torch.maximum(row_future, col_future)
    return child_g + future_lb.to(dtype=child_g.dtype)


def _greedy_complete_row_maps_from_score(
    runner,
    row_maps,
    pair_idx,
    completion_score,
    n1,
    n2,
    max_n2,
    row_ids,
    col_ids,
):
    if row_maps.shape[0] == 0:
        return row_maps
    batch_n1 = n1.index_select(0, pair_idx)
    batch_n2 = n2.index_select(0, pair_idx)
    if runner.assignment_backend == 'row_top1_unique_n2':
        return runner._complete_row_maps_row_top1_unique(
            completion_score,
            row_maps,
            batch_n1,
            batch_n2,
        )

    partial, matched_mask, blocked_rows = runner._build_partial_from_row_mapping(row_maps, max_n2)
    selected = partial > 0.5
    col_taken = selected.any(dim=1)
    remaining_rows = ((~matched_mask) & (~blocked_rows) & (row_ids.view(1, -1) < batch_n1.view(-1, 1))).sum(dim=1).long()
    remaining_cols = ((~col_taken) & (col_ids.view(1, -1) < batch_n2.view(-1, 1))).sum(dim=1).long()
    matched_count = matched_mask.sum(dim=1).long()
    full_sizes = matched_count + torch.minimum(remaining_rows, remaining_cols)
    completed = runner._complete_partial_matchings_greedy(
        completion_score,
        partial,
        blocked_rows | (row_ids.view(1, -1) >= batch_n1.view(-1, 1)),
        full_sizes,
    )
    return runner._dense_matchings_to_row_maps(completed > 0.5, batch_n1)


def _build_ged_aware_completion_score(
    runner,
    row_maps,
    pair_idx,
    node_cost_batch,
    ged_adj1_batch,
    ged_adj2_batch,
    n2,
    max_n2,
):
    device = row_maps.device
    num_states, max_n1 = row_maps.shape
    if num_states == 0 or max_n2 == 0:
        return torch.empty((num_states, max_n1, max_n2), dtype=torch.float32, device=device)

    batch_n2 = n2.index_select(0, pair_idx)
    node_cost = node_cost_batch.index_select(0, pair_idx).to(dtype=torch.float32)
    adj1 = ged_adj1_batch.index_select(0, pair_idx) > 0.5
    adj2 = ged_adj2_batch.index_select(0, pair_idx) > 0.5

    matched_mask = (row_maps >= 0) & (row_maps < batch_n2.view(-1, 1))
    total_cost = node_cost.clone()
    if bool(matched_mask.any()):
        safe_cols = row_maps.clamp(min=0, max=max(max_n2 - 1, 0))
        chunk_size = max(1, int(os.environ.get('PARTIALDIFF_GED_AWARE_COMPLETION_CHUNK', '256')))
        state_arange = torch.arange(num_states, device=device)
        for start in range(0, num_states, chunk_size):
            end = min(num_states, start + chunk_size)
            local_idx = state_arange[start:end]
            edge_cost = torch.zeros((end - start, max_n1, max_n2), dtype=torch.float32, device=device)
            adj1_chunk = adj1[start:end]
            adj2_chunk = adj2[start:end]
            safe_cols_chunk = safe_cols[start:end]
            matched_chunk = matched_mask[start:end]
            chunk_rows = torch.arange(end - start, device=device)
            for mapped_row in range(max_n1):
                mapped_valid = matched_chunk[:, mapped_row]
                if not bool(mapped_valid.any()):
                    continue
                q_edges = adj1_chunk[:, :, mapped_row]
                mapped_cols = safe_cols_chunk[:, mapped_row]
                g_edges = adj2_chunk[chunk_rows, :, mapped_cols]
                diff = q_edges.unsqueeze(2) ^ g_edges.unsqueeze(1)
                edge_cost.add_(diff.to(dtype=torch.float32) * mapped_valid.view(-1, 1, 1).to(dtype=torch.float32))
            total_cost[start:end] = total_cost[start:end] + edge_cost

    score = -total_cost
    valid_cols = torch.arange(max_n2, device=device).view(1, 1, -1) < batch_n2.view(-1, 1, 1)
    return score.masked_fill(~valid_cols, float('-inf'))




def _run_gpu_refine_search(runner, contexts, variant_name, completion_mode):
    if not contexts:
        return

    cfg = _resolve_gpu_refine_runtime_config_base(runner)
    if cfg['lb_type'] != 'row_col_min':
        raise RuntimeError(f'{variant_name} currently only supports --v9-lb-type row_col_min, got {cfg["lb_type"]!r}.')

    device = runner.trainer.device
    num_contexts = len(contexts)
    max_n1 = max(context.n1 for context in contexts)
    max_n2 = max(context.n2 for context in contexts)
    feat_dim = int(runner.trainer.features[0].shape[-1])
    beam_width = int(cfg['beam_width'])
    branch_width = int(cfg['branch_width'])
    candidate_cap = int(cfg['candidate_cap'])
    rerank_pool = max(beam_width, int(cfg['rerank_pool']))
    lb_tiebreak_weight = float(cfg['lb_tiebreak_weight'])
    prefix_fraction = float(cfg.get('search_prefix_fraction', 1.0))
    prefix_min_rows = int(cfg.get('search_prefix_min_rows', 1))
    max_search_depth = int(cfg.get('max_search_depth', 0))
    anchor_ratio = float(getattr(runner.trainer.args, 'app_bmao_anchor_ratio', 0.0))
    anchor_enable = anchor_ratio > 0.0
    step_limit = max_n1 if max_search_depth <= 0 else min(max_n1, max_search_depth)
    refine_diagnostics_enable = bool(getattr(runner.trainer.args, 'app_bmao_refine_diagnostics_enable', False))
    fast_exact_update_enable = bool(getattr(runner.trainer.args, 'app_bmao_fast_exact_update', False))
    action_count = candidate_cap + 1
    top_r = min(branch_width, action_count)
    pool_size = min(max(beam_width, rerank_pool), beam_width * top_r)
    final_chunk_size = max(1, num_contexts * pool_size)
    action_lb_slack_env = os.environ.get('PARTIALDIFF_ACTION_LB_SLACK')
    action_lb_slack = float(cfg.get('action_lb_slack', -1.0))
    if action_lb_slack_env not in (None, ''):
        action_lb_slack = float(action_lb_slack_env)
    action_lb_slack = None if action_lb_slack < 0.0 else action_lb_slack
    runner.profile_metadata['search.gpu_refine_completion_mode'] = str(completion_mode)
    runner.profile_metadata['search.gpu_refine_search_prefix_fraction'] = float(prefix_fraction)
    runner.profile_metadata['search.gpu_refine_search_prefix_min_rows'] = int(prefix_min_rows)
    runner.profile_metadata['search.gpu_refine_max_search_depth'] = int(max_search_depth)

    n1 = torch.tensor([context.n1 for context in contexts], dtype=torch.long, device=device)
    n2 = torch.tensor([context.n2 for context in contexts], dtype=torch.long, device=device)
    search_row_count = torch.tensor([context.search_row_count for context in contexts], dtype=torch.long, device=device)
    row_mask = torch.arange(max_n1, device=device).view(1, -1) < n1.view(-1, 1)
    col_mask = torch.arange(max_n2, device=device).view(1, -1) < n2.view(-1, 1)
    row_ids = torch.arange(max_n1, device=device)
    col_ids = torch.arange(max_n2, device=device)

    x1_batch = torch.zeros((num_contexts, max_n1, feat_dim), dtype=torch.float, device=device)
    x2_batch = torch.zeros((num_contexts, max_n2, feat_dim), dtype=torch.float, device=device)
    ged_adj1_batch = torch.zeros((num_contexts, max_n1, max_n1), dtype=torch.float, device=device)
    ged_adj2_batch = torch.zeros((num_contexts, max_n2, max_n2), dtype=torch.float, device=device)
    row_order = torch.zeros((num_contexts, max_n1), dtype=torch.long, device=device)
    score_batch = torch.full((num_contexts, max_n1, max_n2), float('-inf'), dtype=torch.float, device=device)
    matched_prob_batch = torch.full((num_contexts, max_n1), float('-inf'), dtype=torch.float, device=device) if anchor_enable else None
    baseline_row_maps = torch.full((num_contexts, max_n1), -3, dtype=torch.long, device=device)
    best_row_maps = torch.full((num_contexts, max_n1), -3, dtype=torch.long, device=device)
    best_ub = torch.tensor([context.best_ub for context in contexts], dtype=torch.float, device=device)

    eval_counts = torch.zeros((num_contexts,), dtype=torch.long, device=device)
    expansion_counts = torch.zeros((num_contexts,), dtype=torch.long, device=device)
    step_counts = torch.zeros((num_contexts,), dtype=torch.long, device=device)
    max_frontier = torch.ones((num_contexts,), dtype=torch.long, device=device)
    if refine_diagnostics_enable:
        complete_counts = torch.zeros((num_contexts,), dtype=torch.long, device=device)
        equal_best_counts = torch.zeros((num_contexts,), dtype=torch.long, device=device)
        equal_best_diff_counts = torch.zeros((num_contexts,), dtype=torch.long, device=device)
        better_than_best_counts = torch.zeros((num_contexts,), dtype=torch.long, device=device)
    else:
        complete_counts = None
        equal_best_counts = None
        equal_best_diff_counts = None
        better_than_best_counts = None
    max_depth_reached = torch.zeros((num_contexts,), dtype=torch.long, device=device)
    ub_prune_stats_enable = os.environ.get('PARTIALDIFF_UB_PRUNE_STATS', '0') == '1'
    ub_pre_valid_total = torch.zeros((), dtype=torch.long, device=device) if ub_prune_stats_enable else None
    ub_pruned_total = torch.zeros((), dtype=torch.long, device=device) if ub_prune_stats_enable else None
    dense_lb_enable = os.environ.get('PARTIALDIFF_DENSE_LB', '0') == '1'

    for idx, context in enumerate(contexts):
        x1_batch[idx, :context.n1] = context.x1
        x2_batch[idx, :context.n2] = context.x2
        ged_adj1_batch[idx, :context.n1, :context.n1] = context.ged_adj1
        ged_adj2_batch[idx, :context.n2, :context.n2] = context.ged_adj2
        row_order[idx, :context.search_row_count] = context.row_order_tensor[:context.search_row_count]
        score_batch[idx, :context.n1, :context.n2] = context.score_matrix
        if matched_prob_batch is not None:
            matched_prob_batch[idx, :context.n1] = context.matched_prob_t[:context.n1]
        if context.baseline_row_mapping_t is not None:
            baseline_row_maps[idx, :context.n1] = context.baseline_row_mapping_t[:context.n1]
            best_row_maps[idx, :context.n1] = context.baseline_row_mapping_t[:context.n1]
        elif context.best_matching_tensor is not None:
            best_row_maps[idx, :context.n1] = runner._dense_matchings_to_row_maps(
                context.best_matching_tensor[:context.n1, :context.n2].unsqueeze(0) > 0.5,
                torch.tensor([context.n1], device=device, dtype=torch.long),
            )[0, :context.n1]

    label1_batch = torch.zeros((num_contexts, max_n1), dtype=torch.long, device=device)
    label2_batch = torch.zeros((num_contexts, max_n2), dtype=torch.long, device=device)
    for idx, context in enumerate(contexts):
        if context.query_labels_t is not None:
            label1_batch[idx, :context.n1] = context.query_labels_t[:context.n1].to(device=device, dtype=torch.long)
        else:
            label1_batch[idx, :context.n1] = runner.trainer._node_labels_dense(context.x1).to(device=device, dtype=torch.long)[:context.n1]
        if context.target_labels_t is not None:
            label2_batch[idx, :context.n2] = context.target_labels_t[:context.n2].to(device=device, dtype=torch.long)
        else:
            label2_batch[idx, :context.n2] = runner.trainer._node_labels_dense(context.x2).to(device=device, dtype=torch.long)[:context.n2]
    node_cost_batch = (label1_batch.unsqueeze(2) != label2_batch.unsqueeze(1)).to(dtype=torch.float32)
    left_upper_batch = torch.triu(ged_adj1_batch > 0.5, diagonal=1)
    left_edge_count_batch = left_upper_batch.sum(dim=(1, 2)).to(dtype=torch.float)
    right_edge_count_batch = torch.triu(ged_adj2_batch > 0.5, diagonal=1).sum(dim=(1, 2)).to(dtype=torch.float)
    candidate_cols = _build_batched_candidate_cols(score_batch, col_mask, candidate_cap)

    if anchor_enable:
        anchor_targets, anchor_mask = _select_batched_anchor_targets(
            baseline_row_maps=baseline_row_maps,
            matched_prob_batch=matched_prob_batch,
            row_mask=row_mask,
            n2=n2,
            anchor_ratio=anchor_ratio,
        )
    else:
        anchor_targets = torch.full((num_contexts, max_n1), -2, dtype=torch.long, device=device)
        anchor_mask = torch.zeros((num_contexts, max_n1), dtype=torch.bool, device=device)
    row_order, search_row_count = _filter_row_order_by_anchor(row_order, search_row_count, anchor_mask)
    search_row_count = _apply_search_prefix(search_row_count, prefix_fraction, prefix_min_rows)
    if max_search_depth > 0:
        search_row_count = torch.minimum(search_row_count, torch.full_like(search_row_count, max_search_depth))
    initial_cost = _compute_batched_anchor_initial_cost(
        anchor_targets=anchor_targets,
        anchor_mask=anchor_mask,
        n2=n2,
        node_cost_batch=node_cost_batch,
        ged_adj1_batch=ged_adj1_batch,
        ged_adj2_batch=ged_adj2_batch,
    )
    if anchor_enable:
        anchor_edge_cost, anchor_delete_cost = _compute_anchor_action_cost_cache(
            anchor_targets=anchor_targets,
            anchor_mask=anchor_mask,
            ged_adj1_batch=ged_adj1_batch,
            ged_adj2_batch=ged_adj2_batch,
        )
    else:
        anchor_edge_cost = None
        anchor_delete_cost = None

    row_mapping = torch.full((num_contexts, beam_width, max_n1), -3, dtype=torch.long, device=device)
    current_cost = torch.full((num_contexts, beam_width), float('inf'), dtype=torch.float, device=device)
    depth = torch.zeros((num_contexts, beam_width), dtype=torch.long, device=device)
    active = torch.zeros((num_contexts, beam_width), dtype=torch.bool, device=device)
    used_cols = torch.zeros((num_contexts, beam_width, max_n2), dtype=torch.bool, device=device)
    active[:, 0] = True
    row_mapping[:, 0, :] = anchor_targets
    current_cost[:, 0] = initial_cost
    if max_n2 > 0:
        initial_mapped = (row_mapping >= 0) & row_mask.view(num_contexts, 1, max_n1) & (row_mapping < n2.view(-1, 1, 1))
        if bool(initial_mapped.any()):
            state_idx, beam_idx, row_idx = torch.nonzero(initial_mapped, as_tuple=True)
            used_cols[state_idx, beam_idx, row_mapping[state_idx, beam_idx, row_idx]] = True

    pair_flat_full = torch.arange(num_contexts, device=device).view(-1, 1).expand(num_contexts, beam_width).reshape(-1)
    action_cols = torch.empty((num_contexts, beam_width, action_count), dtype=torch.long, device=device)
    action_valid = torch.empty((num_contexts, beam_width, action_count), dtype=torch.bool, device=device)
    action_lb = torch.empty((num_contexts, beam_width, action_count), dtype=torch.float32, device=device)
    rerank_score = torch.empty((num_contexts, pool_size), dtype=torch.float32, device=device)

    for _ in range(step_limit):
        valid_parent = active & (depth < search_row_count.view(-1, 1))
        if not bool(valid_parent.any()):
            break

        current_rows = row_order.gather(1, depth.clamp(max=max(max_n1 - 1, 0)))
        gather_rows = current_rows.unsqueeze(-1).expand(-1, -1, candidate_cap)
        action_cols_real = candidate_cols.gather(1, gather_rows)
        action_cols[:, :, :candidate_cap] = action_cols_real
        action_cols[:, :, candidate_cap] = max_n2

        action_immediate_start = runner._profile_start()
        action_immediate = _compute_gpu_refine_immediate_costs(
            runner=runner,
            match_cols=row_mapping,
            current_rows=current_rows,
            action_cols=action_cols,
            valid_parent=valid_parent,
            node_cost_batch=node_cost_batch,
            ged_adj1_batch=ged_adj1_batch,
            ged_adj2_batch=ged_adj2_batch,
            pair_flat=pair_flat_full,
            anchor_mask_batch=anchor_mask if anchor_enable else None,
            anchor_edge_cost_batch=anchor_edge_cost,
            anchor_delete_cost_batch=anchor_delete_cost,
        )
        runner._profile_stop(action_immediate_start, 'search.action_cost')

        action_valid.zero_()
        real_valid = action_cols_real >= 0
        if max_n2 > 0:
            safe_real_cols = action_cols_real.clamp(min=0, max=max_n2 - 1)
            real_valid = real_valid & (~used_cols.gather(2, safe_real_cols))
            real_valid = real_valid & (safe_real_cols < n2.view(-1, 1, 1))
        action_valid[:, :, :candidate_cap] = real_valid & valid_parent.unsqueeze(-1)
        action_valid[:, :, candidate_cap] = valid_parent

        child_g = current_cost.unsqueeze(-1) + action_immediate.masked_fill(~action_valid, float('inf'))

        lower_bound_start = runner._profile_start()
        if dense_lb_enable:
            flat_lb = _compute_gpu_refine_row_col_lb(
                runner=runner,
                pair_idx=pair_flat_full,
                row_maps=row_mapping.reshape(num_contexts * beam_width, max_n1),
                used_cols=used_cols.reshape(num_contexts * beam_width, max_n2),
                current_rows=current_rows.reshape(-1),
                action_cols=action_cols.reshape(num_contexts * beam_width, action_count),
                child_g=child_g.reshape(num_contexts * beam_width, action_count),
                node_cost_batch=node_cost_batch,
                row_mask=row_mask,
                col_mask=col_mask,
                max_n2=max_n2,
            )
            action_lb = flat_lb.view(num_contexts, beam_width, action_count).masked_fill(~action_valid, float('inf'))
        else:
            flat_pair_idx, flat_beam_idx = torch.nonzero(valid_parent, as_tuple=True)
            if flat_pair_idx.numel() == 0:
                break
            flat_lb = _compute_gpu_refine_row_col_lb(
                runner=runner,
                pair_idx=flat_pair_idx,
                row_maps=row_mapping[flat_pair_idx, flat_beam_idx],
                used_cols=used_cols[flat_pair_idx, flat_beam_idx],
                current_rows=current_rows[flat_pair_idx, flat_beam_idx],
                action_cols=action_cols[flat_pair_idx, flat_beam_idx],
                child_g=child_g[flat_pair_idx, flat_beam_idx],
                node_cost_batch=node_cost_batch,
                row_mask=row_mask,
                col_mask=col_mask,
                max_n2=max_n2,
            )
            flat_action_valid = action_valid[flat_pair_idx, flat_beam_idx]
            action_lb.fill_(float('inf'))
            action_lb[flat_pair_idx, flat_beam_idx] = flat_lb.masked_fill(~flat_action_valid, float('inf'))
        runner._profile_stop(lower_bound_start, 'search.lower_bound')

        action_score = action_lb.masked_fill(~action_valid, float('inf'))
        ub_prune_mask = action_valid & (action_lb >= best_ub.view(-1, 1, 1))
        if ub_prune_stats_enable:
            ub_pre_valid_total += action_valid.sum()
            ub_pruned_total += ub_prune_mask.sum()
        action_score = action_score.masked_fill(ub_prune_mask, float('inf'))
        if action_lb_slack is not None and action_lb_slack >= 0.0:
            parent_best_score = action_score.min(dim=2, keepdim=True).values
            slack_keep = action_score <= (parent_best_score + float(action_lb_slack))
            action_score = action_score.masked_fill(~slack_keep, float('inf'))

        top_score, top_action_idx = torch.topk(action_score, k=top_r, dim=2, largest=False)
        top_action_cols = action_cols.gather(2, top_action_idx)
        top_action_g = child_g.gather(2, top_action_idx)
        top_action_lb = action_lb.gather(2, top_action_idx)

        flat_cheap = top_score.reshape(num_contexts, -1)
        pool_score, pool_idx = torch.topk(flat_cheap, k=pool_size, dim=1, largest=False)
        pool_active = torch.isfinite(pool_score)
        pool_parent = torch.div(pool_idx, top_r, rounding_mode='floor')
        pool_cols = top_action_cols.reshape(num_contexts, -1).gather(1, pool_idx)
        pool_g = top_action_g.reshape(num_contexts, -1).gather(1, pool_idx)
        pool_lb = top_action_lb.reshape(num_contexts, -1).gather(1, pool_idx)
        pool_rows = current_rows.gather(1, pool_parent)

        pool_maps = row_mapping.gather(1, pool_parent.unsqueeze(-1).expand(-1, -1, max_n1)).clone()
        pool_real_cols = torch.where(pool_cols == max_n2, torch.full_like(pool_cols, -1), pool_cols)
        valid_pool_materialize = pool_active & (pool_rows >= 0) & (pool_rows < max_n1)
        safe_pool_rows = pool_rows.clamp(min=0, max=max(max_n1 - 1, 0)).unsqueeze(-1)
        pool_old_val = pool_maps.gather(2, safe_pool_rows)
        pool_src_val = torch.where(valid_pool_materialize.unsqueeze(-1), pool_real_cols.unsqueeze(-1), pool_old_val)
        pool_maps.scatter_(2, safe_pool_rows, pool_src_val)
        pool_used_cols = used_cols.gather(1, pool_parent.unsqueeze(-1).expand(-1, -1, max_n2)).clone()
        if max_n2 > 0:
            pool_real_assign = pool_active & (pool_cols >= 0) & (pool_cols < max_n2)
            safe_pool_cols = pool_cols.clamp(min=0, max=max_n2 - 1).unsqueeze(-1)
            pool_old_used = pool_used_cols.gather(2, safe_pool_cols)
            pool_src_used = pool_old_used | pool_real_assign.unsqueeze(-1)
            pool_used_cols.scatter_(2, safe_pool_cols, pool_src_used)

        flat_pool_pair, flat_pool_slot = torch.nonzero(pool_active, as_tuple=True)
        if flat_pool_pair.numel() == 0:
            break
        flat_pool_maps = pool_maps[flat_pool_pair, flat_pool_slot]
        if completion_mode == 'diffusion_score':
            completion_score = score_batch.index_select(0, flat_pool_pair)
        elif completion_mode == 'ged_aware':
            completion_score = _build_ged_aware_completion_score(
                runner=runner,
                row_maps=flat_pool_maps,
                pair_idx=flat_pool_pair,
                node_cost_batch=node_cost_batch,
                ged_adj1_batch=ged_adj1_batch,
                ged_adj2_batch=ged_adj2_batch,
                n2=n2,
                max_n2=max_n2,
            )
        else:
            raise RuntimeError(f'Unsupported completion mode for {variant_name}: {completion_mode!r}')

        completion_start = runner._profile_start()
        completed_pool_maps = _greedy_complete_row_maps_from_score(
            runner=runner,
            row_maps=flat_pool_maps,
            pair_idx=flat_pool_pair,
            completion_score=completion_score,
            n1=n1,
            n2=n2,
            max_n2=max_n2,
            row_ids=row_ids,
            col_ids=col_ids,
        )
        runner._profile_stop(completion_start, 'search.greedy_completion')

        exact_score_start = runner._profile_start()
        pool_exact_cost = _score_compact_row_maps_exact(
            runner=runner,
            pair_idx_all=flat_pool_pair,
            row_maps_all=completed_pool_maps,
            n1=n1,
            n2=n2,
            ged_adj2_batch=ged_adj2_batch,
            label1_batch=label1_batch,
            label2_batch=label2_batch,
            left_upper_batch=left_upper_batch,
            left_edge_count_batch=left_edge_count_batch,
            right_edge_count_batch=right_edge_count_batch,
        )
        runner._profile_stop(exact_score_start, 'search.exact_score')

        rerank_score.fill_(float('inf'))
        rerank_score[flat_pool_pair, flat_pool_slot] = pool_exact_cost

        update_best_start = runner._profile_start()
        dense_pool_update = (
            fast_exact_update_enable
            and os.environ.get('PARTIALDIFF_DENSE_POOL_UPDATE', '0') == '1'
            and complete_counts is None
            and equal_best_counts is None
            and equal_best_diff_counts is None
            and better_than_best_counts is None
        )
        if dense_pool_update:
            pool_best_cost, pool_best_slot = torch.min(rerank_score, dim=1)
            improved = pool_best_cost < best_ub
            if bool(improved.any()):
                improved_idx = torch.nonzero(improved, as_tuple=False).flatten()
                best_ub[improved_idx] = pool_best_cost.index_select(0, improved_idx)
                best_slots = pool_best_slot.index_select(0, improved_idx)
                dense_pool_maps = torch.full(
                    (num_contexts, pool_size, max_n1),
                    -3,
                    dtype=completed_pool_maps.dtype,
                    device=device,
                )
                dense_pool_maps[flat_pool_pair, flat_pool_slot] = completed_pool_maps
                best_row_maps[improved_idx] = dense_pool_maps[improved_idx, best_slots]
        else:
            best_ub, best_row_maps = runner._update_best_from_compact_row_maps(
                pair_idx_all=flat_pool_pair,
                row_maps_all=completed_pool_maps,
                best_ub=best_ub,
                best_row_maps=best_row_maps,
                n1=n1,
                n2=n2,
                x1_batch=x1_batch,
                x2_batch=x2_batch,
                ged_adj1_batch=ged_adj1_batch,
                ged_adj2_batch=ged_adj2_batch,
                chunk_size=final_chunk_size,
                baseline_row_maps=baseline_row_maps,
                complete_counts=complete_counts,
                equal_best_counts=equal_best_counts,
                equal_best_diff_counts=equal_best_diff_counts,
                better_than_best_counts=better_than_best_counts,
                label1_batch=label1_batch,
                label2_batch=label2_batch,
                left_upper_batch=left_upper_batch,
                left_edge_count_batch=left_edge_count_batch,
                right_edge_count_batch=right_edge_count_batch,
                ged_values_all=pool_exact_cost,
            )
        runner._profile_stop(update_best_start, 'search.update_best')

        if lb_tiebreak_weight != 0.0:
            rerank_score.add_(lb_tiebreak_weight * pool_lb.masked_fill(~pool_active, 0.0))

        beam_update_start = runner._profile_start()
        keep_score, keep_pool_idx = torch.topk(rerank_score, k=beam_width, dim=1, largest=False)
        new_active = torch.isfinite(keep_score)
        keep_parent = pool_parent.gather(1, keep_pool_idx)
        next_row_mapping = pool_maps.gather(1, keep_pool_idx.unsqueeze(-1).expand(-1, -1, max_n1))
        next_used_cols = pool_used_cols.gather(1, keep_pool_idx.unsqueeze(-1).expand(-1, -1, max_n2))
        next_current_cost = pool_g.gather(1, keep_pool_idx)
        next_depth = depth.gather(1, keep_parent) + new_active.long()

        eval_counts += action_valid.sum(dim=(1, 2)).long()
        expansion_counts += valid_parent.sum(dim=1).long()
        step_counts += valid_parent.any(dim=1).long()
        max_depth_reached = torch.maximum(max_depth_reached, next_depth.masked_fill(~new_active, 0).max(dim=1).values)
        row_mapping = next_row_mapping
        row_mapping.masked_fill_(~new_active.unsqueeze(-1), -3)
        current_cost = next_current_cost
        current_cost.masked_fill_(~new_active, float('inf'))
        depth = next_depth
        depth.masked_fill_(~new_active, 0)
        used_cols = next_used_cols
        used_cols.masked_fill_(~new_active.unsqueeze(-1), False)
        active = new_active
        max_frontier = torch.maximum(max_frontier, active.sum(dim=1).long())
        runner._profile_stop(beam_update_start, 'search.state_update')

    best_ub_cpu = best_ub.detach().cpu()
    if refine_diagnostics_enable:
        complete_counts_cpu = complete_counts.detach().cpu()
        equal_best_counts_cpu = equal_best_counts.detach().cpu()
        equal_best_diff_counts_cpu = equal_best_diff_counts.detach().cpu()
        better_than_best_counts_cpu = better_than_best_counts.detach().cpu()
    else:
        complete_counts_cpu = None
        equal_best_counts_cpu = None
        equal_best_diff_counts_cpu = None
        better_than_best_counts_cpu = None
    eval_counts_cpu = eval_counts.detach().cpu()
    expansion_counts_cpu = expansion_counts.detach().cpu()
    step_counts_cpu = step_counts.detach().cpu()
    max_frontier_cpu = max_frontier.detach().cpu()
    max_depth_reached_cpu = max_depth_reached.detach().cpu()
    best_row_maps_cpu = best_row_maps.detach()
    anchor_targets_cpu = anchor_targets.detach().cpu()
    anchor_counts_cpu = anchor_mask.sum(dim=1).detach().cpu()
    search_row_count_cpu = search_row_count.detach().cpu()
    if ub_prune_stats_enable:
        runner.profile_metadata['search.gpu_refine_ub_pre_valid_total'] = int(ub_pre_valid_total.detach().cpu().item())
        runner.profile_metadata['search.gpu_refine_ub_pruned_total'] = int(ub_pruned_total.detach().cpu().item())
        pre_valid_total = int(runner.profile_metadata['search.gpu_refine_ub_pre_valid_total'])
        if pre_valid_total > 0:
            runner.profile_metadata['search.gpu_refine_ub_pruned_ratio_ppm'] = int(round(
                1_000_000.0 * int(runner.profile_metadata['search.gpu_refine_ub_pruned_total']) / pre_valid_total
            ))

    for idx, context in enumerate(contexts):
        context.best_ub = float(best_ub_cpu[idx])
        context.best_row_mapping_t = best_row_maps_cpu[idx, :context.n1].clone()
        context.best_matching_tensor = None
        context.anchor_count = int(anchor_counts_cpu[idx])
        context.search_row_count = int(search_row_count_cpu[idx])
        context.anchor_target_t = anchor_targets[idx, :context.n1].clone()
        context.anchor_target_for_q = tuple(int(x) for x in anchor_targets_cpu[idx, :context.n1].tolist())
        context.states_evaluated = int(eval_counts_cpu[idx])
        context.expansions = int(expansion_counts_cpu[idx])
        context.max_queue_size = int(max_frontier_cpu[idx])
        context.steps = int(step_counts_cpu[idx])
        context.complete_states_evaluated = 0 if complete_counts_cpu is None else int(complete_counts_cpu[idx])
        context.complete_states_equal_best = 0 if equal_best_counts_cpu is None else int(equal_best_counts_cpu[idx])
        context.complete_states_equal_best_diff_mapping = 0 if equal_best_diff_counts_cpu is None else int(equal_best_diff_counts_cpu[idx])
        context.complete_states_better_than_best = 0 if better_than_best_counts_cpu is None else int(better_than_best_counts_cpu[idx])
        context.max_depth_reached = int(max_depth_reached_cpu[idx])


@torch.inference_mode()
def run_gpu_refine(runner, contexts):
    cfg = _resolve_gpu_refine_runtime_config_base(runner)
    completion_mode = str(cfg.get('completion_mode', 'diffusion_score'))
    if completion_mode not in {'diffusion_score', 'ged_aware'}:
        raise RuntimeError(
            "--gpu-completion-mode/PARTIALDIFF_GPU_REFINE_COMPLETION_MODE must be 'diffusion_score' or 'ged_aware', "
            f"got {completion_mode!r}."
        )
    return _run_gpu_refine_search(
        runner=runner,
        contexts=contexts,
        variant_name='gpu_refine',
        completion_mode=completion_mode,
    )


@dataclass(slots=True)
# 单个搜索状态保存当前 partial mapping 与当前搜索排序信息。
class SearchState:
    row_mapping: torch.Tensor
    next_pos: int
    current_cost: float
    lower_bound: float = 0.0
    strict_completed_mapping: Optional[torch.Tensor] = None
    strict_completion_ged: float = float('inf')


@dataclass(slots=True)
# 单个 graph pair / candidate 对应的一份静态上下文与搜索统计。
# 搜索过程中会反复用到的张量都提前缓存到 context 中,
# 避免每个 step 再回到 Python 里重新组装。
class SearchContext:
    pair_index: int
    candidate_index: int
    graph_1: int
    graph_2: int
    graph_1_gid: int
    graph_2_gid: int
    gt_ged: float
    baseline_ged: float
    n1: int
    n2: int
    search_row_count: int
    anchor_count: int
    row_order_tensor: torch.Tensor
    score_matrix: torch.Tensor
    prob_matrix: torch.Tensor
    matched_prob_t: torch.Tensor
    x1: torch.Tensor
    x2: torch.Tensor
    ged_adj1: torch.Tensor
    ged_adj2: torch.Tensor
    candidate_cols_tensor: torch.Tensor
    search_budget: int
    best_ub: float
    best_matching_pairs: List[List[int]]
    query_labels: Tuple[int, ...] = field(default_factory=tuple)
    target_labels: Tuple[int, ...] = field(default_factory=tuple)
    query_adj_bits: Tuple[int, ...] = field(default_factory=tuple)
    target_adj_bits: Tuple[int, ...] = field(default_factory=tuple)
    query_edge_set: frozenset = field(default_factory=frozenset)
    target_edge_set: frozenset = field(default_factory=frozenset)
    query_degrees: Tuple[int, ...] = field(default_factory=tuple)
    target_degrees: Tuple[int, ...] = field(default_factory=tuple)
    anchor_target_for_q: Tuple[int, ...] = field(default_factory=tuple)
    strict_search_rows: Tuple[int, ...] = field(default_factory=tuple)
    query_labels_t: Optional[torch.Tensor] = None
    target_labels_t: Optional[torch.Tensor] = None
    query_adj_matrix_t: Optional[torch.Tensor] = None
    target_adj_matrix_t: Optional[torch.Tensor] = None
    query_degrees_t: Optional[torch.Tensor] = None
    target_degrees_t: Optional[torch.Tensor] = None
    anchor_target_t: Optional[torch.Tensor] = None
    strict_search_rows_t: Optional[torch.Tensor] = None
    baseline_row_mapping_t: Optional[torch.Tensor] = None
    best_row_mapping_t: Optional[torch.Tensor] = None
    best_matching_tensor: Optional[torch.Tensor] = None
    selection_metrics: Dict = field(default_factory=dict)
    best_candidate_metrics: Optional[Dict] = None
    expansions: int = 0
    states_evaluated: int = 0
    max_queue_size: int = 0
    steps: int = 0
    strict_generated_children: int = 0
    strict_frontier_before_prune: int = 0
    strict_frontier_after_prune: int = 0
    strict_pruned_children: int = 0
    complete_states_evaluated: int = 0
    complete_states_equal_best: int = 0
    complete_states_equal_best_diff_mapping: int = 0
    complete_states_better_than_best: int = 0
    max_depth_reached: int = 0
    beam: List[SearchState] = field(default_factory=list)


class InternalAppBmaoSearchRunner:
    # Minimal internal refine runner kept for gpu_refine.
    def __init__(self, trainer, backend, branch_topk, unmatched_cost, beam_width, beam_steps, score_source, assignment_backend="auto", bestfirst_topm=32):
        self.trainer = trainer
        self.backend = str(backend)
        self.branch_topk = max(1, int(branch_topk))
        self.unmatched_cost = float(unmatched_cost)
        self.beam_width = max(1, int(beam_width))
        self.score_source = str(score_source)
        self.assignment_backend = str(assignment_backend)
        self.profile_enabled = bool(getattr(self.trainer.args, 'app_bmao_profile_enable', False))
        self.last_profile_report = None
        self._reset_profile_state()
        self._graph_cache = {}
        self._graph_tensor_cache = {}
        self._row_order_cache = {}
        self._anchor_zero_cache = {}
        self._empty_candidate_cols_cache = {}
        self._profile_cuda_sync_enabled = True

    def _initialize_run_profile(self, postprocess_mode, candidate_budget, beam_score):
        if self.backend not in {'gpu_refine'}:
            raise ValueError(f'Unsupported internal App-BMao backend: {self.backend}')

        self._reset_profile_state()
        self.profile_metadata.update({
            'postprocess_mode': str(postprocess_mode),
            'candidate_budget': int(candidate_budget),
            'beam_score': str(beam_score),
            'assignment_backend': str(self.assignment_backend),
        })

    @staticmethod
    def _move_tensor(value, device):
        if isinstance(value, torch.Tensor):
            return value.to(device=device)
        return value

    def _tensor_on_device(self, value, device):
        if not isinstance(value, torch.Tensor):
            return True
        return self._device_cache_key(value.device) == self._device_cache_key(device)

    def _contexts_on_device(self, contexts, device):
        tensor_attrs = (
            'row_order_tensor',
            'score_matrix',
            'matched_prob_t',
            'x1',
            'x2',
            'ged_adj1',
            'ged_adj2',
        )
        for context in contexts:
            for attr in tensor_attrs:
                if not self._tensor_on_device(getattr(context, attr), device):
                    return False
            for state in context.beam:
                if not self._tensor_on_device(state.row_mapping, device):
                    return False
        return True

    def _move_contexts_to_device(self, contexts, device):
        if self._contexts_on_device(contexts, device):
            return contexts
        tensor_attrs = (
            'row_order_tensor',
            'score_matrix',
            'prob_matrix',
            'matched_prob_t',
            'x1',
            'x2',
            'ged_adj1',
            'ged_adj2',
            'candidate_cols_tensor',
            'query_labels_t',
            'target_labels_t',
            'query_adj_matrix_t',
            'target_adj_matrix_t',
            'query_degrees_t',
            'target_degrees_t',
            'anchor_target_t',
            'strict_search_rows_t',
            'baseline_row_mapping_t',
            'best_row_mapping_t',
            'best_matching_tensor',
        )
        for context in contexts:
            for attr in tensor_attrs:
                setattr(context, attr, self._move_tensor(getattr(context, attr), device))
            for state in context.beam:
                state.row_mapping = self._move_tensor(state.row_mapping, device)
                state.strict_completed_mapping = self._move_tensor(state.strict_completed_mapping, device)
        return contexts

    def prepare_contexts(self, candidate_pairs, postprocess_mode, candidate_budget, beam_score, context_device=None):
        self._initialize_run_profile(postprocess_mode, candidate_budget, beam_score)
        previous_profile_cuda_sync_enabled = self._profile_cuda_sync_enabled
        if context_device is not None and torch.device(context_device).type == 'cpu':
            self._profile_cuda_sync_enabled = False
        context_start = self._profile_start()
        # 先把每个 pair 的 diffusion candidate 打包成 context。
        # 这一步会完成:
        # 1. anchor 固定
        # 2. row 搜索顺序构造
        # 3. 每个 row 的静态 top-k 候选列预缓存
        try:
            contexts = self._build_contexts(
                candidate_pairs=candidate_pairs,
                postprocess_mode=postprocess_mode,
                candidate_budget=candidate_budget,
                beam_score=beam_score,
                context_device=context_device,
            )
        finally:
            self._profile_stop(context_start, 'run.context_build')
            self._profile_cuda_sync_enabled = previous_profile_cuda_sync_enabled
        return contexts

    def run_prepared_contexts(self, candidate_pairs, contexts, postprocess_mode, candidate_budget, beam_score):
        run_start = self._profile_start()
        if not contexts:
            self._profile_stop(run_start, 'run.total')
            self._finalize_profile_report(num_pairs=len(candidate_pairs), num_contexts=0)
            return []

        self._move_contexts_to_device(contexts, self.trainer.device)
        search_start = self._profile_start()
        run_gpu_refine(self, contexts)
        self._profile_stop(search_start, 'run.search')

        finalize_start = self._profile_start()
        # 把每个 pair 的最优结果与统计信息回收成 trainer 统一使用的结果字典。
        pair_results = self._finalize_pair_results(candidate_pairs, contexts, postprocess_mode, candidate_budget, beam_score)
        self._profile_stop(finalize_start, 'run.finalize_results')
        self._profile_stop(run_start, 'run.total')
        if self.profile_enabled:
            self.profile_totals['run.total'] = (
                float(self.profile_totals.get('run.context_build', 0.0))
                + float(self.profile_totals.get('run.search', 0.0))
                + float(self.profile_totals.get('run.finalize_results', 0.0))
            )
            self.profile_counts['run.total'] = max(1, int(self.profile_counts.get('run.total', 0)))
        self._finalize_profile_report(num_pairs=len(candidate_pairs), num_contexts=len(contexts))
        return pair_results

    def run(self, candidate_pairs, postprocess_mode, candidate_budget, beam_score):
        contexts = self.prepare_contexts(
            candidate_pairs=candidate_pairs,
            postprocess_mode=postprocess_mode,
            candidate_budget=candidate_budget,
            beam_score=beam_score,
        )
        return self.run_prepared_contexts(
            candidate_pairs=candidate_pairs,
            contexts=contexts,
            postprocess_mode=postprocess_mode,
            candidate_budget=candidate_budget,
            beam_score=beam_score,
        )

    # 从 diffusion candidate 中挑高置信 anchor, 直接写入初始 row mapping。
    # 这些 row 后续不再参与剩余搜索, 相当于先把一部分 mapping 固定下来缩小搜索空间。
    def _reset_profile_state(self):
        self.profile_totals = defaultdict(float)
        self.profile_counts = defaultdict(int)
        self.profile_chunk_records = []
        self.profile_metadata = {}
        self.profile_samples = defaultdict(list)

    def _profile_start(self):
        if not self.profile_enabled:
            return None
        if self._profile_cuda_sync_enabled and self.trainer.use_gpu:
            torch.cuda.synchronize(self.trainer.device)
        return time.perf_counter()

    def _profile_stop(self, start_time, key=None, count=1):
        if start_time is None:
            return 0.0
        if self._profile_cuda_sync_enabled and self.trainer.use_gpu:
            torch.cuda.synchronize(self.trainer.device)
        elapsed = time.perf_counter() - start_time
        if key is not None:
            self.profile_totals[key] += float(elapsed)
            self.profile_counts[key] += int(count)
        return float(elapsed)

    @staticmethod
    def _profile_percentile(values, pct):
        if not values:
            return None
        ordered = sorted(float(v) for v in values)
        if len(ordered) == 1:
            return ordered[0]
        pos = (len(ordered) - 1) * max(0.0, min(100.0, float(pct))) / 100.0
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            return ordered[lo]
        frac = pos - lo
        return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


    def _finalize_profile_report(self, num_pairs, num_contexts):
        if not self.profile_enabled:
            self.last_profile_report = None
            return
        totals = {key: float(self.profile_totals[key]) for key in sorted(self.profile_totals.keys())}
        counts = {key: int(self.profile_counts.get(key, 0)) for key in sorted(self.profile_totals.keys())}
        avg_ms = {key: (1000.0 * totals[key] / counts[key]) if counts[key] > 0 else None for key in totals}
        metadata = dict(self.profile_metadata)
        lap_call_samples = list(self.profile_samples.get('search.v5_lap_matrices_per_call', []))
        if lap_call_samples:
            metadata['search.v5_lap_matrices_per_call_samples'] = [int(v) for v in lap_call_samples]
            metadata['search.v5_lap_matrices_per_call_mean'] = float(sum(lap_call_samples) / len(lap_call_samples))
            metadata['search.v5_lap_matrices_per_call_p50'] = float(self._profile_percentile(lap_call_samples, 50.0))
            metadata['search.v5_lap_matrices_per_call_p95'] = float(self._profile_percentile(lap_call_samples, 95.0))
            metadata['search.v5_lap_matrices_per_call_max'] = int(max(lap_call_samples))
        parent_valid_samples = list(self.profile_samples.get('search.v5_valid_actions_per_parent', []))
        if parent_valid_samples:
            metadata['search.v5_valid_actions_per_parent_samples'] = [int(v) for v in parent_valid_samples]
            metadata['search.v5_valid_actions_per_parent_mean'] = float(sum(parent_valid_samples) / len(parent_valid_samples))
            metadata['search.v5_valid_actions_per_parent_p50'] = float(self._profile_percentile(parent_valid_samples, 50.0))
            metadata['search.v5_valid_actions_per_parent_p95'] = float(self._profile_percentile(parent_valid_samples, 95.0))
            metadata['search.v5_valid_actions_per_parent_max'] = int(max(parent_valid_samples))
        iter_valid_samples = list(self.profile_samples.get('search.v5_valid_actions_per_iter', []))
        if iter_valid_samples:
            metadata['search.v5_valid_actions_per_iter_samples'] = [int(v) for v in iter_valid_samples]
            metadata['search.v5_valid_actions_per_iter_mean'] = float(sum(iter_valid_samples) / len(iter_valid_samples))
            metadata['search.v5_valid_actions_per_iter_p50'] = float(self._profile_percentile(iter_valid_samples, 50.0))
            metadata['search.v5_valid_actions_per_iter_p95'] = float(self._profile_percentile(iter_valid_samples, 95.0))
            metadata['search.v5_valid_actions_per_iter_max'] = int(max(iter_valid_samples))
        lap_matrix_count = int(metadata.get('search.v5_lap_matrix_size_count', 0) or 0)
        if lap_matrix_count > 0:
            metadata['search.v5_lap_matrix_size_mean'] = float(metadata.get('search.v5_lap_matrix_size_weighted_sum', 0.0) / lap_matrix_count)
        completion_total = int(metadata.get('search.v5_completion_candidates_total', 0) or 0)
        completion_unique = int(metadata.get('search.v5_completion_unique_total', 0) or 0)
        if completion_total > 0:
            metadata['search.v5_completion_unique_ratio_ppm'] = int(round(1_000_000.0 * completion_unique / completion_total))
        pre_ub_total = int(metadata.get('search.v5_pre_ub_selected_total', 0) or 0)
        post_ub_total = int(metadata.get('search.v5_post_ub_selected_total', 0) or 0)
        if pre_ub_total >= post_ub_total:
            metadata['search.v5_pruned_by_ub_total'] = int(pre_ub_total - post_ub_total)
        metadata.pop('search.v5_lap_matrix_size_weighted_sum', None)
        metadata.pop('search.v5_lap_matrix_size_count', None)
        self.last_profile_report = {
            'enabled': True,
            'backend': self.backend,
            'num_pairs': int(num_pairs),
            'num_contexts': int(num_contexts),
            'stage_totals_s': totals,
            'stage_counts': counts,
            'stage_avg_ms': avg_ms,
            'chunk_records': list(self.profile_chunk_records),
            'metadata': metadata,
        }






    # 构造每个 candidate 的静态搜索上下文。
    # 关键工作包括:
    # - 读取图特征 / GED 邻接矩阵
    # - 选择 diffusion 输出的 final_scores 作为打分矩阵
    # - 构建包含 anchor 的初始搜索状态

    def _graph_struct(self, graph_index):
        graph_index = int(graph_index)
        cached = self._graph_cache.get(graph_index)
        if cached is not None:
            return cached
        graph = self.trainer.graphs[graph_index]
        n = int(graph['n'])
        graph_gid = int(graph.get('gid', graph_index))
        labels = tuple(int(x) for x in self.trainer._resolve_graph_labels_for_app_bmao(graph, graph_gid))
        adj_bits = [0] * n
        degrees = [0] * n
        edge_set = set()
        for src, dst in graph.get('graph', []):
            src = int(src)
            dst = int(dst)
            if src == dst:
                continue
            adj_bits[src] |= (1 << dst)
            adj_bits[dst] |= (1 << src)
            degrees[src] += 1
            degrees[dst] += 1
            if src < dst:
                edge_set.add((src, dst))
            else:
                edge_set.add((dst, src))
        cached = {
            'n': n,
            'labels': labels,
            'adj_bits': tuple(adj_bits),
            'degrees': tuple(degrees),
            'edge_set': frozenset(edge_set),
            'directed_edge_count': sum(degrees),
        }
        self._graph_cache[graph_index] = cached
        return cached

    def _graph_struct_from_dense(self, x, ged_adj):
        x = x.detach().cpu()
        ged_adj = ged_adj.detach().cpu()
        n = int(x.shape[0])
        labels = tuple(int(v) for v in self.trainer._node_labels_dense(x).detach().cpu().tolist())
        adj_bits = [0] * n
        degrees = [0] * n
        edge_set = set()
        edge_indices = torch.nonzero(torch.triu(ged_adj > 0.5, diagonal=1), as_tuple=False)
        for src_t, dst_t in edge_indices:
            src = int(src_t.item())
            dst = int(dst_t.item())
            adj_bits[src] |= (1 << dst)
            adj_bits[dst] |= (1 << src)
            degrees[src] += 1
            degrees[dst] += 1
            edge_set.add((src, dst))
        return {
            'n': n,
            'labels': labels,
            'adj_bits': tuple(adj_bits),
            'degrees': tuple(degrees),
            'edge_set': frozenset(edge_set),
            'directed_edge_count': sum(degrees),
        }

    def _device_cache_key(self, device):
        device = torch.device(device)
        if device.type == 'cuda':
            return (device.type, torch.cuda.current_device() if device.index is None else int(device.index))
        return (device.type, device.index)

    def _graph_runtime_tensors(self, graph_index, n, device):
        graph_index = int(graph_index)
        n = int(n)
        device = torch.device(device)
        cache_key = (graph_index, n, self._device_cache_key(device))
        cached = self._graph_tensor_cache.get(cache_key)
        if cached is not None:
            return cached

        graph_struct = self._graph_struct(graph_index)
        x = self.trainer.features[graph_index][:n].to(device=device, dtype=torch.float)
        ged_adj = self.trainer.ged_adj[graph_index][:n, :n].to(device=device, dtype=torch.float)
        labels_t = torch.tensor(graph_struct['labels'][:n], dtype=torch.long, device=device)
        adj_bool_t = ged_adj > 0.5
        degrees_t = torch.tensor(graph_struct['degrees'][:n], dtype=torch.long, device=device)
        cached = {
            'graph': graph_struct,
            'x': x,
            'ged_adj': ged_adj,
            'labels_t': labels_t,
            'adj_bool_t': adj_bool_t,
            'degrees_t': degrees_t,
        }
        self._graph_tensor_cache[cache_key] = cached
        return cached

    def _anchor_zero_initialization(self, n1, device):
        n1 = int(n1)
        device = torch.device(device)
        cache_key = (n1, self._device_cache_key(device))
        cached = self._anchor_zero_cache.get(cache_key)
        if cached is not None:
            return cached
        anchor_target_for_q = tuple([-2] * n1)
        row_mapping_t = torch.full((n1,), -2, dtype=torch.long, device=device)
        anchor_target_t = row_mapping_t
        cached = (anchor_target_for_q, row_mapping_t, anchor_target_t)
        self._anchor_zero_cache[cache_key] = cached
        return cached

    def _empty_candidate_cols(self, device):
        device = torch.device(device)
        cache_key = self._device_cache_key(device)
        cached = self._empty_candidate_cols_cache.get(cache_key)
        if cached is None:
            cached = torch.empty((0, 0), dtype=torch.long, device=device)
            self._empty_candidate_cols_cache[cache_key] = cached
        return cached

    def _row_mapping_from_dense_matching_fast(self, final_matching, n1, n2, device):
        n1 = int(n1)
        n2 = int(n2)
        row_map = torch.full((n1,), -1, dtype=torch.long)
        if n1 > 0 and n2 > 0:
            matching = final_matching[:n1, :n2] > 0.5
            has_match = matching.any(dim=1)
            matched_cols = matching.to(dtype=torch.long).argmax(dim=1)
            row_map = torch.where(has_match, matched_cols, row_map.to(device=matching.device))
        return row_map.to(device=device)

    def _row_order_for_context(self, q_graph, g_graph, graph_1, graph_2, n1, device, anchored_rows):
        device = torch.device(device)
        base_key = (int(graph_1), int(graph_2), int(n1), self._device_cache_key(device))
        cached = self._row_order_cache.get(base_key)
        if cached is None:
            order_tuple = self._compute_bmao_mapping_order(q_graph, g_graph)
            row_order_tensor = torch.tensor(order_tuple, dtype=torch.long, device=device)
            cached = (order_tuple, row_order_tensor)
            self._row_order_cache[base_key] = cached
        order_tuple, row_order_tensor = cached
        if not anchored_rows:
            return row_order_tensor, order_tuple

        anchored_set = {int(row) for row in anchored_rows}
        filtered_tuple = tuple(int(row) for row in order_tuple if int(row) not in anchored_set)
        filtered_tensor = torch.tensor(filtered_tuple, dtype=torch.long, device=device)
        return filtered_tensor, filtered_tuple


    def _compute_bmao_mapping_order(self, query_graph, target_graph):
        q_n = int(query_graph['n'])
        g_n = int(target_graph['n'])
        g_m = int(len(target_graph['edge_set']))
        target_label_freq = {}
        for label in target_graph['labels']:
            target_label_freq[label] = target_label_freq.get(label, 0) + 1
        vertex_rarity = [1.0 - target_label_freq.get(query_graph['labels'][u], 0) / float(max(g_n, 1)) for u in range(q_n)]
        # Our dense graphs do not expose edge labels here, so we use the unlabeled-edge BMao analogue:
        # every incident edge contributes the same rarity term derived from the target edge count.
        edge_rarity = 1.0 - (1.0 / float(g_m)) if g_m > 0 else 0.0
        root = 0
        root_weight = float('-inf')
        for u in range(q_n):
            weight = vertex_rarity[u] + int(query_graph['degrees'][u]) * edge_rarity
            if weight > root_weight:
                root = u
                root_weight = weight

        scores = [0.0] * q_n
        scores[root] = root_weight
        picked = [False] * q_n
        order = []
        for _ in range(q_n):
            best_u = -1
            best_score = float('-inf')
            for u in range(q_n):
                if picked[u]:
                    continue
                score = scores[u]
                if score > best_score or (score == best_score and (best_u < 0 or u < best_u)):
                    best_u = u
                    best_score = score
            if best_u < 0:
                break
            picked[best_u] = True
            order.append(best_u)
            nbr_bits = query_graph['adj_bits'][best_u]
            for v in range(q_n):
                if picked[v] or (((nbr_bits >> v) & 1) == 0):
                    continue
                if scores[v] <= 1e-12:
                    scores[v] += vertex_rarity[v]
                scores[v] += edge_rarity
        return tuple(order)

    def _build_contexts(self, candidate_pairs, postprocess_mode, candidate_budget, beam_score, context_device=None):
            contexts = []
            device = self.trainer.device if context_device is None else torch.device(context_device)
            anchor_ratio = float(getattr(self.trainer.args, 'app_bmao_anchor_ratio', 0.6))
            build_start = self._profile_start()
            for pair_index, candidate_pair in enumerate(candidate_pairs):
                pair_start = self._profile_start()
                select_start = self._profile_start()
                candidate_specs, baseline_ged, selection_metadata = self.trainer._prepare_app_bmao_refine_candidates(
                    candidate_pair=candidate_pair,
                    mode=postprocess_mode,
                    candidate_budget=candidate_budget,
                    beam_score=beam_score,
                )
                self._profile_stop(select_start, 'context.select_candidates')
                tensor_start = self._profile_start()
                graph_1 = int(candidate_pair['pair'][0].item())
                graph_2 = int(candidate_pair['pair'][1].item())
                graph_1_gid = int(candidate_pair['pair_gid'][0].item())
                graph_2_gid = int(candidate_pair['pair_gid'][1].item())
                gt_ged = float(candidate_pair['gt_ged'])
                n1 = int(candidate_pair['n1'])
                n2 = int(candidate_pair['n2'])
                has_dense_graph_payload = all(key in candidate_pair for key in ('x1', 'x2', 'ged_adj1', 'ged_adj2'))
                if has_dense_graph_payload:
                    x1 = candidate_pair['x1'][:n1].to(device=device, dtype=torch.float)
                    x2 = candidate_pair['x2'][:n2].to(device=device, dtype=torch.float)
                    ged_adj1 = candidate_pair['ged_adj1'][:n1, :n1].to(device=device, dtype=torch.float)
                    ged_adj2 = candidate_pair['ged_adj2'][:n2, :n2].to(device=device, dtype=torch.float)
                    q_graph = self._graph_struct_from_dense(candidate_pair['x1'][:n1], candidate_pair['ged_adj1'][:n1, :n1])
                    g_graph = self._graph_struct_from_dense(candidate_pair['x2'][:n2], candidate_pair['ged_adj2'][:n2, :n2])
                    query_labels_t = torch.tensor(q_graph['labels'], dtype=torch.long, device=device)
                    target_labels_t = torch.tensor(g_graph['labels'], dtype=torch.long, device=device)
                    query_adj_matrix_t = (ged_adj1 > 0.5).to(dtype=torch.bool)
                    target_adj_matrix_t = (ged_adj2 > 0.5).to(dtype=torch.bool)
                    query_degrees_t = torch.tensor(q_graph['degrees'], dtype=torch.long, device=device)
                    target_degrees_t = torch.tensor(g_graph['degrees'], dtype=torch.long, device=device)
                else:
                    q_runtime = self._graph_runtime_tensors(graph_1, n1, device)
                    g_runtime = self._graph_runtime_tensors(graph_2, n2, device)
                    q_graph = q_runtime['graph']
                    g_graph = g_runtime['graph']
                    x1 = q_runtime['x']
                    x2 = g_runtime['x']
                    ged_adj1 = q_runtime['ged_adj']
                    ged_adj2 = g_runtime['ged_adj']
                    query_labels_t = q_runtime['labels_t']
                    target_labels_t = g_runtime['labels_t']
                    query_adj_matrix_t = q_runtime['adj_bool_t']
                    target_adj_matrix_t = g_runtime['adj_bool_t']
                    query_degrees_t = q_runtime['degrees_t']
                    target_degrees_t = g_runtime['degrees_t']
                self._profile_stop(tensor_start, 'context.graph_tensor_materialize')
                for candidate_spec in candidate_specs:
                    candidate_start = self._profile_start()
                    candidate_index = int(candidate_spec['candidate_index'])
                    final_matching = candidate_spec['final_matching']
                    score_matrix = candidate_spec['final_scores'].to(device=device, dtype=torch.float)
                    matched_pairs = []
                    initial_matching_tensor = None
                    baseline_row_mapping_t = self._row_mapping_from_dense_matching_fast(final_matching, n1, n2, device)
                    if anchor_ratio > 0.0:
                        matched_prob_t = _matched_probabilities_from_dense_matching(
                            final_matching=final_matching,
                            final_probabilities=candidate_spec['final_probabilities'],
                            device=device,
                        ).to(device=device, dtype=torch.float)
                    else:
                        matched_prob_t = torch.full((n1,), float('-inf'), dtype=torch.float, device=device)
                    prob_matrix = score_matrix
                    anchor_start = self._profile_start()
                    anchor_target_for_q, initial_row_mapping_t, anchor_target_t = self._anchor_zero_initialization(n1, device)
                    anchor_count = 0
                    anchored_rows = ()
                    self._profile_stop(anchor_start, 'context.anchor_prepare')
                    row_order_start = self._profile_start()
                    if has_dense_graph_payload:
                        mo = self._compute_bmao_mapping_order(q_graph, g_graph)
                        row_order_tensor = torch.tensor(mo, dtype=torch.long, device=device)
                        search_rows = mo
                    else:
                        row_order_tensor, search_rows = self._row_order_for_context(
                            q_graph=q_graph,
                            g_graph=g_graph,
                            graph_1=graph_1,
                            graph_2=graph_2,
                            n1=n1,
                            device=device,
                            anchored_rows=(),
                        )
                    self._profile_stop(row_order_start, 'context.row_order_prepare')
                    empty_candidate_cols = self._empty_candidate_cols(device)
                    initial_state = SearchState(
                        row_mapping=initial_row_mapping_t,
                        next_pos=0,
                        current_cost=0.0,
                    )
                    context = SearchContext(
                        pair_index=pair_index,
                        candidate_index=int(candidate_index),
                        graph_1=graph_1,
                        graph_2=graph_2,
                        graph_1_gid=graph_1_gid,
                        graph_2_gid=graph_2_gid,
                        gt_ged=gt_ged,
                        baseline_ged=float(baseline_ged),
                        n1=n1,
                        n2=n2,
                        search_row_count=len(search_rows),
                        anchor_count=int(anchor_count),
                        row_order_tensor=row_order_tensor,
                        score_matrix=score_matrix,
                        prob_matrix=prob_matrix,
                        matched_prob_t=matched_prob_t,
                        x1=x1,
                        x2=x2,
                        ged_adj1=ged_adj1,
                        ged_adj2=ged_adj2,
                        candidate_cols_tensor=empty_candidate_cols,
                        search_budget=1,
                        best_ub=float(candidate_spec['candidate_ged']),
                        best_matching_pairs=matched_pairs,
                        query_labels=q_graph['labels'],
                        target_labels=g_graph['labels'],
                        query_adj_bits=q_graph['adj_bits'],
                        target_adj_bits=g_graph['adj_bits'],
                        query_edge_set=q_graph['edge_set'],
                        target_edge_set=g_graph['edge_set'],
                        query_degrees=q_graph['degrees'],
                        target_degrees=g_graph['degrees'],
                        anchor_target_for_q=anchor_target_for_q,
                        strict_search_rows=search_rows,
                        query_labels_t=query_labels_t,
                        target_labels_t=target_labels_t,
                        query_adj_matrix_t=query_adj_matrix_t,
                        target_adj_matrix_t=target_adj_matrix_t,
                        query_degrees_t=query_degrees_t,
                        target_degrees_t=target_degrees_t,
                        anchor_target_t=anchor_target_t,
                        strict_search_rows_t=None,
                        baseline_row_mapping_t=baseline_row_mapping_t,
                        best_row_mapping_t=None,
                        best_matching_tensor=initial_matching_tensor,
                        selection_metrics=selection_metadata.get(int(candidate_index), {}),
                        best_candidate_metrics=selection_metadata.get(int(candidate_index)),
                        beam=[initial_state],
                        max_queue_size=1,
                    )
                    initial_lb_start = self._profile_start()
                    initial_state.current_cost = 0.0
                    initial_state.lower_bound = initial_state.current_cost
                    self._profile_stop(initial_lb_start, 'context.initial_bound_setup')
                    self._profile_stop(candidate_start, 'context.candidate_total')
                    contexts.append(context)
                self._profile_stop(pair_start, 'context.pair_total')
            self._profile_stop(build_start, 'context.total')
            return contexts
    @staticmethod
    # 把 row_mapping 转成 dense partial matching tensor, 方便后续 batch completion / GED 计算。
    def _build_partial_from_row_mapping(row_mapping, max_n2):
        num_states, max_n1 = row_mapping.shape
        device = row_mapping.device
        partial = torch.zeros((num_states, max_n1, max_n2), dtype=torch.float, device=device)
        matched_mask = row_mapping >= 0
        if max_n2 > 0 and bool(matched_mask.any()):
            state_idx, row_idx = torch.nonzero(matched_mask, as_tuple=True)
            partial[state_idx, row_idx, row_mapping[state_idx, row_idx]] = 1.0
        blocked_rows = row_mapping == -1
        return partial, matched_mask, blocked_rows

    @staticmethod
    def _dense_matchings_to_row_maps(dense_matchings, batch_n1):
        device = dense_matchings.device
        num_states, max_n1, max_n2 = dense_matchings.shape
        row_maps = torch.full((num_states, max_n1), -1, dtype=torch.long, device=device)
        if max_n2 <= 0 or num_states == 0:
            return row_maps
        has_match = dense_matchings.any(dim=2)
        if bool(has_match.any()):
            row_maps[has_match] = dense_matchings.float().argmax(dim=2)[has_match].to(dtype=torch.long)
        row_ids = torch.arange(max_n1, device=device).view(1, -1)
        row_maps[row_ids >= batch_n1.view(-1, 1)] = -1
        return row_maps

    def _pairwise_node_substitution_cost(self, x1_valid, x2_valid):
        labels1 = self.trainer._node_labels_dense(x1_valid)
        labels2 = self.trainer._node_labels_dense(x2_valid)
        return (labels1.unsqueeze(2) != labels2.unsqueeze(1)).to(dtype=torch.float32)



    @staticmethod


    @staticmethod

    # 对一整池 child states 做张量化评估:
    # 1. 从 child_mapping 构出 partial matching
    # 2. 计算松弛 lower bound
    # 3. 做 greedy completion 得到可行 full matching
    # 4. 调 dense GED 计算 completed matching 的真实 GED

    @staticmethod
    def _complete_partial_matchings_greedy(score_batch, partial_matchings, blocked_rows, full_sizes):
        decoded = partial_matchings.clone()
        selected = decoded > 0.5
        row_taken = selected.any(dim=2) | blocked_rows
        col_taken = selected.any(dim=1)
        work_scores = score_batch.clone()
        work_scores = work_scores.masked_fill(row_taken.unsqueeze(2), float('-inf'))
        work_scores = work_scores.masked_fill(col_taken.unsqueeze(1), float('-inf'))
        selected_counts = selected.reshape(selected.shape[0], -1).sum(dim=1).long()
        batch_idx = torch.arange(score_batch.shape[0], device=score_batch.device)
        max_steps = int(full_sizes.max().item()) if full_sizes.numel() > 0 else 0
        n2 = score_batch.shape[2]
        for _ in range(max_steps):
            active = selected_counts < full_sizes
            if not bool(active.any()):
                break
            flat_scores = work_scores.view(score_batch.shape[0], -1)
            best_idx = torch.argmax(flat_scores, dim=1)
            best_scores = flat_scores[batch_idx, best_idx]
            active = active & torch.isfinite(best_scores)
            if not bool(active.any()):
                break
            rows = best_idx // n2
            cols = best_idx % n2
            active_batch = batch_idx[active]
            active_rows = rows[active]
            active_cols = cols[active]
            decoded[active_batch, active_rows, active_cols] = 1.0
            work_scores[active_batch, active_rows, :] = float('-inf')
            work_scores[active_batch, :, active_cols] = float('-inf')
            selected_counts[active_batch] += 1
        return decoded

    @staticmethod
    def _complete_partial_matchings_row_top1_unique(score_batch, partial_matchings, blocked_rows, full_sizes):
        decoded = partial_matchings.clone()
        selected = decoded > 0.5
        row_taken = selected.any(dim=2) | blocked_rows
        col_taken = selected.any(dim=1)
        selected_counts = selected.reshape(selected.shape[0], -1).sum(dim=1).long()
        num_states, n1, n2 = score_batch.shape
        if num_states == 0 or n1 == 0 or n2 == 0:
            return decoded

        work_scores = score_batch.clone()
        work_scores = work_scores.masked_fill(row_taken.unsqueeze(2), float('-inf'))
        work_scores = work_scores.masked_fill(col_taken.unsqueeze(1), float('-inf'))
        row_ids = torch.arange(n1, device=score_batch.device, dtype=torch.long).view(1, -1).expand(num_states, -1)
        row_sentinel = torch.full_like(row_ids, n1)
        max_steps = int(full_sizes.max().item()) if full_sizes.numel() > 0 else 0
        for _ in range(max_steps):
            active_batch_mask = selected_counts < full_sizes
            if not bool(active_batch_mask.any()):
                break

            row_best_scores, row_best_cols = torch.max(work_scores, dim=2)
            valid = active_batch_mask.view(-1, 1) & torch.isfinite(row_best_scores)
            if not bool(valid.any()):
                break

            safe_scores = row_best_scores.masked_fill(~valid, float('-inf'))
            col_best_scores = torch.full(
                (num_states, n2),
                float('-inf'),
                dtype=score_batch.dtype,
                device=score_batch.device,
            )
            col_best_scores.scatter_reduce_(
                1,
                row_best_cols,
                safe_scores,
                reduce='amax',
                include_self=True,
            )
            winners = valid & (row_best_scores == col_best_scores.gather(1, row_best_cols))
            candidate_rows = torch.where(winners, row_ids, row_sentinel)
            col_best_rows = torch.full((num_states, n2), n1, dtype=torch.long, device=score_batch.device)
            col_best_rows.scatter_reduce_(
                1,
                row_best_cols,
                candidate_rows,
                reduce='amin',
                include_self=True,
            )
            final_winners = winners & (row_ids == col_best_rows.gather(1, row_best_cols))
            if not bool(final_winners.any()):
                break

            active_state, active_rows = torch.nonzero(final_winners, as_tuple=True)
            active_cols = row_best_cols[active_state, active_rows]
            decoded[active_state, active_rows, active_cols] = 1.0
            row_taken[active_state, active_rows] = True
            col_taken[active_state, active_cols] = True
            work_scores[active_state, active_rows, :] = float('-inf')
            work_scores[active_state, :, active_cols] = float('-inf')
            selected_counts += torch.bincount(active_state, minlength=num_states).to(dtype=selected_counts.dtype)
        return decoded

    @staticmethod
    def _complete_row_maps_row_top1_unique(score_batch, row_maps, batch_n1, batch_n2):
        completed = row_maps.clone()
        num_states, n1, n2 = score_batch.shape
        if num_states == 0 or n1 == 0 or n2 == 0:
            valid_rows = torch.arange(n1, device=row_maps.device).view(1, -1) < batch_n1.view(-1, 1)
            completed = completed.masked_fill(valid_rows & (completed < 0), -1)
            completed = completed.masked_fill(~valid_rows, -1)
            return completed

        device = score_batch.device
        row_ids_1d = torch.arange(n1, device=device, dtype=torch.long)
        row_valid = row_ids_1d.view(1, -1) < batch_n1.view(-1, 1)
        col_valid = torch.arange(n2, device=device, dtype=torch.long).view(1, -1) < batch_n2.view(-1, 1)
        row_taken = (completed >= 0) | (completed == -1) | (~row_valid)
        selected_counts = ((completed >= 0) & row_valid).sum(dim=1).long()

        col_taken = torch.zeros((num_states, n2), dtype=torch.bool, device=device)
        matched = (completed >= 0) & row_valid
        if bool(matched.any()):
            state_idx, row_idx = torch.nonzero(matched, as_tuple=True)
            col_taken[state_idx, completed[state_idx, row_idx].clamp(min=0, max=n2 - 1)] = True
        col_taken |= ~col_valid

        remaining_rows = (~row_taken).sum(dim=1).long()
        remaining_cols = (~col_taken).sum(dim=1).long()
        full_sizes = selected_counts + torch.minimum(remaining_rows, remaining_cols)
        max_steps = int((full_sizes - selected_counts).clamp_min(0).max().item()) if full_sizes.numel() > 0 else 0
        if max_steps <= 0:
            completed = completed.masked_fill(row_valid & (completed < 0), -1)
            completed = completed.masked_fill(~row_valid, -1)
            return completed

        work_scores = score_batch.clone()
        work_scores.masked_fill_(row_taken.unsqueeze(2), float('-inf'))
        work_scores.masked_fill_(col_taken.unsqueeze(1), float('-inf'))
        row_ids = row_ids_1d.view(1, -1).expand(num_states, -1)
        row_sentinel = torch.full_like(row_ids, n1)

        for _ in range(max_steps):
            active_batch_mask = selected_counts < full_sizes
            if not bool(active_batch_mask.any()):
                break

            row_best_scores, row_best_cols = torch.max(work_scores, dim=2)
            valid = active_batch_mask.view(-1, 1) & torch.isfinite(row_best_scores)
            if not bool(valid.any()):
                break

            safe_scores = row_best_scores.masked_fill(~valid, float('-inf'))
            col_best_scores = torch.full(
                (num_states, n2),
                float('-inf'),
                dtype=score_batch.dtype,
                device=device,
            )
            col_best_scores.scatter_reduce_(
                1,
                row_best_cols,
                safe_scores,
                reduce='amax',
                include_self=True,
            )
            winners = valid & (row_best_scores == col_best_scores.gather(1, row_best_cols))
            candidate_rows = torch.where(winners, row_ids, row_sentinel)
            col_best_rows = torch.full((num_states, n2), n1, dtype=torch.long, device=device)
            col_best_rows.scatter_reduce_(
                1,
                row_best_cols,
                candidate_rows,
                reduce='amin',
                include_self=True,
            )
            final_winners = winners & (row_ids == col_best_rows.gather(1, row_best_cols))
            if not bool(final_winners.any()):
                break

            active_state, active_rows = torch.nonzero(final_winners, as_tuple=True)
            active_cols = row_best_cols[active_state, active_rows]
            completed[active_state, active_rows] = active_cols
            row_taken[active_state, active_rows] = True
            col_taken[active_state, active_cols] = True
            work_scores[active_state, active_rows, :] = float('-inf')
            work_scores[active_state, :, active_cols] = float('-inf')
            selected_counts += torch.bincount(active_state, minlength=num_states).to(dtype=selected_counts.dtype)

        completed = completed.masked_fill(row_valid & (completed < 0), -1)
        completed = completed.masked_fill(~row_valid, -1)
        return completed

    def _update_best_from_compact_row_maps(self, pair_idx_all, row_maps_all, best_ub, best_row_maps, n1, n2, x1_batch, x2_batch, ged_adj1_batch, ged_adj2_batch, chunk_size, candidate_lb_all=None, baseline_row_maps=None, complete_counts=None, equal_best_counts=None, equal_best_diff_counts=None, better_than_best_counts=None, label1_batch=None, label2_batch=None, left_upper_batch=None, left_edge_count_batch=None, right_edge_count_batch=None, ged_values_all=None):
        device = row_maps_all.device
        num_contexts = best_ub.shape[0]
        max_n1 = row_maps_all.shape[1]
        if pair_idx_all.numel() == 0:
            return best_ub, best_row_maps

        row_ids = torch.arange(max_n1, device=device).view(1, -1)
        current_best_map_full = best_row_maps
        candidate_best = torch.full((num_contexts,), float('inf'), dtype=torch.float, device=device)
        candidate_best_map = torch.full((num_contexts, max_n1), -3, dtype=torch.long, device=device)
        fast_exact_update = bool(getattr(self.trainer.args, 'app_bmao_fast_exact_update', True))
        equal_replace_found = torch.zeros((num_contexts,), dtype=torch.bool, device=device)
        equal_replace_map = torch.full((num_contexts, max_n1), -3, dtype=torch.long, device=device)
        total_items = int(pair_idx_all.shape[0])
        for chunk_start in range(0, total_items, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total_items)
            pair_idx = pair_idx_all[chunk_start:chunk_end]
            row_map = row_maps_all[chunk_start:chunk_end]
            chunk_ged_values = None if ged_values_all is None else ged_values_all[chunk_start:chunk_end]
            if candidate_lb_all is not None:
                candidate_lb = candidate_lb_all[chunk_start:chunk_end]
                current_best_for_lb = best_ub.index_select(0, pair_idx)
                keep = candidate_lb < current_best_for_lb
                if not bool(keep.any()):
                    continue
                pair_idx = pair_idx[keep]
                row_map = row_map[keep]
                if chunk_ged_values is not None:
                    chunk_ged_values = chunk_ged_values[keep]
                if pair_idx.numel() == 0:
                    continue
            batch_n1 = n1.index_select(0, pair_idx)
            batch_n2 = n2.index_select(0, pair_idx)
            if chunk_ged_values is None:
                right_adj_batch = ged_adj2_batch.index_select(0, pair_idx)
                ged_values = self._ged_from_compact_row_maps_precomputed(
                    row_maps=row_map,
                    pair_idx=pair_idx,
                    batch_n1=batch_n1,
                    batch_n2=batch_n2,
                    right_adj_batch=right_adj_batch,
                    label1_batch=label1_batch,
                    label2_batch=label2_batch,
                    left_upper_batch=left_upper_batch,
                    left_edge_count_batch=left_edge_count_batch,
                    right_edge_count_batch=right_edge_count_batch,
                )
            else:
                ged_values = chunk_ged_values
            current_best = best_ub.index_select(0, pair_idx)
            better_best = ged_values < current_best
            equal_best = None
            different_from_current = None
            if (not fast_exact_update) or (complete_counts is not None):
                current_best_map = current_best_map_full.index_select(0, pair_idx)
                valid_rows = row_ids < batch_n1.view(-1, 1)
                different_from_current = ((row_map != current_best_map) & valid_rows).any(dim=1)
                equal_best = torch.isclose(ged_values, current_best, atol=1e-6, rtol=0.0)
            if complete_counts is not None:
                ones = torch.ones_like(pair_idx, dtype=torch.long)
                complete_counts.index_add_(0, pair_idx, ones)
                if equal_best is not None and bool(equal_best.any()):
                    equal_best_counts.index_add_(0, pair_idx[equal_best], ones[equal_best])
                    equal_diff = equal_best & different_from_current
                    if bool(equal_diff.any()):
                        equal_best_diff_counts.index_add_(0, pair_idx[equal_diff], ones[equal_diff])
                if bool(better_best.any()):
                    better_than_best_counts.index_add_(0, pair_idx[better_best], ones[better_best])
            if not fast_exact_update and equal_best is not None:
                equal_diff = equal_best & different_from_current
                if bool(equal_diff.any()):
                    equal_diff_pairs = pair_idx[equal_diff]
                    equal_diff_maps = row_map[equal_diff]
                    unique_pairs, counts = torch.unique_consecutive(equal_diff_pairs, return_counts=True)
                    first_pos = torch.cumsum(counts, dim=0) - counts
                    new_pair_mask = ~equal_replace_found[unique_pairs]
                    if bool(new_pair_mask.any()):
                        new_pairs = unique_pairs[new_pair_mask]
                        equal_replace_found[new_pairs] = True
                        equal_replace_map[new_pairs] = equal_diff_maps[first_pos[new_pair_mask]]
            prev_best = candidate_best.index_select(0, pair_idx)
            if fast_exact_update:
                choose = ged_values < prev_best
                if bool(choose.any()):
                    cand_pairs = pair_idx[choose]
                    cand_values = ged_values[choose]
                    cand_maps = row_map[choose]
                    num_candidates = cand_pairs.shape[0]
                    pair_min = torch.full((num_contexts,), float('inf'), dtype=cand_values.dtype, device=device)
                    pair_min.scatter_reduce_(0, cand_pairs, cand_values, reduce='amin', include_self=True)
                    improved_pairs = torch.nonzero(pair_min < candidate_best, as_tuple=False).flatten()
                    if improved_pairs.numel() > 0:
                        positions = torch.arange(num_candidates, device=device, dtype=torch.long)
                        is_pair_min = torch.isclose(cand_values, pair_min.index_select(0, cand_pairs), atol=1e-6, rtol=0.0)
                        first_pos = torch.full((num_contexts,), num_candidates, dtype=torch.long, device=device)
                        first_pos.scatter_reduce_(0, cand_pairs[is_pair_min], positions[is_pair_min], reduce='amin', include_self=True)
                        chosen_pos = first_pos.index_select(0, improved_pairs)
                        candidate_best[improved_pairs] = pair_min.index_select(0, improved_pairs)
                        candidate_best_map[improved_pairs] = cand_maps[chosen_pos]
            else:
                prev_map = candidate_best_map.index_select(0, pair_idx)
                valid_rows = row_ids < batch_n1.view(-1, 1)
                improved = ged_values < prev_best
                tied_replace = torch.isclose(ged_values, prev_best, atol=1e-6, rtol=0.0) & ((row_map != prev_map) & valid_rows).any(dim=1)
                choose = improved | tied_replace
                if bool(choose.any()):
                    cand_pairs = pair_idx[choose]
                    cand_values = ged_values[choose]
                    cand_maps = row_map[choose]
                    cand_valid_rows = valid_rows[choose]
                    order = torch.argsort(cand_pairs, stable=True)
                    cand_pairs = cand_pairs[order]
                    cand_values = cand_values[order]
                    cand_maps = cand_maps[order]
                    cand_valid_rows = cand_valid_rows[order]
                    unique_pairs = torch.unique_consecutive(cand_pairs)
                    pair_min = torch.full((num_contexts,), float('inf'), dtype=cand_values.dtype, device=device)
                    pair_min.scatter_reduce_(0, cand_pairs, cand_values, reduce='amin', include_self=True)
                    pair_min_unique = pair_min.index_select(0, unique_pairs)
                    pair_best = candidate_best.index_select(0, unique_pairs)
                    positions = torch.arange(cand_pairs.shape[0], device=device, dtype=torch.long)
                    invalid_pos = torch.full_like(positions, cand_pairs.shape[0])
                    min_mask = torch.isclose(cand_values, pair_min.index_select(0, cand_pairs), atol=1e-6, rtol=0.0)
                    first_min_pos_all = torch.full((num_contexts,), cand_pairs.shape[0], dtype=torch.long, device=device)
                    first_min_pos_all.scatter_reduce_(0, cand_pairs, torch.where(min_mask, positions, invalid_pos), reduce='amin', include_self=True)
                    better_pairs = pair_min_unique < pair_best
                    if bool(better_pairs.any()):
                        better_pair_idx = unique_pairs[better_pairs]
                        better_pos = first_min_pos_all.index_select(0, better_pair_idx)
                        candidate_best[better_pair_idx] = pair_min_unique[better_pairs]
                        candidate_best_map[better_pair_idx] = cand_maps[better_pos]
                    equal_pairs = torch.isclose(pair_min_unique, pair_best, atol=1e-6, rtol=0.0)
                    if bool(equal_pairs.any()):
                        current_pair_maps = candidate_best_map.index_select(0, cand_pairs)
                        diff_mask = ((cand_maps != current_pair_maps) & cand_valid_rows).any(dim=1)
                        tie_mask = min_mask & diff_mask
                        first_tie_pos_all = torch.full((num_contexts,), cand_pairs.shape[0], dtype=torch.long, device=device)
                        first_tie_pos_all.scatter_reduce_(0, cand_pairs, torch.where(tie_mask, positions, invalid_pos), reduce='amin', include_self=True)
                        equal_pair_idx = unique_pairs[equal_pairs]
                        equal_pos = first_tie_pos_all.index_select(0, equal_pair_idx)
                        valid_equal = equal_pos < cand_pairs.shape[0]
                        if bool(valid_equal.any()):
                            equal_pair_idx = equal_pair_idx[valid_equal]
                            equal_pos = equal_pos[valid_equal]
                            candidate_best_map[equal_pair_idx] = cand_maps[equal_pos]

        improved_pairs = torch.isfinite(candidate_best) & (candidate_best < best_ub)
        equal_replace_pairs = (~improved_pairs) & equal_replace_found if not fast_exact_update else torch.zeros_like(improved_pairs)
        if bool(improved_pairs.any()):
            improved_idx = torch.nonzero(improved_pairs, as_tuple=False).flatten()
            best_ub[improved_idx] = candidate_best.index_select(0, improved_idx)
            chosen_row_map = candidate_best_map.index_select(0, improved_idx)
            best_row_maps[improved_idx] = chosen_row_map
        if bool(equal_replace_pairs.any()):
            equal_idx = torch.nonzero(equal_replace_pairs, as_tuple=False).flatten()
            chosen_row_map = equal_replace_map.index_select(0, equal_idx)
            best_row_maps[equal_idx] = chosen_row_map
        return best_ub, best_row_maps

    @staticmethod
    def _ged_from_compact_row_maps_precomputed(
        row_maps,
        pair_idx,
        batch_n1,
        batch_n2,
        right_adj_batch,
        label1_batch,
        label2_batch,
        left_upper_batch,
        left_edge_count_batch,
        right_edge_count_batch,
    ):
        device = row_maps.device
        num_states, max_n1 = row_maps.shape
        if num_states == 0:
            return torch.empty((0,), dtype=torch.float, device=device)

        row_ids = torch.arange(max_n1, device=device).view(1, -1)
        matched_mask = (row_ids < batch_n1.view(-1, 1)) & (row_maps >= 0) & (row_maps < batch_n2.view(-1, 1))
        matched_count = matched_mask.sum(dim=1).to(dtype=torch.float)

        left_labels = label1_batch.index_select(0, pair_idx)
        right_labels = label2_batch.index_select(0, pair_idx)
        max_label_cols = int(right_labels.shape[1])
        safe_cols = row_maps.clamp(min=0, max=max(max_label_cols - 1, 0)) if max_label_cols > 0 else torch.zeros_like(row_maps)
        if max_label_cols > 0:
            mapped_right_labels = right_labels.gather(1, safe_cols)
            node_sub_cost = (matched_mask & (left_labels != mapped_right_labels)).sum(dim=1).to(dtype=torch.float)
        else:
            node_sub_cost = torch.zeros((num_states,), dtype=torch.float, device=device)

        node_cost = (
            batch_n1.to(dtype=torch.float) - matched_count
            + batch_n2.to(dtype=torch.float) - matched_count
            + node_sub_cost
        )

        left_upper = left_upper_batch.index_select(0, pair_idx)
        left_edge_count = left_edge_count_batch.index_select(0, pair_idx)
        right_edge_count = right_edge_count_batch.index_select(0, pair_idx)

        if right_adj_batch.shape[1] > 0:
            gather_rows = safe_cols.unsqueeze(2).expand(-1, -1, right_adj_batch.shape[2])
            selected_rows = right_adj_batch.gather(1, gather_rows)
            gather_cols = safe_cols.unsqueeze(1).expand(-1, max_n1, -1)
            mapped_right_adj = selected_rows.gather(2, gather_cols) > 0.5
            matched_pair_mask = matched_mask.unsqueeze(1) & matched_mask.unsqueeze(2)
            preserved_edge_count = (left_upper & mapped_right_adj & matched_pair_mask).sum(dim=(1, 2)).to(dtype=torch.float)
        else:
            preserved_edge_count = torch.zeros((num_states,), dtype=torch.float, device=device)

        edge_delete_cost = left_edge_count - preserved_edge_count
        edge_insert_cost = right_edge_count - preserved_edge_count
        return node_cost + edge_delete_cost + edge_insert_cost

    @staticmethod
    def _row_mapping_to_dense_matching(row_mapping, n1, n2, device=None):
        if device is None and isinstance(row_mapping, torch.Tensor):
            device = row_mapping.device
        if device is None:
            device = torch.device('cpu')
        matching = torch.zeros((n1, n2), dtype=torch.float, device=device)
        if n1 == 0 or n2 == 0:
            return matching
        row_mapping_t = row_mapping.to(device=device, dtype=torch.long) if isinstance(row_mapping, torch.Tensor) else torch.tensor(row_mapping, dtype=torch.long, device=device)
        rows = torch.arange(n1, device=device, dtype=torch.long)
        valid = (row_mapping_t[:n1] >= 0) & (row_mapping_t[:n1] < n2)
        if bool(valid.any()):
            matching[rows[valid], row_mapping_t[:n1][valid]] = 1.0
        return matching

    def _matching_pairs_from_tensor(self, matching_tensor):
        if matching_tensor is None:
            return []
        _row_map, matched_pairs = self.trainer._dense_matching_to_row_mapping(matching_tensor.detach().cpu())
        return matched_pairs

    @staticmethod
    def _matching_pairs_from_row_mapping(row_mapping, n1, n2):
        if row_mapping is None:
            return []
        row_mapping_t = row_mapping.detach().cpu().to(dtype=torch.long) if isinstance(row_mapping, torch.Tensor) else torch.tensor(row_mapping, dtype=torch.long)
        matched_pairs = []
        limit = min(int(n1), int(row_mapping_t.shape[0]))
        for row in range(limit):
            col = int(row_mapping_t[row].item())
            if 0 <= col < int(n2):
                matched_pairs.append([row, col])
        return matched_pairs

    def _finalize_pair_results(self, candidate_pairs, contexts, postprocess_mode, candidate_budget, beam_score):
        grouped = {}
        for context in contexts:
            grouped.setdefault(context.pair_index, []).append(context)
        pair_results = []
        for pair_index, pair_contexts in sorted(grouped.items(), key=lambda item: item[0]):
            best_context = min(pair_contexts, key=lambda ctx: (ctx.best_ub, ctx.candidate_index))
            candidate_pair = candidate_pairs[pair_index]
            aggregated_source_candidate_indices = sorted(
                {
                    int(idx)
                    for ctx in pair_contexts
                    for idx in ctx.selection_metrics.get('source_candidate_indices', [ctx.candidate_index])
                }
            )
            if best_context.best_row_mapping_t is not None:
                best_matching_pairs = self._matching_pairs_from_row_mapping(best_context.best_row_mapping_t, best_context.n1, best_context.n2)
            elif best_context.best_matching_tensor is not None:
                best_matching_pairs = self._matching_pairs_from_tensor(best_context.best_matching_tensor)
            else:
                best_matching_pairs = best_context.best_matching_pairs
            best_row_mapping = None
            if best_context.best_row_mapping_t is not None:
                best_row_mapping = [
                    (None if int(col) < 0 or int(col) >= int(best_context.n2) else int(col))
                    for col in best_context.best_row_mapping_t[:best_context.n1].detach().cpu().tolist()
                ]
            elif best_context.best_matching_tensor is not None:
                best_row_mapping, _ = self.trainer._dense_matching_to_row_mapping(best_context.best_matching_tensor)
            anchor_rows = [idx for idx, col in enumerate(best_context.anchor_target_for_q) if col >= 0]
            anchor_targets = {str(idx): int(best_context.anchor_target_for_q[idx]) for idx in anchor_rows}
            anchors_respected = None
            if best_row_mapping is not None:
                anchors_respected = all((best_row_mapping[idx] is not None) and (int(best_row_mapping[idx]) == int(best_context.anchor_target_for_q[idx])) for idx in anchor_rows)
            pair_results.append({
                'pair_index': pair_index,
                'graph_1': best_context.graph_1,
                'graph_2': best_context.graph_2,
                'graph_1_gid': best_context.graph_1_gid,
                'graph_2_gid': best_context.graph_2_gid,
                'mode': postprocess_mode,
                'gt_ged': best_context.gt_ged,
                'n1': int(best_context.n1),
                'n2': int(best_context.n2),
                'search_row_count': int(best_context.search_row_count),
                'baseline_candidate_ged': float(best_context.baseline_ged),
                'refined_ged': float(best_context.best_ub),
                'app_bmao_reported_ged': float(best_context.best_ub),
                'delta_refined_minus_baseline': float(best_context.best_ub - best_context.baseline_ged),
                'delta_refined_minus_gt': float(best_context.best_ub - best_context.gt_ged),
                'best_candidate_index': int(best_context.candidate_index),
                'best_anchor_count': int(best_context.anchor_count),
                'num_candidates_available': int(candidate_pair['candidate_ged'].shape[0]),
                'num_candidates_processed': int(len(pair_contexts)),
                'aggregated_source_candidate_count': int(len(aggregated_source_candidate_indices)),
                'aggregated_source_candidate_indices': aggregated_source_candidate_indices,
                'sum_wall_time_s': 0.0,
                'sum_solver_time_s': 0.0,
                'best_matching_payload': {
                    'query_graph_id': str(best_context.graph_1_gid),
                    'db_graph_id': str(best_context.graph_2_gid),
                    'q_g_swapped': False,
                    'query_to_db': best_matching_pairs,
                },
                'beam_candidate_budget': int(candidate_budget),
                'beam_score_name': beam_score,
                'selected_candidate_indices': [int(ctx.candidate_index) for ctx in pair_contexts],
                'selected_candidate_metrics': {str(ctx.candidate_index): ctx.selection_metrics for ctx in pair_contexts},
                'best_candidate_metrics': best_context.best_candidate_metrics,
                'anchor_rows': anchor_rows,
                'anchor_targets': anchor_targets,
                'anchors_respected_by_best_matching': anchors_respected,
                'search_backend': self.backend,
                'search_expansions': int(sum(ctx.expansions for ctx in pair_contexts)),
                'search_states_evaluated': int(sum(ctx.states_evaluated for ctx in pair_contexts)),
                'search_max_frontier': int(max(ctx.max_queue_size for ctx in pair_contexts)),
                'search_steps': int(max(ctx.steps for ctx in pair_contexts)),
                'search_max_depth_reached': int(max(ctx.max_depth_reached for ctx in pair_contexts)),
                'strict_generated_children': int(sum(ctx.strict_generated_children for ctx in pair_contexts)),
                'strict_frontier_before_prune': int(max(ctx.strict_frontier_before_prune for ctx in pair_contexts)),
                'complete_states_evaluated': int(sum(ctx.complete_states_evaluated for ctx in pair_contexts)),
                'complete_states_equal_best': int(sum(ctx.complete_states_equal_best for ctx in pair_contexts)),
                'complete_states_equal_best_diff_mapping': int(sum(ctx.complete_states_equal_best_diff_mapping for ctx in pair_contexts)),
                'complete_states_better_than_best': int(sum(ctx.complete_states_better_than_best for ctx in pair_contexts)),
                'strict_frontier_after_prune': int(max(ctx.strict_frontier_after_prune for ctx in pair_contexts)),
                'strict_pruned_children': int(sum(ctx.strict_pruned_children for ctx in pair_contexts)),
            })
        return pair_results
