import json
import math
import os
import random
import re
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import kendalltau, spearmanr
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torch_geometric.nn.conv import GINConv
from torch_geometric.nn.norm import GraphNorm
from torch_geometric.nn.pool import global_add_pool

from app_bmao_internal_search import InternalAppBmaoSearchRunner
from diffusion_schedulers import CategoricalDiffusion, InferenceSchedule
from layers import ScalarEmbeddingSine, timestep_embedding
from layers_lightgt import DenseCrossGraphTransformer
from utils import load_all_graphs, load_ged_map, load_labels, load_pair_manifest, resolve_dataset_root


class DensePairDataset(Dataset):
    def __init__(self, trainer, pair_specs):
        self.trainer = trainer
        self.pair_specs = pair_specs

    def __len__(self):
        return len(self.pair_specs)

    def __getitem__(self, idx):
        return self.trainer.pack_dense_graph_pair(self.pair_specs[idx])


class TestSizeBucketedBatchSampler:
    def __init__(self, dataset, batch_size, test_nodes=0):
        self.dataset = dataset
        self.batch_size = max(1, int(batch_size))
        self.test_nodes = max(0, int(test_nodes))
        self.batches = self._build_batches()

    def _pair_size(self, pair_spec):
        pair_type, id_1, id_2 = pair_spec[:3]
        n1 = int(self.dataset.trainer.gn[id_1])
        if int(pair_type) == 1:
            n2 = int(self.dataset.trainer.delta_graphs[id_1][id_2]["n"])
        else:
            n2 = int(self.dataset.trainer.gn[id_2])
        return n1, n2

    def _pair_sort_key(self, pair_spec):
        n1, n2 = self._pair_size(pair_spec)
        long_side = max(n1, n2)
        short_side = min(n1, n2)
        pair_type, id_1, id_2 = pair_spec[:3]
        return (long_side * short_side, long_side, short_side, int(id_1), int(id_2))

    def _build_batches(self):
        ordered_indices = sorted(range(len(self.dataset.pair_specs)), key=lambda idx: self._pair_sort_key(self.dataset.pair_specs[idx]))
        if self.test_nodes <= 0:
            return [ordered_indices[start:start + self.batch_size] for start in range(0, len(ordered_indices), self.batch_size)]

        batches = []
        start = 0
        while start < len(ordered_indices):
            batch = []
            max_n1 = 0
            max_n2 = 0
            max_batch_size = None
            cursor = start
            while cursor < len(ordered_indices) and (max_batch_size is None or len(batch) < max_batch_size):
                pair_idx = ordered_indices[cursor]
                n1, n2 = self._pair_size(self.dataset.pair_specs[pair_idx])
                max_n1 = max(max_n1, n1)
                max_n2 = max(max_n2, n2)
                max_batch_size = max(1, self.test_nodes // max(1, max_n1 * max_n2))
                if batch and len(batch) >= max_batch_size:
                    break
                batch.append(pair_idx)
                cursor += 1
            batches.append(batch)
            start += len(batch)
        return batches

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


class TimestepEmbeddingSine(torch.nn.Module):
    def __init__(self, dim, max_period=10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, timesteps):
        return timestep_embedding(timesteps, self.dim, max_period=self.max_period)


class SparseMappingAGNN(torch.nn.Module):
    """DiffGED-style sparse matching-edge message passing adapted for dense batches."""

    def __init__(self, hidden_dim, time_emb_dim, noise_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.edge_transform = torch.nn.Linear(noise_dim, hidden_dim)
        self.P = torch.nn.Linear(hidden_dim, hidden_dim)
        self.Q = torch.nn.Linear(hidden_dim, hidden_dim)
        self.R = torch.nn.Linear(hidden_dim, hidden_dim)
        self.U = torch.nn.Linear(hidden_dim, hidden_dim)
        self.V = torch.nn.Linear(hidden_dim, hidden_dim)
        self.bn_bip_h = GraphNorm(hidden_dim)
        self.bn_bip_e = GraphNorm(hidden_dim)
        self.time_emb_layer = torch.nn.Sequential(torch.nn.ReLU(), torch.nn.Linear(time_emb_dim, hidden_dim))
        self.out_layer = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_dim, elementwise_affine=True),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, features, edge_mapping_idx, noise_mapping_emb, time_emb, node_batch):
        if edge_mapping_idx.numel() == 0:
            return features, noise_mapping_emb
        noise_mapping_emb = self.edge_transform(noise_mapping_emb)
        q_h = self.Q(features)
        r_h = self.R(features)
        mapping_e_hat = self.P(noise_mapping_emb) + q_h[edge_mapping_idx[0]] + r_h[edge_mapping_idx[1]]
        gates = torch.sigmoid(mapping_e_hat)

        u_h = self.U(features)
        v_h = self.V(features)
        aggr = global_add_pool(v_h[edge_mapping_idx[1]] * gates, edge_mapping_idx[0], size=features.shape[0])
        h = u_h + aggr

        h = self.bn_bip_h(h, node_batch)
        e = self.bn_bip_e(mapping_e_hat, node_batch[edge_mapping_idx[0]])
        h = torch.relu(h)
        e = torch.relu(e)

        e = e + self.time_emb_layer(time_emb)[node_batch[edge_mapping_idx[0]]]
        h = features + h
        e = noise_mapping_emb + self.out_layer(e)
        return h, e


class DenseSparseGNNDiffMatchDenseIO(torch.nn.Module):
    """Sparse DiffGED denoiser with the same dense-batch IO contract as LightGT."""

    def __init__(self, args, number_of_labels):
        super().__init__()
        self.args = args
        self.number_labels = number_of_labels
        self.setup_layers()

    def setup_layers(self):
        self.hidden_dims = list(self.args.hidden_dim)
        self.num_layers = len(self.hidden_dims)
        self.time_emb_dim = self.hidden_dims[0] // 2
        self.conv_layers = torch.nn.ModuleList()
        self.agnn_layers = torch.nn.ModuleList()
        self.gns = torch.nn.ModuleList()

        for layer_idx in range(self.num_layers):
            if layer_idx == 0:
                gin_mlp = torch.nn.Sequential(
                    torch.nn.Linear(self.number_labels, self.hidden_dims[layer_idx]),
                    torch.nn.ReLU(),
                    torch.nn.Linear(self.hidden_dims[layer_idx], self.hidden_dims[layer_idx]),
                )
                noise_dim = self.hidden_dims[layer_idx]
            else:
                gin_mlp = torch.nn.Sequential(
                    torch.nn.Linear(self.hidden_dims[layer_idx - 1], self.hidden_dims[layer_idx]),
                    torch.nn.ReLU(),
                    torch.nn.Linear(self.hidden_dims[layer_idx], self.hidden_dims[layer_idx]),
                )
                noise_dim = self.hidden_dims[layer_idx - 1]
            self.conv_layers.append(GINConv(gin_mlp, train_eps=True))
            self.agnn_layers.append(SparseMappingAGNN(self.hidden_dims[layer_idx], self.time_emb_dim, noise_dim))
            self.gns.append(GraphNorm(self.hidden_dims[layer_idx]))

        self.time_embed = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dims[0], self.time_emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.time_emb_dim, self.time_emb_dim),
        )
        self.edge_pos_embed = ScalarEmbeddingSine(self.hidden_dims[0], normalize=False)
        self.edge_embed = torch.nn.Linear(self.hidden_dims[0], self.hidden_dims[0])
        self.mapMatrix = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dims[-1], self.hidden_dims[-1] * 2),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dims[-1] * 2, self.hidden_dims[-1]),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dims[-1], 1),
        )

    def pop_last_topk_pruning_stats(self):
        return None

    @staticmethod
    def _dense_adj_to_edge_index(adj, offset):
        edge = torch.nonzero(adj > 0, as_tuple=False)
        if edge.numel() == 0:
            return edge.new_empty((2, 0))
        return (edge + int(offset)).t().contiguous()

    def _build_sparse_view(self, batch, noise_matching):
        x1 = batch["x1"]
        x2 = batch["x2"]
        adj1 = batch["adj1"]
        adj2 = batch["adj2"]
        n1_values = batch["n1"].detach().cpu().tolist()
        n2_values = batch["n2"].detach().cpu().tolist()
        device = x1.device

        node_features = []
        node_batch = []
        graph2_flags = []
        graph_edges = []
        mapping_edges = []
        mapping_values = []
        mapping_meta = []
        offset = 0

        for batch_idx, (n1_raw, n2_raw) in enumerate(zip(n1_values, n2_values)):
            n1 = int(n1_raw)
            n2 = int(n2_raw)
            left_offset = offset
            right_offset = offset + n1
            node_features.append(x1[batch_idx, :n1])
            node_features.append(x2[batch_idx, :n2])
            node_batch.extend([batch_idx] * (n1 + n2))
            graph2_flags.extend([False] * n1 + [True] * n2)

            graph_edges.append(self._dense_adj_to_edge_index(adj1[batch_idx, :n1, :n1], left_offset))
            graph_edges.append(self._dense_adj_to_edge_index(adj2[batch_idx, :n2, :n2], right_offset))

            if n1 > 0 and n2 > 0:
                rows = torch.arange(n1, device=device).repeat_interleave(n2)
                cols = torch.arange(n2, device=device).repeat(n1)
                mapping_edges.append(torch.stack([rows + left_offset, cols + right_offset], dim=0))
                mapping_values.append(noise_matching[batch_idx, :n1, :n2].reshape(-1, 1).float())
                mapping_meta.append(
                    torch.stack(
                        [
                            torch.full((n1 * n2,), batch_idx, device=device, dtype=torch.long),
                            rows,
                            cols,
                        ],
                        dim=1,
                    )
                )
            offset += n1 + n2

        if node_features:
            features = torch.cat(node_features, dim=0)
        else:
            features = x1.new_zeros((0, x1.shape[-1]))
        node_batch_tensor = torch.tensor(node_batch, device=device, dtype=torch.long)
        graph2_tensor = torch.tensor(graph2_flags, device=device, dtype=torch.bool)
        edge_index = torch.cat([edge for edge in graph_edges if edge.numel() > 0], dim=1) if any(edge.numel() > 0 for edge in graph_edges) else torch.empty((2, 0), device=device, dtype=torch.long)
        edge_mapping_idx = torch.cat(mapping_edges, dim=1) if mapping_edges else torch.empty((2, 0), device=device, dtype=torch.long)
        noise_values = torch.cat(mapping_values, dim=0) if mapping_values else x1.new_zeros((0, 1))
        meta = torch.cat(mapping_meta, dim=0) if mapping_meta else torch.empty((0, 3), device=device, dtype=torch.long)
        return features, edge_index, edge_mapping_idx, noise_values, meta, node_batch_tensor, graph2_tensor

    def forward(self, batch, noise_matching, t):
        features, edge_index, edge_mapping_idx, noise_values, meta, node_batch, graph2 = self._build_sparse_view(
            batch,
            noise_matching,
        )
        if features.numel() == 0 or edge_mapping_idx.numel() == 0:
            return noise_matching.new_zeros(noise_matching.shape)

        time_emb = self.time_embed(timestep_embedding(t, self.hidden_dims[0]).to(features.device))
        noise_mapping_emb = self.edge_embed(self.edge_pos_embed(noise_values))
        bn_batch = node_batch * 2
        bn_batch = bn_batch + graph2.long()

        for layer_idx in range(self.num_layers):
            features = torch.relu(self.gns[layer_idx](self.conv_layers[layer_idx](features, edge_index), batch=bn_batch))
            features, noise_mapping_emb = self.agnn_layers[layer_idx](
                features,
                edge_mapping_idx,
                noise_mapping_emb,
                time_emb,
                node_batch,
            )

        edge_logits = self.mapMatrix(noise_mapping_emb).squeeze(-1)
        out = noise_matching.new_zeros(noise_matching.shape)
        out[meta[:, 0], meta[:, 1], meta[:, 2]] = edge_logits
        pair_mask = batch["mask1"][:, :, None] & batch["mask2"][:, None, :]
        return out.masked_fill(~pair_mask, 0.0)


class DenseLightGTDiffMatchDenseIO(torch.nn.Module):
    """Dense-input / dense-output variant that keeps checkpoint-compatible parameter names."""

    def __init__(self, args, number_of_labels):
        super().__init__()
        self.args = args
        self.number_labels = number_of_labels
        self.setup_layers()

    def _get_heads_for_dim(self, dim):
        requested = int(getattr(self.args, "gt_heads", getattr(self.args, "num_heads", 4)))
        if requested <= 0:
            requested = 1
        if dim % requested == 0:
            return requested
        return math.gcd(dim, requested) or 1

    def setup_layers(self):
        self.hidden_dims = list(self.args.hidden_dim)
        self.num_layers = len(self.hidden_dims)
        self.dropout = float(getattr(self.args, "dropout", 0.0))
        self.time_emb_dim = self.hidden_dims[0] // 2
        self.dense_topk_enable = bool(getattr(self.args, "dense_topk_enable", False))
        self.dense_topk_start_layer = int(getattr(self.args, "dense_topk_start_layer", 2))
        self.dense_topk_row = int(getattr(self.args, "dense_topk_row", 16))
        self.dense_topk_col = int(getattr(self.args, "dense_topk_col", 16))
        self.dense_topk_score_source = str(getattr(self.args, "dense_topk_score_source", "qk_mean"))
        self.dense_topk_force_current_matching = bool(
            getattr(self.args, "dense_topk_force_current_matching", True)
        )
        self.last_dense_topk_stats = None

        self.conv_layers = torch.nn.ModuleList()
        self.cross_layers = torch.nn.ModuleList()
        self.gns = torch.nn.ModuleList()

        for layer_idx in range(self.num_layers):
            if layer_idx == 0:
                gin_mlp = torch.nn.Sequential(
                    torch.nn.Linear(self.number_labels, self.hidden_dims[layer_idx]),
                    torch.nn.ReLU(),
                    torch.nn.Linear(self.hidden_dims[layer_idx], self.hidden_dims[layer_idx]),
                )
                pair_in_dim = self.hidden_dims[layer_idx]
            else:
                gin_mlp = torch.nn.Sequential(
                    torch.nn.Linear(self.hidden_dims[layer_idx - 1], self.hidden_dims[layer_idx]),
                    torch.nn.ReLU(),
                    torch.nn.Linear(self.hidden_dims[layer_idx], self.hidden_dims[layer_idx]),
                )
                pair_in_dim = self.hidden_dims[layer_idx - 1]

            self.conv_layers.append(GINConv(gin_mlp, train_eps=True))
            self.gns.append(GraphNorm(self.hidden_dims[layer_idx]))
            self.cross_layers.append(
                DenseCrossGraphTransformer(
                    hidden_dim=self.hidden_dims[layer_idx],
                    time_emb_dim=self.time_emb_dim,
                    edge_dim=pair_in_dim,
                    num_heads=self._get_heads_for_dim(self.hidden_dims[layer_idx]),
                    dropout=self.dropout,
                )
            )

        self.time_embed = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dims[0], self.time_emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.time_emb_dim, self.time_emb_dim),
        )
        self.time_pos_embed = TimestepEmbeddingSine(self.hidden_dims[0])
        self.edge_pos_embed = ScalarEmbeddingSine(self.hidden_dims[0], normalize=False)
        self.edge_embed = torch.nn.Linear(self.hidden_dims[0], self.hidden_dims[0])
        self.mapMatrix = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dims[-1], self.hidden_dims[-1] * 2),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dims[-1] * 2, self.hidden_dims[-1]),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dims[-1], 1),
        )

    def pop_last_topk_pruning_stats(self):
        stats = self.last_dense_topk_stats
        self.last_dense_topk_stats = None
        return stats

    def build_time_context(self, t, batch_size, device, dtype):
        del batch_size
        time_pos = self.time_pos_embed(t)
        return self.time_embed(time_pos).to(device=device, dtype=dtype)

    def _dense_gin(self, x, adj, conv, mask):
        eps = conv.eps
        agg = torch.matmul(adj.to(dtype=x.dtype), x)
        out = conv.nn((1.0 + eps) * x + agg)
        return out * mask.unsqueeze(-1).to(out.dtype)

    @staticmethod
    def _apply_graph_norm_two_sides(x1, x2, mask1, mask2, graph_norm):
        batch_size, n1, dim = x1.shape
        _, n2, _ = x2.shape
        device = x1.device

        batch_ids = torch.arange(batch_size, device=device, dtype=torch.long)
        left_batch = (batch_ids * 2).unsqueeze(1).expand(batch_size, n1)
        right_batch = (batch_ids * 2 + 1).unsqueeze(1).expand(batch_size, n2)

        flat_x = torch.cat([x1[mask1], x2[mask2]], dim=0)
        flat_batch = torch.cat([left_batch[mask1], right_batch[mask2]], dim=0)
        if flat_x.numel() == 0:
            return x1, x2

        flat_norm = graph_norm(flat_x, flat_batch)
        out1 = x1.new_zeros(x1.shape)
        out2 = x2.new_zeros(x2.shape)
        left_count = int(mask1.sum().item())
        out1[mask1] = flat_norm[:left_count]
        out2[mask2] = flat_norm[left_count:]
        return out1, out2

    def _build_row_col_topk_mask(self, score_pair, mask1, mask2, forced_mask=None):
        _, n1, n2 = score_pair.shape
        pair_mask = mask1[:, :, None] & mask2[:, None, :]
        s = score_pair.masked_fill(~pair_mask, torch.finfo(score_pair.dtype).min)
        candidate_mask = torch.zeros_like(pair_mask, dtype=torch.bool)

        k_row = max(1, min(self.dense_topk_row, n2))
        row_idx = torch.topk(s, k=k_row, dim=-1).indices
        candidate_mask.scatter_(2, row_idx, True)

        k_col = max(1, min(self.dense_topk_col, n1))
        col_idx = torch.topk(s, k=k_col, dim=1).indices
        candidate_mask.scatter_(1, col_idx, True)

        if forced_mask is not None:
            candidate_mask = candidate_mask | forced_mask
        return candidate_mask & pair_mask

    def forward(self, batch, noise_matching, t):
        x1 = batch["x1"]
        x2 = batch["x2"]
        adj1 = batch["adj1"]
        adj2 = batch["adj2"]
        mask1 = batch["mask1"]
        mask2 = batch["mask2"]
        pair_mask = mask1[:, :, None] & mask2[:, None, :]

        dense_noise_attr = noise_matching.float().unsqueeze(-1)
        edge_pos = self.edge_pos_embed(dense_noise_attr.reshape(-1, 1))
        edge_pair = self.edge_embed(edge_pos).view(
            dense_noise_attr.shape[0],
            dense_noise_attr.shape[1],
            dense_noise_attr.shape[2],
            -1,
        )
        edge_pair = edge_pair * pair_mask.unsqueeze(-1).to(edge_pair.dtype)
        dense_noise_scalar = noise_matching.float()

        candidate_stats = {
            "enabled": self.dense_topk_enable,
            "num_layers": self.num_layers,
            "active_edges": [0.0] * self.num_layers,
            "full_edges": [float(pair_mask.sum().item())] * self.num_layers,
            "active_ratio": [1.0] * self.num_layers,
        }

        time_emb = self.build_time_context(
            t=t,
            batch_size=x1.shape[0],
            device=x1.device,
            dtype=x1.dtype,
        )

        features1 = x1
        features2 = x2
        for layer_idx in range(self.num_layers):
            left_conv = self._dense_gin(features1, adj1, self.conv_layers[layer_idx], mask1)
            right_conv = self._dense_gin(features2, adj2, self.conv_layers[layer_idx], mask2)
            left_norm, right_norm = self._apply_graph_norm_two_sides(
                left_conv,
                right_conv,
                mask1,
                mask2,
                self.gns[layer_idx],
            )
            features1 = torch.relu(left_norm) * mask1.unsqueeze(-1).to(left_norm.dtype)
            features2 = torch.relu(right_norm) * mask2.unsqueeze(-1).to(right_norm.dtype)

            if self.dense_topk_enable and layer_idx >= self.dense_topk_start_layer:
                h = self.cross_layers[layer_idx].num_heads
                dh = self.cross_layers[layer_idx].head_dim
                q1 = self.cross_layers[layer_idx].q_proj(features1).view(features1.shape[0], features1.shape[1], h, dh)
                k2 = self.cross_layers[layer_idx].k_proj(features2).view(features2.shape[0], features2.shape[1], h, dh)
                with torch.no_grad():
                    if self.dense_topk_score_source == "z_l2":
                        score_pair = torch.linalg.vector_norm(edge_pair, ord=2, dim=-1)
                    else:
                        score = torch.einsum("bihd,bjhd->bhij", q1, k2) * self.cross_layers[layer_idx].scale
                        if self.dense_topk_score_source == "qk_max":
                            score_pair = score.max(dim=1).values
                        else:
                            score_pair = score.mean(dim=1)
                    forced_mask = None
                    if self.dense_topk_force_current_matching:
                        forced_mask = (dense_noise_scalar > 0.5) & pair_mask
                    candidate_mask = self._build_row_col_topk_mask(
                        score_pair=score_pair,
                        mask1=mask1,
                        mask2=mask2,
                        forced_mask=forced_mask,
                    )
                active_edges = float(candidate_mask.sum().item())
                candidate_stats["active_edges"][layer_idx] = active_edges
                candidate_stats["active_ratio"][layer_idx] = active_edges / max(candidate_stats["full_edges"][layer_idx], 1.0)
                features1, features2, edge_pair = self.cross_layers[layer_idx](
                    x1=features1,
                    x2=features2,
                    edge_pair=edge_pair,
                    mask1=mask1,
                    mask2=mask2,
                    time_emb=time_emb,
                    edge_is_projected=False,
                    candidate_mask=candidate_mask,
                    precomputed_q1=q1,
                    precomputed_k2=k2,
                )
            else:
                features1, features2, edge_pair = self.cross_layers[layer_idx](
                    x1=features1,
                    x2=features2,
                    edge_pair=edge_pair,
                    mask1=mask1,
                    mask2=mask2,
                    time_emb=time_emb,
                    edge_is_projected=False,
                )

        score_dense = self.mapMatrix(edge_pair).squeeze(-1)
        score_dense = score_dense.masked_fill(~pair_mask, 0.0)
        self.last_dense_topk_stats = candidate_stats if self.dense_topk_enable else None
        return score_dense


class StreamingAppBmaoPostprocessor:
    def __init__(self, trainer, testing_graph_set, top_k_approach, test_k):
        self.trainer = trainer
        self.testing_graph_set = testing_graph_set
        self.top_k_approach = top_k_approach
        self.test_k = int(test_k)
        self.anchor_ratio = float(getattr(trainer.args, "app_bmao_anchor_ratio", 0.6))
        if self.anchor_ratio < 0.0 or self.anchor_ratio > 1.0:
            raise ValueError("--app-bmao-anchor-ratio must be in [0, 1].")
        self.postprocess_mode = str(getattr(trainer.args, "app_bmao_postprocess_mode", "best"))
        self.search_states = int(getattr(trainer.args, "app_bmao_search_states", 100))
        self.workers = max(1, int(getattr(trainer.args, "app_bmao_workers", 1)))
        self.candidate_budget = int(getattr(trainer.args, "app_bmao_candidate_budget", 4))
        self.beam_score = "ged"
        self.ged_bin = str(getattr(trainer.args, "app_bmao_ged_bin", ""))
        self.timeout_seconds = float(getattr(trainer.args, "app_bmao_timeout_seconds", 30.0))
        if not self.ged_bin or not os.path.exists(self.ged_bin):
            raise FileNotFoundError(f"Anchored App-BMao binary not found: {self.ged_bin}")
        self.solver_config = trainer._resolve_external_bmao_backend_config(
            getattr(trainer.args, "app_bmao_search_backend", "external_app_bmao")
        )
        self.model_dir = str(Path(self.ged_bin).resolve().parent)
        self.dataset_root, _resolved_dataset_name = resolve_dataset_root(trainer.data_path, trainer.args.dataset)
        self.stdin_enable = bool(getattr(trainer.args, "app_bmao_stdin_enable", False))
        self.diffusion_ub_enable = bool(getattr(trainer.args, "app_bmao_diffusion_ub_enable", False))
        self.disable_incumbent_ub = bool(getattr(trainer.args, "app_bmao_disable_incumbent_ub", False))
        self.temp_dir_ctx = tempfile.TemporaryDirectory(prefix="dense_app_bmao_stream_")
        self.temp_dir = Path(self.temp_dir_ctx.name)
        self.graph_cache = {}
        self.graph_cache_lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=self.workers)
        self.futures = {}
        self.pair_results = []
        self.total_submitted = 0
        self.progress_start = time.time()

        print(
            "Run streaming App-BMao postprocess: mode={} anchor_ratio={} search_states={} workers={} candidate_budget={} beam_score={}".format(
                self.postprocess_mode,
                self.anchor_ratio,
                self.search_states,
                self.workers,
                self.candidate_budget,
                self.beam_score,
            ),
            flush=True,
        )

    def _ensure_graph_cached(self, graph_id):
        graph_id = int(graph_id)
        cached = self.graph_cache.get(graph_id)
        if cached is not None:
            return cached
        with self.graph_cache_lock:
            cached = self.graph_cache.get(graph_id)
            if cached is not None:
                return cached
            graph_json_path = self.trainer._resolve_graph_json_path(self.dataset_root, graph_id)
            with graph_json_path.open("r", encoding="utf-8") as handle:
                graph_obj = json.load(handle)
            graph_text = self.trainer._render_graph_entry_from_json(graph_obj, graph_id)
            if self.stdin_enable:
                self.graph_cache[graph_id] = graph_text
                return graph_text
            graph_txt_path = self.temp_dir / f"graph_{graph_id}.txt"
            graph_txt_path.write_text(graph_text, encoding="utf-8")
            self.graph_cache[graph_id] = graph_txt_path
            return graph_txt_path

    def submit_candidate_pairs(self, candidate_pairs, start_pair_index):
        for local_idx, candidate_pair in enumerate(candidate_pairs):
            graph_1_gid = int(candidate_pair["pair_gid"][0].item())
            graph_2_gid = int(candidate_pair["pair_gid"][1].item())
            self._ensure_graph_cached(graph_1_gid)
            self._ensure_graph_cached(graph_2_gid)
            pair_index = int(start_pair_index + local_idx)
            task = {
                "pair_index": pair_index,
                "candidate_pair": candidate_pair,
                "mode": self.postprocess_mode,
                "search_states": self.search_states,
                "anchor_ratio": self.anchor_ratio,
                "candidate_budget": self.candidate_budget,
                "beam_score": self.beam_score,
                "ged_bin": self.ged_bin,
                "solver_config": self.solver_config,
                "model_dir": self.model_dir,
                "timeout_seconds": self.timeout_seconds,
                "graph_cache": self.graph_cache,
                "work_dir": str(self.temp_dir),
                "stdin_enable": self.stdin_enable,
                "diffusion_ub_enable": self.diffusion_ub_enable,
                "disable_incumbent_ub": self.disable_incumbent_ub,
            }
            future = self.executor.submit(self.trainer._run_app_bmao_for_pair, task)
            self.futures[future] = pair_index
            self.total_submitted += 1
        self._drain_completed(block=False)

    def _drain_completed(self, block):
        if not self.futures:
            return
        if block:
            iterator = as_completed(list(self.futures))
        else:
            iterator = [future for future in list(self.futures) if future.done()]
        for future in iterator:
            self.pair_results.append(future.result())
            self.futures.pop(future, None)
            completed = len(self.pair_results)
            total = self.total_submitted
            if completed % 50 == 0 or completed == total:
                elapsed_seconds = time.time() - self.progress_start
                eta_seconds = 0.0
                if completed > 0 and completed < total:
                    eta_seconds = (elapsed_seconds / completed) * (total - completed)
                sys.stdout.write(
                    "\r[App-BMao overlap] progress {}/{} elapsed={} eta={}".format(
                        completed,
                        total,
                        self.trainer._format_progress_seconds(elapsed_seconds),
                        self.trainer._format_progress_seconds(eta_seconds),
                    )
                )
                sys.stdout.flush()

    def finalize(self):
        try:
            self._drain_completed(block=True)
            if self.total_submitted > 0:
                sys.stdout.write("\n")
                sys.stdout.flush()
            return self.pair_results
        finally:
            self.executor.shutdown(wait=True)
            self.temp_dir_ctx.cleanup()


class GpuRefineOverlapPostprocessor:
    def __init__(self, trainer, testing_graph_set, top_k_approach, test_k):
        self.trainer = trainer
        self.testing_graph_set = testing_graph_set
        self.top_k_approach = top_k_approach
        self.test_k = int(test_k)
        self.delay_batches = max(1, int(getattr(trainer.args, "app_bmao_gpu_refine_overlap_delay_batches", 0)))
        self.anchor_ratio = float(getattr(trainer.args, "app_bmao_anchor_ratio", 0.6))
        if self.anchor_ratio < 0.0 or self.anchor_ratio > 1.0:
            raise ValueError("--app-bmao-anchor-ratio must be in [0, 1].")
        self.postprocess_mode = str(getattr(trainer.args, "app_bmao_postprocess_mode", "best"))
        self.candidate_budget = int(getattr(trainer.args, "app_bmao_candidate_budget", 4))
        self.beam_score = "ged"
        self.search_backend = str(getattr(trainer.args, "app_bmao_search_backend", "gpu_refine"))
        self.profile_enable = bool(getattr(trainer.args, "app_bmao_profile_enable", False))
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.pending_chunks = []
        self.pair_results = []
        self.profiling_reports = []
        self.total_submitted = 0
        self.total_completed = 0
        self.total_chunks_submitted = 0
        self.progress_start = time.time()
        self.visible_postprocess_time_s = 0.0

        print(
            "Run overlapped gpu_refine postprocess: mode={} anchor_ratio={} delay_batches={} candidate_budget={} beam_score={}".format(
                self.postprocess_mode,
                self.anchor_ratio,
                self.delay_batches,
                self.candidate_budget,
                self.beam_score,
            ),
            flush=True,
        )

    def _make_runner(self):
        return InternalAppBmaoSearchRunner(
            trainer=self.trainer,
            backend=self.search_backend,
            branch_topk=int(getattr(self.trainer.args, "v9_branch_width", 4)),
            unmatched_cost=1.0,
            beam_width=int(getattr(self.trainer.args, "v9_beam_width", 4)),
            beam_steps=0,
            score_source="scores",
            assignment_backend=str(getattr(self.trainer.args, "app_bmao_assignment_backend", "auto")),
            bestfirst_topm=int(getattr(self.trainer.args, "v9_rerank_pool", 32)),
        )

    def _prepare_chunk(self, runner, candidate_pairs):
        return runner.prepare_contexts(
            candidate_pairs=candidate_pairs,
            postprocess_mode=self.postprocess_mode,
            candidate_budget=self.candidate_budget,
            beam_score=self.beam_score,
            context_device=torch.device("cpu"),
        )

    def submit_candidate_pairs(self, candidate_pairs, start_pair_index):
        if not candidate_pairs:
            return 0.0
        chunk_pairs = list(candidate_pairs)
        runner = self._make_runner()
        chunk_index = self.total_chunks_submitted
        self.total_chunks_submitted += 1
        prepare_start = time.perf_counter()
        future = self.executor.submit(self._prepare_chunk, runner, chunk_pairs)
        self.pending_chunks.append(
            {
                "chunk_index": int(chunk_index),
                "pair_start": int(start_pair_index),
                "candidate_pairs": chunk_pairs,
                "runner": runner,
                "future": future,
                "prepare_submit_time": float(prepare_start),
            }
        )
        self.total_submitted += len(chunk_pairs)
        return self._drain_due_chunks()

    def _drain_due_chunks(self):
        drained_time_s = 0.0
        while len(self.pending_chunks) > self.delay_batches:
            drained_time_s += self._run_prepared_chunk(self.pending_chunks.pop(0))
        return drained_time_s

    def _run_prepared_chunk(self, chunk):
        visible_start = time.perf_counter()
        wait_start = time.perf_counter()
        contexts = chunk["future"].result()
        prepare_wait_time_s = time.perf_counter() - wait_start
        runner = chunk["runner"]
        chunk_pairs = chunk["candidate_pairs"]
        chunk_run_start = time.perf_counter()
        chunk_results = runner.run_prepared_contexts(
            candidate_pairs=chunk_pairs,
            contexts=contexts,
            postprocess_mode=self.postprocess_mode,
            candidate_budget=self.candidate_budget,
            beam_score=self.beam_score,
        )
        chunk_wall_s = time.perf_counter() - chunk_run_start
        visible_time_s = time.perf_counter() - visible_start
        pair_start = int(chunk["pair_start"])
        if pair_start != 0:
            for item in chunk_results:
                item["pair_index"] = int(item["pair_index"] + pair_start)
        if chunk_results:
            per_pair_wall = visible_time_s / float(len(chunk_results))
            for item in chunk_results:
                item["sum_wall_time_s"] = per_pair_wall
                item["sum_solver_time_s"] = per_pair_wall
        self.pair_results.extend(chunk_results)
        self.total_completed += len(chunk_pairs)
        self.visible_postprocess_time_s += visible_time_s

        if self.profile_enable and runner.last_profile_report is not None:
            profile = runner.last_profile_report
            metadata = dict(profile.get("metadata") or {})
            metadata.update(
                {
                    "pipeline.overlap_delay_batches": int(self.delay_batches),
                    "pipeline.prepare_wait_time_s": float(prepare_wait_time_s),
                    "pipeline.visible_block_time_s": float(visible_time_s),
                    "pipeline.chunk_gpu_refine_wall_time_s": float(chunk_wall_s),
                }
            )
            self.profiling_reports.append(
                {
                    "enabled": True,
                    "search_batch_size": int(len(chunk_pairs)),
                    "outer_total_wall_time_s": float(visible_time_s),
                    "outer_chunk_records": [
                        {
                            "chunk_index": int(chunk["chunk_index"]),
                            "pair_start": pair_start,
                            "pair_count": int(len(chunk_pairs)),
                            "prepare_wait_time_s": float(prepare_wait_time_s),
                            "chunk_gpu_refine_wall_time_s": float(chunk_wall_s),
                            "visible_block_time_s": float(visible_time_s),
                            "profile": profile,
                        }
                    ],
                    "aggregate_stage_totals_s": dict(profile.get("stage_totals_s") or {}),
                    "aggregate_stage_counts": dict(profile.get("stage_counts") or {}),
                    "aggregate_metadata": metadata,
                }
            )

        elapsed_seconds = time.time() - self.progress_start
        if self.total_submitted > 0:
            sys.stdout.write(
                "\r[gpu_refine overlap] progress {}/{} elapsed={} delay_batches={}".format(
                    self.total_completed,
                    self.total_submitted,
                    self.trainer._format_progress_seconds(elapsed_seconds),
                    self.delay_batches,
                )
            )
            sys.stdout.flush()
        return visible_time_s

    def finalize(self):
        try:
            while self.pending_chunks:
                self._run_prepared_chunk(self.pending_chunks.pop(0))
            if self.total_submitted > 0:
                sys.stdout.write("\n")
                sys.stdout.flush()
            profiling_report = self.trainer._merge_app_bmao_profiling_reports(self.profiling_reports)
            return self.pair_results, profiling_report
        finally:
            self.executor.shutdown(wait=True)


class TrainerDense(object):
    APP_BMAO_MATCHING_PAT = re.compile(r"^matching\s+(\d+)\s+\((.*?),\s(.*?)\):\s*(\{.*\})$")

    @staticmethod
    def _node_labels_dense(x):
        if x.dim() > 1 and x.size(-1) > 1:
            return x.argmax(dim=-1).long()
        if x.dim() > 1 and x.size(-1) == 1:
            return x.squeeze(-1).long()
        return x.long()

    def _graph_node_labels_dense(self, graph_index):
        graph = self.graphs[int(graph_index)]
        labels = graph.get("labels") if isinstance(graph, dict) else None
        if labels is not None:
            mapped_labels = []
            for label in labels:
                try:
                    mapped_labels.append(int(label))
                    continue
                except (TypeError, ValueError):
                    pass
                if hasattr(self, "global_labels") and label in self.global_labels:
                    mapped_labels.append(int(self.global_labels[label]))
                    continue
                label_key = str(label)
                if hasattr(self, "global_labels") and label_key in self.global_labels:
                    mapped_labels.append(int(self.global_labels[label_key]))
                    continue
                raise ValueError(f"Unable to map graph label {label!r} to an integer id")
            return torch.tensor(mapped_labels, dtype=torch.long)
        return self._node_labels_dense(self.features[int(graph_index)]).detach().cpu().long()

    def _node_labels_from_graph_indices(self, graph_indices, fallback_x):
        labels = self._node_labels_dense(fallback_x)
        if graph_indices is None or not hasattr(self, "node_labels"):
            return labels
        labels = labels.clone()
        for batch_idx, graph_index in enumerate(graph_indices.detach().cpu().tolist()):
            graph_index = int(graph_index)
            if not (0 <= graph_index < len(self.node_labels)):
                continue
            graph_labels = self.node_labels[graph_index].to(device=labels.device, dtype=torch.long)
            keep = min(labels.shape[1], graph_labels.numel()) if labels.dim() > 1 else min(labels.numel(), graph_labels.numel())
            if labels.dim() > 1:
                labels[batch_idx, :keep] = graph_labels[:keep]
            else:
                labels[:keep] = graph_labels[:keep]
        return labels

    def __init__(self, args):
        self.args = args
        if self.args.denoise_network not in {"lightgt_dense", "diffged_sparse"}:
            raise ValueError("Unsupported --denoise-network {}.".format(self.args.denoise_network))
        if getattr(self.args, "diffusion_mechanism", "partialdiff") not in {"partialdiff", "diffged"}:
            raise ValueError("Unsupported --diffusion-mechanism {}.".format(self.args.diffusion_mechanism))

        self.use_gpu = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if self.use_gpu else "cpu")
        self.results = []
        self.timing_totals = defaultdict(float)
        self.timing_counts = defaultdict(int)
        self.cur_epoch = 0
        self.data_path = self.args.data_path if getattr(self.args, "data_path", None) else self.args.abs_path

        self.load_data()
        self.transfer_data_to_torch()
        self.gen_delta_graphs()
        self.setup_model()
        self.init_graph_pairs()
        self._build_dataloaders()

    def _eval_autocast_context(self):
        precision = str(getattr(self.args, "eval_precision", "fp32")).lower()
        if not self.use_gpu or precision == "fp32":
            return nullcontext()
        if precision == "fp16":
            dtype = torch.float16
        elif precision == "bf16":
            dtype = torch.bfloat16
        else:
            raise ValueError(f"Unsupported --eval-precision: {precision}")
        return torch.autocast(device_type="cuda", dtype=dtype)

    def start_timer(self):
        if self.use_gpu:
            torch.cuda.synchronize(self.device)
        return time.perf_counter()

    def stop_timer(self, start_time, key=None):
        if self.use_gpu:
            torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - start_time
        if key is not None:
            self.timing_totals[key] += elapsed
            self.timing_counts[key] += 1
        return elapsed

    def reset_timing_stats(self):
        self.timing_totals.clear()
        self.timing_counts.clear()

    def _cuda_memory_snapshot(self):
        if not self.use_gpu:
            return None
        torch.cuda.synchronize(self.device)
        return {
            "allocated_mb": round(torch.cuda.memory_allocated(self.device) / (1024.0 * 1024.0), 3),
            "reserved_mb": round(torch.cuda.memory_reserved(self.device) / (1024.0 * 1024.0), 3),
            "peak_allocated_mb": round(torch.cuda.max_memory_allocated(self.device) / (1024.0 * 1024.0), 3),
            "peak_reserved_mb": round(torch.cuda.max_memory_reserved(self.device) / (1024.0 * 1024.0), 3),
        }

    def _reset_cuda_peak_memory(self):
        if self.use_gpu:
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)

    @staticmethod
    def _summarize_runtime_profile(records):
        if not records:
            return None

        def _sum(key):
            return round(sum(float(record.get(key, 0.0) or 0.0) for record in records), 5)

        def _max_nested(stage_key, metric_key):
            values = []
            for record in records:
                stage = record.get(stage_key)
                if isinstance(stage, dict) and stage.get(metric_key) is not None:
                    values.append(float(stage[metric_key]))
            return round(max(values), 3) if values else None

        def _max_any_nested(stage_keys, metric_key):
            values = []
            for record in records:
                for stage_key in stage_keys:
                    stage = record.get(stage_key)
                    if isinstance(stage, dict) and stage.get(metric_key) is not None:
                        values.append(float(stage[metric_key]))
            return round(max(values), 3) if values else None

        slow_batches = sorted(
            records,
            key=lambda item: float(item.get("batch_wall_time_s", 0.0) or 0.0),
            reverse=True,
        )[:5]
        return {
            "num_profiled_batches": int(len(records)),
            "total_batch_wall_time_s": _sum("batch_wall_time_s"),
            "total_diffusion_wall_time_s": _sum("diffusion_wall_time_s"),
            "total_app_bmao_wall_time_s": _sum("app_bmao_wall_time_s"),
            "cuda_peak_allocated_mb": _max_any_nested(
                ["cuda_after_move", "diffusion_cuda_memory", "app_bmao_cuda_memory", "batch_cuda_memory"],
                "peak_allocated_mb",
            ),
            "cuda_peak_reserved_mb": _max_any_nested(
                ["cuda_after_move", "diffusion_cuda_memory", "app_bmao_cuda_memory", "batch_cuda_memory"],
                "peak_reserved_mb",
            ),
            "diffusion_peak_allocated_mb": _max_nested("diffusion_cuda_memory", "peak_allocated_mb"),
            "app_bmao_peak_allocated_mb": _max_nested("app_bmao_cuda_memory", "peak_allocated_mb"),
            "top_slowest_batches": [
                {
                    "batch_idx": int(record.get("batch_idx", -1)),
                    "batch_size": int(record.get("batch_size", 0)),
                    "max_n1": int(record.get("max_n1", 0)),
                    "max_n2": int(record.get("max_n2", 0)),
                    "batch_wall_time_s": round(float(record.get("batch_wall_time_s", 0.0) or 0.0), 5),
                    "diffusion_wall_time_s": round(float(record.get("diffusion_wall_time_s", 0.0) or 0.0), 5),
                    "app_bmao_wall_time_s": round(float(record.get("app_bmao_wall_time_s", 0.0) or 0.0), 5),
                }
                for record in slow_batches
            ],
        }

    def load_data(self):
        if getattr(self.args, "fixed_pair_root", None):
            self.load_fixed_pair_data()
            return

        dataset_name = self.args.dataset
        dataset_root, resolved_dataset_name = resolve_dataset_root(self.data_path, dataset_name)
        load_workers = int(getattr(self.args, "load_workers", 0))
        with ThreadPoolExecutor(max_workers=3) as executor:
            graphs_future = executor.submit(load_all_graphs, self.data_path, dataset_name, load_workers)
            manifest_future = executor.submit(load_pair_manifest, self.data_path, dataset_name)
            ged_future = executor.submit(load_ged_map, self.data_path, dataset_name, "TaGED.json")

            self.train_num, self.val_num, self.test_num, self.graphs = graphs_future.result()
            precomputed_pair_splits = manifest_future.result()
            ged_dict = ged_future.result()

        print("Load {} graphs. ({} for training)".format(len(self.graphs), self.train_num))

        self.has_precomputed_pairs = precomputed_pair_splits is not None
        self.precomputed_pair_splits = None
        if self.has_precomputed_pairs:
            self.precomputed_pair_splits = precomputed_pair_splits
            print("Load {} precomputed graph pairs.".format(len(self.precomputed_pair_splits)))

        labels_path = os.path.join(dataset_root, "labels.json")
        self.number_of_labels = 0
        if os.path.isfile(labels_path):
            self.global_labels, self.features = load_labels(
                self.data_path,
                dataset_name,
                graphs=self.graphs,
                load_workers=load_workers,
            )
            if self.features and self.features[0] and self.features[0][0]:
                self.number_of_labels = len(self.features[0][0])
        if self.number_of_labels == 0:
            self.number_of_labels = 1
            self.features = [[[2.0] for _ in range(g["n"])] for g in self.graphs]

        self.ged_dict = ged_dict
        if resolved_dataset_name != dataset_name:
            print("Load ged dict. (resolved dataset alias: {} -> {})".format(dataset_name, resolved_dataset_name))
        else:
            print("Load ged dict.")

    @staticmethod
    def _load_fixed_text_graph(path, gid):
        with open(path, "r", encoding="utf-8") as handle:
            header = handle.readline().strip().split()
            if len(header) < 2:
                raise ValueError(f"Invalid fixed-pair graph header: {path}")
            n = int(header[0])
            m = int(header[1])
            feature_dim = int(header[2]) if len(header) >= 3 else 0
            labels = [0] * n
            for _ in range(n):
                parts = handle.readline().strip().split()
                if len(parts) < 2:
                    raise ValueError(f"Invalid fixed-pair node row: {path}")
                labels[int(parts[0])] = int(parts[1])
            edges = []
            for line in handle:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                edges.append([int(parts[0]), int(parts[1])])
        if m != len(edges):
            m = len(edges)
        return {"gid": int(gid), "n": n, "m": m, "labels": labels, "graph": edges, "fixed_feature_dim": feature_dim}

    @staticmethod
    def _fixed_graph_local_path(dataset_root, graph_key):
        split_name, local_name = str(graph_key).split("/", 1)
        return os.path.join(dataset_root, split_name, local_name)

    def _fixed_pair_dataset_root(self):
        root = Path(self.args.fixed_pair_root).expanduser()
        if (root / "benchmark_train_pairs.jsonl").is_file():
            return root
        dataset_root = root / self.args.dataset
        if not dataset_root.is_dir():
            raise FileNotFoundError(f"Fixed-pair dataset root not found: {dataset_root}")
        return dataset_root

    def _fixed_pair_splits_to_load(self):
        if int(getattr(self.args, "model_train", 1)) != 0:
            return ("train", "val", "test")
        if str(getattr(self.args, "experiment", "test")) != "test":
            return ("train", "val", "test")

        testset = str(getattr(self.args, "testset", "test"))
        if testset in {"test", "small", "large"}:
            return ("test",)
        if testset == "val":
            return ("val",)
        return ("train", "val", "test")

    def load_fixed_pair_data(self):
        dataset_root = self._fixed_pair_dataset_root()
        splits_to_load = self._fixed_pair_splits_to_load()
        pair_files = {
            split: dataset_root / f"benchmark_{split}_pairs.jsonl"
            for split in splits_to_load
        }
        rows = []
        graph_keys = []
        seen_graph_keys = set()
        for split_name, pair_path in pair_files.items():
            if not pair_path.is_file():
                continue
            with pair_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    row.setdefault("benchmark_split", split_name)
                    rows.append(row)
                    for key_name in ("graph_1", "graph_2"):
                        graph_key = str(row[key_name])
                        if graph_key not in seen_graph_keys:
                            seen_graph_keys.add(graph_key)
                            graph_keys.append(graph_key)

        if not rows:
            raise FileNotFoundError(f"No fixed-pair rows found under {dataset_root}")

        fixed_feature_dim = 0
        summary_path = dataset_root / "summary.json"
        if summary_path.is_file():
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                fixed_feature_dim = int(summary.get("feature_dim") or len(summary.get("label_map", {})) or 0)
            except Exception:
                fixed_feature_dim = 0

        self.fixed_pair_rows = rows
        self.fixed_graph_key_to_index = {}
        self.fixed_index_to_graph_key = []
        self.graphs = []
        label_values = set()
        for graph_key in graph_keys:
            graph_path = self._fixed_graph_local_path(str(dataset_root), graph_key)
            if not os.path.isfile(graph_path):
                raise FileNotFoundError(f"Fixed-pair graph file not found: {graph_path}")
            graph_idx = len(self.graphs)
            graph = self._load_fixed_text_graph(graph_path, gid=graph_idx)
            self.fixed_graph_key_to_index[graph_key] = graph_idx
            self.fixed_index_to_graph_key.append(graph_key)
            self.graphs.append(graph)
            label_values.update(int(label) for label in graph["labels"])
            fixed_feature_dim = max(fixed_feature_dim, int(graph.get("fixed_feature_dim", 0) or 0))

        if fixed_feature_dim > 0:
            sorted_labels = list(range(fixed_feature_dim))
        else:
            sorted_labels = sorted(label_values) if label_values else [0]
        label_to_index = {label: idx for idx, label in enumerate(sorted_labels)}
        feature_dim = len(sorted_labels)
        self.global_labels = {str(label): idx for label, idx in label_to_index.items()}
        self.features = []
        for graph in self.graphs:
            graph_features = []
            for label in graph["labels"]:
                onehot = [0.0] * feature_dim
                onehot[label_to_index[int(label)]] = 1.0
                graph_features.append(onehot)
            self.features.append(graph_features)

        self.train_num = sum(1 for key in graph_keys if key.startswith("train/"))
        self.val_num = 0
        self.test_num = len(self.graphs) - self.train_num
        self.number_of_labels = feature_dim
        self.ged_dict = {}
        self.has_precomputed_pairs = False
        self.precomputed_pair_splits = None
        print(
            "Load fixed-pair benchmark {} from {}: {} graphs, {} pairs, splits={}.".format(
                self.args.dataset,
                dataset_root,
                len(self.graphs),
                len(self.fixed_pair_rows),
                ",".join(splits_to_load),
            )
        )

    def transfer_data_to_torch(self):
        self.features = [torch.tensor(x).float() for x in self.features]
        print("Feature shape of 1st graph:", self.features[0].shape)

        self.gid = [g["gid"] for g in self.graphs]
        self.gid_to_index = {graph_gid: idx for idx, graph_gid in enumerate(self.gid)}
        self.gn = [g["n"] for g in self.graphs]
        self.node_labels = [self._graph_node_labels_dense(idx) for idx in range(len(self.graphs))]
        self.identity_mappings = [torch.eye(graph_n, dtype=torch.float) for graph_n in self.gn]
        self.zero_ta_ged = (0.0, 0.0, 0.0, 0.0)
        self.pair_metadata = {}

        self.conv_adj = []
        self.ged_adj = []
        self.ged_edge_counts = []
        self.degrees = []
        for graph in self.graphs:
            n = graph["n"]
            conv_adj = torch.zeros((n, n), dtype=torch.float)
            ged_adj = torch.zeros((n, n), dtype=torch.float)
            for src, dst in graph["graph"]:
                conv_adj[src, dst] = 1.0
                conv_adj[dst, src] = 1.0
                ged_adj[src, dst] = 1.0
                ged_adj[dst, src] = 1.0
            conv_adj.fill_diagonal_(1.0)
            self.conv_adj.append(conv_adj)
            self.ged_adj.append(ged_adj)
            self.ged_edge_counts.append(len(graph["graph"]))
            self.degrees.append(conv_adj.sum(dim=1))

        for (gid_1, gid_2), (ta_ged, gt_mappings) in self.ged_dict.items():
            idx_1 = self.gid_to_index.get(gid_1)
            idx_2 = self.gid_to_index.get(gid_2)
            if idx_1 is None or idx_2 is None:
                continue
            self.pair_metadata[(idx_1, idx_2)] = {
                "ta_ged": ta_ged,
                "node_mappings": gt_mappings,
            }

    @staticmethod
    def node_mapping_to_matrix(node_mapping, n1, n2):
        mapping_matrix = torch.zeros((n1, n2), dtype=torch.float)
        for row_idx, col_idx in enumerate(node_mapping):
            if col_idx is None:
                continue
            mapping_matrix[row_idx, col_idx] = 1.0
        return mapping_matrix

    @staticmethod
    def node_mapping_pairs_to_matrix(node_mapping_pairs, n1, n2):
        mapping_matrix = torch.zeros((n1, n2), dtype=torch.float)
        for row_idx, col_idx in node_mapping_pairs or []:
            row_idx = int(row_idx)
            col_idx = int(col_idx)
            if 0 <= row_idx < n1 and 0 <= col_idx < n2:
                mapping_matrix[row_idx, col_idx] = 1.0
        return mapping_matrix

    def get_pair_metadata(self, id_1, id_2):
        if id_1 == id_2:
            return {
                "ta_ged": self.zero_ta_ged,
                "mapping": self.identity_mappings[id_1],
                "mapping_candidates": [self.identity_mappings[id_1]],
            }

        pair_data = self.pair_metadata.get((id_1, id_2))
        if pair_data is not None:
            n1 = self.gn[id_1]
            n2 = self.gn[id_2]
            dense_mappings = [self.node_mapping_to_matrix(gt_mapping, n1, n2) for gt_mapping in pair_data["node_mappings"]]
            return {
                "ta_ged": pair_data["ta_ged"],
                "mapping": dense_mappings[0],
                "mapping_candidates": dense_mappings,
            }

        reverse_pair_data = self.pair_metadata.get((id_2, id_1))
        if reverse_pair_data is None:
            return None

        n1 = self.gn[id_2]
        n2 = self.gn[id_1]
        reverse_dense_mappings = [self.node_mapping_to_matrix(gt_mapping, n1, n2) for gt_mapping in reverse_pair_data["node_mappings"]]
        return {
            "ta_ged": reverse_pair_data["ta_ged"],
            "mapping": reverse_dense_mappings[0].t().contiguous(),
            "mapping_candidates": [mapping.t().contiguous() for mapping in reverse_dense_mappings],
        }

    @staticmethod
    def _adj_from_edges(num_nodes, edges, include_self_loops):
        adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float)
        for src, dst in edges:
            src = int(src)
            dst = int(dst)
            adj[src, dst] = 1.0
            adj[dst, src] = 1.0
        if include_self_loops:
            adj.fill_diagonal_(1.0)
        return adj

    @staticmethod
    def delta_graph(graph, features):
        n = int(graph["n"])
        permute = list(range(n))
        random.shuffle(permute)
        mapping = torch.zeros((n, n), dtype=torch.float)
        for row, col in enumerate(permute):
            mapping[row, col] = 1.0

        edges = [list(edge) for edge in graph["graph"]]
        edge_set = set()
        for src, dst in edges:
            edge_set.add((int(src), int(dst)))
            edge_set.add((int(dst), int(src)))

        random.shuffle(edges)
        original_edges = len(edges)
        ged_budget = random.randint(1, 5) if n <= 20 else random.randint(1, 10)
        del_num = min(original_edges, random.randint(0, ged_budget))
        edges = edges[: original_edges - del_num]
        add_num = ged_budget - del_num
        if (add_num + original_edges) * 2 > n * (n - 1):
            add_num = n * (n - 1) // 2 - original_edges

        added = 0
        while added < add_num:
            src = random.randint(0, n - 1)
            dst = random.randint(0, n - 1)
            if src != dst and (src, dst) not in edge_set:
                edge_set.add((src, dst))
                edge_set.add((dst, src))
                edges.append([src, dst])
                added += 1

        permuted_edges = [[permute[src], permute[dst]] for src, dst in edges]
        synthetic_features = torch.zeros_like(features)
        for src, dst in enumerate(permute):
            synthetic_features[dst] = features[src]

        ged = del_num + add_num
        return {
            "n": n,
            "m": len(permuted_edges),
            "features": synthetic_features,
            "conv_adj": TrainerDense._adj_from_edges(n, permuted_edges, include_self_loops=True),
            "ged_adj": TrainerDense._adj_from_edges(n, permuted_edges, include_self_loops=False),
            "mapping": mapping,
            "ta_ged": (ged, 0, 0, ged),
        }

    def gen_delta_graphs(self):
        if getattr(self.args, "fixed_pair_root", None):
            self.delta_graphs = [None] * len(self.graphs)
            return
        if getattr(self, "has_precomputed_pairs", False):
            self.delta_graphs = [None] * len(self.graphs)
            return
        random.seed(0)
        delta_count = max(0, int(getattr(self.args, "num_delta_graphs", 0)))
        self.delta_graphs = [None] * len(self.graphs)
        if delta_count <= 0:
            return
        for graph_idx, graph in enumerate(self.graphs):
            if int(graph["n"]) <= 10:
                continue
            self.delta_graphs[graph_idx] = [
                self.delta_graph(graph, self.features[graph_idx]) for _ in range(delta_count)
            ]

    def base_match_score_adjustment_dense_tensors(self, id_1, x1, x2, degree1, degree2, label2=None):
        n1 = x1.shape[0]
        n2 = x2.shape[0]
        if hasattr(self, "node_labels"):
            left_label = self.node_labels[int(id_1)][:n1].to(dtype=torch.long)
        else:
            left_label = self._node_labels_dense(x1)
        right_label = label2.to(dtype=torch.long) if label2 is not None else self._node_labels_dense(x2)

        degree_scale = torch.maximum(degree1.max(), degree2.max()).clamp(min=1.0)
        left_sim = x1 @ x2.t() if x1.dim() == 2 and x2.dim() == 2 else torch.zeros((n1, n2), dtype=torch.float)
        label_cost = (left_label[:, None] != right_label[None, :]).float()
        degree_cost = (degree1[:, None] - degree2[None, :]).abs() / degree_scale
        return self.args.match_cost_scale * (
            self.args.match_label_weight * label_cost
            + self.args.match_degree_weight * degree_cost
            - self.args.match_similarity_weight * left_sim
        )

    def base_match_score_adjustment_dense(self, id_1, id_2, x1, x2):
        n1 = x1.shape[0]
        n2 = x2.shape[0]
        if hasattr(self, "node_labels"):
            left_label = self.node_labels[int(id_1)][:n1].to(device=x1.device, dtype=torch.long)
            right_label = self.node_labels[int(id_2)][:n2].to(device=x2.device, dtype=torch.long)
        else:
            left_label = self._node_labels_dense(x1)
            right_label = self._node_labels_dense(x2)

        left_degree = self.degrees[id_1][:n1]
        right_degree = self.degrees[id_2][:n2]
        degree_scale = torch.maximum(left_degree.max(), right_degree.max()).clamp(min=1.0)

        left_sim = x1 @ x2.t() if x1.dim() == 2 and x2.dim() == 2 else torch.zeros((n1, n2), dtype=torch.float)
        label_cost = (left_label[:, None] != right_label[None, :]).float()
        degree_cost = (left_degree[:, None] - right_degree[None, :]).abs() / degree_scale
        return self.args.match_cost_scale * (
            self.args.match_label_weight * label_cost
            + self.args.match_degree_weight * degree_cost
            - self.args.match_similarity_weight * left_sim
        )

    def _pair_side_metadata(self, graph_id, x, degree=None, labels=None):
        n = int(x.shape[0])
        if degree is None:
            degree = self.degrees[int(graph_id)][:n]
        else:
            degree = degree[:n]
        if labels is None:
            if hasattr(self, "node_labels"):
                labels = self.node_labels[int(graph_id)][:n]
            else:
                labels = self._node_labels_dense(x)
        else:
            labels = labels[:n]
        return degree.to(dtype=torch.float), labels.to(dtype=torch.long)

    def pack_dense_graph_pair(self, pair):
        pair_type, id_1, id_2 = pair[:3]
        if pair_type == 2:
            gt_ged = float(pair[3])
            x1 = self.features[id_1]
            x2 = self.features[id_2]
            n1 = x1.shape[0]
            n2 = x2.shape[0]
            gt_matching = self.node_mapping_pairs_to_matrix(pair[4], n1, n2) if len(pair) > 4 else torch.zeros((n1, n2), dtype=torch.float)
            left_degree, left_label = self._pair_side_metadata(id_1, x1)
            right_degree, right_label = self._pair_side_metadata(id_2, x2)
            return {
                "pair": torch.tensor([id_1, id_2], dtype=torch.long),
                "pair_gid": torch.tensor([self.gid[id_1], self.gid[id_2]], dtype=torch.long),
                "x1": x1,
                "x2": x2,
                "adj1": self.conv_adj[id_1],
                "adj2": self.conv_adj[id_2],
                "ged_adj1": self.ged_adj[id_1],
                "ged_adj2": self.ged_adj[id_2],
                "gt_matching": gt_matching,
                "ged": torch.tensor(gt_ged, dtype=torch.float),
                "left_degree": left_degree,
                "right_degree": right_degree,
                "left_label": left_label,
                "right_label": right_label,
                "n1": torch.tensor(n1, dtype=torch.long),
                "n2": torch.tensor(n2, dtype=torch.long),
            }
        if pair_type == 1:
            delta_graphs = self.delta_graphs[id_1]
            if delta_graphs is None:
                raise KeyError("Missing synthetic delta graphs for graph index {}.".format(id_1))
            delta_data = delta_graphs[id_2]
            x1 = self.features[id_1]
            x2 = delta_data["features"]
            degree1 = self.degrees[id_1]
            degree2 = delta_data["conv_adj"].sum(dim=1)
            label2 = self._node_labels_dense(x2)
            left_degree, left_label = self._pair_side_metadata(id_1, x1, degree=degree1)
            right_degree, right_label = self._pair_side_metadata(id_2, x2, degree=degree2, labels=label2)
            return {
                "pair": torch.tensor([id_1, id_2], dtype=torch.long),
                "pair_gid": torch.tensor([self.gid[id_1], -1 - int(id_2)], dtype=torch.long),
                "x1": x1,
                "x2": x2,
                "adj1": self.conv_adj[id_1],
                "adj2": delta_data["conv_adj"],
                "ged_adj1": self.ged_adj[id_1],
                "ged_adj2": delta_data["ged_adj"],
                "gt_matching": delta_data["mapping"].float(),
                "ged": torch.tensor(float(delta_data["ta_ged"][0]), dtype=torch.float),
                "left_degree": left_degree,
                "right_degree": right_degree,
                "left_label": left_label,
                "right_label": right_label,
                "n1": torch.tensor(x1.shape[0], dtype=torch.long),
                "n2": torch.tensor(x2.shape[0], dtype=torch.long),
            }
        if pair_type != 0:
            raise ValueError("Unsupported dense graph pair type: {}".format(pair_type))

        pair_data = self.get_pair_metadata(id_1, id_2)
        if pair_data is None:
            raise KeyError("Missing graph pair metadata for ({}, {}).".format(id_1, id_2))

        x1 = self.features[id_1]
        x2 = self.features[id_2]
        left_degree, left_label = self._pair_side_metadata(id_1, x1)
        right_degree, right_label = self._pair_side_metadata(id_2, x2)
        return {
            "pair": torch.tensor([id_1, id_2], dtype=torch.long),
            "pair_gid": torch.tensor([self.gid[id_1], self.gid[id_2]], dtype=torch.long),
            "x1": x1,
            "x2": x2,
            "adj1": self.conv_adj[id_1],
            "adj2": self.conv_adj[id_2],
            "ged_adj1": self.ged_adj[id_1],
            "ged_adj2": self.ged_adj[id_2],
            "gt_matching": pair_data["mapping"].float(),
            "ged": torch.tensor(float(pair_data["ta_ged"][0]), dtype=torch.float),
            "left_degree": left_degree,
            "right_degree": right_degree,
            "left_label": left_label,
            "right_label": right_label,
            "n1": torch.tensor(x1.shape[0], dtype=torch.long),
            "n2": torch.tensor(x2.shape[0], dtype=torch.long),
        }

    def dense_collate(self, items):
        batch_size = len(items)
        feat_dim = items[0]["x1"].shape[1]
        pad_n1 = max(int(item["n1"].item()) for item in items)
        pad_n2 = max(int(item["n2"].item()) for item in items)

        batch = {
            "x1": torch.zeros((batch_size, pad_n1, feat_dim), dtype=torch.float),
            "x2": torch.zeros((batch_size, pad_n2, feat_dim), dtype=torch.float),
            "adj1": torch.zeros((batch_size, pad_n1, pad_n1), dtype=torch.float),
            "adj2": torch.zeros((batch_size, pad_n2, pad_n2), dtype=torch.float),
            "ged_adj1": torch.zeros((batch_size, pad_n1, pad_n1), dtype=torch.float),
            "ged_adj2": torch.zeros((batch_size, pad_n2, pad_n2), dtype=torch.float),
            "gt_matching": torch.zeros((batch_size, pad_n1, pad_n2), dtype=torch.float),
            "base_cost": torch.zeros((batch_size, pad_n1, pad_n2), dtype=torch.float),
            "left_degree": torch.zeros((batch_size, pad_n1), dtype=torch.float),
            "right_degree": torch.zeros((batch_size, pad_n2), dtype=torch.float),
            "left_label": torch.zeros((batch_size, pad_n1), dtype=torch.long),
            "right_label": torch.zeros((batch_size, pad_n2), dtype=torch.long),
            "ged": torch.zeros((batch_size,), dtype=torch.float),
            "n1": torch.zeros((batch_size,), dtype=torch.long),
            "n2": torch.zeros((batch_size,), dtype=torch.long),
            "pair": torch.zeros((batch_size, 2), dtype=torch.long),
            "pair_gid": torch.zeros((batch_size, 2), dtype=torch.long),
            "mask1": torch.zeros((batch_size, pad_n1), dtype=torch.bool),
            "mask2": torch.zeros((batch_size, pad_n2), dtype=torch.bool),
        }

        for batch_idx, item in enumerate(items):
            n1 = int(item["n1"].item())
            n2 = int(item["n2"].item())
            batch["x1"][batch_idx, :n1] = item["x1"]
            batch["x2"][batch_idx, :n2] = item["x2"]
            batch["adj1"][batch_idx, :n1, :n1] = item["adj1"]
            batch["adj2"][batch_idx, :n2, :n2] = item["adj2"]
            batch["ged_adj1"][batch_idx, :n1, :n1] = item["ged_adj1"]
            batch["ged_adj2"][batch_idx, :n2, :n2] = item["ged_adj2"]
            batch["gt_matching"][batch_idx, :n1, :n2] = item["gt_matching"]
            if "base_cost" in item:
                batch["base_cost"][batch_idx, :n1, :n2] = item["base_cost"]
            if "left_degree" in item:
                batch["left_degree"][batch_idx, :n1] = item["left_degree"]
            if "right_degree" in item:
                batch["right_degree"][batch_idx, :n2] = item["right_degree"]
            if "left_label" in item:
                batch["left_label"][batch_idx, :n1] = item["left_label"]
            if "right_label" in item:
                batch["right_label"][batch_idx, :n2] = item["right_label"]
            batch["ged"][batch_idx] = item["ged"]
            batch["n1"][batch_idx] = item["n1"]
            batch["n2"][batch_idx] = item["n2"]
            batch["pair"][batch_idx] = item["pair"]
            batch["pair_gid"][batch_idx] = item["pair_gid"]
            batch["mask1"][batch_idx, :n1] = True
            batch["mask2"][batch_idx, :n2] = True

        return batch

    def _compute_base_cost_on_device(self, batch):
        required = {"left_degree", "right_degree", "left_label", "right_label", "mask1", "mask2"}
        if not required.issubset(batch.keys()):
            return batch
        left_degree = batch["left_degree"]
        right_degree = batch["right_degree"]
        left_label = batch["left_label"].to(dtype=torch.long)
        right_label = batch["right_label"].to(dtype=torch.long)
        pair_mask = batch["mask1"][:, :, None] & batch["mask2"][:, None, :]

        left_sim = torch.bmm(batch["x1"], batch["x2"].transpose(1, 2))
        label_cost = (left_label[:, :, None] != right_label[:, None, :]).to(dtype=torch.float)
        degree_scale = torch.maximum(
            left_degree.max(dim=1).values,
            right_degree.max(dim=1).values,
        ).clamp(min=1.0)
        degree_cost = (left_degree[:, :, None] - right_degree[:, None, :]).abs() / degree_scale[:, None, None]
        base_cost = self.args.match_cost_scale * (
            self.args.match_label_weight * label_cost
            + self.args.match_degree_weight * degree_cost
            - self.args.match_similarity_weight * left_sim
        )
        batch["base_cost"] = base_cost.masked_fill(~pair_mask, 0.0)
        return batch

    def _move_batch_to_device(self, batch):
        moved = {key: value.to(self.device) if torch.is_tensor(value) else value for key, value in batch.items()}
        return self._compute_base_cost_on_device(moved)

    def setup_model(self):
        if self.args.denoise_network == "diffged_sparse":
            self.model = DenseSparseGNNDiffMatchDenseIO(self.args, self.number_of_labels).to(self.device)
        else:
            self.model = DenseLightGTDiffMatchDenseIO(self.args, self.number_of_labels).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )
        self.diffusion = CategoricalDiffusion(T=self.args.diffusion_steps)

    def _graph_index_pool_for_split(self, split_name):
        train_end = self.train_num
        val_end = self.train_num + self.val_num
        graph_total = len(self.graphs)
        if split_name == "train":
            return list(range(0, train_end))
        if split_name == "val":
            return list(range(train_end, val_end))
        if split_name == "test":
            return list(range(val_end, graph_total))
        return list(range(graph_total))

    def _init_k_graph_cross_product_pairs(self):
        graph_count = int(getattr(self.args, "inference_graph_count", 0))
        if graph_count <= 0:
            raise ValueError("--inference-graph-count must be > 0 when using --inference-pair-mode k_graph_cross_product.")

        split_name = str(getattr(self.args, "inference_graph_split", "train"))
        start_offset = max(0, int(getattr(self.args, "inference_graph_offset", 0)))
        source_pool = self._graph_index_pool_for_split(split_name)
        if start_offset >= len(source_pool):
            raise ValueError(
                f"--inference-graph-offset={start_offset} is out of range for split {split_name} with {len(source_pool)} graphs."
            )
        selected = source_pool[start_offset:start_offset + graph_count]
        if len(selected) < graph_count:
            raise ValueError(
                f"Requested K={graph_count} graphs from split {split_name} at offset {start_offset}, "
                f"but only {len(selected)} are available."
            )

        symmetry = str(getattr(self.args, "inference_pair_symmetry", "unique"))
        testing_specs = []
        if symmetry == "ordered":
            for left_idx in selected:
                for right_idx in selected:
                    if self.get_pair_metadata(left_idx, right_idx) is not None:
                        testing_specs.append((0, left_idx, right_idx))
        else:
            for left_pos, left_idx in enumerate(selected):
                for right_idx in selected[left_pos + 1:]:
                    if self.get_pair_metadata(left_idx, right_idx) is not None:
                        testing_specs.append((0, left_idx, right_idx))

        print(
            "[Inference][k_graph_cross_product][baseline] split={} offset={} K={} pair_symmetry={} pairs={}".format(
                split_name,
                start_offset,
                len(selected),
                symmetry,
                len(testing_specs),
            )
        )

        empty_dataset = DensePairDataset(self, [])
        self.training_graphs = empty_dataset
        self.val_graphs = empty_dataset
        self.testing_graphs = DensePairDataset(self, testing_specs)
        self.testing_graphs_small = self.testing_graphs
        self.testing_graphs_large = self.testing_graphs

    def init_graph_pairs(self):
        random.seed(1)
        if getattr(self.args, "fixed_pair_root", None):
            split_specs = {"train": [], "val": [], "test": []}
            missing = []
            for row in self.fixed_pair_rows:
                graph_1 = str(row["graph_1"])
                graph_2 = str(row["graph_2"])
                id_1 = self.fixed_graph_key_to_index.get(graph_1)
                id_2 = self.fixed_graph_key_to_index.get(graph_2)
                if id_1 is None or id_2 is None:
                    missing.append((graph_1, graph_2))
                    continue
                split_name = str(row.get("benchmark_split", "test"))
                if split_name not in split_specs:
                    raise ValueError(f"Unknown fixed-pair split: {split_name}")
                split_specs[split_name].append((2, id_1, id_2, float(row["ged"]), row.get("node_mapping_pairs", [])))
            if missing:
                raise ValueError("Fixed-pair rows reference missing graph keys: {}".format(missing[:5]))
            self.training_graphs = DensePairDataset(self, split_specs["train"])
            self.val_graphs = DensePairDataset(self, split_specs["val"])
            self.testing_graphs = DensePairDataset(self, split_specs["test"])
            self.testing_graphs_small = DensePairDataset(
                self,
                [spec for spec, row in zip(split_specs["test"], [r for r in self.fixed_pair_rows if r.get("benchmark_split", "test") == "test"]) if row.get("pair_type") == "small"],
            )
            self.testing_graphs_large = DensePairDataset(
                self,
                [spec for spec, row in zip(split_specs["test"], [r for r in self.fixed_pair_rows if r.get("benchmark_split", "test") == "test"]) if row.get("pair_type") == "large"],
            )
            print(
                "Use fixed graph pairs: train={} val={} test={} small={} large={}.".format(
                    len(self.training_graphs),
                    len(self.val_graphs),
                    len(self.testing_graphs),
                    len(self.testing_graphs_small),
                    len(self.testing_graphs_large),
                )
            )
            return
        if self.args.model_train == 0 and getattr(self.args, "inference_pair_mode", "dataset_pairs") == "k_graph_cross_product":
            self._init_k_graph_cross_product_pairs()
            return
        if self.has_precomputed_pairs:
            training_specs = []
            val_specs = []
            testing_specs = []
            missing_gids = []
            for pair_item in self.precomputed_pair_splits:
                query_gid = int(pair_item["new_query_gid"])
                db_gid = int(pair_item["new_db_gid"])
                query_idx = self.gid_to_index.get(query_gid)
                db_idx = self.gid_to_index.get(db_gid)
                if query_idx is None or db_idx is None:
                    missing_gids.append((query_gid, db_gid))
                    continue
                pair_spec = (0, query_idx, db_idx)
                split_name = pair_item["benchmark_split"]
                if split_name == "train":
                    training_specs.append(pair_spec)
                elif split_name == "val":
                    val_specs.append(pair_spec)
                elif split_name == "test":
                    testing_specs.append(pair_spec)
                else:
                    raise ValueError("Unknown benchmark split: {}".format(split_name))
            if missing_gids:
                raise ValueError("Manifest references missing graph gid pairs: {}".format(missing_gids[:5]))

            self.training_graphs = DensePairDataset(self, training_specs)
            self.val_graphs = DensePairDataset(self, val_specs)
            self.testing_graphs = DensePairDataset(self, testing_specs)
            self.testing_graphs_small = DensePairDataset(self, testing_specs)
            self.testing_graphs_large = DensePairDataset(self, testing_specs)
            return

        training_specs = []
        val_specs = []
        testing_specs = []

        train_num = self.train_num
        val_num = train_num + self.val_num
        test_num = len(self.graphs)

        for i in range(train_num):
            if self.gn[i] <= 10:
                for j in range(i, train_num):
                    if self.get_pair_metadata(i, j) is not None:
                        training_specs.append((0, i, j))
            elif self.delta_graphs[i] is not None:
                for j in range(len(self.delta_graphs[i])):
                    training_specs.append((1, i, j))
        candidate_train = [idx for idx in range(train_num) if self.gn[idx] <= 10]
        for i in range(train_num, val_num):
            if self.gn[i] <= 10:
                local = list(candidate_train)
                random.shuffle(local)
                for j in local[:self.args.num_testing_graphs]:
                    if self.get_pair_metadata(i, j) is not None:
                        val_specs.append((0, i, j))
            elif self.delta_graphs[i] is not None:
                for j in range(len(self.delta_graphs[i])):
                    val_specs.append((1, i, j))
        for i in range(val_num, test_num):
            if self.gn[i] <= 10:
                local = list(candidate_train)
                random.shuffle(local)
                for j in local[:self.args.num_testing_graphs]:
                    if self.get_pair_metadata(i, j) is not None:
                        testing_specs.append((0, i, j))
            elif self.delta_graphs[i] is not None:
                for j in range(len(self.delta_graphs[i])):
                    testing_specs.append((1, i, j))

        self.training_graphs = DensePairDataset(self, training_specs)
        self.val_graphs = DensePairDataset(self, val_specs)
        self.testing_graphs = DensePairDataset(self, testing_specs)
        self.testing_graphs_small = DensePairDataset(self, [spec for spec in testing_specs if spec[0] == 0])
        self.testing_graphs_large = DensePairDataset(self, [spec for spec in testing_specs if spec[0] == 1])

    def _make_testing_dataloader(self, dataset):
        if getattr(self.args, "test_batch_size_bucketing", False):
            batch_sampler = TestSizeBucketedBatchSampler(dataset, self.args.test_batch_size, getattr(self.args, "test_nodes", 0))
            return DataLoader(
                dataset,
                batch_sampler=batch_sampler,
                num_workers=self.args.num_workers,
                collate_fn=self.dense_collate,
            )
        return DataLoader(
            dataset,
            batch_size=self.args.test_batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            collate_fn=self.dense_collate,
        )

    def _build_dataloaders(self):
        self.training_data_loader = DataLoader(
            self.training_graphs,
            batch_size=self.args.batch_size,
            shuffle=len(self.training_graphs) > 0,
            num_workers=self.args.num_workers,
            collate_fn=self.dense_collate,
        )
        self.val_data_loader = self._make_testing_dataloader(self.val_graphs)
        self.testing_data_loader = self._make_testing_dataloader(self.testing_graphs)
        self.testing_data_small_loader = self._make_testing_dataloader(self.testing_graphs_small)
        self.testing_data_large_loader = self._make_testing_dataloader(self.testing_graphs_large)

    def dense_mapping_loss(self, pred_logits, gt_matching, pair_mask):
        loss = F.binary_cross_entropy_with_logits(pred_logits, gt_matching, reduction="none")
        valid = pair_mask.float()
        pos_count = (gt_matching * valid).sum(dim=(1, 2), keepdim=True)
        neg_count = ((1.0 - gt_matching) * valid).sum(dim=(1, 2), keepdim=True)
        pos_weight = torch.where(pos_count > 0, (neg_count / pos_count).clamp(min=1.0), torch.ones_like(pos_count))
        weights = torch.where(gt_matching > 0.5, pos_weight, torch.ones_like(gt_matching))
        weighted = loss * weights * valid
        per_pair = weighted.sum(dim=(1, 2)) / valid.sum(dim=(1, 2)).clamp_min(1.0)
        return per_pair.mean()

    def sample_partial_pair_matchings(self, pair_matchings, t):
        partial_matchings = torch.zeros_like(pair_matchings)
        for batch_idx, pair_t in enumerate(t.tolist()):
            gt_matching = pair_matchings[batch_idx]
            positive_idx = torch.nonzero(gt_matching > 0.5, as_tuple=False)
            keep_ratio = self.partialdiff_keep_ratio(pair_t)
            if str(getattr(self.args, "partialdiff_noise_mode", "fixed_count")) == "bernoulli_drop":
                # Independent one-way Bernoulli corruption: only existing 1-edges can survive.
                keep_mask = torch.rand(positive_idx.shape[0], device=gt_matching.device) < keep_ratio
                kept_edges = positive_idx[keep_mask]
                if kept_edges.numel() > 0:
                    partial_matchings[batch_idx, kept_edges[:, 0], kept_edges[:, 1]] = 1.0
                continue
            keep_count = int(round(keep_ratio * positive_idx.shape[0]))
            keep_count = min(max(keep_count, 0), positive_idx.shape[0])
            if keep_count <= 0:
                continue
            if keep_count >= positive_idx.shape[0]:
                partial_matchings[batch_idx] = gt_matching
                continue
            keep_perm = torch.randperm(positive_idx.shape[0], device=gt_matching.device)[:keep_count]
            kept_edges = positive_idx[keep_perm]
            partial_matchings[batch_idx, kept_edges[:, 0], kept_edges[:, 1]] = 1.0
        return partial_matchings

    def partialdiff_keep_ratio(self, t):
        """Return the GT-edge survival probability used by PartialDiff."""
        if str(getattr(self.args, "partialdiff_keep_schedule", "linear")) == "alpha_bar":
            return float(self.diffusion.alpha_bar[int(t)])
        return float(self.diffusion.keep_ratio(t))

    def sample_diffged_pair_matchings(self, pair_matchings, t, pair_mask):
        labels = (pair_matchings > 0.5).long()
        onehot = F.one_hot(labels, num_classes=2).float()
        batch_size = int(pair_matchings.shape[0])
        batch_ids = torch.arange(batch_size, device=pair_matchings.device, dtype=torch.long).view(-1, 1, 1)
        edge_batch = batch_ids.expand_as(labels)
        flat_valid = pair_mask.reshape(-1)
        sampled = self.diffusion.sample(
            onehot.reshape(-1, 2)[flat_valid].unsqueeze(1),
            t.detach().cpu().numpy().astype(int),
            edge_batch.reshape(-1)[flat_valid],
        )
        out = torch.zeros_like(pair_matchings)
        out.reshape(-1)[flat_valid] = sampled.to(device=pair_matchings.device, dtype=out.dtype).reshape(-1)
        return out

    @staticmethod
    def batched_constrained_matching_decode(score_matrices, full_size):
        num_pairs, _, n2 = score_matrices.shape
        work_scores = score_matrices.clone()
        decoded = torch.zeros_like(work_scores)
        batch_idx = torch.arange(num_pairs, device=work_scores.device)
        if torch.is_tensor(full_size):
            full_sizes = full_size.to(device=work_scores.device, dtype=torch.long).view(-1)
        else:
            full_sizes = torch.full((num_pairs,), int(full_size), device=work_scores.device, dtype=torch.long)
        max_steps = int(full_sizes.max().item()) if full_sizes.numel() > 0 else 0
        selected_counts = torch.zeros_like(full_sizes)

        for _ in range(max_steps):
            active = selected_counts < full_sizes
            if not bool(active.any()):
                break
            flat_scores = work_scores.view(num_pairs, -1)
            argmax_result = torch.argmax(flat_scores, dim=-1)
            best_scores = flat_scores[batch_idx, argmax_result]
            active = active & torch.isfinite(best_scores)
            if not bool(active.any()):
                break
            rows = argmax_result // n2
            cols = argmax_result % n2
            valid_batch = batch_idx[active]
            valid_rows = rows[active]
            valid_cols = cols[active]
            decoded[valid_batch, valid_rows, valid_cols] = 1.0
            work_scores[valid_batch, valid_rows, :] = float("-inf")
            work_scores[valid_batch, :, valid_cols] = float("-inf")
            selected_counts[valid_batch] += 1
        return decoded

    def batched_row_top1_unique(self, score_matrices, full_size=None):
        del full_size
        num_pairs, n1, n2 = score_matrices.shape
        decoded = torch.zeros_like(score_matrices)
        if n1 == 0 or n2 == 0:
            return decoded

        row_best_scores, row_best_cols = torch.max(score_matrices, dim=2)
        valid = torch.isfinite(row_best_scores)
        if not bool(valid.any()):
            return decoded

        col_best_scores = torch.full(
            (num_pairs, n2),
            float("-inf"),
            device=score_matrices.device,
            dtype=score_matrices.dtype,
        )
        safe_scores = row_best_scores.masked_fill(~valid, float("-inf"))
        col_best_scores.scatter_reduce_(
            1,
            row_best_cols,
            safe_scores,
            reduce="amax",
            include_self=True,
        )

        winners = valid & (row_best_scores == col_best_scores.gather(1, row_best_cols))
        if not bool(winners.any()):
            return decoded

        row_ids = torch.arange(n1, device=score_matrices.device, dtype=torch.long).unsqueeze(0).expand(num_pairs, -1)
        sentinel = torch.full_like(row_ids, n1)
        candidate_rows = torch.where(winners, row_ids, sentinel)
        col_best_rows = torch.full((num_pairs, n2), n1, device=score_matrices.device, dtype=torch.long)
        col_best_rows.scatter_reduce_(
            1,
            row_best_cols,
            candidate_rows,
            reduce="amin",
            include_self=True,
        )

        final_winners = winners & (row_ids == col_best_rows.gather(1, row_best_cols))
        batch_idx, row_idx = torch.nonzero(final_winners, as_tuple=True)
        col_idx = row_best_cols[batch_idx, row_idx]
        decoded[batch_idx, row_idx, col_idx] = 1.0
        return decoded

    def batched_col_top1_unique(self, score_matrices, full_size=None):
        decoded_t = self.batched_row_top1_unique(score_matrices.transpose(1, 2), full_size)
        return decoded_t.transpose(1, 2)

    @staticmethod
    def _uses_top1_decode_mode(mode):
        return mode in {
            "row_top1_unique_n2",
            "col_top1_unique_n2",
            "alternating_row_col_top1_n2",
            "alternating_col_row_top1_n2",
        }

    @staticmethod
    def _effective_top1_decode_mode(mode, step_idx=None):
        if mode == "alternating_row_col_top1_n2":
            if step_idx is None or int(step_idx) % 2 == 0:
                return "row_top1_unique_n2"
            return "col_top1_unique_n2"
        if mode == "alternating_col_row_top1_n2":
            if step_idx is None or int(step_idx) % 2 == 0:
                return "col_top1_unique_n2"
            return "row_top1_unique_n2"
        return mode

    def batched_constrained_matching_decode_by_mode(self, score_matrices, full_size, step_idx=None):
        mode = str(getattr(self.args, "constrained_greedy_mode", "global_n3"))
        effective_mode = self._effective_top1_decode_mode(mode, step_idx=step_idx)
        if effective_mode == "row_top1_unique_n2":
            return self.batched_row_top1_unique(score_matrices, full_size)
        if effective_mode == "col_top1_unique_n2":
            return self.batched_col_top1_unique(score_matrices, full_size)
        return self.batched_constrained_matching_decode(score_matrices, full_size)

    def batched_blockwise_autoregressive_decode(
        self,
        score_matrices,
        fixed_matchings,
        full_size,
        step_idx=None,
        fixed_mask=None,
        fixed_counts=None,
    ):
        if fixed_matchings.shape != score_matrices.shape:
            raise ValueError("fixed_matchings and score_matrices must share the same shape.")

        if fixed_mask is None:
            fixed_mask = fixed_matchings > 0.5
        if fixed_counts is None:
            fixed_counts = fixed_mask.reshape(fixed_mask.shape[0], -1).sum(dim=1).to(dtype=torch.long)
        remaining_sizes = (full_size.to(device=score_matrices.device, dtype=torch.long) - fixed_counts).clamp(min=0)
        if not bool((remaining_sizes > 0).any()):
            return fixed_mask.to(dtype=score_matrices.dtype)

        row_taken = fixed_mask.any(dim=2)
        col_taken = fixed_mask.any(dim=1)
        locked_mask = row_taken.unsqueeze(2) | col_taken.unsqueeze(1)
        work_scores = score_matrices.masked_fill(locked_mask, float("-inf"))

        decoded_remaining = self.batched_constrained_matching_decode_by_mode(
            work_scores,
            remaining_sizes,
            step_idx=step_idx,
        )
        merged = fixed_mask.to(dtype=score_matrices.dtype) + decoded_remaining
        return merged.clamp_(min=0.0, max=1.0)

    @staticmethod
    def validate_blockwise_autoregressive_step(previous_partial, next_partial, previous_mask=None, next_mask=None):
        if previous_mask is None:
            previous_mask = previous_partial > 0.5
        if next_mask is None:
            next_mask = next_partial > 0.5

        if bool((previous_mask & ~next_mask).any()):
            raise RuntimeError("blockwise_autoregressive decode violated monotonicity: an anchored match disappeared.")

        if bool((next_mask.sum(dim=2) > 1).any()):
            raise RuntimeError("blockwise_autoregressive decode produced an invalid matching: a row was matched multiple times.")

        if bool((next_mask.sum(dim=1) > 1).any()):
            raise RuntimeError("blockwise_autoregressive decode produced an invalid matching: a column was matched multiple times.")

    @staticmethod
    def batched_repair_matching_completion(score_matrices, partial_matchings, full_sizes):
        num_pairs, _, n2 = score_matrices.shape
        if num_pairs == 0:
            return partial_matchings

        decoded = partial_matchings.clone()
        selected = decoded > 0.5
        selected_counts = selected.reshape(num_pairs, -1).sum(dim=1)
        full_sizes = full_sizes.to(device=score_matrices.device, dtype=torch.long)
        work_scores = score_matrices.clone()
        row_taken = selected.any(dim=2)
        col_taken = selected.any(dim=1)
        work_scores = work_scores.masked_fill(row_taken.unsqueeze(2), float("-inf"))
        work_scores = work_scores.masked_fill(col_taken.unsqueeze(1), float("-inf"))

        batch_idx = torch.arange(num_pairs, device=score_matrices.device)
        max_steps = int(full_sizes.max().item()) if full_sizes.numel() > 0 else 0
        for _ in range(max_steps):
            active = selected_counts < full_sizes
            if not bool(active.any()):
                break
            flat_scores = work_scores.view(num_pairs, -1)
            argmax_result = torch.argmax(flat_scores, dim=-1)
            best_scores = flat_scores[batch_idx, argmax_result]
            active = active & torch.isfinite(best_scores)
            if not bool(active.any()):
                break
            rows = argmax_result // n2
            cols = argmax_result % n2
            valid_batch = batch_idx[active]
            valid_rows = rows[active]
            valid_cols = cols[active]
            decoded[valid_batch, valid_rows, valid_cols] = 1.0
            work_scores[valid_batch, valid_rows, :] = float("-inf")
            work_scores[valid_batch, :, valid_cols] = float("-inf")
            selected_counts[valid_batch] += 1
        return decoded

    @staticmethod
    def batched_bernoulli_drop_new_matchings(new_matchings, keep_probabilities):
        """Independently retain only existing new 1-edges; never creates 0->1 edges."""
        keep_probabilities = keep_probabilities.to(device=new_matchings.device, dtype=new_matchings.dtype).clamp_(0.0, 1.0)
        draws = torch.rand_like(new_matchings) < keep_probabilities.view(-1, 1, 1)
        return ((new_matchings > 0.5) & draws).to(dtype=new_matchings.dtype)

    def batched_sample_partial_from_clean_matching_variable_size(self, clean_matchings, score_matrices, target_sizes):
        num_pairs, n1, n2 = clean_matchings.shape
        target_sizes = target_sizes.to(device=clean_matchings.device, dtype=torch.long).clamp(min=0)
        flat_clean = clean_matchings.reshape(num_pairs, -1) > 0.5
        flat_scores = score_matrices.reshape(num_pairs, -1)
        partial = torch.zeros_like(flat_scores)
        available_counts = flat_clean.sum(dim=1)
        target_sizes = torch.minimum(target_sizes, available_counts)
        if not bool((target_sizes > 0).any()):
            return partial.view(num_pairs, n1, n2)

        work_scores = flat_scores.masked_fill(~flat_clean, float("-inf"))
        selected_counts = torch.zeros_like(target_sizes)
        batch_idx = torch.arange(num_pairs, device=clean_matchings.device)
        max_steps = int(target_sizes.max().item())

        for _ in range(max_steps):
            active = selected_counts < target_sizes
            if not bool(active.any()):
                break
            if self.args.renoise_mode == "topk":
                chosen_idx = torch.argmax(work_scores, dim=1)
                chosen_scores = work_scores[batch_idx, chosen_idx]
                active = active & torch.isfinite(chosen_scores)
            else:
                scaled_scores = work_scores / max(float(self.args.renoise_temperature), 1e-6)
                scaled_scores = scaled_scores.masked_fill(~torch.isfinite(work_scores), float("-inf"))
                weights = torch.softmax(scaled_scores, dim=1)
                active = active & (weights.sum(dim=1) > 0)
                if not bool(active.any()):
                    break
                chosen_idx = torch.zeros((num_pairs,), device=clean_matchings.device, dtype=torch.long)
                chosen_idx[active] = torch.multinomial(weights[active], 1, replacement=False).view(-1)
            if not bool(active.any()):
                break
            active_batches = batch_idx[active]
            active_indices = chosen_idx[active]
            partial[active_batches, active_indices] = 1.0
            work_scores[active_batches, active_indices] = float("-inf")
            selected_counts[active_batches] += 1

        return partial.view(num_pairs, n1, n2)

    @staticmethod
    def _stable_pair_sample_seed(pair_gid_a, pair_gid_b, sample_slot, step_idx, base_seed=0):
        modulus = (1 << 63) - 1
        seed = int(base_seed) % modulus
        seed = (seed * 1000003 + int(pair_gid_a)) % modulus
        seed = (seed * 1000033 + int(pair_gid_b)) % modulus
        seed = (seed * 1000037 + int(sample_slot)) % modulus
        seed = (seed * 1000211 + int(step_idx)) % modulus
        if seed <= 0:
            seed += 1
        return seed

    def _stable_pair_sample_seed_tensor(self, pair_gid, sample_slot, step_idx):
        modulus = (1 << 63) - 1
        seed = torch.full(
            (pair_gid.shape[0],),
            int(getattr(self.args, "seed", 0)) % modulus,
            device=pair_gid.device,
            dtype=torch.long,
        )
        seed = (seed * 1000003 + pair_gid[:, 0].long()) % modulus
        seed = (seed * 1000033 + pair_gid[:, 1].long()) % modulus
        seed = (seed * 1000037 + sample_slot.long()) % modulus
        seed = (seed * 1000211 + int(step_idx)) % modulus
        seed = torch.where(seed <= 0, seed + 1, seed)
        return seed

    def _deterministic_gumbel(self, pair_gid, sample_slot, step_idx, num_cols, device, dtype):
        modulus = 2147483647
        row_seed = (self._stable_pair_sample_seed_tensor(pair_gid, sample_slot, step_idx) % modulus).view(-1, 1)
        col_ids = torch.arange(num_cols, device=device, dtype=torch.long).view(1, -1) + 1
        hashed = (row_seed + col_ids * 104729 + 12345) % modulus
        hashed = (hashed * 48271) % modulus
        hashed = torch.bitwise_xor(hashed, torch.bitwise_right_shift(hashed, 11)) % modulus
        hashed = (hashed * 69621 + 1) % modulus
        uniform = (hashed.to(torch.float64) + 0.5) / float(modulus)
        uniform = uniform.clamp_(1e-6, 1.0 - 1e-6).to(dtype=dtype)
        return -torch.log(-torch.log(uniform))

    def batched_sample_partial_from_clean_matching_variable_size_seeded(
        self,
        clean_matchings,
        score_matrices,
        target_sizes,
        pair_gid,
        sample_slot,
        step_idx,
    ):
        num_pairs, n1, n2 = clean_matchings.shape
        target_sizes = target_sizes.to(device=clean_matchings.device, dtype=torch.long).clamp(min=0)
        flat_clean = clean_matchings.reshape(num_pairs, -1) > 0.5
        flat_scores = score_matrices.reshape(num_pairs, -1)
        partial = torch.zeros_like(flat_scores)
        available_counts = flat_clean.sum(dim=1)
        target_sizes = torch.minimum(target_sizes, available_counts)
        if not bool((target_sizes > 0).any()):
            return partial.view(num_pairs, n1, n2)

        work_scores = flat_scores.masked_fill(~flat_clean, float("-inf"))
        selected_counts = torch.zeros_like(target_sizes)
        batch_idx = torch.arange(num_pairs, device=clean_matchings.device)
        max_steps = int(target_sizes.max().item())

        if self.args.renoise_mode == "topk":
            topk_idx = torch.topk(work_scores, k=max_steps, dim=1).indices
        else:
            logits = work_scores / max(float(self.args.renoise_temperature), 1e-6)
            gumbel = self._deterministic_gumbel(
                pair_gid,
                sample_slot,
                step_idx,
                logits.shape[1],
                clean_matchings.device,
                logits.dtype,
            )
            topk_idx = torch.topk(logits + gumbel, k=max_steps, dim=1).indices

        rank_ids = torch.arange(max_steps, device=clean_matchings.device, dtype=torch.long).view(1, -1)
        select_mask = rank_ids < target_sizes.view(-1, 1)
        row_ids = batch_idx.view(-1, 1).expand(-1, max_steps)[select_mask]
        col_ids = topk_idx[select_mask]
        partial[row_ids, col_ids] = 1.0

        return partial.view(num_pairs, n1, n2)

    def categorical_posterior_dense(self, target_t, t, x0_pred_prob, xt, pair_mask):
        if target_t is None:
            target_t = t - 1
        else:
            target_t = torch.full_like(t, int(target_t))
        t = t.to(device=x0_pred_prob.device, dtype=torch.long)
        target_t = target_t.to(device=x0_pred_prob.device, dtype=torch.long)

        valid = pair_mask.reshape(-1)
        if not bool(valid.any()):
            return torch.zeros_like(xt)

        batch_size = int(x0_pred_prob.shape[0])
        batch_ids = torch.arange(batch_size, device=x0_pred_prob.device, dtype=torch.long).view(-1, 1, 1)
        edge_batch = batch_ids.expand_as(xt).reshape(-1)[valid]

        q_bar = self.diffusion.Q_bar
        q_t_np = np.linalg.inv(q_bar[target_t.detach().cpu().numpy()]) @ q_bar[t.detach().cpu().numpy()]
        q_t = torch.from_numpy(q_t_np).float().to(x0_pred_prob.device)
        q_bar_source = torch.from_numpy(q_bar[t.detach().cpu().numpy()]).float().to(x0_pred_prob.device)
        q_bar_target = torch.from_numpy(q_bar[target_t.detach().cpu().numpy()]).float().to(x0_pred_prob.device)

        flat_x0 = x0_pred_prob.reshape(-1, 2)[valid]
        flat_xt = F.one_hot(xt.reshape(-1)[valid].long(), num_classes=2).float()
        q_t_edge = q_t[edge_batch]
        source_edge = q_bar_source[edge_batch]
        target_edge = q_bar_target[edge_batch]

        x_t_target_prob_part_1 = torch.bmm(flat_xt.unsqueeze(1), q_t_edge.transpose(1, 2)).squeeze(1)
        denom_0 = (source_edge[:, 0] * flat_xt).sum(dim=-1, keepdim=True).clamp_min(1e-12)
        denom_1 = (source_edge[:, 1] * flat_xt).sum(dim=-1, keepdim=True).clamp_min(1e-12)
        x_t_target_prob_0 = (x_t_target_prob_part_1 * target_edge[:, 0]) / denom_0
        x_t_target_prob_1 = (x_t_target_prob_part_1 * target_edge[:, 1]) / denom_1
        class_one_prob = x_t_target_prob_0[:, 1] * flat_x0[:, 0] + x_t_target_prob_1[:, 1] * flat_x0[:, 1]
        class_one_prob = class_one_prob.clamp(0, 1)

        if bool((target_t[edge_batch] > 0).any()):
            sampled = torch.bernoulli(class_one_prob)
            mixed = torch.where(target_t[edge_batch] > 0, sampled, class_one_prob)
        else:
            mixed = class_one_prob
        out = torch.zeros_like(xt)
        out.reshape(-1)[valid] = mixed.to(dtype=out.dtype)
        return out

    def _expand_dense_batch(self, batch, repeat_count):
        expanded = {}
        for key, value in batch.items():
            if not torch.is_tensor(value):
                expanded[key] = value
                continue
            if value.dim() == 0:
                expanded[key] = value
            else:
                expanded[key] = value.repeat_interleave(repeat_count, dim=0)
        pair_count = int(batch["pair"].shape[0])
        expanded["sample_slot"] = torch.arange(repeat_count, dtype=torch.long).repeat(pair_count)
        return expanded

    def dense_ged_from_clean_matchings_direct(self, expanded_batch, clean_matchings):
        matchings = clean_matchings > 0.5
        matchings_f = matchings.to(dtype=expanded_batch["ged_adj1"].dtype)

        matched_count = matchings_f.sum(dim=(1, 2))

        if "left_label" in expanded_batch and "right_label" in expanded_batch:
            left_labels = expanded_batch["left_label"].to(device=matchings.device, dtype=torch.long)
            right_labels = expanded_batch["right_label"].to(device=matchings.device, dtype=torch.long)
        else:
            pair_indices = expanded_batch.get("pair") if isinstance(expanded_batch, dict) else None
            if torch.is_tensor(pair_indices) and pair_indices.dim() == 2 and pair_indices.size(1) >= 2:
                left_labels = self._node_labels_from_graph_indices(pair_indices[:, 0], expanded_batch["x1"])
                right_labels = self._node_labels_from_graph_indices(pair_indices[:, 1], expanded_batch["x2"])
            else:
                left_labels = self._node_labels_dense(expanded_batch["x1"])
                right_labels = self._node_labels_dense(expanded_batch["x2"])
        label_mismatch = left_labels[:, :, None] != right_labels[:, None, :]
        node_sub_cost = (matchings & label_mismatch).sum(dim=(1, 2)).to(dtype=torch.float)

        node_cost = (
            expanded_batch["n1"].to(dtype=torch.float) - matched_count
            + expanded_batch["n2"].to(dtype=torch.float) - matched_count
            + node_sub_cost
        )

        left_adj = expanded_batch["ged_adj1"] > 0.5
        right_adj = expanded_batch["ged_adj2"] > 0.5
        left_upper = torch.triu(left_adj, diagonal=1)
        right_upper = torch.triu(right_adj, diagonal=1)
        left_edge_count = left_upper.sum(dim=(1, 2)).to(dtype=torch.float)
        right_edge_count = right_upper.sum(dim=(1, 2)).to(dtype=torch.float)

        # For a valid one-to-one partial matching M, M @ A2 @ M^T marks left-side
        # edges whose matched right endpoints are adjacent. The number of preserved
        # matched edges is the same whether counted on the left graph or the right
        # graph, so one mapped adjacency is enough to derive both delete and insert.
        mapped_right_adj = torch.bmm(torch.bmm(matchings_f, expanded_batch["ged_adj2"]), matchings_f.transpose(1, 2)) > 0.5
        preserved_edge_count = (left_upper & mapped_right_adj).sum(dim=(1, 2)).to(dtype=torch.float)

        edge_delete_cost = left_edge_count - preserved_edge_count
        edge_insert_cost = right_edge_count - preserved_edge_count
        return node_cost + edge_delete_cost + edge_insert_cost

    @staticmethod
    def _dense_matching_to_row_mapping(matching):
        row_mapping = [None] * int(matching.shape[0])
        matched_pairs = []
        rows, cols = torch.nonzero(matching > 0.5, as_tuple=True)
        for row_idx, col_idx in zip(rows.tolist(), cols.tolist()):
            row_idx = int(row_idx)
            col_idx = int(col_idx)
            row_mapping[row_idx] = col_idx
            matched_pairs.append([row_idx, col_idx])
        return row_mapping, matched_pairs

    @staticmethod
    def _comparison_flags(produced_ged, reference_ged):
        if reference_ged is None:
            return "unknown", False, False, False
        if produced_ged < reference_ged:
            return "better", True, False, False
        if produced_ged == reference_ged:
            return "equal", False, True, False
        return "worse", False, False, True

    @staticmethod
    def _edge_count_from_adj(adj):
        if adj is None:
            return None
        adj_cpu = adj.detach().cpu() if hasattr(adj, "detach") else torch.as_tensor(adj)
        return int(torch.triu(adj_cpu > 0.5, diagonal=1).sum().item())

    def _edge_count_for_graph(self, graph_id):
        graph_id = int(graph_id)
        if hasattr(self, "ged_edge_counts") and 0 <= graph_id < len(self.ged_edge_counts):
            return int(self.ged_edge_counts[graph_id])
        return self._edge_count_from_adj(self.ged_adj[graph_id])

    def _is_ogb_dataset(self):
        return str(self.args.dataset).startswith("ogbg-")

    def _partialdiff_budget_parameters(self, test_k, top_k_approach):
        return {
            "test_k": int(test_k),
            "top_k_approach": str(top_k_approach),
            "denoise_network": str(getattr(self.args, "denoise_network", "")),
            "diffusion_mechanism": str(getattr(self.args, "diffusion_mechanism", "")),
            "inference_diffusion_steps": int(getattr(self.args, "inference_diffusion_steps", 0)),
            "dense_topk_enable": bool(getattr(self.args, "dense_topk_enable", False)),
            "dense_topk_start_layer": int(getattr(self.args, "dense_topk_start_layer", 0)),
            "dense_topk_row": int(getattr(self.args, "dense_topk_row", 0)),
            "dense_topk_col": int(getattr(self.args, "dense_topk_col", 0)),
            "dense_topk_score_source": str(getattr(self.args, "dense_topk_score_source", "")),
            "dense_topk_force_current_matching": int(getattr(self.args, "dense_topk_force_current_matching", 0)),
            "reverse_decode_mode": str(getattr(self.args, "reverse_decode_mode", "")),
            "constrained_greedy_mode": str(getattr(self.args, "constrained_greedy_mode", "")),
            "renoise_mode": str(getattr(self.args, "renoise_mode", "")),
            "score_calibration_mode": str(getattr(self.args, "score_calibration_mode", "")),
            "max_test_pairs": int(getattr(self.args, "max_test_pairs", 0)),
        }

    def _raw_pair_record(
        self,
        method_name,
        graph_1,
        graph_2,
        graph_1_gid,
        graph_2_gid,
        produced_ged,
        reference_ged,
        solver_time,
        total_time,
        n1,
        n2,
        m1,
        m2,
        config_name,
        budget_parameters,
        matching,
        status="success",
        timeout=False,
        extra_fields=None,
    ):
        produced_ged = float(produced_ged)
        reference_value = None if reference_ged is None else float(reference_ged)
        comparison, better, equal, worse = self._comparison_flags(produced_ged, reference_value)
        record = {
            "dataset": self.args.dataset,
            "pair_id": "{}_{}".format(int(graph_1_gid), int(graph_2_gid)),
            "method": method_name,
            "produced_ged": produced_ged,
            "reference_ged": reference_value,
            "gt_ged": None if self._is_ogb_dataset() else reference_value,
            "solver_time": float(solver_time),
            "total_time": float(total_time),
            "status": status,
            "timeout": bool(timeout),
            "n1": int(n1),
            "n2": int(n2),
            "m1": None if m1 is None else int(m1),
            "m2": None if m2 is None else int(m2),
            "comparison": comparison,
            "better": better,
            "equal": equal,
            "worse": worse,
            "config_name": config_name,
            "budget_parameters": budget_parameters,
            "matching": matching,
            "graph_1": int(graph_1),
            "graph_2": int(graph_2),
            "graph_1_gid": int(graph_1_gid),
            "graph_2_gid": int(graph_2_gid),
        }
        if extra_fields:
            record.update(extra_fields)
        return record

    @staticmethod
    def cal_pk(num, pre, gt):
        num = min(num, len(pre))
        if num == 0:
            return 0
        tmp = list(zip(gt, pre))
        tmp.sort()
        beta = []
        for i, p in enumerate(tmp):
            beta.append((p[1], p[0], i))
        beta.sort()
        ans = 0
        for i in range(num):
            if beta[i][2] < num:
                ans += 1
        return ans / num

    def fit(self):
        self.model.train()
        self.reset_timing_stats()
        iterator = tqdm(self.training_data_loader, file=sys.stdout, dynamic_ncols=True) if not self.args.disable_tqdm else self.training_data_loader
        loss_sum = 0.0
        pair_count = 0

        for batch_idx, batch in enumerate(iterator):
            if self.args.max_train_batches > 0 and batch_idx >= self.args.max_train_batches:
                break
            batch = self._move_batch_to_device(batch)
            batch_size = int(batch["x1"].shape[0])
            t = torch.randint(1, self.diffusion.T + 1, (batch_size,), device=self.device)
            pair_mask = batch["mask1"][:, :, None] & batch["mask2"][:, None, :]
            if self.args.diffusion_mechanism == "diffged":
                partial = self.sample_diffged_pair_matchings(batch["gt_matching"], t, pair_mask)
            else:
                partial = self.sample_partial_pair_matchings(batch["gt_matching"], t)

            self.optimizer.zero_grad()
            pred_logits = self.model(batch, partial, t.float())
            loss = self.dense_mapping_loss(pred_logits, batch["gt_matching"], pair_mask)
            loss.backward()
            self.optimizer.step()

            loss_sum += float(loss.item()) * batch_size
            pair_count += batch_size
            if iterator is not self.training_data_loader:
                iterator.set_description("Epoch_{} loss={:.4f}".format(self.cur_epoch + 1, loss_sum / max(pair_count, 1)))

        training_loss = loss_sum / max(pair_count, 1)
        print("Training epoch {}\tloss={:.6f}".format(self.cur_epoch + 1, training_loss))

    def diffusion_ged_diffged_dense(self, batch, test_k=100, collect_pair_outputs=True):
        start_time = self.start_timer()
        batch_size = int(batch["x1"].shape[0])
        num_parallel_sampling = int(test_k)
        expanded_batch = self._expand_dense_batch(batch, num_parallel_sampling)
        pair_mask = expanded_batch["mask1"][:, :, None] & expanded_batch["mask2"][:, None, :]
        mapping_t = (torch.randn_like(expanded_batch["base_cost"], device=self.device) > 0).float()
        mapping_t = mapping_t.masked_fill(~pair_mask, 0.0)
        max_matching_size = torch.minimum(expanded_batch["n1"], expanded_batch["n2"]).long()

        steps = self.args.inference_diffusion_steps
        time_schedule = InferenceSchedule(T=self.diffusion.T, inference_T=steps)
        pad_n1 = int(expanded_batch["x1"].shape[1])
        pad_n2 = int(expanded_batch["x2"].shape[1])
        final_pred_probs = None
        final_scores = None

        for step_idx in range(steps):
            t1, t2 = time_schedule(step_idx)
            step_t = torch.full((batch_size * num_parallel_sampling,), int(t1), device=self.device, dtype=torch.long)
            with torch.no_grad():
                with self._eval_autocast_context():
                    pred_logits = self.model(expanded_batch, mapping_t, step_t.float())
            pred_probs = torch.sigmoid(pred_logits)
            pred_pair_probs = torch.stack([1.0 - pred_probs, pred_probs], dim=-1)
            mapping_t = self.categorical_posterior_dense(
                target_t=t2,
                t=step_t,
                x0_pred_prob=pred_pair_probs,
                xt=mapping_t,
                pair_mask=pair_mask,
            )
            mapping_t = mapping_t.masked_fill(~pair_mask, 0.0)
            final_pred_probs = pred_probs
            final_scores = mapping_t.masked_fill(~pair_mask, float("-inf"))

        decode_scores = mapping_t.masked_fill(~pair_mask, float("-inf"))
        flat_matchings = self.batched_constrained_matching_decode_by_mode(decode_scores, max_matching_size)
        mode_now = str(getattr(self.args, "constrained_greedy_mode", "global_n3"))
        if mode_now == "row_top1_unique_n2":
            flat_matchings = self.batched_repair_matching_completion(
                decode_scores,
                flat_matchings,
                max_matching_size,
            )
        flat_sample_ged = self.dense_ged_from_clean_matchings_direct(expanded_batch, flat_matchings)
        sample_ged = flat_sample_ged.view(batch_size, num_parallel_sampling)
        best_offsets = torch.argmin(sample_ged, dim=1)
        best_values = sample_ged.gather(1, best_offsets.unsqueeze(1)).squeeze(1)

        elapsed = self.stop_timer(start_time)
        per_pair_elapsed = elapsed / max(batch_size, 1)
        if not collect_pair_outputs:
            return {
                "pred_ged": best_values,
                "running_time": torch.full(
                    (batch_size,),
                    float(per_pair_elapsed),
                    device=self.device,
                    dtype=torch.float,
                ),
            }

        final_matchings = flat_matchings.view(batch_size, num_parallel_sampling, pad_n1, pad_n2)
        final_probabilities = final_pred_probs.view(batch_size, num_parallel_sampling, pad_n1, pad_n2)
        final_score_tensor = final_scores.view(batch_size, num_parallel_sampling, pad_n1, pad_n2)
        results = []
        for pair_idx in range(batch_size):
            n1 = int(batch["n1"][pair_idx].item())
            n2 = int(batch["n2"][pair_idx].item())
            best_matching = final_matchings[pair_idx, int(best_offsets[pair_idx].item()), :n1, :n2].unsqueeze(0)
            metadata = {"postprocess_time": 0.0}
            if getattr(self.args, "save_test_k_candidates", False) or getattr(self.args, "app_bmao_postprocess_enable", False):
                candidate_blob = {
                    "pair": batch["pair"][pair_idx].detach().cpu().clone(),
                    "pair_gid": batch["pair_gid"][pair_idx].detach().cpu().clone(),
                    "n1": n1,
                    "n2": n2,
                    "gt_ged": float(batch["ged"][pair_idx].item()),
                    "candidate_ged": sample_ged[pair_idx].detach().cpu().clone(),
                    "best_index": int(best_offsets[pair_idx].item()),
                    "final_probabilities": final_probabilities[pair_idx, :, :n1, :n2].detach().cpu().clone(),
                    "final_matchings": final_matchings[pair_idx, :, :n1, :n2].detach().cpu().clone(),
                    "final_scores": final_score_tensor[pair_idx, :, :n1, :n2].detach().cpu().clone(),
                }
                if getattr(self.args, "save_test_k_candidates", False):
                    candidate_blob.update({
                        "x1": batch["x1"][pair_idx, :n1].detach().cpu().clone(),
                        "x2": batch["x2"][pair_idx, :n2].detach().cpu().clone(),
                        "ged_adj1": batch["ged_adj1"][pair_idx, :n1, :n1].detach().cpu().clone(),
                        "ged_adj2": batch["ged_adj2"][pair_idx, :n2, :n2].detach().cpu().clone(),
                        "gt_matching": batch["gt_matching"][pair_idx, :n1, :n2].detach().cpu().clone(),
                    })
                metadata["all_candidates"] = candidate_blob
            results.append((best_values[pair_idx], best_matching, per_pair_elapsed, None, metadata))
        return results[0] if batch_size == 1 else results

    def diffusion_ged_dense(self, batch, test_k=100, collect_pair_outputs=True):
        if self.args.diffusion_mechanism == "diffged":
            return self.diffusion_ged_diffged_dense(batch, test_k=test_k, collect_pair_outputs=collect_pair_outputs)
        start_time = self.start_timer()
        batch_size = int(batch["x1"].shape[0])
        num_parallel_sampling = int(test_k)
        expanded_batch = self._expand_dense_batch(batch, num_parallel_sampling)
        expanded_n1 = expanded_batch["n1"]
        expanded_n2 = expanded_batch["n2"]
        pair_mask = expanded_batch["mask1"][:, :, None] & expanded_batch["mask2"][:, None, :]
        mapping_t = torch.zeros_like(expanded_batch["base_cost"])
        max_matching_size = torch.minimum(expanded_n1, expanded_n2).long()

        steps = self.args.inference_diffusion_steps
        time_schedule = InferenceSchedule(T=self.diffusion.T, inference_T=steps)
        pad_n1 = int(expanded_batch["x1"].shape[1])
        pad_n2 = int(expanded_batch["x2"].shape[1])
        final_pred_probs = None
        final_adjusted_scores = None
        reverse_decode_mode = str(getattr(self.args, "reverse_decode_mode", "constrained"))

        for step_idx in range(steps):
            t1, t2 = time_schedule(step_idx)
            step_t = torch.full((batch_size * num_parallel_sampling,), float(t1), device=self.device)
            with torch.no_grad():
                with self._eval_autocast_context():
                    pred_logits = self.model(expanded_batch, mapping_t, step_t)
            pred_probs = torch.sigmoid(pred_logits)
            raw_scores = torch.logit(pred_probs.clamp(1e-6, 1 - 1e-6))
            if str(getattr(self.args, "score_calibration_mode", "calibrated")) == "raw":
                adjusted_scores = raw_scores
            else:
                adjusted_scores = raw_scores - expanded_batch["base_cost"]
            adjusted_scores = adjusted_scores.masked_fill(~pair_mask, float("-inf"))
            final_pred_probs = pred_probs
            final_adjusted_scores = adjusted_scores

            previous_partial = mapping_t.masked_fill(~pair_mask, 0.0)
            previous_mask = previous_partial > 0.5
            previous_counts = previous_mask.reshape(previous_mask.shape[0], -1).sum(dim=1).to(dtype=torch.long)
            blockwise_new_matchings = None
            clean_matching_mask = None
            if reverse_decode_mode == "none":
                clean_matchings = pred_probs.masked_fill(~pair_mask, 0.0)
            elif reverse_decode_mode == "blockwise_autoregressive":
                mode_now = str(getattr(self.args, "constrained_greedy_mode", "global_n3"))
                decode_start = self.start_timer()
                clean_matchings = self.batched_blockwise_autoregressive_decode(
                    adjusted_scores,
                    previous_partial,
                    max_matching_size,
                    step_idx=step_idx,
                    fixed_mask=previous_mask,
                    fixed_counts=previous_counts,
                )
                decode_elapsed = self.stop_timer(decode_start)
                self.timing_totals["partialdiff_reverse_decode_total"] += decode_elapsed
                self.timing_counts["partialdiff_reverse_decode_total"] += 1
                self.timing_totals["partialdiff_reverse_decode_" + mode_now] += decode_elapsed
                self.timing_counts["partialdiff_reverse_decode_" + mode_now] += 1
                clean_matching_mask = clean_matchings > 0.5
                blockwise_new_matchings = (clean_matching_mask & ~previous_mask).to(dtype=adjusted_scores.dtype)
            else:
                mode_now = str(getattr(self.args, "constrained_greedy_mode", "global_n3"))
                decode_start = self.start_timer()
                clean_matchings = self.batched_constrained_matching_decode_by_mode(
                    adjusted_scores,
                    max_matching_size,
                    step_idx=step_idx,
                )
                clean_matching_mask = clean_matchings > 0.5
                decode_elapsed = self.stop_timer(decode_start)
                self.timing_totals["partialdiff_reverse_decode_total"] += decode_elapsed
                self.timing_counts["partialdiff_reverse_decode_total"] += 1
                self.timing_totals["partialdiff_reverse_decode_" + mode_now] += decode_elapsed
                self.timing_counts["partialdiff_reverse_decode_" + mode_now] += 1
                if self._uses_top1_decode_mode(mode_now):
                    selected_counts = clean_matching_mask.reshape(clean_matching_mask.shape[0], -1).sum(dim=1).float()
                    valid_full_sizes = max_matching_size.float().clamp_min(1.0)
                    part_ratios = selected_counts / valid_full_sizes
                    valid_pairs = max_matching_size > 0
                    if bool(valid_pairs.any()):
                        ratio_sum = float(part_ratios[valid_pairs].sum().item())
                        ratio_count = int(valid_pairs.sum().item())
                        self.timing_totals["partialdiff_reverse_rowtop1_part_ratio_sum"] += ratio_sum
                        self.timing_counts["partialdiff_reverse_rowtop1_part_ratio_sum"] += ratio_count
                        step_ratio_key = "partialdiff_reverse_rowtop1_part_ratio_step_{}_sum".format(step_idx)
                        self.timing_totals[step_ratio_key] += ratio_sum
                        self.timing_counts[step_ratio_key] += ratio_count
                        repair_needed = (max_matching_size.float() - selected_counts).clamp_min(0.0)
                        repair_needed_sum = float(repair_needed[valid_pairs].sum().item())
                        repair_needed_key = "partialdiff_reverse_rowtop1_repair_needed_step_{}_sum".format(step_idx)
                        self.timing_totals[repair_needed_key] += repair_needed_sum
                        self.timing_counts[repair_needed_key] += ratio_count
                        repair_needed_bins = torch.clamp(
                            repair_needed[valid_pairs].long(),
                            min=0,
                            max=20,
                        )
                        repair_needed_counts = torch.bincount(repair_needed_bins, minlength=21)
                        for hist_idx, hist_count in enumerate(repair_needed_counts.tolist()):
                            hist_key = "partialdiff_reverse_rowtop1_repair_needed_step_{}_hist_missing_{:02d}_count".format(
                                step_idx,
                                hist_idx,
                            )
                            self.timing_totals[hist_key] += int(hist_count)
                            self.timing_counts[hist_key] += 1
                        hist_bins = torch.clamp(
                            torch.floor(part_ratios[valid_pairs] * 20.0).long(),
                            min=0,
                            max=19,
                        )
                        hist_counts = torch.bincount(hist_bins, minlength=20)
                        for hist_idx, hist_count in enumerate(hist_counts.tolist()):
                            hist_key = "partialdiff_reverse_rowtop1_part_ratio_step_{}_hist_bin_{:02d}_count".format(
                                step_idx,
                                hist_idx,
                            )
                            self.timing_totals[hist_key] += int(hist_count)
                            self.timing_counts[hist_key] += 1
                    if bool(getattr(self.args, "diagnostic_rowtop1_global_overlap", False)):
                        global_matchings = self.batched_constrained_matching_decode(
                            adjusted_scores,
                            max_matching_size,
                        )
                        rowtop_selected = clean_matching_mask
                        global_selected = global_matchings > 0.5
                        overlap_counts = (rowtop_selected & global_selected).reshape(clean_matchings.shape[0], -1).sum(dim=1).float()
                        global_counts = global_selected.reshape(global_matchings.shape[0], -1).sum(dim=1).float()
                        union_counts = (rowtop_selected | global_selected).reshape(clean_matchings.shape[0], -1).sum(dim=1).float()
                        rowtop_denom = selected_counts.clamp_min(1.0)
                        global_denom = global_counts.clamp_min(1.0)
                        union_denom = union_counts.clamp_min(1.0)
                        overlap_rowtop_ratios = overlap_counts / rowtop_denom
                        overlap_global_ratios = overlap_counts / global_denom
                        overlap_jaccard = overlap_counts / union_denom
                        overlap_valid = valid_pairs & (selected_counts > 0) & (global_counts > 0)
                        if bool(overlap_valid.any()):
                            overlap_count = int(overlap_valid.sum().item())
                            metrics = {
                                "rowtop_precision": overlap_rowtop_ratios,
                                "global_recall": overlap_global_ratios,
                                "jaccard": overlap_jaccard,
                            }
                            for metric_name, metric_values in metrics.items():
                                metric_sum = float(metric_values[overlap_valid].sum().item())
                                key = "partialdiff_reverse_rowtop1_global_overlap_{}_sum".format(metric_name)
                                step_key = "partialdiff_reverse_rowtop1_global_overlap_{}_step_{}_sum".format(metric_name, step_idx)
                                self.timing_totals[key] += metric_sum
                                self.timing_counts[key] += overlap_count
                                self.timing_totals[step_key] += metric_sum
                                self.timing_counts[step_key] += overlap_count
            mode_now = str(getattr(self.args, "constrained_greedy_mode", "global_n3"))
            row_top1_repair_mode = str(getattr(self.args, "reverse_row_top1_repair_mode", "final_step"))
            should_repair_reverse_row_top1 = (
                reverse_decode_mode != "none"
                and self._uses_top1_decode_mode(mode_now)
                and (row_top1_repair_mode == "every_step" or float(t2) == 0.0)
            )
            if should_repair_reverse_row_top1:
                repair_start = self.start_timer()
                clean_matchings = self.batched_repair_matching_completion(
                    adjusted_scores,
                    clean_matchings,
                    max_matching_size,
                )
                repair_elapsed = self.stop_timer(repair_start)
                self.timing_totals["partialdiff_reverse_repair_total"] += repair_elapsed
                self.timing_counts["partialdiff_reverse_repair_total"] += 1
            if float(t2) == 0.0:
                if reverse_decode_mode == "none":
                    mapping_t = self.batched_constrained_matching_decode(
                        adjusted_scores,
                        max_matching_size,
                    )
                else:
                    mapping_t = clean_matchings
            else:
                if str(getattr(self.args, "renoise_mode", "stochastic")) == "none":
                    mapping_t = clean_matchings
                else:
                    ratio = self.partialdiff_keep_ratio(t2)
                    target_sizes = torch.clamp(torch.round(ratio * max_matching_size.float()).long(), min=0)
                    if reverse_decode_mode == "blockwise_autoregressive":
                        new_target_sizes = (target_sizes - previous_counts).clamp(min=0)
                        if str(getattr(self.args, "blockwise_renoise_mode", "fixed_count")) == "bernoulli_drop":
                            available_new = (blockwise_new_matchings > 0.5).sum(dim=(1, 2)).to(dtype=torch.float)
                            keep_probability = new_target_sizes.to(dtype=torch.float) / available_new.clamp_min(1.0)
                            sampled_new_matchings = self.batched_bernoulli_drop_new_matchings(
                                blockwise_new_matchings,
                                keep_probability,
                            )
                        elif self.args.renoise_mode == "topk":
                            sampled_new_matchings = self.batched_sample_partial_from_clean_matching_variable_size(
                                blockwise_new_matchings,
                                adjusted_scores,
                                new_target_sizes,
                            )
                        else:
                            sampled_new_matchings = self.batched_sample_partial_from_clean_matching_variable_size_seeded(
                                blockwise_new_matchings,
                                adjusted_scores,
                                new_target_sizes,
                                pair_gid=expanded_batch["pair_gid"],
                                sample_slot=expanded_batch["sample_slot"].to(device=self.device),
                                step_idx=step_idx,
                            )
                        mapping_t = previous_mask.to(dtype=sampled_new_matchings.dtype) + sampled_new_matchings
                        mapping_t = mapping_t.clamp_(min=0.0, max=1.0)
                    else:
                        if self.args.renoise_mode == "topk":
                            mapping_t = self.batched_sample_partial_from_clean_matching_variable_size(
                                clean_matchings,
                                adjusted_scores,
                                target_sizes,
                            )
                        else:
                            mapping_t = self.batched_sample_partial_from_clean_matching_variable_size_seeded(
                                clean_matchings,
                                adjusted_scores,
                                target_sizes,
                                pair_gid=expanded_batch["pair_gid"],
                                sample_slot=expanded_batch["sample_slot"].to(device=self.device),
                                step_idx=step_idx,
                            )
            mapping_t = mapping_t.masked_fill(~pair_mask, 0.0)
            if reverse_decode_mode == "blockwise_autoregressive":
                self.validate_blockwise_autoregressive_step(
                    previous_partial,
                    mapping_t,
                    previous_mask=previous_mask,
                    next_mask=mapping_t > 0.5,
                )

        if reverse_decode_mode == "blockwise_autoregressive":
            mapping_t = mapping_t.masked_fill(~pair_mask, 0.0)
        else:
            flat_decode_scores = final_adjusted_scores
            if flat_decode_scores is None:
                flat_decode_scores = mapping_t.masked_fill(~pair_mask, float("-inf"))
            mode_now = str(getattr(self.args, "constrained_greedy_mode", "global_n3"))
            final_decode_start = self.start_timer()
            flat_matchings = self.batched_constrained_matching_decode_by_mode(
                flat_decode_scores,
                max_matching_size,
                step_idx=None,
            )
            final_decode_elapsed = self.stop_timer(final_decode_start)
            self.timing_totals["partialdiff_final_decode_total"] += final_decode_elapsed
            self.timing_counts["partialdiff_final_decode_total"] += 1
            self.timing_totals["partialdiff_final_decode_" + mode_now] += final_decode_elapsed
            self.timing_counts["partialdiff_final_decode_" + mode_now] += 1
            if self._uses_top1_decode_mode(mode_now):
                final_repair_start = self.start_timer()
                flat_matchings = self.batched_repair_matching_completion(
                    flat_decode_scores,
                    flat_matchings,
                    max_matching_size,
                )
                final_repair_elapsed = self.stop_timer(final_repair_start)
                self.timing_totals["partialdiff_final_repair_total"] += final_repair_elapsed
                self.timing_counts["partialdiff_final_repair_total"] += 1
            mapping_t = flat_matchings.masked_fill(~pair_mask, 0.0)

        flat_matchings = mapping_t.view(batch_size * num_parallel_sampling, pad_n1, pad_n2)
        flat_sample_ged = self.dense_ged_from_clean_matchings_direct(expanded_batch, flat_matchings)
        sample_ged = flat_sample_ged.view(batch_size, num_parallel_sampling)
        best_offsets = torch.argmin(sample_ged, dim=1)
        best_values = sample_ged.gather(1, best_offsets.unsqueeze(1)).squeeze(1)

        elapsed = self.stop_timer(start_time)
        per_pair_elapsed = elapsed / max(batch_size, 1)
        if not collect_pair_outputs:
            return {
                "pred_ged": best_values,
                "running_time": torch.full(
                    (batch_size,),
                    float(per_pair_elapsed),
                    device=self.device,
                    dtype=torch.float,
                ),
            }

        final_matchings = mapping_t.view(batch_size, num_parallel_sampling, pad_n1, pad_n2)
        final_probabilities = final_pred_probs.view(batch_size, num_parallel_sampling, pad_n1, pad_n2)
        final_scores = final_adjusted_scores.view(batch_size, num_parallel_sampling, pad_n1, pad_n2)
        results = []
        for pair_idx in range(batch_size):
            n1 = int(batch["n1"][pair_idx].item())
            n2 = int(batch["n2"][pair_idx].item())
            best_matching = final_matchings[pair_idx, int(best_offsets[pair_idx].item()), :n1, :n2].unsqueeze(0)
            metadata = {"postprocess_time": 0.0}
            if getattr(self.args, "save_test_k_candidates", False) or getattr(self.args, "app_bmao_postprocess_enable", False):
                candidate_blob = {
                    "pair": batch["pair"][pair_idx].detach().cpu().clone(),
                    "pair_gid": batch["pair_gid"][pair_idx].detach().cpu().clone(),
                    "n1": n1,
                    "n2": n2,
                    "gt_ged": float(batch["ged"][pair_idx].item()),
                    "candidate_ged": sample_ged[pair_idx].detach().cpu().clone(),
                    "best_index": int(best_offsets[pair_idx].item()),
                    "final_probabilities": final_probabilities[pair_idx, :, :n1, :n2].detach().cpu().clone(),
                    "final_matchings": final_matchings[pair_idx, :, :n1, :n2].detach().cpu().clone(),
                    "final_scores": final_scores[pair_idx, :, :n1, :n2].detach().cpu().clone(),
                }
                if getattr(self.args, "save_test_k_candidates", False):
                    candidate_blob.update({
                        "x1": batch["x1"][pair_idx, :n1].detach().cpu().clone(),
                        "x2": batch["x2"][pair_idx, :n2].detach().cpu().clone(),
                        "ged_adj1": batch["ged_adj1"][pair_idx, :n1, :n1].detach().cpu().clone(),
                        "ged_adj2": batch["ged_adj2"][pair_idx, :n2, :n2].detach().cpu().clone(),
                        "gt_matching": batch["gt_matching"][pair_idx, :n1, :n2].detach().cpu().clone(),
                    })
                metadata["all_candidates"] = candidate_blob
            results.append((best_values[pair_idx], best_matching, per_pair_elapsed, None, metadata))
        return results[0] if batch_size == 1 else results

    def score(self, testing_graph_set="test", test_k=100, top_k_approach="parallel"):
        assert top_k_approach == "parallel"
        if testing_graph_set == "test":
            loader = self.testing_data_loader
        elif testing_graph_set == "val":
            loader = self.val_data_loader
        elif testing_graph_set == "small":
            loader = self.testing_data_small_loader
        else:
            loader = self.testing_data_large_loader

        print("\n\nEvaluate Dense DiffGED with {} topk {} on {} set.\n".format(top_k_approach, test_k, testing_graph_set))
        self.model.eval()
        self.reset_timing_stats()
        pair_limit = int(getattr(self.args, "max_test_pairs", 0))
        pairs_processed = 0
        pred_sum = 0.0
        gt_sum = 0.0
        abs_error_sum = 0.0
        time_sum = 0.0
        pred_lt_gt_sum = 0.0
        acc_sum = 0.0
        graph_ids = []
        pred_values = []
        gt_targets = []
        save_raw_pair_log = bool(getattr(self.args, "save_raw_pair_log", True))
        need_pair_outputs = (
            save_raw_pair_log
            or bool(getattr(self.args, "save_test_k_candidates", False))
            or bool(getattr(self.args, "app_bmao_postprocess_enable", False))
        )
        raw_pair_batches = []
        saved_candidates = []
        app_bmao_pair_results = []
        app_bmao_profiling_reports = []
        app_bmao_streamer = None
        app_bmao_gpu_refine_pipeline = None
        app_bmao_deferred_candidates = []
        candidate_pair_index = 0
        saved_candidates_write_time_s = 0.0
        app_bmao_batch_time_s = 0.0
        app_bmao_finalize_time_s = 0.0
        merged_profile = None
        runtime_profile_enabled = bool(getattr(self.args, "profile_runtime_enable", False))
        runtime_profile_records = []
        app_bmao_search_backend = str(getattr(self.args, "app_bmao_search_backend", "external_app_bmao"))
        gpu_refine_overlap_delay_batches = int(getattr(self.args, "app_bmao_gpu_refine_overlap_delay_batches", 0))
        gpu_refine_defer_enable = False
        if (
            getattr(self.args, "app_bmao_postprocess_enable", False)
            and getattr(self.args, "app_bmao_overlap_enable", False)
            and self._is_external_bmao_backend(app_bmao_search_backend)
        ):
            app_bmao_streamer = StreamingAppBmaoPostprocessor(
                trainer=self,
                testing_graph_set=testing_graph_set,
                top_k_approach=top_k_approach,
                test_k=test_k,
            )
        if (
            getattr(self.args, "app_bmao_postprocess_enable", False)
            and app_bmao_search_backend == "gpu_refine"
            and gpu_refine_overlap_delay_batches > 0
            and not gpu_refine_defer_enable
        ):
            app_bmao_gpu_refine_pipeline = GpuRefineOverlapPostprocessor(
                trainer=self,
                testing_graph_set=testing_graph_set,
                top_k_approach=top_k_approach,
                test_k=test_k,
            )
        iterator = tqdm(loader, file=sys.stdout, dynamic_ncols=True) if not self.args.disable_tqdm else loader
        score_flow_start_time = None
        for batch_idx, batch in enumerate(iterator):
            if self.args.max_test_batches > 0 and batch_idx >= self.args.max_test_batches:
                break
            if pair_limit > 0 and pairs_processed >= pair_limit:
                break

            if pair_limit > 0:
                remaining = pair_limit - pairs_processed
                current_batch = int(batch["ged"].shape[0])
                if current_batch > remaining:
                    trimmed_items = []
                    for item_idx in range(remaining):
                        n1 = int(batch["n1"][item_idx].item())
                        n2 = int(batch["n2"][item_idx].item())
                        trimmed_items.append(
                            {
                                "pair": batch["pair"][item_idx].clone(),
                                "pair_gid": batch["pair_gid"][item_idx].clone(),
                                "x1": batch["x1"][item_idx, :n1].clone(),
                                "x2": batch["x2"][item_idx, :n2].clone(),
                                "adj1": batch["adj1"][item_idx, :n1, :n1].clone(),
                                "adj2": batch["adj2"][item_idx, :n2, :n2].clone(),
                                "ged_adj1": batch["ged_adj1"][item_idx, :n1, :n1].clone(),
                                "ged_adj2": batch["ged_adj2"][item_idx, :n2, :n2].clone(),
                                "gt_matching": batch["gt_matching"][item_idx, :n1, :n2].clone(),
                                "base_cost": batch["base_cost"][item_idx, :n1, :n2].clone(),
                                "left_degree": batch["left_degree"][item_idx, :n1].clone(),
                                "right_degree": batch["right_degree"][item_idx, :n2].clone(),
                                "left_label": batch["left_label"][item_idx, :n1].clone(),
                                "right_label": batch["right_label"][item_idx, :n2].clone(),
                                "ged": batch["ged"][item_idx].clone(),
                                "n1": batch["n1"][item_idx].clone(),
                                "n2": batch["n2"][item_idx].clone(),
                            }
                        )
                    batch = self.dense_collate(trimmed_items)

            batch = self._move_batch_to_device(batch)
            runtime_record = None
            if runtime_profile_enabled:
                self._reset_cuda_peak_memory()
                n1_values_for_profile = [int(v) for v in batch["n1"].detach().cpu().tolist()]
                n2_values_for_profile = [int(v) for v in batch["n2"].detach().cpu().tolist()]
                runtime_record = {
                    "batch_idx": int(batch_idx),
                    "batch_size": int(batch["ged"].shape[0]),
                    "max_n1": int(max(n1_values_for_profile) if n1_values_for_profile else 0),
                    "max_n2": int(max(n2_values_for_profile) if n2_values_for_profile else 0),
                    "pad_n1": int(batch["x1"].shape[1]),
                    "pad_n2": int(batch["x2"].shape[1]),
                    "pairs_processed_start": int(pairs_processed),
                    "cuda_after_move": self._cuda_memory_snapshot(),
                }
                runtime_batch_start = time.perf_counter()
                self._reset_cuda_peak_memory()
            if score_flow_start_time is None:
                score_flow_start_time = time.time()
            runtime_diffusion_start = time.perf_counter() if runtime_profile_enabled else None
            outputs = self.diffusion_ged_dense(batch, test_k=test_k, collect_pair_outputs=need_pair_outputs)
            if runtime_profile_enabled and runtime_record is not None:
                runtime_record["diffusion_wall_time_s"] = round(time.perf_counter() - runtime_diffusion_start, 5)
                runtime_record["diffusion_cuda_memory"] = self._cuda_memory_snapshot()
            if isinstance(outputs, dict):
                batch_outputs = []
                pred_ged = outputs["pred_ged"]
                running_time = outputs["running_time"]
            else:
                batch_outputs = outputs if isinstance(outputs, list) else [outputs]
                pred_ged = torch.stack([torch.as_tensor(out[0], device=self.device, dtype=torch.float) for out in batch_outputs])
                running_time = torch.tensor([float(out[2]) for out in batch_outputs], device=self.device, dtype=torch.float)
            gt_values = batch["ged"].view(-1)
            current_batch_candidates = []
            if save_raw_pair_log:
                pair_values = batch["pair"].detach().cpu().tolist()
                pair_gid_values = batch["pair_gid"].detach().cpu().tolist()
                n1_values = [int(v) for v in batch["n1"].detach().cpu().tolist()]
                n2_values = [int(v) for v in batch["n2"].detach().cpu().tolist()]
                pred_ged_values = [float(v) for v in pred_ged.detach().cpu().tolist()]
                gt_value_list = [float(v) for v in gt_values.detach().cpu().tolist()]
                runtime_values = [float(v) for v in running_time.detach().cpu().tolist()]
                m1_values = [int(v) for v in torch.triu(batch["ged_adj1"] > 0.5, diagonal=1).sum(dim=(1, 2)).detach().cpu().tolist()]
                m2_values = [int(v) for v in torch.triu(batch["ged_adj2"] > 0.5, diagonal=1).sum(dim=(1, 2)).detach().cpu().tolist()]
                config_name = "{}_{}_{}_{}".format(self.args.model_name, self.args.dataset, testing_graph_set, test_k)
                budget_parameters = self._partialdiff_budget_parameters(test_k, top_k_approach)
                matching_tensors = [out[1].squeeze(0) if out[1] is not None else None for out in batch_outputs]
                if all(matching_tensor is not None for matching_tensor in matching_tensors):
                    matching_payload_padded = torch.zeros(
                        (len(matching_tensors), int(batch["x1"].shape[1]), int(batch["x2"].shape[1])),
                        device=self.device,
                        dtype=matching_tensors[0].dtype,
                    )
                    for item_idx, matching_tensor in enumerate(matching_tensors):
                        rows, cols = matching_tensor.shape
                        matching_payload_padded[item_idx, :rows, :cols] = matching_tensor
                    matching_payload = matching_payload_padded.detach().cpu()
                else:
                    matching_payload = [
                        None if matching_tensor is None else matching_tensor.detach().cpu()
                        for matching_tensor in matching_tensors
                    ]
                raw_pair_batches.append(
                    {
                        "pair_values": pair_values,
                        "pair_gid_values": pair_gid_values,
                        "n1_values": n1_values,
                        "n2_values": n2_values,
                        "pred_ged_values": pred_ged_values,
                        "gt_value_list": gt_value_list,
                        "runtime_values": runtime_values,
                        "m1_values": m1_values,
                        "m2_values": m2_values,
                        "matching_payload": matching_payload,
                        "config_name": config_name,
                        "budget_parameters": budget_parameters,
                    }
                )

            pairs_processed += int(gt_values.numel())
            pred_sum += float(pred_ged.sum().item())
            gt_sum += float(gt_values.sum().item())
            abs_error_sum += float(torch.abs(pred_ged - gt_values).sum().item())
            time_sum += float(running_time.sum().item())
            pred_lt_gt_sum += float((pred_ged < gt_values).float().sum().item())
            acc_sum += float((pred_ged == gt_values).float().sum().item())
            graph_ids.extend(int(v) for v in batch["pair"][:, 0].detach().cpu().tolist())
            pred_values.extend(float(v) for v in pred_ged.detach().cpu().tolist())
            gt_targets.extend(float(v) for v in gt_values.detach().cpu().tolist())
            if getattr(self.args, "save_test_k_candidates", False) or getattr(self.args, "app_bmao_postprocess_enable", False):
                for out in batch_outputs:
                    candidate_blob = out[4].get("all_candidates")
                    if candidate_blob is not None:
                        current_batch_candidates.append(candidate_blob)
                        if getattr(self.args, "save_test_k_candidates", False):
                            saved_candidates.append(candidate_blob)
                if app_bmao_streamer is not None and current_batch_candidates:
                    app_bmao_streamer.submit_candidate_pairs(
                        candidate_pairs=current_batch_candidates,
                        start_pair_index=candidate_pair_index,
                    )
                elif app_bmao_gpu_refine_pipeline is not None and current_batch_candidates:
                    runtime_app_bmao_start = time.perf_counter() if runtime_profile_enabled else None
                    if runtime_profile_enabled:
                        self._reset_cuda_peak_memory()
                    drained_wall_time_s = app_bmao_gpu_refine_pipeline.submit_candidate_pairs(
                        candidate_pairs=current_batch_candidates,
                        start_pair_index=candidate_pair_index,
                    )
                    if runtime_profile_enabled and runtime_record is not None:
                        runtime_record["app_bmao_wall_time_s"] = round(time.perf_counter() - runtime_app_bmao_start, 5)
                        runtime_record["app_bmao_visible_refine_time_s"] = round(float(drained_wall_time_s), 5)
                        runtime_record["app_bmao_cuda_memory"] = self._cuda_memory_snapshot()
                    app_bmao_batch_time_s += float(drained_wall_time_s)
                elif gpu_refine_defer_enable and current_batch_candidates:
                    app_bmao_deferred_candidates.extend(current_batch_candidates)
                elif getattr(self.args, "app_bmao_postprocess_enable", False) and current_batch_candidates:
                    batch_postprocess_start_time = time.time()
                    runtime_app_bmao_start = time.perf_counter() if runtime_profile_enabled else None
                    if runtime_profile_enabled:
                        self._reset_cuda_peak_memory()
                    batch_pair_results, batch_profiling_report = self.run_app_bmao_postprocess(
                        candidate_pairs=current_batch_candidates,
                        testing_graph_set=testing_graph_set,
                        top_k_approach=top_k_approach,
                        test_k=test_k,
                        pair_index_offset=candidate_pair_index,
                        finalize=False,
                    )
                    if runtime_profile_enabled and runtime_record is not None:
                        runtime_record["app_bmao_wall_time_s"] = round(time.perf_counter() - runtime_app_bmao_start, 5)
                        runtime_record["app_bmao_cuda_memory"] = self._cuda_memory_snapshot()
                    app_bmao_batch_time_s += time.time() - batch_postprocess_start_time
                    app_bmao_pair_results.extend(batch_pair_results)
                    if batch_profiling_report is not None:
                        app_bmao_profiling_reports.append(batch_profiling_report)
            candidate_pair_index += len(current_batch_candidates)
            if runtime_profile_enabled and runtime_record is not None:
                runtime_record.setdefault("app_bmao_wall_time_s", 0.0)
                runtime_record.setdefault("app_bmao_cuda_memory", self._cuda_memory_snapshot())
                runtime_record["batch_wall_time_s"] = round(time.perf_counter() - runtime_batch_start, 5)
                runtime_record["batch_cuda_memory"] = self._cuda_memory_snapshot()
                runtime_profile_records.append(runtime_record)

        num = max(pairs_processed, 1)
        avg_pred_ged = round(pred_sum / num, 3)
        avg_gt_ged = round(gt_sum / num, 3)
        mae = round(abs_error_sum / num, 3)
        acc = round(acc_sum / num, 3)
        time_usage = round(time_sum / num, 5)
        diffusion_total_time_s = round(time_usage * num, 5)
        pred_lt_gt_count = int(pred_lt_gt_sum)
        pred_lt_gt_ratio = round(pred_lt_gt_sum / num, 3)
        fea = round((sum(1 for pred_value, gt_value in zip(pred_values, gt_targets) if pred_value >= gt_value)) / num, 3)

        pres = {}
        gts = {}
        for gid, pred_value, gt_value in zip(graph_ids, pred_values, gt_targets):
            pres.setdefault(gid, []).append(pred_value)
            gts.setdefault(gid, []).append(gt_value)

        rho_values = []
        tau_values = []
        pk10_values = []
        pk20_values = []
        for graph_id in pres:
            rho_values.append(spearmanr(pres[graph_id], gts[graph_id])[0])
            tau_values.append(kendalltau(pres[graph_id], gts[graph_id])[0])
            pk10_values.append(self.cal_pk(10, pres[graph_id], gts[graph_id]))
            pk20_values.append(self.cal_pk(20, pres[graph_id], gts[graph_id]))

        rho = round(float(np.nanmean(rho_values)), 3) if rho_values else float("nan")
        tau = round(float(np.nanmean(tau_values)), 3) if tau_values else float("nan")
        pk10 = round(float(np.nanmean(pk10_values)), 3) if pk10_values else float("nan")
        pk20 = round(float(np.nanmean(pk20_values)), 3) if pk20_values else float("nan")

        self.results.append((
            "model_name", "topk_approach", "dataset", "graph_set", "#testing_pairs",
            "time_usage(s/p)", "mae", "acc", "avg_pred_ged", "avg_gt_ged", "fea", "rho", "tau", "pk10", "pk20",
            "pred_lt_gt_count", "pred_lt_gt_ratio",
        ))
        self.results.append((
            self.args.model_name, top_k_approach, self.args.dataset, testing_graph_set, num,
            time_usage, mae, acc, avg_pred_ged, avg_gt_ged, fea, rho, tau, pk10, pk20, pred_lt_gt_count, pred_lt_gt_ratio,
        ))
        print(*self.results[-2], sep="\t")
        print(*self.results[-1], sep="\t")

        output_result_path = self._result_path(
            "result_DiffGED_dense_{}_{}_{}_{}.json".format(
                self.args.dataset,
                testing_graph_set,
                top_k_approach,
                test_k,
            )
        )
        result_payload = {
            "time": time_usage,
            "diffusion_total_time_s": diffusion_total_time_s,
            "mae": mae,
            "acc": acc,
            "avg_pred_ged": avg_pred_ged,
            "avg_gt_ged": avg_gt_ged,
            "fea": fea,
            "rho": rho,
            "tau": tau,
            "pk10": pk10,
            "pk20": pk20,
            "pred_lt_gt_count": pred_lt_gt_count,
            "pred_lt_gt_ratio": pred_lt_gt_ratio,
            "num_pairs": num,
        }

        if getattr(self.args, "save_test_k_candidates", False):
            save_candidates_start_time = time.time()
            output_candidates_path = self._result_path(
                "result_DiffGED_dense_candidates_{}_{}_{}_{}.pt".format(
                    self.args.dataset,
                    testing_graph_set,
                    top_k_approach,
                    test_k,
                )
            )
            torch.save(
                {
                    "model_name": self.args.model_name,
                    "dataset": self.args.dataset,
                    "graph_set": testing_graph_set,
                    "topk_approach": top_k_approach,
                    "test_k": int(test_k),
                    "num_pairs": int(len(saved_candidates)),
                    "pairs": saved_candidates,
                },
                output_candidates_path,
            )
            saved_candidates_write_time_s = time.time() - save_candidates_start_time
            print("Saved dense test-k candidates to {}".format(output_candidates_path))

        if app_bmao_streamer is not None:
            app_bmao_finalize_start_time = time.time()
            pair_results = app_bmao_streamer.finalize()
            self._finalize_app_bmao_postprocess(
                pair_results=pair_results,
                testing_graph_set=testing_graph_set,
                top_k_approach=top_k_approach,
                test_k=test_k,
                postprocess_mode=str(getattr(self.args, "app_bmao_postprocess_mode", "best")),
                anchor_ratio=float(getattr(self.args, "app_bmao_anchor_ratio", 0.6)),
                search_states=(int(getattr(self.args, "app_bmao_search_states", 100)) if self._is_external_bmao_backend(app_bmao_search_backend) else 0),
                workers=(max(1, int(getattr(self.args, "app_bmao_workers", 1))) if self._is_external_bmao_backend(app_bmao_search_backend) else 1),
                candidate_budget=int(getattr(self.args, "app_bmao_candidate_budget", 4)),
                beam_score="ged",
                search_backend=app_bmao_search_backend,
            )
            app_bmao_finalize_time_s += time.time() - app_bmao_finalize_start_time
        elif app_bmao_gpu_refine_pipeline is not None:
            app_bmao_finalize_start_time = time.time()
            pair_results, merged_profile = app_bmao_gpu_refine_pipeline.finalize()
            self._finalize_app_bmao_postprocess(
                pair_results=pair_results,
                testing_graph_set=testing_graph_set,
                top_k_approach=top_k_approach,
                test_k=test_k,
                postprocess_mode=str(getattr(self.args, "app_bmao_postprocess_mode", "best")),
                anchor_ratio=float(getattr(self.args, "app_bmao_anchor_ratio", 0.6)),
                search_states=0,
                workers=1,
                candidate_budget=int(getattr(self.args, "app_bmao_candidate_budget", 4)),
                beam_score="ged",
                search_backend=app_bmao_search_backend,
                profiling_report=merged_profile,
            )
            app_bmao_finalize_time_s += time.time() - app_bmao_finalize_start_time
        elif gpu_refine_defer_enable:
            app_bmao_finalize_start_time = time.time()
            search_batch_size_arg = int(getattr(self.args, "app_bmao_pair_chunk_size", 0))
            effective_search_batch_size = search_batch_size_arg
            print(
                "Run deferred gpu_refine postprocess after diffusion: pairs={} search_batch_size={}".format(
                    len(app_bmao_deferred_candidates),
                    effective_search_batch_size,
                ),
                flush=True,
            )
            deferred_output = self.run_app_bmao_postprocess(
                candidate_pairs=app_bmao_deferred_candidates,
                testing_graph_set=testing_graph_set,
                top_k_approach=top_k_approach,
                test_k=test_k,
                pair_index_offset=0,
                finalize=False,
            )
            if deferred_output is None:
                app_bmao_pair_results = []
                merged_profile = None
            else:
                app_bmao_pair_results, merged_profile = deferred_output
            self._finalize_app_bmao_postprocess(
                pair_results=app_bmao_pair_results,
                testing_graph_set=testing_graph_set,
                top_k_approach=top_k_approach,
                test_k=test_k,
                postprocess_mode=str(getattr(self.args, "app_bmao_postprocess_mode", "best")),
                anchor_ratio=float(getattr(self.args, "app_bmao_anchor_ratio", 0.6)),
                search_states=0,
                workers=1,
                candidate_budget=int(getattr(self.args, "app_bmao_candidate_budget", 4)),
                beam_score="ged",
                search_backend=app_bmao_search_backend,
                profiling_report=merged_profile,
            )
            app_bmao_finalize_time_s += time.time() - app_bmao_finalize_start_time
        elif getattr(self.args, "app_bmao_postprocess_enable", False):
            app_bmao_finalize_start_time = time.time()
            merged_profile = self._merge_app_bmao_profiling_reports(app_bmao_profiling_reports)
            self._finalize_app_bmao_postprocess(
                pair_results=app_bmao_pair_results,
                testing_graph_set=testing_graph_set,
                top_k_approach=top_k_approach,
                test_k=test_k,
                postprocess_mode=str(getattr(self.args, "app_bmao_postprocess_mode", "best")),
                anchor_ratio=float(getattr(self.args, "app_bmao_anchor_ratio", 0.6)),
                search_states=int(getattr(self.args, "app_bmao_search_states", 100)),
                workers=max(1, int(getattr(self.args, "app_bmao_workers", 1))),
                candidate_budget=int(getattr(self.args, "app_bmao_candidate_budget", 4)),
                beam_score="ged",
                search_backend=app_bmao_search_backend,
                profiling_report=merged_profile,
            )
            app_bmao_finalize_time_s += time.time() - app_bmao_finalize_start_time

        if score_flow_start_time is None:
            score_flow_start_time = time.time()
        diffusion_to_final_total_time_s = round(time.time() - score_flow_start_time, 5)
        profiled_search_total_time_s = 0.0
        if merged_profile is not None:
            profiled_search_total_time_s = float(merged_profile.get("outer_total_wall_time_s", 0.0) or 0.0)
        accounted_total_time_s = diffusion_total_time_s + saved_candidates_write_time_s + app_bmao_batch_time_s + app_bmao_finalize_time_s
        residual_time_s = max(0.0, diffusion_to_final_total_time_s - accounted_total_time_s)
        result_payload["diffusion_to_final_total_time_s"] = diffusion_to_final_total_time_s
        result_payload["diffusion_to_final_profile"] = {
            "diffusion_total_time_s": round(diffusion_total_time_s, 5),
            "saved_candidates_write_time_s": round(saved_candidates_write_time_s, 5),
            "app_bmao_batch_time_s": round(app_bmao_batch_time_s, 5),
            "app_bmao_finalize_time_s": round(app_bmao_finalize_time_s, 5),
            "app_bmao_profiled_search_total_time_s": round(profiled_search_total_time_s, 5),
            "accounted_total_time_s": round(accounted_total_time_s, 5),
            "residual_time_s": round(residual_time_s, 5),
        }
        decode_profile = {}
        for key in sorted(self.timing_totals):
            if key.startswith("partialdiff_"):
                decode_profile[key] = round(float(self.timing_totals[key]), 5)
                decode_profile[key + "_count"] = int(self.timing_counts.get(key, 0))
        for key in sorted(self.timing_totals):
            if key.startswith("partialdiff_reverse_rowtop1_part_ratio") and key.endswith("_sum"):
                count = int(self.timing_counts.get(key, 0))
                if count > 0:
                    avg_key = key[:-4] + "_avg"
                    decode_profile[avg_key] = round(float(self.timing_totals[key]) / count, 5)
            if key.startswith("partialdiff_reverse_rowtop1_global_overlap") and key.endswith("_sum"):
                count = int(self.timing_counts.get(key, 0))
                if count > 0:
                    avg_key = key[:-4] + "_avg"
                    decode_profile[avg_key] = round(float(self.timing_totals[key]) / count, 5)
            if key.startswith("partialdiff_reverse_rowtop1_repair_needed") and key.endswith("_sum"):
                count = int(self.timing_counts.get(key, 0))
                if count > 0:
                    avg_key = key[:-4] + "_avg"
                    decode_profile[avg_key] = round(float(self.timing_totals[key]) / count, 5)
        if decode_profile:
            decode_total_time_s = (
                float(self.timing_totals.get("partialdiff_reverse_decode_total", 0.0))
                + float(self.timing_totals.get("partialdiff_reverse_repair_total", 0.0))
                + float(self.timing_totals.get("partialdiff_final_decode_total", 0.0))
                + float(self.timing_totals.get("partialdiff_final_repair_total", 0.0))
            )
            decode_profile["partialdiff_decode_total_with_repair"] = round(decode_total_time_s, 5)
            result_payload["diffusion_to_final_profile"]["decode_profile"] = decode_profile
        if merged_profile is not None:
            result_payload["diffusion_to_final_profile"]["app_bmao_profile_summary"] = self._summarize_app_bmao_profile(merged_profile)
            result_payload["diffusion_to_final_profile"]["app_bmao_profile"] = merged_profile
        if runtime_profile_enabled:
            runtime_profile_summary = self._summarize_runtime_profile(runtime_profile_records)
            result_payload["runtime_profile"] = {
                "enabled": True,
                "cuda_available": bool(self.use_gpu),
                "summary": runtime_profile_summary,
                "batch_records": runtime_profile_records,
            }
            if runtime_profile_summary is not None:
                print(
                    "Runtime profile summary: batches={} batch_wall_s={:.3f} diffusion_wall_s={:.3f} app_bmao_wall_s={:.3f} cuda_peak_allocated_mb={}".format(
                        runtime_profile_summary["num_profiled_batches"],
                        runtime_profile_summary["total_batch_wall_time_s"],
                        runtime_profile_summary["total_diffusion_wall_time_s"],
                        runtime_profile_summary["total_app_bmao_wall_time_s"],
                        runtime_profile_summary["cuda_peak_allocated_mb"],
                    ),
                    flush=True,
                )
        if save_raw_pair_log:
            raw_pair_path = self._result_path(
                "raw_pairs_PartialDiff_{}_{}_{}_{}.jsonl".format(
                    self.args.dataset,
                    testing_graph_set,
                    top_k_approach,
                    test_k,
                )
            )
            with raw_pair_path.open("w", encoding="utf-8") as handle:
                for raw_batch in raw_pair_batches:
                    matching_payload = raw_batch["matching_payload"]
                    for item_idx, pair_value in enumerate(raw_batch["pair_values"]):
                        graph_1, graph_2 = pair_value
                        graph_1_gid, graph_2_gid = raw_batch["pair_gid_values"][item_idx]
                        n1 = raw_batch["n1_values"][item_idx]
                        n2 = raw_batch["n2_values"][item_idx]
                        if torch.is_tensor(matching_payload):
                            matching_matrix = matching_payload[item_idx, :n1, :n2]
                        else:
                            matching_matrix = matching_payload[item_idx]
                            if matching_matrix is not None:
                                matching_matrix = matching_matrix[:n1, :n2]
                        if matching_matrix is not None:
                            _row_mapping, matched_pairs = self._dense_matching_to_row_mapping(matching_matrix)
                        else:
                            matched_pairs = None
                        record = self._raw_pair_record(
                            method_name=self.args.model_name,
                            graph_1=graph_1,
                            graph_2=graph_2,
                            graph_1_gid=graph_1_gid,
                            graph_2_gid=graph_2_gid,
                            produced_ged=raw_batch["pred_ged_values"][item_idx],
                            reference_ged=raw_batch["gt_value_list"][item_idx],
                            solver_time=raw_batch["runtime_values"][item_idx],
                            total_time=raw_batch["runtime_values"][item_idx],
                            n1=n1,
                            n2=n2,
                            m1=raw_batch["m1_values"][item_idx],
                            m2=raw_batch["m2_values"][item_idx],
                            config_name=raw_batch["config_name"],
                            budget_parameters=raw_batch["budget_parameters"],
                            matching=matched_pairs,
                        )
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            result_payload["raw_pair_log"] = str(raw_pair_path)
        else:
            result_payload["raw_pair_log"] = None
            result_payload["raw_pair_log_enabled"] = False
        with open(output_result_path, "w") as handle:
            json.dump(result_payload, handle)

    def _resolve_graph_labels_for_app_bmao(self, graph_obj, graph_id):
        graph_index = self.gid_to_index.get(int(graph_id))
        if graph_index is None and 0 <= int(graph_id) < len(self.features):
            graph_index = int(graph_id)

        labels = graph_obj.get("labels") if isinstance(graph_obj, dict) else None
        if labels is not None:
            try:
                return [int(label) for label in labels]
            except (TypeError, ValueError):
                if graph_index is not None and hasattr(self, "node_labels"):
                    return self.node_labels[int(graph_index)].detach().cpu().long().tolist()
                return [int(self.global_labels[str(label)]) for label in labels]

        if graph_index is None:
            raise KeyError(f"Unable to resolve graph index for App-BMao labels: graph_id={graph_id}")
        if hasattr(self, "node_labels"):
            return self.node_labels[int(graph_index)].detach().cpu().long().tolist()

        feature_tensor = self.features[int(graph_index)]
        return self._node_labels_dense(feature_tensor).detach().cpu().long().tolist()

    def _render_graph_entry_from_json(self, graph_obj, graph_id):
        lines = [f"t # {graph_id}\n"]
        for idx, label in enumerate(self._resolve_graph_labels_for_app_bmao(graph_obj, graph_id)):
            lines.append(f"v {idx} {int(label)}\n")
        edge_set = set()
        for src, dst in graph_obj["graph"]:
            src = int(src)
            dst = int(dst)
            if src == dst:
                continue
            if src > dst:
                src, dst = dst, src
            edge_set.add((src, dst))
        for src, dst in sorted(edge_set):
            lines.append(f"e {src} {dst} 1\n")
        return "".join(lines)

    def _render_graph_entry_from_runtime_tensors(self, graph_id):
        graph_index = self.gid_to_index.get(int(graph_id)) if hasattr(self, "gid_to_index") else None
        if graph_index is None and hasattr(self, "gid") and int(graph_id) in set(int(item) for item in self.gid):
            graph_index = [int(item) for item in self.gid].index(int(graph_id))
        if graph_index is None and hasattr(self, "features") and 0 <= int(graph_id) < len(self.features):
            graph_index = int(graph_id)
        if graph_index is None:
            raise KeyError(f"Unable to resolve runtime graph tensors for App-BMao graph_id={graph_id}")

        if hasattr(self, "node_labels"):
            labels = self.node_labels[int(graph_index)].detach().cpu().long().tolist()
        else:
            labels = self._node_labels_dense(self.features[int(graph_index)]).detach().cpu().long().tolist()

        adj = self.ged_adj[int(graph_index)].detach().cpu()
        if adj.dim() == 3:
            adj = adj[..., 0]
        edge_indices = torch.nonzero(adj > 0, as_tuple=False)
        edge_set = set()
        for src, dst in edge_indices.tolist():
            src = int(src)
            dst = int(dst)
            if src == dst:
                continue
            if src > dst:
                src, dst = dst, src
            edge_set.add((src, dst))

        lines = [f"t # {graph_id}\n"]
        for idx, label in enumerate(labels):
            lines.append(f"v {idx} {int(label)}\n")
        for src, dst in sorted(edge_set):
            lines.append(f"e {src} {dst} 1\n")
        return "".join(lines)

    def _resolve_graph_json_path(self, dataset_root, graph_id):
        train_path = Path(dataset_root) / "train" / f"{graph_id}.json"
        if train_path.exists():
            return train_path
        test_path = Path(dataset_root) / "test" / f"{graph_id}.json"
        if test_path.exists():
            return test_path
        raise FileNotFoundError(f"Graph json not found for id={graph_id}: {train_path} or {test_path}")

    def _build_graph_text_cache(self, candidate_pairs, dataset_root, cache_dir):
        graph_ids = set()
        for item in candidate_pairs:
            graph_ids.add(int(item["pair_gid"][0].item()))
            graph_ids.add(int(item["pair_gid"][1].item()))
        graph_ids = sorted(graph_ids)
        stdin_enable = bool(getattr(self.args, "app_bmao_stdin_enable", False))
        workers = max(1, int(getattr(self.args, "app_bmao_workers", 1)))

        def build_one(graph_id):
            if getattr(self.args, "fixed_pair_root", None) and hasattr(self, "graphs") and 0 <= int(graph_id) < len(self.graphs):
                graph_obj = self.graphs[int(graph_id)]
                graph_text = self._render_graph_entry_from_json(graph_obj, graph_id)
            elif getattr(self.args, "fixed_pair_root", None) and hasattr(self, "ged_adj"):
                graph_text = self._render_graph_entry_from_runtime_tensors(graph_id)
            else:
                graph_json_path = self._resolve_graph_json_path(dataset_root, graph_id)
                with graph_json_path.open("r", encoding="utf-8") as handle:
                    graph_obj = json.load(handle)
                graph_text = self._render_graph_entry_from_json(graph_obj, graph_id)
            if stdin_enable:
                return graph_id, graph_text
            graph_txt_path = Path(cache_dir) / f"graph_{graph_id}.txt"
            graph_txt_path.write_text(graph_text, encoding="utf-8")
            return graph_id, graph_txt_path

        if workers <= 1 or len(graph_ids) <= 1:
            return dict(build_one(graph_id) for graph_id in graph_ids)

        graph_cache = {}
        cache_workers = min(workers, len(graph_ids))
        with ThreadPoolExecutor(max_workers=cache_workers) as executor:
            futures = [executor.submit(build_one, graph_id) for graph_id in graph_ids]
            for future in as_completed(futures):
                graph_id, graph_payload = future.result()
                graph_cache[int(graph_id)] = graph_payload
        return graph_cache

    @staticmethod
    def _select_top_probability_anchors(final_matching, final_probabilities, anchor_ratio):
        if float(anchor_ratio) <= 0.0:
            return []
        rows, cols = torch.nonzero(final_matching > 0.5, as_tuple=True)
        if rows.numel() == 0:
            return []
        probs = final_probabilities[rows, cols]
        keep_count = int(math.ceil(rows.numel() * float(anchor_ratio)))
        order = torch.argsort(probs, descending=True)
        anchors = []
        for idx in order[:keep_count].tolist():
            anchors.append((int(rows[idx].item()), int(cols[idx].item()), float(probs[idx].item())))
        return anchors

    @staticmethod
    def _write_anchor_file(anchor_path, anchors):
        used_rows = set()
        used_cols = set()
        with open(anchor_path, "w", encoding="utf-8") as handle:
            for row, col, _prob in anchors:
                if row in used_rows:
                    raise ValueError(f"Duplicate anchor query node {row}")
                if col in used_cols:
                    raise ValueError(f"Duplicate anchor target node {col}")
                used_rows.add(row)
                used_cols.add(col)
                handle.write(f"{row} {col}\n")

    @staticmethod
    def _render_dense_matching_text(final_matching):
        rows, cols = torch.nonzero(final_matching > 0.5, as_tuple=True)
        pairs = sorted((int(row.item()), int(col.item())) for row, col in zip(rows, cols))
        used_rows = set()
        used_cols = set()
        lines = []
        for row, col in pairs:
            if row in used_rows:
                raise ValueError(f"Duplicate incumbent query node {row}")
            if col in used_cols:
                raise ValueError(f"Duplicate incumbent target node {col}")
            used_rows.add(row)
            used_cols.add(col)
            lines.append(f"{row} {col}\n")
        return "".join(lines)

    @staticmethod
    def _write_dense_matching_file(path, final_matching):
        Path(path).write_text(TrainerDense._render_dense_matching_text(final_matching), encoding="utf-8")

    @staticmethod
    def _render_anchor_text(anchors):
        lines = []
        used_rows = set()
        used_cols = set()
        for row, col, _prob in anchors:
            if row in used_rows:
                raise ValueError(f"Duplicate anchor query node {row}")
            if col in used_cols:
                raise ValueError(f"Duplicate anchor target node {col}")
            used_rows.add(row)
            used_cols.add(col)
            lines.append(f"{row} {col}\n")
        return "".join(lines)

    @staticmethod
    def _compose_app_bmao_stdin_payload(query_text, db_text, anchor_text, incumbent_text=""):
        return "".join([
            "__APP_BMAO_DB_BEGIN__\n",
            db_text,
            "__APP_BMAO_DB_END__\n",
            "__APP_BMAO_QUERY_BEGIN__\n",
            query_text,
            "__APP_BMAO_QUERY_END__\n",
            "__APP_BMAO_ANCHOR_BEGIN__\n",
            anchor_text,
            "__APP_BMAO_ANCHOR_END__\n",
            "__APP_BMAO_INCUMBENT_BEGIN__\n",
            incumbent_text,
            "__APP_BMAO_INCUMBENT_END__\n",
        ])

    @staticmethod
    def _parse_app_bmao_solver_output(output_text):
        ged_value = None
        mapping_payload = None
        solver_us = None
        search_space = None
        prune_profile = None
        time_pat = re.compile(r"Total time:\s*([\d,]+)\s*\(microseconds\)")
        search_space_pat = re.compile(r"total search space:\s*([\d,]+)")
        prune_profile_pat = re.compile(r"^prune_profile\s+\d+:\s+(\{.*\})$")
        inline_summary_pat = re.compile(
            r"GED:\s*(-?\d+)\s*,\s*Time:\s*([\d,]+)\s*,\s*Search space:\s*([\d,]+)",
            re.IGNORECASE,
        )
        candidates_pat = re.compile(r"#candidates:\s*([\d,]+)")
        candidates_count = None
        for raw_line in output_text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if re.fullmatch(r"-?\d+", line):
                ged_value = int(line)
                continue
            match = TrainerDense.APP_BMAO_MATCHING_PAT.match(line)
            if match:
                _pair_idx, _g1, _g2, payload = match.groups()
                mapping_payload = json.loads(payload)
                continue
            match = inline_summary_pat.search(line)
            if match:
                ged_value = int(match.group(1))
                solver_us = int(match.group(2).replace(",", ""))
                search_space = int(match.group(3).replace(",", ""))
                continue
            match = time_pat.search(line)
            if match:
                solver_us = int(match.group(1).replace(",", ""))
            match = search_space_pat.search(line)
            if match:
                search_space = int(match.group(1).replace(",", ""))
            match = candidates_pat.search(line)
            if match:
                candidates_count = int(match.group(1).replace(",", ""))
            match = prune_profile_pat.match(line)
            if match:
                prune_profile = json.loads(match.group(1))
        if ged_value is None:
            if candidates_count == 0:
                return -1, mapping_payload, solver_us, search_space, prune_profile
            raise RuntimeError(f"Failed to parse App-BMao GED from output:\n{output_text}")
        return ged_value, mapping_payload, solver_us, search_space, prune_profile

    def _recompute_dense_minimal_ged_for_pair(self, graph_1, graph_2, matching_pairs):
        n1 = int(self.gn[graph_1])
        n2 = int(self.gn[graph_2])
        matching = torch.zeros((1, n1, n2), dtype=torch.float)
        for raw_row, raw_col in matching_pairs:
            row = int(raw_row)
            col = int(raw_col)
            if row == n1 or col == n2:
                continue
            if row < 0 or row > n1:
                raise ValueError(f"App-BMao matching row out of range: row={row}, n1={n1}")
            if col < 0 or col > n2:
                raise ValueError(f"App-BMao matching col out of range: col={col}, n2={n2}")
            if matching[0, row].any():
                raise ValueError(f"App-BMao matching has duplicate row assignment: row={row}")
            if matching[0, :, col].any():
                raise ValueError(f"App-BMao matching has duplicate col assignment: col={col}")
            matching[0, row, col] = 1.0

        batch = {
            "x1": self.features[graph_1].unsqueeze(0),
            "x2": self.features[graph_2].unsqueeze(0),
            "ged_adj1": self.ged_adj[graph_1].unsqueeze(0),
            "ged_adj2": self.ged_adj[graph_2].unsqueeze(0),
            "n1": torch.tensor([n1], dtype=torch.long),
            "n2": torch.tensor([n2], dtype=torch.long),
            "pair": torch.tensor([[graph_1, graph_2]], dtype=torch.long),
        }
        return float(self.dense_ged_from_clean_matchings_direct(batch, matching).item())

    @staticmethod
    def _normalize_app_bmao_matching_payload(mapping_payload):
        matching_pairs = mapping_payload["query_to_db"]
        if not bool(mapping_payload.get("q_g_swapped", False)):
            return matching_pairs

        normalized_pairs = []
        for internal_query, internal_db in matching_pairs:
            normalized_pairs.append([int(internal_db), int(internal_query)])
        return normalized_pairs

    @staticmethod
    def _compute_app_bmao_candidate_confidence(final_matching, final_probabilities):
        matched_mask = final_matching > 0.5
        matched_probs = final_probabilities[matched_mask]
        if matched_probs.numel() == 0:
            return {
                "mean_matched_prob": 0.0,
                "min_matched_prob": 0.0,
                "matched_edge_count": 0,
            }
        return {
            "mean_matched_prob": float(matched_probs.mean().item()),
            "min_matched_prob": float(matched_probs.min().item()),
            "matched_edge_count": int(matched_probs.numel()),
        }

    @staticmethod
    def _is_external_bmao_backend(search_backend):
        return str(search_backend) in {
            "external_app_bmao",
            "external_dfs_bmao",
            "external_astar_bmao",
        }

    def _resolve_external_bmao_backend_config(self, search_backend):
        backend = str(search_backend)
        lower_bound = str(getattr(self.args, "external_bmao_lower_bound", "BMao"))
        if backend == "external_app_bmao":
            return {
                "backend": backend,
                "ged_bin": str(getattr(self.args, "app_bmao_ged_bin", "")),
                "paradigm": "astar",
                "lower_bound": lower_bound,
                "supports_stdin": True,
                "supports_stop": True,
                "supports_incumbent": True,
            }
        if backend == "external_dfs_bmao":
            return {
                "backend": backend,
                "ged_bin": str(getattr(self.args, "dfs_bmao_ged_bin", "")),
                "paradigm": "dfs",
                "lower_bound": lower_bound,
                "supports_stdin": False,
                "supports_stop": True,
                "supports_incumbent": False,
            }
        if backend == "external_astar_bmao":
            return {
                "backend": backend,
                "ged_bin": str(getattr(self.args, "astar_bmao_ged_bin", "")),
                "paradigm": "astar",
                "lower_bound": lower_bound,
                "supports_stdin": False,
                "supports_stop": False,
                "supports_incumbent": False,
            }
        raise ValueError(f"Unsupported external BMao backend: {search_backend}")

    @staticmethod
    def _score_app_bmao_beam_candidate(candidate_ged):
        return float(candidate_ged)

    def _build_avg_refine_candidate(self, candidate_pair):
        final_probabilities = candidate_pair["final_probabilities"].detach().cpu().to(dtype=torch.float)
        final_scores = candidate_pair["final_scores"].detach().cpu().to(dtype=torch.float)
        total_candidates = int(final_probabilities.shape[0])
        if total_candidates <= 0:
            raise ValueError("avg refine mode requires at least one candidate matrix.")

        avg_probabilities = final_probabilities.mean(dim=0)
        avg_scores = final_scores.mean(dim=0)
        n1 = int(candidate_pair["n1"])
        n2 = int(candidate_pair["n2"])
        graph_1 = int(candidate_pair["pair"][0].item())
        graph_2 = int(candidate_pair["pair"][1].item())
        full_size = torch.tensor([min(n1, n2)], dtype=torch.long, device=self.device)
        score_input = avg_scores.unsqueeze(0).to(device=self.device, dtype=torch.float)
        decoded = self.batched_constrained_matching_decode_by_mode(score_input, full_size)
        mode_now = str(getattr(self.args, "constrained_greedy_mode", "global_n3"))
        if self._uses_top1_decode_mode(mode_now):
            decoded = self.batched_repair_matching_completion(score_input, decoded, full_size)
        avg_matching = decoded[0, :n1, :n2].detach().cpu().to(dtype=torch.float)
        _row_mapping, matched_pairs = self._dense_matching_to_row_mapping(avg_matching)
        avg_candidate_ged = self._recompute_dense_minimal_ged_for_pair(graph_1, graph_2, matched_pairs)
        selection_metrics = self._compute_app_bmao_candidate_confidence(avg_matching, avg_probabilities)
        selection_metrics.update(
            {
                "avg_mode": True,
                "source_candidate_count": int(total_candidates),
                "source_candidate_indices": list(range(total_candidates)),
                "synthetic_candidate_ged": float(avg_candidate_ged),
            }
        )
        return {
            "candidate_index": -1,
            "source_candidate_indices": list(range(total_candidates)),
            "final_matching": avg_matching,
            "final_probabilities": avg_probabilities,
            "final_scores": avg_scores,
            "candidate_ged": float(avg_candidate_ged),
            "selection_metrics": selection_metrics,
        }

    def _select_app_bmao_candidate_indices(self, candidate_pair, mode, candidate_budget, beam_score="ged"):
        candidate_geds = candidate_pair["candidate_ged"].detach().cpu()
        total_candidates = int(candidate_geds.shape[0])
        baseline_ged = float(candidate_geds.min().item())

        if mode == "best":
            candidate_indices = [int(candidate_pair["best_index"])]
            return candidate_indices, baseline_ged, {}

        if mode == "all" or candidate_budget <= 0 or candidate_budget >= total_candidates:
            candidate_indices = list(range(total_candidates))
            return candidate_indices, baseline_ged, {}

        if mode != "beam":
            raise ValueError(f"Unsupported --app-bmao-postprocess-mode: {mode}")

        scored_candidates = []
        selection_metadata = {}
        for candidate_index in range(total_candidates):
            confidence_metrics = self._compute_app_bmao_candidate_confidence(
                final_matching=candidate_pair["final_matchings"][candidate_index],
                final_probabilities=candidate_pair["final_probabilities"][candidate_index],
            )
            beam_value = self._score_app_bmao_beam_candidate(
                candidate_ged=candidate_geds[candidate_index].item()
            )
            scored_candidates.append((beam_value, float(candidate_geds[candidate_index].item()), candidate_index))
            selection_metadata[int(candidate_index)] = {
                "beam_score": float(beam_value),
                **confidence_metrics,
            }

        scored_candidates.sort(key=lambda item: (item[0], item[1], item[2]))
        candidate_indices = sorted(item[2] for item in scored_candidates[:candidate_budget])
        return candidate_indices, baseline_ged, selection_metadata

    def _prepare_app_bmao_refine_candidates(self, candidate_pair, mode, candidate_budget, beam_score="ged"):
        if mode == "avg":
            avg_candidate = self._build_avg_refine_candidate(candidate_pair)
            return [avg_candidate], float(avg_candidate["candidate_ged"]), {
                int(avg_candidate["candidate_index"]): dict(avg_candidate["selection_metrics"]),
            }

        candidate_indices, baseline_ged, selection_metadata = self._select_app_bmao_candidate_indices(
            candidate_pair=candidate_pair,
            mode=mode,
            candidate_budget=candidate_budget,
            beam_score=beam_score,
        )
        candidate_specs = []
        for candidate_index in candidate_indices:
            candidate_specs.append(
                {
                    "candidate_index": int(candidate_index),
                    "source_candidate_indices": [int(candidate_index)],
                    "final_matching": candidate_pair["final_matchings"][candidate_index],
                    "final_probabilities": candidate_pair["final_probabilities"][candidate_index],
                    "final_scores": candidate_pair["final_scores"][candidate_index],
                    "candidate_ged": float(candidate_pair["candidate_ged"][candidate_index]),
                    "selection_metrics": dict(selection_metadata.get(int(candidate_index), {})),
                }
            )
        return candidate_specs, baseline_ged, selection_metadata

    def _run_app_bmao_for_pair(self, task):
        candidate_pair = task["candidate_pair"]
        pair_index = int(task["pair_index"])
        graph_1 = int(candidate_pair["pair"][0].item())
        graph_2 = int(candidate_pair["pair"][1].item())
        graph_1_gid = int(candidate_pair["pair_gid"][0].item())
        graph_2_gid = int(candidate_pair["pair_gid"][1].item())
        gt_ged = float(candidate_pair["gt_ged"])
        candidate_budget = int(task.get("candidate_budget", 0))
        beam_score = "ged"
        candidate_specs, baseline_ged, selection_metadata = self._prepare_app_bmao_refine_candidates(
            candidate_pair=candidate_pair,
            mode=task["mode"],
            candidate_budget=candidate_budget,
            beam_score=beam_score,
        )

        solver_config = task["solver_config"]
        stdin_enable = bool(task.get("stdin_enable", False))
        query_payload = task["graph_cache"][graph_1_gid]
        db_payload = task["graph_cache"][graph_2_gid]
        query_txt = None if stdin_enable else Path(query_payload)
        db_txt = None if stdin_enable else Path(db_payload)

        best_refined = None
        best_candidate_index = None
        best_anchor_count = None
        best_reported_ged = None
        best_matching_payload = None
        best_candidate_metrics = None
        total_wall_time_s = 0.0
        total_solver_time_s = 0.0
        total_search_space = 0
        total_external_prune_profile = defaultdict(int)
        best_external_prune_profile = None
        diffusion_ub_enable = bool(task.get("diffusion_ub_enable", False))
        best_diffusion_ub_threshold = None
        if diffusion_ub_enable:
            best_refined = float(baseline_ged)
            best_reported_ged = int(round(float(baseline_ged)))
            best_candidate_index = int(candidate_specs[0]["candidate_index"]) if candidate_specs else None
            best_anchor_count = 0
            best_candidate_metrics = selection_metadata.get(int(best_candidate_index)) if best_candidate_index is not None else None

        for candidate_spec in candidate_specs:
            candidate_index = int(candidate_spec["candidate_index"])
            final_matching = candidate_spec["final_matching"]
            final_probabilities = candidate_spec["final_probabilities"]
            candidate_ged = float(candidate_spec["candidate_ged"])
            disable_incumbent_ub = bool(task.get("disable_incumbent_ub", False))
            incumbent_enable = bool(solver_config.get("supports_incumbent", False)) and not disable_incumbent_ub
            incumbent_ged = int(math.ceil(candidate_ged))
            anchors = self._select_top_probability_anchors(
                final_matching=final_matching,
                final_probabilities=final_probabilities,
                anchor_ratio=task["anchor_ratio"],
            )
            diffusion_ub_threshold = None
            if diffusion_ub_enable:
                diffusion_ub_threshold = max(0, int(math.ceil(candidate_ged)) - 1)
            if stdin_enable:
                anchor_text = self._render_anchor_text(anchors)
                incumbent_text = self._render_dense_matching_text(final_matching) if incumbent_enable else ""
                payload = self._compose_app_bmao_stdin_payload(
                    query_text=str(query_payload),
                    db_text=str(db_payload),
                    anchor_text=anchor_text,
                    incumbent_text=incumbent_text,
                )
                command = [
                    task["ged_bin"],
                    "--stdin-payload",
                    "-m",
                    "pair",
                    "-p",
                    str(solver_config["paradigm"]),
                    "-l",
                    str(solver_config["lower_bound"]),
                    "-g",
                    "-x",
                ]
                if bool(solver_config.get("supports_stop", True)):
                    command.extend(["-k", str(int(task["search_states"]))])
                if incumbent_enable:
                    command.extend(["--incumbent-ged", str(int(incumbent_ged))])
            else:
                anchor_path = Path(task["work_dir"]) / f"anchor_{pair_index}_{candidate_index}.txt"
                self._write_anchor_file(anchor_path, anchors)
                incumbent_path = Path(task["work_dir"]) / f"incumbent_{pair_index}_{candidate_index}.txt"
                if incumbent_enable:
                    self._write_dense_matching_file(incumbent_path, final_matching)
                payload = None
                command = [
                    task["ged_bin"],
                    "-d",
                    str(db_txt),
                    "-q",
                    str(query_txt),
                    "-m",
                    "pair",
                    "-p",
                    str(solver_config["paradigm"]),
                    "-l",
                    str(solver_config["lower_bound"]),
                    "-g",
                    "-x",
                    "-a",
                    str(anchor_path),
                ]
                if bool(solver_config.get("supports_stop", True)):
                    command.extend(["-k", str(int(task["search_states"]))])
                if incumbent_enable:
                    command.extend(["--incumbent", str(incumbent_path), "--incumbent-ged", str(int(incumbent_ged))])
            if diffusion_ub_threshold is not None:
                command.extend(["-t", str(int(diffusion_ub_threshold))])
            wall_start = time.time()
            proc = subprocess.run(
                command,
                cwd=task["model_dir"],
                text=True,
                input=payload,
                capture_output=True,
                timeout=float(task["timeout_seconds"]),
                check=True,
            )
            wall_elapsed = time.time() - wall_start
            total_wall_time_s += wall_elapsed
            output_text = (proc.stdout or "") + (proc.stderr or "")
            if not output_text.strip() and diffusion_ub_enable:
                constrained_ged, mapping_payload, solver_us, search_space, prune_profile = -1, None, None, None, None
            else:
                constrained_ged, mapping_payload, solver_us, search_space, prune_profile = self._parse_app_bmao_solver_output(output_text)
            if solver_us is not None:
                total_solver_time_s += float(solver_us) / 1e6
            if search_space is not None:
                total_search_space += int(search_space)
            if prune_profile:
                for key, value in prune_profile.items():
                    if isinstance(value, (int, float)):
                        total_external_prune_profile[key] += int(value)
            if constrained_ged == -1 and diffusion_ub_enable:
                continue
            if mapping_payload is None or "query_to_db" not in mapping_payload:
                raise RuntimeError(
                    f"{solver_config['backend']} postprocess did not return a usable matching payload for dense GED recomputation."
                )
            normalized_matching_pairs = self._normalize_app_bmao_matching_payload(mapping_payload)
            normalized_matching_set = {(int(row), int(col)) for row, col in normalized_matching_pairs}
            anchor_pairs = [(int(row), int(col)) for row, col, _prob in anchors]
            anchor_violations = [
                [int(row), int(col)]
                for row, col in anchor_pairs
                if (int(row), int(col)) not in normalized_matching_set
            ]
            if anchor_violations:
                raise RuntimeError(
                    f"{solver_config['backend']} returned a mapping that violates anchored pairs: "
                    f"pair_index={pair_index}, candidate_index={candidate_index}, "
                    f"violations={anchor_violations[:10]}, num_violations={len(anchor_violations)}"
                )
            recomputed_ged = self._recompute_dense_minimal_ged_for_pair(
                graph_1=graph_1,
                graph_2=graph_2,
                matching_pairs=normalized_matching_pairs,
            )
            if best_refined is None or recomputed_ged < best_refined:
                best_refined = float(recomputed_ged)
                best_candidate_index = int(candidate_index)
                best_anchor_count = int(len(anchors))
                best_reported_ged = int(constrained_ged)
                best_matching_payload = mapping_payload
                best_candidate_metrics = dict(candidate_spec.get("selection_metrics", {}))
                best_diffusion_ub_threshold = diffusion_ub_threshold
                best_anchor_violations = anchor_violations
                best_external_prune_profile = prune_profile

        if best_refined is None or best_reported_ged is None:
            raise RuntimeError(f"App-BMao postprocess produced no valid refined result for pair_index={pair_index}.")
        aggregated_source_candidate_indices = sorted(
            {
                int(idx)
                for spec in candidate_specs
                for idx in spec.get("source_candidate_indices", [])
            }
        )

        return {
            "pair_index": pair_index,
            "graph_1": graph_1,
            "graph_2": graph_2,
            "graph_1_gid": graph_1_gid,
            "graph_2_gid": graph_2_gid,
            "mode": task["mode"],
            "gt_ged": gt_ged,
            "baseline_candidate_ged": baseline_ged,
            "refined_ged": float(best_refined),
            "app_bmao_reported_ged": float(best_reported_ged),
            "delta_refined_minus_baseline": float(best_refined - baseline_ged),
            "delta_refined_minus_gt": float(best_refined - gt_ged),
            "best_candidate_index": best_candidate_index,
            "best_anchor_count": best_anchor_count,
            "best_anchor_violations": best_anchor_violations if "best_anchor_violations" in locals() else [],
            "best_anchor_violation_count": len(best_anchor_violations) if "best_anchor_violations" in locals() else 0,
            "num_candidates_available": int(candidate_pair["candidate_ged"].shape[0]),
            "num_candidates_processed": int(len(candidate_specs)),
            "aggregated_source_candidate_count": int(len(aggregated_source_candidate_indices)),
            "aggregated_source_candidate_indices": aggregated_source_candidate_indices,
            "sum_wall_time_s": total_wall_time_s,
            "sum_solver_time_s": total_solver_time_s,
            "external_search_space": int(total_search_space),
            "external_prune_profile": {key: int(value) for key, value in total_external_prune_profile.items()},
            "best_external_prune_profile": best_external_prune_profile,
            "diffusion_ub_enable": bool(diffusion_ub_enable),
            "disable_incumbent_ub": bool(task.get("disable_incumbent_ub", False)),
            "diffusion_ub_threshold": best_diffusion_ub_threshold,
            "best_matching_payload": best_matching_payload,
            "beam_candidate_budget": int(candidate_budget),
            "beam_score_name": beam_score,
            "selected_candidate_indices": [int(spec["candidate_index"]) for spec in candidate_specs],
            "selected_candidate_metrics": selection_metadata,
            "best_candidate_metrics": best_candidate_metrics,
            "solver_backend": solver_config["backend"],
            "solver_paradigm": solver_config["paradigm"],
            "solver_lower_bound": solver_config["lower_bound"],
        }

    def run_app_bmao_postprocess(self, candidate_pairs, testing_graph_set, top_k_approach, test_k, pair_index_offset=0, finalize=True):
        if not candidate_pairs:
            print("Skip integrated App-BMao postprocess: no candidate blobs were collected.")
            return

        anchor_ratio = float(getattr(self.args, "app_bmao_anchor_ratio", 0.6))
        if anchor_ratio < 0.0 or anchor_ratio > 1.0:
            raise ValueError("--app-bmao-anchor-ratio must be in [0, 1].")

        postprocess_mode = str(getattr(self.args, "app_bmao_postprocess_mode", "best"))
        search_states = int(getattr(self.args, "app_bmao_search_states", 100))
        workers = max(1, int(getattr(self.args, "app_bmao_workers", 1)))
        search_batch_size_arg = int(getattr(self.args, "app_bmao_pair_chunk_size", 0))
        search_batch_size = search_batch_size_arg
        candidate_budget = int(getattr(self.args, "app_bmao_candidate_budget", 4))
        beam_score = "ged"
        search_backend = str(getattr(self.args, "app_bmao_search_backend", "external_app_bmao"))
        profile_enable = bool(getattr(self.args, "app_bmao_profile_enable", False))
        profiling_report = None

        if self._is_external_bmao_backend(search_backend):
            solver_config = self._resolve_external_bmao_backend_config(search_backend)
            ged_bin = str(solver_config["ged_bin"])
            timeout_seconds = float(getattr(self.args, "app_bmao_timeout_seconds", 30.0))
            if not ged_bin or not os.path.exists(ged_bin):
                raise FileNotFoundError(f"Anchored {search_backend} binary not found: {ged_bin}")
            stdin_enable = bool(getattr(self.args, "app_bmao_stdin_enable", False))
            if stdin_enable and not bool(solver_config.get("supports_stdin", False)):
                raise ValueError(f"{search_backend} does not support --app-bmao-stdin-enable; use file-based graph payloads.")
            model_dir = str(Path(ged_bin).resolve().parent)
            dataset_root, _resolved_dataset_name = resolve_dataset_root(self.data_path, self.args.dataset)

            with tempfile.TemporaryDirectory(prefix="dense_app_bmao_") as temp_dir:
                graph_cache = self._build_graph_text_cache(candidate_pairs, dataset_root, temp_dir)
                tasks = []
                for pair_index, candidate_pair in enumerate(candidate_pairs):
                    tasks.append(
                        {
                            "pair_index": pair_index,
                            "candidate_pair": candidate_pair,
                            "mode": postprocess_mode,
                            "search_states": search_states,
                            "anchor_ratio": anchor_ratio,
                            "candidate_budget": candidate_budget,
                            "beam_score": beam_score,
                            "ged_bin": ged_bin,
                            "solver_config": solver_config,
                            "model_dir": model_dir,
                            "timeout_seconds": timeout_seconds,
                            "graph_cache": graph_cache,
                            "work_dir": temp_dir,
                            "stdin_enable": stdin_enable,
                            "diffusion_ub_enable": bool(getattr(self.args, "app_bmao_diffusion_ub_enable", False)),
                            "disable_incumbent_ub": bool(getattr(self.args, "app_bmao_disable_incumbent_ub", False)),
                        }
                    )

                pair_results = []
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    futures = {executor.submit(self._run_app_bmao_for_pair, task): task for task in tasks}
                    total = len(tasks)
                    for future in as_completed(futures):
                        pair_results.append(future.result())
        else:
            runner = InternalAppBmaoSearchRunner(
                trainer=self,
                backend=search_backend,
                branch_topk=int(getattr(self.args, "v9_branch_width", 4)),
                unmatched_cost=1.0,
                beam_width=int(getattr(self.args, "v9_beam_width", 4)),
                beam_steps=0,
                score_source="scores",
                assignment_backend=str(getattr(self.args, "app_bmao_assignment_backend", "auto")),
                bestfirst_topm=int(getattr(self.args, "v9_rerank_pool", 32)),
            )
            progress_start = time.time()
            pair_results = []
            total = len(candidate_pairs)
            chunk_size = total if search_batch_size <= 0 else max(1, search_batch_size)
            completed = 0
            profile_chunk_reports = []
            for chunk_index, chunk_start in enumerate(range(0, total, chunk_size)):
                chunk_pairs = candidate_pairs[chunk_start: chunk_start + chunk_size]
                chunk_run_start = time.perf_counter()
                chunk_results = runner.run(
                    candidate_pairs=chunk_pairs,
                    postprocess_mode=postprocess_mode,
                    candidate_budget=candidate_budget,
                    beam_score=beam_score,
                )
                chunk_elapsed = time.time() - progress_start
                chunk_wall_s = time.perf_counter() - chunk_run_start
                if profile_enable and runner.last_profile_report is not None:
                    profile_chunk_reports.append({
                        "chunk_index": int(chunk_index),
                        "pair_start": int(chunk_start),
                        "pair_count": int(len(chunk_pairs)),
                        "chunk_wall_time_s": float(chunk_wall_s),
                        "profile": runner.last_profile_report,
                    })
                pair_results.extend(chunk_results)
                completed += len(chunk_pairs)
                if total > 0:
                    sys.stdout.write(
                        "\r[Internal App-BMao] backend={} progress {}/{} elapsed={}".format(
                            search_backend,
                            completed,
                            total,
                            self._format_progress_seconds(chunk_elapsed),
                        )
                    )
                    sys.stdout.flush()
            elapsed_seconds = time.time() - progress_start
            if total > 0:
                sys.stdout.write("\n")
                sys.stdout.flush()
            if total > 0:
                sys.stdout.write("\n")
                sys.stdout.flush()
            if pair_results:
                per_pair_wall = elapsed_seconds / float(len(pair_results))
                for item in pair_results:
                    item["sum_wall_time_s"] = per_pair_wall
                    item["sum_solver_time_s"] = per_pair_wall
                print(
                    "[Internal App-BMao] backend={} pairs={} elapsed={} search_batch_size={}".format(
                        search_backend,
                        len(pair_results),
                        self._format_progress_seconds(elapsed_seconds),
                        chunk_size,
                    ),
                    flush=True,
                )
            if profile_enable:
                agg_totals = defaultdict(float)
                agg_counts = defaultdict(int)
                agg_metadata = defaultdict(float)
                agg_metadata_lists = defaultdict(list)
                agg_metadata_min = {}
                agg_metadata_max = {}
                agg_metadata_fixed = {}

                def _percentile(values, pct):
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

                for chunk_item in profile_chunk_reports:
                    profile = chunk_item.get("profile") or {}
                    for key, value in (profile.get("stage_totals_s") or {}).items():
                        agg_totals[key] += float(value)
                    for key, value in (profile.get("stage_counts") or {}).items():
                        agg_counts[key] += int(value)
                    metadata = profile.get("metadata") or {}
                    for key, value in metadata.items():
                        if key in {'search.v5_lap_matrices_per_call_samples', 'search.v5_valid_actions_per_parent_samples', 'search.v5_valid_actions_per_iter_samples'} and isinstance(value, list):
                            agg_metadata_lists[key].extend(int(v) for v in value)
                        elif key in {'search.v5_lap_num_calls', 'search.v5_lap_num_matrices', 'search.v5_lap_matrix_size_weighted_sum', 'search.v5_lap_matrix_size_count', 'v5_lap_total_actions', 'v5_lap_valid_actions', 'search.v5_pre_cap_valid_actions_total', 'search.v5_post_cap_valid_actions_total', 'search.v5_pre_ub_selected_total', 'search.v5_post_ub_selected_total', 'search.v5_completion_candidates_total', 'search.v5_completion_unique_total', 'search.v5_expanded_parents_total'}:
                            agg_metadata[key] += float(value)
                        elif key in {'search.v5_lap_matrix_size_min'}:
                            agg_metadata_min[key] = float(value) if key not in agg_metadata_min else min(agg_metadata_min[key], float(value))
                        elif key in {'search.v5_lap_matrix_size_max'}:
                            agg_metadata_max[key] = float(value) if key not in agg_metadata_max else max(agg_metadata_max[key], float(value))
                        elif key in {'search.v5_lap_chunk_size'}:
                            agg_metadata_fixed.setdefault(key, int(value))
                aggregate_metadata = {key: float(agg_metadata[key]) for key in sorted(agg_metadata.keys())}
                for key, value in agg_metadata_min.items():
                    aggregate_metadata[key] = float(value)
                for key, value in agg_metadata_max.items():
                    aggregate_metadata[key] = float(value)
                for key, value in agg_metadata_fixed.items():
                    aggregate_metadata[key] = int(value)
                lap_call_samples = agg_metadata_lists.get('search.v5_lap_matrices_per_call_samples', [])
                if lap_call_samples:
                    aggregate_metadata['search.v5_lap_matrices_per_call_mean'] = float(sum(lap_call_samples) / len(lap_call_samples))
                    aggregate_metadata['search.v5_lap_matrices_per_call_p50'] = float(_percentile(lap_call_samples, 50.0))
                    aggregate_metadata['search.v5_lap_matrices_per_call_p95'] = float(_percentile(lap_call_samples, 95.0))
                    aggregate_metadata['search.v5_lap_matrices_per_call_max'] = int(max(lap_call_samples))
                parent_valid_samples = agg_metadata_lists.get('search.v5_valid_actions_per_parent_samples', [])
                if parent_valid_samples:
                    aggregate_metadata['search.v5_valid_actions_per_parent_mean'] = float(sum(parent_valid_samples) / len(parent_valid_samples))
                    aggregate_metadata['search.v5_valid_actions_per_parent_p50'] = float(_percentile(parent_valid_samples, 50.0))
                    aggregate_metadata['search.v5_valid_actions_per_parent_p95'] = float(_percentile(parent_valid_samples, 95.0))
                    aggregate_metadata['search.v5_valid_actions_per_parent_max'] = int(max(parent_valid_samples))
                iter_valid_samples = agg_metadata_lists.get('search.v5_valid_actions_per_iter_samples', [])
                if iter_valid_samples:
                    aggregate_metadata['search.v5_valid_actions_per_iter_mean'] = float(sum(iter_valid_samples) / len(iter_valid_samples))
                    aggregate_metadata['search.v5_valid_actions_per_iter_p50'] = float(_percentile(iter_valid_samples, 50.0))
                    aggregate_metadata['search.v5_valid_actions_per_iter_p95'] = float(_percentile(iter_valid_samples, 95.0))
                    aggregate_metadata['search.v5_valid_actions_per_iter_max'] = int(max(iter_valid_samples))
                lap_matrix_count = int(aggregate_metadata.get('search.v5_lap_matrix_size_count', 0) or 0)
                if lap_matrix_count > 0:
                    aggregate_metadata['search.v5_lap_matrix_size_mean'] = float(aggregate_metadata.get('search.v5_lap_matrix_size_weighted_sum', 0.0) / lap_matrix_count)
                total_actions = int(aggregate_metadata.get('v5_lap_total_actions', 0) or 0)
                valid_actions = int(aggregate_metadata.get('v5_lap_valid_actions', 0) or 0)
                if total_actions > 0:
                    aggregate_metadata['v5_lap_valid_ratio_ppm'] = int(round(1_000_000.0 * valid_actions / total_actions))
                pre_cap_total = int(aggregate_metadata.get('search.v5_pre_cap_valid_actions_total', 0) or 0)
                post_cap_total = int(aggregate_metadata.get('search.v5_post_cap_valid_actions_total', 0) or 0)
                if pre_cap_total > 0:
                    aggregate_metadata['search.v5_post_cap_valid_ratio_ppm'] = int(round(1_000_000.0 * post_cap_total / pre_cap_total))
                    aggregate_metadata['search.v5_pre_lap_action_pruned_total'] = int(max(pre_cap_total - post_cap_total, 0))
                pre_ub_total = int(aggregate_metadata.get('search.v5_pre_ub_selected_total', 0) or 0)
                post_ub_total = int(aggregate_metadata.get('search.v5_post_ub_selected_total', 0) or 0)
                if pre_ub_total >= post_ub_total:
                    aggregate_metadata['search.v5_pruned_by_ub_total'] = int(pre_ub_total - post_ub_total)
                completion_total = int(aggregate_metadata.get('search.v5_completion_candidates_total', 0) or 0)
                completion_unique = int(aggregate_metadata.get('search.v5_completion_unique_total', 0) or 0)
                if completion_total > 0:
                    aggregate_metadata['search.v5_completion_unique_ratio_ppm'] = int(round(1_000_000.0 * completion_unique / completion_total))
                profiling_report = {
                    "enabled": True,
                    "search_batch_size": int(chunk_size),
                    "outer_total_wall_time_s": float(elapsed_seconds),
                    "outer_chunk_records": profile_chunk_reports,
                    "aggregate_stage_totals_s": {key: float(agg_totals[key]) for key in sorted(agg_totals.keys())},
                    "aggregate_stage_counts": {key: int(agg_counts[key]) for key in sorted(agg_counts.keys())},
                    "aggregate_stage_avg_ms": {key: (1000.0 * agg_totals[key] / agg_counts[key]) if agg_counts[key] > 0 else None for key in sorted(agg_totals.keys())},
                    "aggregate_metadata": aggregate_metadata,
                }

        if pair_index_offset != 0:
            for item in pair_results:
                item["pair_index"] = int(item["pair_index"] + pair_index_offset)
            if profiling_report is not None:
                shifted_records = []
                for record in profiling_report.get("outer_chunk_records", []):
                    updated = dict(record)
                    if "pair_start" in updated:
                        updated["pair_start"] = int(updated["pair_start"] + pair_index_offset)
                    shifted_records.append(updated)
                profiling_report = dict(profiling_report)
                profiling_report["outer_chunk_records"] = shifted_records
        if not finalize:
            return pair_results, profiling_report

        self._finalize_app_bmao_postprocess(
            pair_results=pair_results,
            testing_graph_set=testing_graph_set,
            top_k_approach=top_k_approach,
            test_k=test_k,
            postprocess_mode=postprocess_mode,
            anchor_ratio=anchor_ratio,
            search_states=(search_states if self._is_external_bmao_backend(search_backend) else 0),
            workers=(workers if self._is_external_bmao_backend(search_backend) else 1),
            candidate_budget=candidate_budget,
            beam_score=beam_score,
            search_backend=search_backend,
            profiling_report=profiling_report,
        )

    @staticmethod
    def _merge_app_bmao_profiling_reports(profiling_reports):
        reports = [report for report in profiling_reports if report is not None]
        if not reports:
            return None
        if len(reports) == 1:
            return reports[0]

        merged_stage_totals = defaultdict(float)
        merged_stage_counts = defaultdict(int)
        merged_outer_chunk_records = []
        merged_metadata = {}
        numeric_sum_keys = {
            'search.v5_lap_num_calls', 'search.v5_lap_num_matrices', 'search.v5_lap_matrix_size_weighted_sum',
            'search.v5_lap_matrix_size_count', 'v5_lap_total_actions', 'v5_lap_valid_actions',
            'search.v5_pre_cap_valid_actions_total', 'search.v5_post_cap_valid_actions_total',
            'search.v5_pre_ub_selected_total', 'search.v5_post_ub_selected_total',
            'search.v5_completion_candidates_total', 'search.v5_completion_unique_total',
            'search.v5_expanded_parents_total', 'outer_total_wall_time_s',
            'pipeline.prepare_wait_time_s', 'pipeline.visible_block_time_s',
            'pipeline.chunk_gpu_refine_wall_time_s',
        }
        numeric_min_keys = {'search.v5_lap_matrix_size_min'}
        numeric_max_keys = {'search.v5_lap_matrix_size_max'}
        fixed_keys = {
            'search.v5_lap_chunk_size',
            'pipeline.overlap_delay_batches',
        }

        for report in reports:
            for key, value in (report.get('aggregate_stage_totals_s') or {}).items():
                merged_stage_totals[key] += float(value)
            for key, value in (report.get('aggregate_stage_counts') or {}).items():
                merged_stage_counts[key] += int(value)
            merged_outer_chunk_records.extend(report.get('outer_chunk_records', []))
            metadata = report.get('aggregate_metadata') or {}
            for key, value in metadata.items():
                if key in numeric_sum_keys:
                    merged_metadata[key] = float(merged_metadata.get(key, 0.0)) + float(value)
                elif key in numeric_min_keys:
                    merged_metadata[key] = float(value) if key not in merged_metadata else min(float(merged_metadata[key]), float(value))
                elif key in numeric_max_keys:
                    merged_metadata[key] = float(value) if key not in merged_metadata else max(float(merged_metadata[key]), float(value))
                elif key in fixed_keys:
                    merged_metadata.setdefault(key, int(value))

        lap_matrix_count = int(merged_metadata.get('search.v5_lap_matrix_size_count', 0) or 0)
        if lap_matrix_count > 0:
            merged_metadata['search.v5_lap_matrix_size_mean'] = float(merged_metadata.get('search.v5_lap_matrix_size_weighted_sum', 0.0) / lap_matrix_count)
        total_actions = int(merged_metadata.get('v5_lap_total_actions', 0) or 0)
        valid_actions = int(merged_metadata.get('v5_lap_valid_actions', 0) or 0)
        if total_actions > 0:
            merged_metadata['v5_lap_valid_ratio_ppm'] = int(round(1_000_000.0 * valid_actions / total_actions))
        pre_cap_total = int(merged_metadata.get('search.v5_pre_cap_valid_actions_total', 0) or 0)
        post_cap_total = int(merged_metadata.get('search.v5_post_cap_valid_actions_total', 0) or 0)
        if pre_cap_total > 0:
            merged_metadata['search.v5_post_cap_valid_ratio_ppm'] = int(round(1_000_000.0 * post_cap_total / pre_cap_total))
            merged_metadata['search.v5_pre_lap_action_pruned_total'] = int(max(pre_cap_total - post_cap_total, 0))
        pre_ub_total = int(merged_metadata.get('search.v5_pre_ub_selected_total', 0) or 0)
        post_ub_total = int(merged_metadata.get('search.v5_post_ub_selected_total', 0) or 0)
        if pre_ub_total >= post_ub_total:
            merged_metadata['search.v5_pruned_by_ub_total'] = int(pre_ub_total - post_ub_total)
        completion_total = int(merged_metadata.get('search.v5_completion_candidates_total', 0) or 0)
        completion_unique = int(merged_metadata.get('search.v5_completion_unique_total', 0) or 0)
        if completion_total > 0:
            merged_metadata['search.v5_completion_unique_ratio_ppm'] = int(round(1_000_000.0 * completion_unique / completion_total))

        return {
            'enabled': all(bool(report.get('enabled', False)) for report in reports),
            'search_batch_size': int(reports[0].get('search_batch_size', reports[0].get('pair_chunk_size', 0)) or 0),
            'outer_total_wall_time_s': float(sum(float(report.get('outer_total_wall_time_s', 0.0) or 0.0) for report in reports)),
            'outer_chunk_records': merged_outer_chunk_records,
            'aggregate_stage_totals_s': {key: float(merged_stage_totals[key]) for key in sorted(merged_stage_totals.keys())},
            'aggregate_stage_counts': {key: int(merged_stage_counts[key]) for key in sorted(merged_stage_counts.keys())},
            'aggregate_stage_avg_ms': {key: (1000.0 * merged_stage_totals[key] / merged_stage_counts[key]) if merged_stage_counts[key] > 0 else None for key in sorted(merged_stage_totals.keys())},
            'aggregate_metadata': merged_metadata,
        }

    @staticmethod
    def _summarize_app_bmao_profile(profiling_report):
        if profiling_report is None:
            return None

        stage_totals = profiling_report.get("aggregate_stage_totals_s") or {}
        outer_total = float(profiling_report.get("outer_total_wall_time_s", 0.0) or 0.0)
        run_context_build = float(stage_totals.get("run.context_build", 0.0) or 0.0)
        run_search = float(stage_totals.get("run.search", 0.0) or 0.0)
        run_finalize = float(stage_totals.get("run.finalize_results", 0.0) or 0.0)
        run_total = float(stage_totals.get("run.total", 0.0) or 0.0)
        context_summary = {}
        search_summary = {}
        for key, value in stage_totals.items():
            if key.startswith("context."):
                if key in {"context.total", "context.pair_total"}:
                    continue
                context_summary[key[len("context."):]] = round(float(value), 5)
            elif key.startswith("search."):
                search_summary[key[len("search."):]] = round(float(value), 5)
        summarized_total = run_context_build + run_search + run_finalize
        return {
            "outer_total_wall_time_s": round(outer_total, 5),
            "run_total_s": round(run_total, 5),
            "run_context_build_s": round(run_context_build, 5),
            "run_search_s": round(run_search, 5),
            "run_finalize_results_s": round(run_finalize, 5),
            "run_other_s": round(max(0.0, run_total - summarized_total), 5),
            "context_total_s": round(float(stage_totals.get("context.total", 0.0) or 0.0), 5),
            "context_pair_total_s": round(float(stage_totals.get("context.pair_total", 0.0) or 0.0), 5),
            "context_stage_totals_s": context_summary,
            "search_stage_totals_s": search_summary,
        }

    @staticmethod
    def _format_progress_seconds(seconds):
        seconds = max(0.0, float(seconds))
        total_seconds = int(round(seconds))
        hours, remainder = divmod(total_seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        if hours > 0:
            return f"{hours}h{minutes:02d}m{secs:02d}s"
        if minutes > 0:
            return f"{minutes}m{secs:02d}s"
        return f"{secs}s"

    def _finalize_app_bmao_postprocess(
        self,
        pair_results,
        testing_graph_set,
        top_k_approach,
        test_k,
        postprocess_mode,
        anchor_ratio,
        search_states,
        workers,
        candidate_budget,
        beam_score,
        search_backend,
        profiling_report=None,
    ):
        pair_results.sort(key=lambda item: item["pair_index"])
        num_pairs = max(len(pair_results), 1)
        avg_baseline = sum(item["baseline_candidate_ged"] for item in pair_results) / num_pairs
        avg_refined = sum(item["refined_ged"] for item in pair_results) / num_pairs
        avg_app_bmao_reported = sum(item["app_bmao_reported_ged"] for item in pair_results) / num_pairs
        avg_gt = sum(item["gt_ged"] for item in pair_results) / num_pairs
        total_solver_time_s = sum(item["sum_solver_time_s"] for item in pair_results)
        total_wall_time_s = sum(item["sum_wall_time_s"] for item in pair_results)
        external_search_space_total = sum(int(item.get("external_search_space", 0) or 0) for item in pair_results)
        external_prune_profile_totals = defaultdict(int)
        for item in pair_results:
            for key, value in (item.get("external_prune_profile") or {}).items():
                if isinstance(value, (int, float)):
                    external_prune_profile_totals[key] += int(value)
        total_candidates_available = sum(int(item.get("num_candidates_available", item["num_candidates_processed"])) for item in pair_results)
        total_candidates_processed = sum(int(item["num_candidates_processed"]) for item in pair_results)
        improved_count = sum(1 for item in pair_results if item["delta_refined_minus_baseline"] < 0)
        equal_count = sum(1 for item in pair_results if item["delta_refined_minus_baseline"] == 0)
        worse_count = sum(1 for item in pair_results if item["delta_refined_minus_baseline"] > 0)
        summary = {
            "dataset": self.args.dataset,
            "graph_set": testing_graph_set,
            "topk_approach": top_k_approach,
            "test_k": int(test_k),
            "mode": postprocess_mode,
            "anchor_ratio": anchor_ratio,
            "search_states": search_states,
            "workers": workers,
            "candidate_budget": candidate_budget,
            "beam_score": beam_score,
            "search_backend": search_backend,
            "num_pairs": len(pair_results),
            "total_candidates_available": total_candidates_available,
            "total_candidates_processed": total_candidates_processed,
            "avg_baseline_candidate_ged": avg_baseline,
            "avg_refined_ged": avg_refined,
            "avg_app_bmao_reported_ged": avg_app_bmao_reported,
            "avg_gt_ged": avg_gt,
            "avg_delta_refined_minus_baseline": avg_refined - avg_baseline,
            "avg_delta_refined_minus_gt": avg_refined - avg_gt,
            "total_solver_time_s": total_solver_time_s,
            "total_wall_time_s": total_wall_time_s,
            "external_search_space_total": int(external_search_space_total),
            "external_prune_profile_totals": {key: int(value) for key, value in sorted(external_prune_profile_totals.items())},
            "improved_pair_count": improved_count,
            "equal_pair_count": equal_count,
            "worse_pair_count": worse_count,
            "pairs": pair_results,
        }
        if profiling_report is not None:
            summary["profiling"] = profiling_report
        output_json_path = self._result_path(
            "result_DiffGED_dense_app_bmao_{}_{}_{}_{}_backend-{}_mode-{}_anchor-{}.json".format(
                self.args.dataset,
                testing_graph_set,
                top_k_approach,
                test_k,
                search_backend,
                postprocess_mode,
                str(anchor_ratio).replace(".", "p"),
            )
        )
        config_name = "PartialDiff_refine_{}_{}_{}_{}_backend-{}_mode-{}_anchor-{}".format(
            self.args.dataset,
            testing_graph_set,
            top_k_approach,
            test_k,
            search_backend,
            postprocess_mode,
            str(anchor_ratio).replace(".", "p"),
        )
        budget_parameters = self._partialdiff_budget_parameters(test_k, top_k_approach)
        budget_parameters.update({
            "postprocess_mode": postprocess_mode,
            "anchor_ratio": float(anchor_ratio),
            "search_states": int(search_states),
            "workers": int(workers),
            "candidate_budget": int(candidate_budget),
            "beam_score": beam_score,
            "search_backend": search_backend,
            "beam_width": int(getattr(self.args, "v9_beam_width", 0)),
        })
        if bool(getattr(self.args, "save_raw_pair_log", True)):
            raw_pair_path = self._result_path(
                "raw_pairs_PartialDiff_refine_{}_{}_{}_{}_backend-{}_mode-{}_anchor-{}.jsonl".format(
                    self.args.dataset,
                    testing_graph_set,
                    top_k_approach,
                    test_k,
                    search_backend,
                    postprocess_mode,
                    str(anchor_ratio).replace(".", "p"),
                )
            )
            with raw_pair_path.open("w", encoding="utf-8") as handle:
                for item in pair_results:
                    matching_payload = item.get("best_matching_payload")
                    matching = self._normalize_app_bmao_matching_payload(matching_payload) if matching_payload else None
                    graph_1 = int(item["graph_1"])
                    graph_2 = int(item["graph_2"])
                    n1 = int(self.gn[graph_1])
                    n2 = int(self.gn[graph_2])
                    m1 = self._edge_count_for_graph(graph_1)
                    m2 = self._edge_count_for_graph(graph_2)
                    record = self._raw_pair_record(
                        method_name="PartialDiff+refine",
                        graph_1=graph_1,
                        graph_2=graph_2,
                        graph_1_gid=int(item["graph_1_gid"]),
                        graph_2_gid=int(item["graph_2_gid"]),
                        produced_ged=float(item["refined_ged"]),
                        reference_ged=float(item["gt_ged"]),
                        solver_time=float(item["sum_solver_time_s"]),
                        total_time=float(item["sum_wall_time_s"]),
                        n1=n1,
                        n2=n2,
                        m1=m1,
                        m2=m2,
                        config_name=config_name,
                        budget_parameters=budget_parameters,
                        matching=matching,
                        extra_fields={
                            "before_refine_ged": float(item["baseline_candidate_ged"]),
                            "after_refine_ged": float(item["refined_ged"]),
                            "anchor_count": item.get("best_anchor_count"),
                            "search_states": int(search_states),
                            "beam_width": int(getattr(self.args, "v9_beam_width", 0)),
                            "expanded_states": item.get("expanded_states"),
                            "external_search_space": item.get("external_search_space"),
                            "external_prune_profile": item.get("external_prune_profile"),
                            "best_external_prune_profile": item.get("best_external_prune_profile"),
                            "num_candidates_available": int(item.get("num_candidates_available", item["num_candidates_processed"])),
                            "num_candidates_processed": int(item["num_candidates_processed"]),
                            "app_bmao_reported_ged": int(item["app_bmao_reported_ged"]),
                        },
                    )
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            summary["raw_pair_log"] = str(raw_pair_path)
        else:
            summary["raw_pair_log"] = None
            summary["raw_pair_log_enabled"] = False
        with output_json_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)

        refine_time_usage = round(total_wall_time_s / num_pairs, 5)
        refine_mae = round(sum(abs(float(item["refined_ged"]) - float(item["gt_ged"])) for item in pair_results) / num_pairs, 3)
        refine_acc = round(sum(1.0 for item in pair_results if float(item["refined_ged"]) == float(item["gt_ged"])) / num_pairs, 3)
        refine_pred_lt_gt_count = int(sum(1 for item in pair_results if float(item["refined_ged"]) < float(item["gt_ged"])))
        refine_pred_lt_gt_ratio = round(refine_pred_lt_gt_count / num_pairs, 3)
        refine_fea = round(sum(1.0 for item in pair_results if float(item["refined_ged"]) >= float(item["gt_ged"])) / num_pairs, 3)

        refine_pres = {}
        refine_gts = {}
        for item in pair_results:
            gid = int(item["graph_1"])
            refine_pres.setdefault(gid, []).append(float(item["refined_ged"]))
            refine_gts.setdefault(gid, []).append(float(item["gt_ged"]))

        refine_rho_values = []
        refine_tau_values = []
        refine_pk10_values = []
        refine_pk20_values = []
        for graph_id in refine_pres:
            refine_rho_values.append(spearmanr(refine_pres[graph_id], refine_gts[graph_id])[0])
            refine_tau_values.append(kendalltau(refine_pres[graph_id], refine_gts[graph_id])[0])
            refine_pk10_values.append(self.cal_pk(10, refine_pres[graph_id], refine_gts[graph_id]))
            refine_pk20_values.append(self.cal_pk(20, refine_pres[graph_id], refine_gts[graph_id]))

        refine_rho = round(float(np.nanmean(refine_rho_values)), 3) if refine_rho_values else float("nan")
        refine_tau = round(float(np.nanmean(refine_tau_values)), 3) if refine_tau_values else float("nan")
        refine_pk10 = round(float(np.nanmean(refine_pk10_values)), 3) if refine_pk10_values else float("nan")
        refine_pk20 = round(float(np.nanmean(refine_pk20_values)), 3) if refine_pk20_values else float("nan")

        print(
            "Integrated App-BMao summary: backend={} avg_baseline_candidate_ged={:.3f} avg_refined_ged={:.3f} avg_app_bmao_reported_ged={:.3f} total_solver_time_s={:.3f} total_wall_time_s={:.3f} candidates={}/{}".format(
                search_backend,
                avg_baseline,
                avg_refined,
                avg_app_bmao_reported,
                total_solver_time_s,
                total_wall_time_s,
                total_candidates_processed,
                total_candidates_available,
            ),
            flush=True,
        )
        print(*self.results[-2], sep="	")
        print(
            *(
                f"{self.args.model_name}+{search_backend}",
                top_k_approach,
                self.args.dataset,
                testing_graph_set,
                len(pair_results),
                refine_time_usage,
                refine_mae,
                refine_acc,
                round(avg_refined, 3),
                round(avg_gt, 3),
                refine_fea,
                refine_rho,
                refine_tau,
                refine_pk10,
                refine_pk20,
                refine_pred_lt_gt_count,
                refine_pred_lt_gt_ratio,
            ),
            sep="	",
        )
        print(f"Saved integrated App-BMao results to {output_json_path}", flush=True)

    def save(self, epoch):
        output_path = self._checkpoint_path(epoch)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), output_path)
        metadata_path = self._checkpoint_metadata_path(epoch)
        with metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "epoch": int(epoch),
                    "dataset": self.args.dataset,
                    "model_name": self.args.model_name,
                    "hyperparameters": self._serialize_args(),
                },
                handle,
                indent=2,
                sort_keys=True,
            )

    def _serialize_args(self):
        serialized = {}
        for key, value in vars(self.args).items():
            if isinstance(value, Path):
                serialized[key] = str(value)
            else:
                serialized[key] = value
        return serialized

    @staticmethod
    def _infer_hidden_dims_from_state_dict(state_dict):
        hidden_dims = []
        pattern = re.compile(r"^gns\.(\d+)\.weight$")
        indexed_dims = {}
        for key, value in state_dict.items():
            match = pattern.match(key)
            if match and hasattr(value, "shape") and len(value.shape) == 1:
                indexed_dims[int(match.group(1))] = int(value.shape[0])
        if indexed_dims:
            max_idx = max(indexed_dims)
            hidden_dims = [indexed_dims[idx] for idx in range(max_idx + 1) if idx in indexed_dims]
        return hidden_dims

    def load(self, epoch):
        checkpoint_path = self._checkpoint_path(epoch)
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        try:
            load_result = self.model.load_state_dict(state_dict, strict=True)
        except RuntimeError as exc:
            inferred_hidden_dims = self._infer_hidden_dims_from_state_dict(state_dict)
            current_hidden_dims = list(getattr(self.args, "hidden_dim", []))
            hint_lines = [
                f"Failed to load checkpoint: {checkpoint_path}",
                f"Current --hidden-dim: {current_hidden_dims}",
            ]
            if inferred_hidden_dims:
                hint_lines.append(f"Checkpoint appears to use --hidden-dim: {inferred_hidden_dims}")
            raise RuntimeError(str(exc) + "\n" + "\n".join(hint_lines)) from exc
        missing = len(getattr(load_result, "missing_keys", []))
        unexpected = len(getattr(load_result, "unexpected_keys", []))
        print(f"[Load][dense-io] strict load complete: missing_keys={missing}, unexpected_keys={unexpected}")

    def _resolve_output_dir(self, path_value):
        path = Path(path_value)
        if path.is_absolute():
            return path
        return Path(self.args.abs_path) / path

    def _checkpoint_path(self, epoch):
        model_dir = self._resolve_output_dir(self.args.model_path)
        return model_dir / f"{self.args.dataset}_{epoch}_{self.args.model_name}.pt"

    def _checkpoint_metadata_path(self, epoch):
        model_dir = self._resolve_output_dir(self.args.model_path)
        return model_dir / f"{self.args.dataset}_{epoch}_{self.args.model_name}.json"

    def _result_path(self, filename):
        result_dir = self._resolve_output_dir(self.args.result_path)
        result_dir.mkdir(parents=True, exist_ok=True)
        return result_dir / filename
