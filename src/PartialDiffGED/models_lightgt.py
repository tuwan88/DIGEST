"""DiffMatch variant with a lightweight cross-graph transformer denoiser.

Usage:
    from models_lightgt import LightGTDiffMatch
    model = LightGTDiffMatch(args, number_of_labels)

It keeps the same forward signature as the original DiffMatch:
    logits = model(data, noise_mapping_attr, t)
"""

import math

import torch
import torch.nn.functional as F
from torch_geometric.nn.conv import GINConv
from torch_geometric.nn.norm import GraphNorm
from torch_geometric.utils import to_undirected, to_dense_batch

from layers import timestep_embedding, ScalarEmbeddingSine
from layers_lightgt import (
    SparseCrossGraphTransformer,
    DenseCrossGraphTransformer,
)


def adaptive_k(n_other, ratio=0.8, k_min=16, k_max=50):
    k = math.ceil(float(ratio) * int(n_other))
    k = max(int(k_min), k)
    k = min(int(k_max), k)
    k = min(int(n_other), k)
    return k


class TimestepEmbeddingSine(torch.nn.Module):
    """Module wrapper around the functional timestep embedding for profiling."""

    def __init__(self, dim, max_period=10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, timesteps):
        return timestep_embedding(timesteps, self.dim, max_period=self.max_period)


class PairToUndirected(torch.nn.Module):
    """Module wrapper around torch_geometric.to_undirected for profiling."""

    def forward(self, edge_index, edge_attr):
        return to_undirected(edge_index, edge_attr)


class TensorFloatCast(torch.nn.Module):
    """Profiles explicit dtype casts that otherwise disappear into Python code."""

    def forward(self, tensor):
        return tensor.float()


class LightGTDiffMatch(torch.nn.Module):
    """Lightweight Graph Transformer denoising network.

    The intra-graph part remains a small shared GIN encoder for local structure.
    The inter-graph AGNN block is replaced by SparseCrossGraphTransformer, which
    performs noise-conditioned cross attention on the bipartite matching edge list.
    """

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
        # Fall back to a valid divisor instead of failing during hyperparameter sweeps.
        return math.gcd(dim, requested) or 1

    def setup_layers(self):
        self.hidden_dims = self.args.hidden_dim
        self.num_layers = len(self.hidden_dims)
        self.dropout = float(getattr(self.args, "dropout", 0.0))
        self.time_emb_dim = self.hidden_dims[0] // 2
        self.enable_topk_pruning = bool(getattr(self.args, "enable_topk_pruning", False))
        self.topk_k_min = int(getattr(self.args, "topk_min", 16))
        self.topk_k_max = int(getattr(self.args, "topk_max", 50))
        self.topk_anchor_bias = float(getattr(self.args, "topk_anchor_bias", 2.0))
        self.topk_score_source = str(getattr(self.args, "topk_score_source", "base_cost"))
        self.log_candidate_recall = bool(getattr(self.args, "log_candidate_recall", False))
        self.topk_ratios = self._normalize_topk_ratios(getattr(self.args, "topk_ratios", [0.9, 0.8, 0.7, 0.7]))
        self.last_topk_pruning_stats = None

        self.conv_layers = torch.nn.ModuleList()
        self.cross_layers = torch.nn.ModuleList()
        self.gns = torch.nn.ModuleList()

        for l in range(self.num_layers):
            if l == 0:
                gin_mlp = torch.nn.Sequential(
                    torch.nn.Linear(self.number_labels, self.hidden_dims[l]),
                    torch.nn.ReLU(),
                    torch.nn.Linear(self.hidden_dims[l], self.hidden_dims[l]),
                )
                pair_in_dim = self.hidden_dims[l]
            else:
                gin_mlp = torch.nn.Sequential(
                    torch.nn.Linear(self.hidden_dims[l - 1], self.hidden_dims[l]),
                    torch.nn.ReLU(),
                    torch.nn.Linear(self.hidden_dims[l], self.hidden_dims[l]),
                )
                pair_in_dim = self.hidden_dims[l - 1]

            self.conv_layers.append(GINConv(gin_mlp, train_eps=True))
            self.gns.append(GraphNorm(self.hidden_dims[l]))
            self.cross_layers.append(
                SparseCrossGraphTransformer(
                    hidden_dim=self.hidden_dims[l],
                    time_emb_dim=self.time_emb_dim,
                    edge_dim=pair_in_dim,
                    num_heads=self._get_heads_for_dim(self.hidden_dims[l]),
                    dropout=self.dropout,
                )
            )

        self.time_embed = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dims[0], self.time_emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.time_emb_dim, self.time_emb_dim),
        )

        self.pair_to_undirected = PairToUndirected()
        self.noise_attr_cast = TensorFloatCast()
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

    def _normalize_topk_ratios(self, ratios):
        if isinstance(ratios, (float, int)):
            ratios = [float(ratios)]
        else:
            ratios = [float(item) for item in ratios]
        if not ratios:
            ratios = [0.9]
        if len(ratios) < self.num_layers:
            ratios = ratios + [ratios[-1]] * (self.num_layers - len(ratios))
        return ratios[:self.num_layers]

    def pop_last_topk_pruning_stats(self):
        stats = self.last_topk_pruning_stats
        self.last_topk_pruning_stats = None
        return stats

    def _empty_topk_stats(self):
        return {
            "enabled": self.enable_topk_pruning,
            "num_layers": self.num_layers,
            "active_edges": [0.0] * self.num_layers,
            "full_edges": [0.0] * self.num_layers,
            "recall_hits": [0.0] * self.num_layers,
            "recall_total": [0.0] * self.num_layers,
        }

    def _build_pruning_score(self, dense_noise, dense_base_cost, edge_available):
        score = self.topk_anchor_bias * dense_noise
        if self.topk_score_source == "base_cost" and dense_base_cost is not None:
            score = score - dense_base_cost
        score = score.masked_fill(~edge_available, float("-inf"))
        return score

    @staticmethod
    def _topk_mask_along_dim(score, available_mask, k, dim):
        output_mask = torch.zeros_like(available_mask)
        size_along_dim = int(score.shape[dim])
        if size_along_dim == 0 or k <= 0:
            return output_mask
        topk_idx = torch.topk(score, k=min(int(k), size_along_dim), dim=dim).indices
        output_mask.scatter_(dim, topk_idx, True)
        return output_mask & available_mask

    def _build_group_specs(self, data):
        pair_sizes = data.n.long()
        unique_sizes, inverse = torch.unique(pair_sizes, dim=0, return_inverse=True)
        grouped_specs = {}
        for group_idx in range(unique_sizes.size(0)):
            pair_tensor = torch.nonzero(inverse == group_idx, as_tuple=False).view(-1)
            if pair_tensor.numel() == 0:
                continue
            n1 = int(unique_sizes[group_idx, 0].item())
            n2 = int(unique_sizes[group_idx, 1].item())
            grouped_specs[(n1, n2)] = {"pair_tensor": pair_tensor}
        return grouped_specs

    def _group_dense_edge_attrs(self, data, mapping_attr_dict, grouped_specs):
        edge_mapping_idx = data.edge_index_mapping
        src, dst = edge_mapping_idx
        device = edge_mapping_idx.device
        mapping_batch = data.batch[src].long()
        node_offsets = data.ptr[mapping_batch]
        n1_per_edge = data.n[mapping_batch, 0].long()
        local_rows = (src - node_offsets).long()
        local_cols = (dst - node_offsets - n1_per_edge).long()

        dense_attr_by_group = {name: {} for name in mapping_attr_dict}
        flat_index_by_group = {}
        available_by_group = {}
        edge_count_by_group = {}
        total_pairs = int(data.n.shape[0])
        for key, group_spec in grouped_specs.items():
            n1, n2 = key
            pair_tensor = group_spec["pair_tensor"]
            group_size = int(pair_tensor.shape[0])
            pair_to_group = torch.full((total_pairs,), -1, device=device, dtype=torch.long)
            pair_to_group[pair_tensor] = torch.arange(group_size, device=device, dtype=torch.long)

            edge_group_ids = pair_to_group[mapping_batch]
            group_mask = edge_group_ids >= 0
            edge_group = edge_group_ids[group_mask]
            edge_rows = local_rows[group_mask]
            edge_cols = local_cols[group_mask]

            for name, mapping_attr in mapping_attr_dict.items():
                if mapping_attr is None:
                    dense_attr_by_group[name][key] = None
                    continue
                dense_attr = mapping_attr.new_zeros((group_size, n1, n2))
                dense_attr[edge_group, edge_rows, edge_cols] = mapping_attr[group_mask, 0]
                dense_attr_by_group[name][key] = dense_attr

            flat_index_by_group[key] = (
                edge_group,
                edge_rows,
                edge_cols,
                group_mask,
            )
            dense_available = torch.zeros((group_size, n1, n2), dtype=torch.bool, device=device)
            dense_available[edge_group, edge_rows, edge_cols] = True
            available_by_group[key] = dense_available
            edge_count_by_group[key] = int(group_mask.sum().item())

        return dense_attr_by_group, flat_index_by_group, available_by_group, edge_count_by_group

    def _build_layer_candidate_masks(self, data, noise_mapping_attr):
        num_edges = int(data.edge_index_mapping.shape[1])
        device = data.edge_index_mapping.device
        layer_masks = [torch.ones(num_edges, dtype=torch.bool, device=device) for _ in range(self.num_layers)]
        stats = self._empty_topk_stats()
        if not self.enable_topk_pruning:
            full_edges = float(num_edges)
            for layer_idx in range(self.num_layers):
                stats["active_edges"][layer_idx] = full_edges
                stats["full_edges"][layer_idx] = full_edges
            return layer_masks, stats

        grouped_specs = self._build_group_specs(data)
        mapping_attr_dict = {
            "noise": noise_mapping_attr,
            "base_cost": getattr(data, "edge_base_cost", None) if self.topk_score_source == "base_cost" else None,
            "gt": getattr(data, "edge_attr_mapping", None),
        }
        dense_attr_by_group, flat_index_by_group, available_by_group, edge_count_by_group = self._group_dense_edge_attrs(
            data,
            mapping_attr_dict,
            grouped_specs,
        )

        for (n1, n2), _group_spec in grouped_specs.items():
            dense_noise = dense_attr_by_group["noise"][(n1, n2)]
            dense_base_cost = dense_attr_by_group["base_cost"][(n1, n2)]
            dense_gt = dense_attr_by_group["gt"][(n1, n2)]
            edge_available = available_by_group[(n1, n2)]

            score = self._build_pruning_score(dense_noise, dense_base_cost, edge_available)
            mandatory_mask = (dense_noise > 0.5) & edge_available
            gt_positive = None
            if dense_gt is not None:
                gt_positive = (dense_gt > 0.5) & edge_available
                if self.training:
                    mandatory_mask = mandatory_mask | gt_positive

            previous_mask = edge_available
            edge_group, edge_rows, edge_cols, group_mask = flat_index_by_group[(n1, n2)]
            full_edges_group = float(edge_count_by_group[(n1, n2)])

            for layer_idx, ratio in enumerate(self.topk_ratios):
                row_k = adaptive_k(n2, ratio=ratio, k_min=self.topk_k_min, k_max=self.topk_k_max)
                col_k = adaptive_k(n1, ratio=ratio, k_min=self.topk_k_min, k_max=self.topk_k_max)
                masked_score = score.masked_fill(~previous_mask, float("-inf"))

                row_topk_idx = torch.topk(masked_score, k=min(row_k, n2), dim=2).indices
                row_topk = torch.zeros_like(previous_mask)
                row_topk.scatter_(2, row_topk_idx, True)
                row_topk &= previous_mask

                col_score = masked_score.transpose(1, 2)
                col_prev = previous_mask.transpose(1, 2)
                col_topk_idx = torch.topk(col_score, k=min(col_k, n1), dim=2).indices
                col_topk = torch.zeros_like(col_prev)
                col_topk.scatter_(2, col_topk_idx, True)
                col_topk = (col_topk & col_prev).transpose(1, 2)

                candidate_mask = (row_topk | col_topk | mandatory_mask) & edge_available
                previous_mask = candidate_mask

                layer_masks[layer_idx][group_mask] = candidate_mask[edge_group, edge_rows, edge_cols]
                stats["active_edges"][layer_idx] += float(candidate_mask.sum().item())
                stats["full_edges"][layer_idx] += full_edges_group
                if self.training and self.log_candidate_recall and gt_positive is not None:
                    stats["recall_hits"][layer_idx] += float((candidate_mask & gt_positive).sum().item())
                    stats["recall_total"][layer_idx] += float(gt_positive.sum().item())

        return layer_masks, stats

    def convolutional_pass(self, features, graph_edge_index, edge_mapping_idx, noise_mapping_emb, time_emb, batch, graph_2, active_masks=None, full_active_layers=None):
        # Keep the two graphs in a graph-pair separated for node normalization,
        # exactly as in the original local GIN part.
        node_norm_batch = batch * 2
        node_norm_batch[graph_2] += 1

        for l in range(self.num_layers):
            features = torch.relu(self.gns[l](self.conv_layers[l](features, graph_edge_index), batch=node_norm_batch))
            if active_masks is None:
                features, noise_mapping_emb = self.cross_layers[l](
                    features=features,
                    edge_mapping_idx=edge_mapping_idx,
                    noise_mapping_emb=noise_mapping_emb,
                    time_emb=time_emb,
                    batch=batch,
                    node_norm_batch=node_norm_batch,
                )
                continue

            if full_active_layers is not None and full_active_layers[l]:
                features, noise_mapping_emb = self.cross_layers[l](
                    features=features,
                    edge_mapping_idx=edge_mapping_idx,
                    noise_mapping_emb=noise_mapping_emb,
                    time_emb=time_emb,
                    batch=batch,
                    node_norm_batch=node_norm_batch,
                )
                continue

            directed_mask = active_masks[l]
            undirected_mask = torch.cat([directed_mask, directed_mask], dim=0)
            full_projected_edges = self.cross_layers[l].project_edge(noise_mapping_emb)
            active_edge_mapping_idx = edge_mapping_idx[:, undirected_mask]
            active_noise_mapping_emb = full_projected_edges[undirected_mask]
            features, updated_active_edges = self.cross_layers[l](
                features=features,
                edge_mapping_idx=active_edge_mapping_idx,
                noise_mapping_emb=active_noise_mapping_emb,
                time_emb=time_emb,
                batch=batch,
                node_norm_batch=node_norm_batch,
                edge_is_projected=True,
            )
            full_projected_edges[undirected_mask] = updated_active_edges
            noise_mapping_emb = full_projected_edges

        return features, noise_mapping_emb

    def build_time_context(self, t, batch_size, device, dtype):
        del batch_size
        time_pos = self.time_pos_embed(t)
        return self.time_embed(time_pos).to(device=device, dtype=dtype)

    def forward(self, data, noise_mapping_attr, t):
        graph_edge_index = data.edge_index
        graph_x = data.x
        batch = data.batch
        edge_mapping_idx = data.edge_index_mapping

        pair_indicator = data.x_indicator
        graph_2 = (pair_indicator == 1).squeeze(1)

        time_emb = self.build_time_context(
            t=t,
            batch_size=int(data.n.shape[0]),
            device=graph_x.device,
            dtype=graph_x.dtype,
        )
        if not self.enable_topk_pruning:
            # Same symmetry trick as the original model: use M_t and M_t^T together.
            undirected_edge_mapping_idx, undirected_noise_mapping_attr = self.pair_to_undirected(
                edge_mapping_idx,
                noise_mapping_attr,
            )
            undirected_noise_mapping_attr = self.noise_attr_cast(undirected_noise_mapping_attr)
            undirected_noise_mapping_emb = self.edge_embed(self.edge_pos_embed(undirected_noise_mapping_attr))

            _, noise_mapping_emb = self.convolutional_pass(
                graph_x,
                graph_edge_index,
                undirected_edge_mapping_idx,
                undirected_noise_mapping_emb,
                time_emb,
                batch,
                graph_2,
            )

            map_matrix = self.mapMatrix(noise_mapping_emb)

            # Sum logits for (v, v') and (v', v), then keep the graph1->graph2 half.
            _, map_matrix = to_undirected(undirected_edge_mapping_idx, map_matrix)
            map_matrix = map_matrix[(pair_indicator[undirected_edge_mapping_idx[0]] == 0).squeeze(1)]
            self.last_topk_pruning_stats = None
            return map_matrix

        directed_noise_mapping_attr = self.noise_attr_cast(noise_mapping_attr)
        manual_undirected_edge_mapping_idx = torch.cat(
            [edge_mapping_idx, edge_mapping_idx.flip(0)],
            dim=1,
        )
        manual_undirected_noise_mapping_attr = torch.cat(
            [directed_noise_mapping_attr, directed_noise_mapping_attr],
            dim=0,
        )
        active_masks, pruning_stats = self._build_layer_candidate_masks(data, directed_noise_mapping_attr)
        full_active_layers = [
            active >= full - 0.5
            for active, full in zip(pruning_stats["active_edges"], pruning_stats["full_edges"])
        ]
        undirected_noise_mapping_emb = self.edge_embed(self.edge_pos_embed(manual_undirected_noise_mapping_attr))

        _, noise_mapping_emb = self.convolutional_pass(
            graph_x,
            graph_edge_index,
            manual_undirected_edge_mapping_idx,
            undirected_noise_mapping_emb,
            time_emb,
            batch,
            graph_2,
            active_masks=active_masks,
            full_active_layers=full_active_layers,
        )

        map_matrix = self.mapMatrix(noise_mapping_emb)
        num_directed_edges = int(edge_mapping_idx.shape[1])
        pruning_stats["active_ratio"] = [
            active / max(full, 1.0)
            for active, full in zip(pruning_stats["active_edges"], pruning_stats["full_edges"])
        ]
        pruning_stats["candidate_recall"] = [
            hits / max(total, 1e-8)
            for hits, total in zip(pruning_stats["recall_hits"], pruning_stats["recall_total"])
        ]
        self.last_topk_pruning_stats = pruning_stats
        return map_matrix[:num_directed_edges] + map_matrix[num_directed_edges:]


class DenseLightGTDiffMatch(torch.nn.Module):
    """Dense full-grid variant of LightGT cross layers (no top-k pruning / buckets)."""

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
        self.hidden_dims = self.args.hidden_dim
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

        for l in range(self.num_layers):
            if l == 0:
                gin_mlp = torch.nn.Sequential(
                    torch.nn.Linear(self.number_labels, self.hidden_dims[l]),
                    torch.nn.ReLU(),
                    torch.nn.Linear(self.hidden_dims[l], self.hidden_dims[l]),
                )
                pair_in_dim = self.hidden_dims[l]
            else:
                gin_mlp = torch.nn.Sequential(
                    torch.nn.Linear(self.hidden_dims[l - 1], self.hidden_dims[l]),
                    torch.nn.ReLU(),
                    torch.nn.Linear(self.hidden_dims[l], self.hidden_dims[l]),
                )
                pair_in_dim = self.hidden_dims[l - 1]

            self.conv_layers.append(GINConv(gin_mlp, train_eps=True))
            self.gns.append(GraphNorm(self.hidden_dims[l]))
            self.cross_layers.append(
                DenseCrossGraphTransformer(
                    hidden_dim=self.hidden_dims[l],
                    time_emb_dim=self.time_emb_dim,
                    edge_dim=pair_in_dim,
                    num_heads=self._get_heads_for_dim(self.hidden_dims[l]),
                    dropout=self.dropout,
                )
            )

        self.time_embed = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dims[0], self.time_emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.time_emb_dim, self.time_emb_dim),
        )
        self.time_pos_embed = TimestepEmbeddingSine(self.hidden_dims[0])
        self.noise_attr_cast = TensorFloatCast()
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
        if not self.dense_topk_enable:
            return None
        stats = self.last_dense_topk_stats
        self.last_dense_topk_stats = None
        return stats

    def build_time_context(self, t, batch_size, device, dtype):
        del batch_size
        time_pos = self.time_pos_embed(t)
        return self.time_embed(time_pos).to(device=device, dtype=dtype)

    @staticmethod
    def _build_dense_edge_tensor(data, noise_mapping_attr, n1_max, n2_max):
        # Fill dense candidate noise tensor from directed edge list.
        edge_mapping_idx = data.edge_index_mapping
        src, dst = edge_mapping_idx
        mapping_batch = data.batch[src].long()
        node_offsets = data.ptr[mapping_batch]
        n1_per_edge = data.n[mapping_batch, 0].long()
        local_rows = (src - node_offsets).long()
        local_cols = (dst - node_offsets - n1_per_edge).long()

        bsz = int(data.n.shape[0])
        dense = torch.zeros(
            (bsz, n1_max, n2_max, 1),
            device=noise_mapping_attr.device,
            dtype=noise_mapping_attr.dtype,
        )
        dense[mapping_batch, local_rows, local_cols, 0] = noise_mapping_attr[:, 0]
        return dense, mapping_batch, local_rows, local_cols

    def _build_row_col_topk_mask(self, score_pair, mask1, mask2, forced_mask=None):
        # score_pair: [B, N1, N2]
        bsz, n1, n2 = score_pair.shape
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
        candidate_mask = candidate_mask & pair_mask
        return candidate_mask

    def forward(self, data, noise_mapping_attr, t):
        graph_edge_index = data.edge_index
        graph_x = data.x
        batch = data.batch
        edge_mapping_idx = data.edge_index_mapping
        pair_indicator = data.x_indicator
        graph_2 = (pair_indicator == 1).squeeze(1)

        # Node norm id follows original LightGT.
        node_norm_batch = batch * 2
        node_norm_batch[graph_2] += 1

        # Build per-side dense node tensors once; per-layer updates are written back.
        x1_flat = graph_x[~graph_2]
        x2_flat = graph_x[graph_2]
        b1 = batch[~graph_2]
        b2 = batch[graph_2]
        x1, mask1 = to_dense_batch(x1_flat, b1)  # [B, N1, D0]
        x2, mask2 = to_dense_batch(x2_flat, b2)  # [B, N2, D0]
        bsz, n1_max, _ = x1.shape
        _, n2_max, _ = x2.shape
        pair_mask = mask1[:, :, None] & mask2[:, None, :]

        directed_noise_mapping_attr = self.noise_attr_cast(noise_mapping_attr)
        dense_noise_attr, mapping_batch, local_rows, local_cols = self._build_dense_edge_tensor(
            data,
            directed_noise_mapping_attr,
            n1_max,
            n2_max,
        )
        edge_pos = self.edge_pos_embed(dense_noise_attr.reshape(-1, 1))
        edge_pair = self.edge_embed(edge_pos).view(bsz, n1_max, n2_max, -1)
        edge_pair = edge_pair * pair_mask.unsqueeze(-1).to(edge_pair.dtype)
        dense_noise_scalar = dense_noise_attr[..., 0]
        candidate_stats = {
            "enabled": self.dense_topk_enable,
            "num_layers": self.num_layers,
            "active_edges": [0.0] * self.num_layers,
            "full_edges": [0.0] * self.num_layers,
            "active_ratio": [1.0] * self.num_layers,
            "recall_hits": [0.0] * self.num_layers,
            "recall_total": [0.0] * self.num_layers,
            "candidate_recall": [0.0] * self.num_layers,
        }

        time_emb = self.build_time_context(
            t=t,
            batch_size=int(data.n.shape[0]),
            device=graph_x.device,
            dtype=graph_x.dtype,
        )

        features = graph_x
        for l in range(self.num_layers):
            features = torch.relu(self.gns[l](self.conv_layers[l](features, graph_edge_index), batch=node_norm_batch))
            x1_flat = features[~graph_2]
            x2_flat = features[graph_2]
            x1, mask1 = to_dense_batch(x1_flat, b1)
            x2, mask2 = to_dense_batch(x2_flat, b2)
            full_edges = float((mask1[:, :, None] & mask2[:, None, :]).sum().item())
            candidate_stats["full_edges"][l] = full_edges

            if self.dense_topk_enable and l >= self.dense_topk_start_layer:
                h = self.cross_layers[l].num_heads
                dh = self.cross_layers[l].head_dim
                q1 = self.cross_layers[l].q_proj(x1).view(bsz, n1_max, h, dh)
                k2 = self.cross_layers[l].k_proj(x2).view(bsz, n2_max, h, dh)
                with torch.no_grad():
                    score = torch.einsum("bihd,bjhd->bhij", q1, k2) * self.cross_layers[l].scale
                    if self.dense_topk_score_source == "qk_max":
                        score_pair = score.max(dim=1).values
                    else:
                        score_pair = score.mean(dim=1)
                    forced_mask = None
                    if self.dense_topk_force_current_matching:
                        forced_mask = (dense_noise_scalar > 0.5) & (mask1[:, :, None] & mask2[:, None, :])
                    candidate_mask = self._build_row_col_topk_mask(
                        score_pair=score_pair,
                        mask1=mask1,
                        mask2=mask2,
                        forced_mask=forced_mask,
                    )
                active_edges = float(candidate_mask.sum().item())
                candidate_stats["active_edges"][l] = active_edges
                candidate_stats["active_ratio"][l] = active_edges / max(full_edges, 1.0)
                x1, x2, edge_pair = self.cross_layers[l](
                    x1=x1,
                    x2=x2,
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
                candidate_stats["active_edges"][l] = full_edges
                candidate_stats["active_ratio"][l] = 1.0
                x1, x2, edge_pair = self.cross_layers[l](
                    x1=x1,
                    x2=x2,
                    edge_pair=edge_pair,
                    mask1=mask1,
                    mask2=mask2,
                    time_emb=time_emb,
                    edge_is_projected=False,
                )
            features_next = torch.zeros_like(features)
            features_next[~graph_2] = x1[mask1]
            features_next[graph_2] = x2[mask2]
            features = features_next

        score_dense = self.mapMatrix(edge_pair).squeeze(-1)
        score_dense = score_dense.masked_fill(~pair_mask, 0.0)
        map_matrix = score_dense[mapping_batch, local_rows, local_cols].unsqueeze(-1)
        self.last_dense_topk_stats = candidate_stats if self.dense_topk_enable else None
        return map_matrix


class DisjointLightGTDiffMatch(LightGTDiffMatch):
    """LightGT backbone for the sparse disjoint trainer/decode path."""

    @staticmethod
    def _node_labels(x):
        if x.dim() > 1 and x.size(-1) > 1:
            return x.argmax(dim=-1).long()
        return x.view(-1).long()

    @staticmethod
    def build_disjoint_batch_info(data):
        match_src = data.edge_index_mapping[0].long()
        match_dst = data.edge_index_mapping[1].long()
        node_pair_id = data.batch.long()
        edge_pair_id = node_pair_id[match_src]
        node_side = data.x_indicator.view(-1).long()
        node_offset = data.ptr[edge_pair_id].long()
        n1_per_edge = data.n[edge_pair_id, 0].long()
        local_left = match_src - node_offset
        local_right = match_dst - node_offset - n1_per_edge
        return {
            "match_src": match_src,
            "match_dst": match_dst,
            "node_pair_id": node_pair_id,
            "edge_pair_id": edge_pair_id,
            "node_side": node_side,
            "local_left": local_left,
            "local_right": local_right,
            "num_pairs": int(data.n.shape[0]),
            "num_nodes": int(data.x.shape[0]),
        }

    @classmethod
    def sample_partial_edge_attr(cls, data, mapping_attr, timesteps, keep_ratio_fn):
        mapping_attr = mapping_attr.view(-1)
        info = cls.build_disjoint_batch_info(data)
        pair_id = info["edge_pair_id"]
        partial = torch.zeros_like(mapping_attr)
        positive_mask = mapping_attr > 0.5
        for pair_idx in range(info["num_pairs"]):
            positive_idx = torch.nonzero(positive_mask & (pair_id == pair_idx), as_tuple=False).view(-1)
            if positive_idx.numel() == 0:
                continue
            keep_ratio = float(keep_ratio_fn(int(timesteps[pair_idx])))
            keep_count = int(round(keep_ratio * positive_idx.numel()))
            keep_count = min(max(keep_count, 0), positive_idx.numel())
            if keep_count <= 0:
                continue
            if keep_count >= positive_idx.numel():
                partial[positive_idx] = 1.0
                continue
            perm = torch.randperm(positive_idx.numel(), device=positive_idx.device)[:keep_count]
            partial[positive_idx[perm]] = 1.0
        return partial.unsqueeze(-1)

    @staticmethod
    def _segment_argmax_edges(edge_ids, scores, group, dim_size):
        if edge_ids.numel() == 0:
            return (
                torch.full((dim_size,), -1, device=group.device, dtype=torch.long),
                torch.full((dim_size,), float("-inf"), device=scores.device, dtype=scores.dtype),
            )
        max_scores = torch.full((dim_size,), float("-inf"), device=scores.device, dtype=scores.dtype)
        max_scores.scatter_reduce_(0, group, scores, reduce="amax", include_self=True)
        sentinel = torch.iinfo(edge_ids.dtype).max
        candidate_edges = torch.where(
            scores == max_scores[group],
            edge_ids,
            edge_ids.new_full(edge_ids.shape, sentinel),
        )
        winner_edges = edge_ids.new_full((dim_size,), sentinel)
        winner_edges.scatter_reduce_(0, group, candidate_edges, reduce="amin", include_self=True)
        winner_edges[winner_edges == sentinel] = -1
        return winner_edges, max_scores

    @classmethod
    def _segment_top2_edges(cls, edge_ids, scores, group, dim_size):
        top1_edge, top1_val = cls._segment_argmax_edges(edge_ids, scores, group, dim_size)
        if edge_ids.numel() == 0:
            return top1_edge, top1_val, torch.zeros_like(top1_val)
        second_scores = scores.masked_fill(edge_ids == top1_edge[group], float("-inf"))
        _, top2_val = cls._segment_argmax_edges(edge_ids, second_scores, group, dim_size)
        top1_val = torch.where(torch.isfinite(top1_val), top1_val, torch.zeros_like(top1_val))
        top2_val = torch.where(torch.isfinite(top2_val), top2_val, torch.zeros_like(top2_val))
        return top1_edge, top1_val, top2_val

    @staticmethod
    def edge_attr_from_matched_edges(num_edges, matched_edges, reference_tensor):
        edge_attr = reference_tensor.new_zeros((num_edges, 1))
        if matched_edges.numel() > 0:
            edge_attr[matched_edges, 0] = 1.0
        return edge_attr

    @staticmethod
    def _cap_edges_per_pair(edge_ids, edge_scores, pair_id, pair_remaining):
        if edge_ids.numel() == 0:
            return edge_ids.new_empty((0,), dtype=torch.long)

        score_order = torch.argsort(edge_scores, descending=True, stable=True)
        sorted_pairs_by_score = pair_id[score_order]
        pair_order = torch.argsort(sorted_pairs_by_score, stable=True)
        final_order = score_order[pair_order]

        sorted_pairs = pair_id[final_order]
        _, counts = torch.unique_consecutive(sorted_pairs, return_counts=True)
        group_starts = torch.repeat_interleave(counts.cumsum(0) - counts, counts)
        ranks = torch.arange(final_order.numel(), device=edge_ids.device) - group_starts
        keep = ranks < pair_remaining[sorted_pairs]
        return edge_ids[final_order[keep]]

  

    @classmethod
    def sparse_matchings_for_selected_pairs(cls, data, selected_pair_ids, selected_edge_mask):
        info = cls.build_disjoint_batch_info(data)
        selected_pair_ids = selected_pair_ids.long()
        num_selected = int(selected_pair_ids.numel())
        if num_selected == 0:
            return []

        selected_sizes = data.n[selected_pair_ids].long()
        n1_sizes = selected_sizes[:, 0]
        n2_sizes = selected_sizes[:, 1]
        n1_max = int(n1_sizes.max().item())
        n2_max = int(n2_sizes.max().item())

        pair_to_pos = torch.full(
            (int(data.n.shape[0]),),
            -1,
            device=selected_pair_ids.device,
            dtype=torch.long,
        )
        pair_to_pos[selected_pair_ids] = torch.arange(num_selected, device=selected_pair_ids.device, dtype=torch.long)

        edge_pos = pair_to_pos[info["edge_pair_id"]]
        active_mask = (edge_pos >= 0) & selected_edge_mask
        dense_all = torch.zeros(
            (num_selected, n1_max, n2_max),
            device=selected_edge_mask.device,
            dtype=torch.float,
        )
        if bool(active_mask.any()):
            dense_all[
                edge_pos[active_mask],
                info["local_left"][active_mask],
                info["local_right"][active_mask],
            ] = 1.0

        dense_matchings = []
        for idx in range(num_selected):
            n1 = int(n1_sizes[idx].item())
            n2 = int(n2_sizes[idx].item())
            dense_matchings.append(dense_all[idx:idx + 1, :n1, :n2])
        return dense_matchings

    @classmethod
    def sparse_probabilities_for_selected_pairs(cls, data, selected_pair_ids, edge_values):
        info = cls.build_disjoint_batch_info(data)
        selected_pair_ids = selected_pair_ids.long()
        edge_values = edge_values.view(-1)
        num_selected = int(selected_pair_ids.numel())
        if num_selected == 0:
            return []

        selected_sizes = data.n[selected_pair_ids].long()
        n1_sizes = selected_sizes[:, 0]
        n2_sizes = selected_sizes[:, 1]
        n1_max = int(n1_sizes.max().item())
        n2_max = int(n2_sizes.max().item())

        pair_to_pos = torch.full(
            (int(data.n.shape[0]),),
            -1,
            device=selected_pair_ids.device,
            dtype=torch.long,
        )
        pair_to_pos[selected_pair_ids] = torch.arange(num_selected, device=selected_pair_ids.device, dtype=torch.long)

        edge_pos = pair_to_pos[info["edge_pair_id"]]
        active_mask = edge_pos >= 0
        dense_all = torch.zeros(
            (num_selected, n1_max, n2_max),
            device=edge_values.device,
            dtype=edge_values.dtype,
        )
        if bool(active_mask.any()):
            dense_all[
                edge_pos[active_mask],
                info["local_left"][active_mask],
                info["local_right"][active_mask],
            ] = edge_values[active_mask]

        dense_probs = []
        for idx in range(num_selected):
            n1 = int(n1_sizes[idx].item())
            n2 = int(n2_sizes[idx].item())
            dense_probs.append(dense_all[idx, :n1, :n2])
        return dense_probs

    @staticmethod
    def _canonical_unique_keys(edge_index, node_side, side_value, num_nodes):
        src = edge_index[0].long()
        dst = edge_index[1].long()
        mask = (node_side[src] == side_value) & (node_side[dst] == side_value) & (src != dst)
        src = src[mask]
        dst = dst[mask]
        lo = torch.minimum(src, dst)
        hi = torch.maximum(src, dst)
        return torch.unique(lo * num_nodes + hi, sorted=True)

    @staticmethod
    def _sorted_membership(keys, sorted_universe):
        if keys.numel() == 0 or sorted_universe.numel() == 0:
            return torch.zeros_like(keys, dtype=torch.bool)
        pos = torch.searchsorted(sorted_universe, keys)
        valid = pos < sorted_universe.numel()
        out = torch.zeros_like(keys, dtype=torch.bool)
        out[valid] = sorted_universe[pos[valid]] == keys[valid]
        return out

    @classmethod
    def induced_ged_from_selected_edges(cls, data, selected_edge_mask):
        info = cls.build_disjoint_batch_info(data)
        match_src = info["match_src"]
        match_dst = info["match_dst"]
        edge_pair_id = info["edge_pair_id"]
        node_pair_id = info["node_pair_id"]
        node_side = info["node_side"]
        num_pairs = info["num_pairs"]
        num_nodes = info["num_nodes"]
        device = data.x.device

        selected_edges = torch.nonzero(selected_edge_mask, as_tuple=False).view(-1)
        map_l_to_r = torch.full((num_nodes,), -1, device=device, dtype=torch.long)
        map_r_to_l = torch.full((num_nodes,), -1, device=device, dtype=torch.long)
        matched_count = torch.zeros((num_pairs,), device=device, dtype=torch.float)
        node_sub_cost = torch.zeros((num_pairs,), device=device, dtype=torch.float)
        if selected_edges.numel() > 0:
            matched_l = match_src[selected_edges]
            matched_r = match_dst[selected_edges]
            matched_pair = edge_pair_id[selected_edges]
            map_l_to_r[matched_l] = matched_r
            map_r_to_l[matched_r] = matched_l
            matched_count.scatter_add_(0, matched_pair, torch.ones_like(matched_pair, dtype=torch.float))
            left_labels = cls._node_labels(data.x[matched_l])
            right_labels = cls._node_labels(data.x[matched_r])
            node_sub = (left_labels != right_labels).float()
            node_sub_cost.scatter_add_(0, matched_pair, node_sub)

        node_cost = (data.n[:, 0].float() - matched_count) + (data.n[:, 1].float() - matched_count) + node_sub_cost

        left_keys = cls._canonical_unique_keys(data.edge_index, node_side, 0, num_nodes)
        right_keys = cls._canonical_unique_keys(data.edge_index, node_side, 1, num_nodes)

        left_u = left_keys // num_nodes
        left_v = left_keys % num_nodes
        left_pair_id = node_pair_id[left_u]
        mapped_ru = map_l_to_r[left_u]
        mapped_rv = map_l_to_r[left_v]
        left_both_matched = (mapped_ru >= 0) & (mapped_rv >= 0)
        mapped_right_keys = (
            torch.minimum(mapped_ru[left_both_matched], mapped_rv[left_both_matched]) * num_nodes
            + torch.maximum(mapped_ru[left_both_matched], mapped_rv[left_both_matched])
        )
        left_exists = torch.zeros_like(left_both_matched, dtype=torch.bool)
        left_exists[left_both_matched] = cls._sorted_membership(mapped_right_keys, right_keys)
        edge_delete_cost = torch.zeros((num_pairs,), device=device, dtype=torch.float)
        edge_delete_cost.scatter_add_(0, left_pair_id, ((~left_both_matched) | (~left_exists)).float())

        right_u = right_keys // num_nodes
        right_v = right_keys % num_nodes
        right_pair_id = node_pair_id[right_u]
        mapped_lu = map_r_to_l[right_u]
        mapped_lv = map_r_to_l[right_v]
        right_both_matched = (mapped_lu >= 0) & (mapped_lv >= 0)
        mapped_left_keys = (
            torch.minimum(mapped_lu[right_both_matched], mapped_lv[right_both_matched]) * num_nodes
            + torch.maximum(mapped_lu[right_both_matched], mapped_lv[right_both_matched])
        )
        right_exists = torch.zeros_like(right_both_matched, dtype=torch.bool)
        right_exists[right_both_matched] = cls._sorted_membership(mapped_left_keys, left_keys)
        edge_insert_cost = torch.zeros((num_pairs,), device=device, dtype=torch.float)
        edge_insert_cost.scatter_add_(0, right_pair_id, ((~right_both_matched) | (~right_exists)).float())

        return node_cost + edge_delete_cost + edge_insert_cost

    def forward(self, data, noise_mapping_attr, t):
        if bool(getattr(self.args, "enable_topk_pruning", False)):
            raise ValueError(
                "lightgt_disjoint does not support --enable-topk-pruning because "
                "that path rebuilds dense candidate tensors."
            )
        return super().forward(data, noise_mapping_attr, t)
