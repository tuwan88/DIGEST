"""Minimal dense LightGT cross-graph layers used by main_dense.py."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.norm import GraphNorm
from torch_geometric.utils import softmax


class DenseMaskedEdgeNorm(nn.Module):
    def __init__(self, hidden_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_dim))
        self.bias = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, x, pair_mask):
        mask = pair_mask.unsqueeze(-1).to(dtype=x.dtype)
        denom = mask.sum(dim=(1, 2), keepdim=True).clamp_min(1.0)
        mean = (x * mask).sum(dim=(1, 2), keepdim=True) / denom
        var = ((x - mean).pow(2) * mask).sum(dim=(1, 2), keepdim=True) / denom
        x = (x - mean) * torch.rsqrt(var + self.eps)
        x = x * self.weight.view(1, 1, 1, -1) + self.bias.view(1, 1, 1, -1)
        return x * mask


class DenseCrossGraphTransformer(nn.Module):
    def __init__(self, hidden_dim, time_emb_dim, edge_dim, num_heads=4, dropout=0.0):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.edge_in = nn.Linear(edge_dim, hidden_dim)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.edge_bias = nn.Linear(hidden_dim, num_heads, bias=False)
        self.time_bias = nn.Linear(time_emb_dim, num_heads, bias=False)

        self.src_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.dst_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.qk_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.attn_proj = nn.Linear(2 * num_heads, hidden_dim, bias=False)
        self.edge_time = nn.Linear(time_emb_dim, hidden_dim)

        self.edge_ffn = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.edge_norm = DenseMaskedEdgeNorm(hidden_dim)
        self.edge_norm_sparse = GraphNorm(hidden_dim)

        self.edge_value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.edge_gate = nn.Linear(hidden_dim, hidden_dim)
        self.node_time = nn.Linear(time_emb_dim, hidden_dim)
        self.node_ffn = nn.Sequential(
            nn.LayerNorm(hidden_dim * 3),
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.node_norm = GraphNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def project_edge(self, edge_pair):
        return self.edge_in(edge_pair)

    def forward(
        self,
        x1,
        x2,
        edge_pair,
        mask1,
        mask2,
        time_emb=None,
        edge_is_projected=False,
        candidate_mask=None,
        precomputed_q1=None,
        precomputed_k2=None,
    ):
        if candidate_mask is not None:
            return self.forward_topk(
                x1=x1,
                x2=x2,
                edge_pair=edge_pair,
                mask1=mask1,
                mask2=mask2,
                candidate_mask=candidate_mask,
                time_emb=time_emb,
                edge_is_projected=edge_is_projected,
                precomputed_q1=precomputed_q1,
                precomputed_k2=precomputed_k2,
            )

        bsz, n1, _ = x1.shape
        _, n2, _ = x2.shape
        h = self.num_heads
        dh = self.head_dim

        pair_mask = mask1[:, :, None] & mask2[:, None, :]
        edge = edge_pair if edge_is_projected else self.project_edge(edge_pair)
        edge = edge * pair_mask.unsqueeze(-1).to(edge.dtype)

        q1 = self.q_proj(x1).view(bsz, n1, h, dh)
        k2 = self.k_proj(x2).view(bsz, n2, h, dh)
        v1 = self.v_proj(x1).view(bsz, n1, h, dh)
        v2 = self.v_proj(x2).view(bsz, n2, h, dh)

        score = torch.einsum("bihd,bjhd->bhij", q1, k2) * self.scale
        score = score + self.edge_bias(edge).permute(0, 3, 1, 2)
        if time_emb is not None:
            score = score + self.time_bias(time_emb)[:, :, None, None]
        score = score.masked_fill(~pair_mask[:, None, :, :], torch.finfo(score.dtype).min)

        row_alpha = torch.softmax(score, dim=-1).masked_fill(~pair_mask[:, None, :, :], 0.0)
        col_alpha = torch.softmax(score, dim=-2).masked_fill(~pair_mask[:, None, :, :], 0.0)

        edge_context = self.src_proj(x1)[:, :, None, :] + self.dst_proj(x2)[:, None, :, :]
        qk_pair = (q1[:, :, None, :, :] * k2[:, None, :, :, :]).reshape(bsz, n1, n2, self.hidden_dim)
        edge_context = edge_context + self.qk_proj(qk_pair)
        edge_context = edge_context + self.attn_proj(
            torch.cat([row_alpha.permute(0, 2, 3, 1), col_alpha.permute(0, 2, 3, 1)], dim=-1)
        )
        if time_emb is not None:
            edge_context = edge_context + self.edge_time(time_emb)[:, None, None, :]
        edge_context = (edge_context + edge) * pair_mask.unsqueeze(-1).to(edge.dtype)

        edge_delta = self.edge_norm(self.edge_ffn(edge_context), pair_mask)
        if bool(getattr(self, "diagnostic_disable_selected_edge_residual", False)):
            edge_pair_new = self.dropout(edge_delta) * pair_mask.unsqueeze(-1).to(edge.dtype)
        else:
            edge_pair_new = (edge + self.dropout(edge_delta)) * pair_mask.unsqueeze(-1).to(edge.dtype)

        edge_val = self.edge_value(edge_pair_new).view(bsz, n1, n2, h, dh)
        edge_gate = torch.sigmoid(self.edge_gate(edge_pair_new)).view(bsz, n1, n2, h, dh)

        msg1 = (
            row_alpha.permute(0, 2, 3, 1).unsqueeze(-1) * (edge_gate * edge_val + v2[:, None, :, :, :])
        ).sum(dim=2).reshape(bsz, n1, self.hidden_dim)
        msg2 = (
            col_alpha.permute(0, 2, 3, 1).unsqueeze(-1) * (edge_gate * edge_val + v1[:, :, None, :, :])
        ).sum(dim=1).reshape(bsz, n2, self.hidden_dim)

        node_time = x1.new_zeros((bsz, self.hidden_dim)) if time_emb is None else self.node_time(time_emb)
        x1_delta = self.node_ffn(torch.cat([x1, msg1, node_time[:, None, :].expand(-1, n1, -1)], dim=-1))
        x2_delta = self.node_ffn(torch.cat([x2, msg2, node_time[:, None, :].expand(-1, n2, -1)], dim=-1))
        x1_new, x2_new = self._normalize_nodes(x1_delta, x2_delta, mask1, mask2, x1, x2)
        return x1_new, x2_new, edge_pair_new

    def _normalize_nodes(self, x1_delta, x2_delta, mask1, mask2, x1, x2):
        bsz, n1, _ = x1.shape
        n2 = x2.shape[1]
        xcat_delta = torch.cat([x1_delta, x2_delta], dim=1)
        mcat = torch.cat([mask1, mask2], dim=1)
        xcat_delta = xcat_delta * mcat.unsqueeze(-1).to(xcat_delta.dtype)
        bid = torch.arange(bsz, device=x1.device).unsqueeze(1).expand(bsz, n1 + n2)
        flat_delta = xcat_delta[mcat]
        flat_batch = bid[mcat]
        flat_delta = self.node_norm(flat_delta, flat_batch)
        xcat_norm = xcat_delta.new_zeros(xcat_delta.shape)
        xcat_norm[mcat] = flat_delta
        x1_norm = xcat_norm[:, :n1, :]
        x2_norm = xcat_norm[:, n1:, :]
        x1_new = (x1 + self.dropout(F.silu(x1_norm))) * mask1.unsqueeze(-1).to(x1.dtype)
        x2_new = (x2 + self.dropout(F.silu(x2_norm))) * mask2.unsqueeze(-1).to(x2.dtype)
        return x1_new, x2_new

    def forward_topk(
        self,
        x1,
        x2,
        edge_pair,
        mask1,
        mask2,
        candidate_mask,
        time_emb=None,
        edge_is_projected=False,
        precomputed_q1=None,
        precomputed_k2=None,
    ):
        bsz, n1, _ = x1.shape
        _, n2, _ = x2.shape
        h = self.num_heads
        dh = self.head_dim

        pair_mask = mask1[:, :, None] & mask2[:, None, :]
        candidate_mask = candidate_mask & pair_mask
        edge = edge_pair if edge_is_projected else self.project_edge(edge_pair)
        edge = edge * pair_mask.unsqueeze(-1).to(edge.dtype)

        q1 = self.q_proj(x1).view(bsz, n1, h, dh) if precomputed_q1 is None else precomputed_q1
        k2 = self.k_proj(x2).view(bsz, n2, h, dh) if precomputed_k2 is None else precomputed_k2
        v1 = self.v_proj(x1).view(bsz, n1, h, dh)
        v2 = self.v_proj(x2).view(bsz, n2, h, dh)

        b_idx, i_idx, j_idx = candidate_mask.nonzero(as_tuple=True)
        if b_idx.numel() == 0:
            return x1, x2, edge

        q_e = q1[b_idx, i_idx]
        k_e = k2[b_idx, j_idx]
        v1_e = v1[b_idx, i_idx]
        v2_e = v2[b_idx, j_idx]
        edge_e = edge[b_idx, i_idx, j_idx]

        score_e = (q_e * k_e).sum(dim=-1) * self.scale + self.edge_bias(edge_e)
        if time_emb is not None:
            score_e = score_e + self.time_bias(time_emb)[b_idx]

        row_group = b_idx * n1 + i_idx
        col_group = b_idx * n2 + j_idx
        row_alpha = softmax(score_e, row_group, num_nodes=bsz * n1)
        col_alpha = softmax(score_e, col_group, num_nodes=bsz * n2)

        edge_val = self.edge_value(edge_e).view(-1, h, dh)
        edge_gate = torch.sigmoid(self.edge_gate(edge_e)).view(-1, h, dh)
        msg1_e = (row_alpha.unsqueeze(-1) * (edge_gate * edge_val + v2_e)).reshape(-1, self.hidden_dim)
        msg2_e = (col_alpha.unsqueeze(-1) * (edge_gate * edge_val + v1_e)).reshape(-1, self.hidden_dim)

        msg1_flat = x1.new_zeros((bsz * n1, self.hidden_dim))
        msg1_flat.index_add_(0, row_group, msg1_e)
        msg2_flat = x2.new_zeros((bsz * n2, self.hidden_dim))
        msg2_flat.index_add_(0, col_group, msg2_e)
        msg1 = msg1_flat.view(bsz, n1, self.hidden_dim)
        msg2 = msg2_flat.view(bsz, n2, self.hidden_dim)

        node_time = x1.new_zeros((bsz, self.hidden_dim)) if time_emb is None else self.node_time(time_emb)
        x1_delta = self.node_ffn(torch.cat([x1, msg1, node_time[:, None, :].expand(-1, n1, -1)], dim=-1))
        x2_delta = self.node_ffn(torch.cat([x2, msg2, node_time[:, None, :].expand(-1, n2, -1)], dim=-1))
        x1_new, x2_new = self._normalize_nodes(x1_delta, x2_delta, mask1, mask2, x1, x2)

        edge_context = self.src_proj(x1_new)[b_idx, i_idx] + self.dst_proj(x2_new)[b_idx, j_idx]
        edge_context = edge_context + self.qk_proj((q_e * k_e).reshape(-1, self.hidden_dim))
        edge_context = edge_context + self.attn_proj(torch.cat([row_alpha, col_alpha], dim=-1))
        if time_emb is not None:
            edge_context = edge_context + self.edge_time(time_emb)[b_idx]
        edge_context = edge_context + edge_e

        edge_delta = self.edge_norm_sparse(self.edge_ffn(edge_context), b_idx)
        edge_delta = self.dropout(F.silu(edge_delta))
        if bool(getattr(self, "diagnostic_disable_selected_edge_residual", False)):
            edge_new = edge_delta
        else:
            edge_new = edge_e + edge_delta
        if bool(getattr(self, "diagnostic_disable_inactive_edge_carry", False)):
            edge_pair_new = edge.new_zeros(edge.shape)
        else:
            edge_pair_new = edge.clone()
        edge_pair_new[b_idx, i_idx, j_idx] = edge_new
        edge_pair_new = edge_pair_new * pair_mask.unsqueeze(-1).to(edge_pair_new.dtype)
        return x1_new, x2_new, edge_pair_new
