"""Sparse persistent pair-state attention for the 85M RNA encoder.

The pair state is initialized once at the first routed layer, updated after
every routed layer, and contributes an additive bias to a sparse attention
distribution.  No structure or clean-token input is required at inference.
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gather(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather [B,L,D] values with [B,L,K] indices."""
    batch, length, dim = values.shape
    offset = torch.arange(batch, device=values.device)[:, None, None] * length
    flat = values.reshape(batch * length, dim)
    return flat[offset + indices.clamp(min=0)]


class CandidateRetriever(nn.Module):
    """Sequence-only native Top-K retrieval; supervision edges never enter here."""

    def __init__(self, hidden_size: int, pair_dim: int, max_candidates: int,
                 forced_slots: int, min_distance: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.q_proj = nn.Linear(hidden_size, pair_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, pair_dim, bias=False)
        self.max_candidates = max_candidates
        self.forced_slots = forced_slots
        self.min_distance = min_distance

    def forward(self, hidden: torch.Tensor, input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor]) -> Dict[str, torch.Tensor]:
        batch, length, _ = hidden.shape
        normalized = self.norm(hidden)
        q = F.normalize(self.q_proj(normalized).float(), dim=-1)
        k = F.normalize(self.k_proj(normalized).float(), dim=-1)
        scores = torch.matmul(q, k.transpose(-1, -2)) * math.sqrt(q.shape[-1])

        # Base tokens and MASK may route; specials, N and padding may not.
        valid_token = ((input_ids >= 3) & (input_ids <= 7))
        if attention_mask is not None:
            valid_token &= attention_mask.bool()
        positions = torch.arange(length, device=hidden.device)
        distance_ok = (positions[:, None] - positions[None, :]).abs() >= self.min_distance
        valid = valid_token[:, :, None] & valid_token[:, None, :] & distance_ok[None]
        scores = scores.masked_fill(~valid, -1e4)

        # The model graph is always the full learned L x K graph.  Training-only
        # objective edges are handled by a separate shadow-state path.
        fresh_k = min(self.max_candidates, length)
        if fresh_k:
            fresh_scores, fresh_indices = torch.topk(scores, fresh_k, dim=-1)
            fresh_mask = valid.gather(-1, fresh_indices)
        else:
            fresh_indices = torch.empty(batch, length, 0, dtype=torch.long, device=hidden.device)
            fresh_mask = torch.empty(batch, length, 0, dtype=torch.bool, device=hidden.device)

        indices, mask = fresh_indices, fresh_mask

        if indices.shape[-1] < self.max_candidates:
            pad = self.max_candidates - indices.shape[-1]
            indices = F.pad(indices, (0, pad))
            mask = F.pad(mask, (0, pad), value=False)
        return {
            "candidate_indices": indices[:, :, :self.max_candidates].masked_fill(
                ~mask[:, :, :self.max_candidates], -1
            ),
            "candidate_mask": mask[:, :, :self.max_candidates],
            "retrieval_q": q,
            "retrieval_k": k,
        }


class PairStateStep(nn.Module):
    """Update persistent z_ij, then use it as sparse token-attention bias."""

    def __init__(self, hidden_size: int, pair_dim: int, pair_heads: int,
                 dropout: float, residual_cap: float, pair_mass_cap: float):
        super().__init__()
        if pair_dim % pair_heads:
            raise ValueError("pair_dim must be divisible by pair_heads")
        self.pair_heads = pair_heads
        self.head_dim = pair_dim // pair_heads
        self.residual_cap = residual_cap
        self.pair_mass_cap = pair_mass_cap
        self.node_norm = nn.LayerNorm(hidden_size)
        self.node_proj = nn.Linear(hidden_size, pair_dim, bias=False)
        self.state_norm = nn.LayerNorm(pair_dim)
        self.state_update = nn.Sequential(
            nn.Linear(pair_dim * 4, pair_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(pair_dim * 2, pair_dim),
        )
        self.q_proj = nn.Linear(hidden_size, pair_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, pair_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, pair_dim, bias=False)
        self.state_bias = nn.Linear(pair_dim, pair_heads, bias=False)
        self.state_value = nn.Linear(pair_dim, pair_dim, bias=False)
        self.null_logit = nn.Linear(hidden_size, pair_heads)
        self.out_proj = nn.Linear(pair_dim, hidden_size, bias=False)
        self.gate = nn.Linear(hidden_size + pair_dim, 1)
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.state_bias.weight, mean=0.0, std=1e-3)
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -2.0)

    def update_pair_state(self, hidden: torch.Tensor, state: torch.Tensor,
                          candidate_indices: torch.Tensor,
                          candidate_mask: torch.Tensor) -> torch.Tensor:
        candidates = candidate_indices.shape[-1]
        normalized = self.node_norm(hidden)
        node = self.node_proj(normalized)
        anchor = node[:, :, None, :].expand(-1, -1, candidates, -1)
        partner = _gather(node, candidate_indices)
        normalized_state = self.state_norm(state)
        state_input = torch.cat((
            normalized_state, anchor + partner,
            (anchor - partner).abs(), anchor * partner,
        ), dim=-1)
        # Re-anchor the persistent state at every routed layer.  Reusing the
        # existing normalization keeps the state fed to bias/value projections
        # bounded without adding parameters or another normalization pass.
        state = normalized_state + self.state_update(state_input)
        return state * candidate_mask[:, :, :, None].to(state.dtype)

    def update_objective_state(self, hidden: torch.Tensor, state: torch.Tensor,
                               edge_row: torch.Tensor, edge_i: torch.Tensor,
                               edge_j: torch.Tensor) -> torch.Tensor:
        """Update sparse shadow relations without applying any token residual."""
        if not state.numel():
            return state
        normalized = self.node_norm(hidden)
        node = self.node_proj(normalized)
        anchor = node[edge_row, edge_i]
        partner = node[edge_row, edge_j]
        normalized_state = self.state_norm(state)
        state_input = torch.cat((
            normalized_state, anchor + partner,
            (anchor - partner).abs(), anchor * partner,
        ), dim=-1)
        return normalized_state + self.state_update(state_input)

    def apply_pair_to_hidden(self, hidden: torch.Tensor, state: torch.Tensor,
                             candidate_indices: torch.Tensor,
                             candidate_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch, length, candidates = candidate_indices.shape
        normalized = self.node_norm(hidden)

        q = self.q_proj(normalized).view(batch, length, self.pair_heads, self.head_dim)
        k = self.k_proj(normalized)
        v = self.v_proj(normalized)
        gathered_k = _gather(k, candidate_indices).view(
            batch, length, candidates, self.pair_heads, self.head_dim
        )
        gathered_v = _gather(v, candidate_indices).view(
            batch, length, candidates, self.pair_heads, self.head_dim
        )
        logits = (
            q[:, :, None, :, :] * gathered_k
        ).sum(-1).float() / math.sqrt(self.head_dim)
        logits = logits + self.state_bias(state.float())
        logits = logits.masked_fill(~candidate_mask[:, :, :, None], -1e4)
        null = self.null_logit(normalized).float()[:, :, None, :]
        weights_all = F.softmax(torch.cat((logits, null), dim=2), dim=2)
        raw_candidate_weights = (
            weights_all[:, :, :candidates, :] * candidate_mask[:, :, :, None]
        )
        true_null_probability = weights_all[:, :, candidates, :]
        # Preserve null competition: effective mass is cap * (1 - p_null).
        weights = float(self.pair_mass_cap) * raw_candidate_weights

        state_v = self.state_value(state).view(
            batch, length, candidates, self.pair_heads, self.head_dim
        )
        values = gathered_v + state_v.to(gathered_v.dtype)
        context = (weights.to(values.dtype)[..., None] * values).sum(2)
        context_flat = context.reshape(batch, length, -1)
        gate = torch.sigmoid(self.gate(torch.cat((normalized, context_flat), dim=-1)))
        # Diagnostics only: how much each valid native slot participates in the
        # pair-to-token route. These tensors do not alter routing or gradients.
        slot_attention = weights.mean(-1)
        slot_utilization = (
            slot_attention
            * gate.squeeze(-1)[:, :, None].to(slot_attention.dtype)
        )
        raw_update = self.dropout(self.out_proj(context_flat)).float()
        # Strict relative-norm bound: learned projections cannot silently grow
        # past the configured fraction of the pretrained token representation.
        direction = F.normalize(raw_update, dim=-1)
        hidden_norm = hidden.float().norm(dim=-1, keepdim=True).clamp_min(1e-8)
        update = float(self.residual_cap) * gate.float() * hidden_norm * direction
        hidden = hidden + update.to(hidden.dtype)
        valid = candidate_mask.any(-1)
        return {
            "hidden_states": hidden,
            "pair_state": state,
            "attention_weights": weights,
            "slot_attention": slot_attention,
            "slot_utilization": slot_utilization,
            "true_null_probability": true_null_probability,
            "null_probability": true_null_probability,
            "gate_mean": gate[valid].mean() if valid.any() else gate.new_zeros(()),
            "effective_pair_attention_mass": weights.sum(2),
            "attention_mass": weights.sum(2).mean(-1)[valid].mean() if valid.any() else gate.new_zeros(()),
            "update_ratio": (
                update[valid].norm(dim=-1).mean()
                / hidden.float()[valid].norm(dim=-1).mean().clamp_min(1e-8)
                if valid.any() else gate.new_zeros(())
            ),
        }

    def forward(self, hidden: torch.Tensor, state: torch.Tensor,
                candidate_indices: torch.Tensor,
                candidate_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        state = self.update_pair_state(hidden, state, candidate_indices, candidate_mask)
        return self.apply_pair_to_hidden(hidden, state, candidate_indices, candidate_mask)


# class PersistentPairStateAttentionModel(nn.Module):
#     """Wrap an existing RNA MLM and route a persistent pair state in top layers."""

#     def __init__(self, base_model: nn.Module, pair_layer_start: int = 8,
#                  pair_dim: int = 128, pair_heads: int = 4,
#                  max_candidates: int = 4, forced_slots: int = 2,
#                  min_distance: int = 4, dropout: float = 0.1,
#                  residual_cap: float = 0.1, pair_mass_cap: float = 0.25):
#         super().__init__()
#         self.base = base_model
#         self.config = base_model.config
#         self.pair_layer_start = pair_layer_start
#         self.max_candidates = max_candidates
#         self.forced_slots = forced_slots
#         routed_layers = self.config.num_hidden_layers - pair_layer_start
#         if routed_layers <= 0:
#             raise ValueError("pair_layer_start must be inside the encoder")
#         self.retriever = CandidateRetriever(
#             self.config.hidden_size, pair_dim, max_candidates, forced_slots, min_distance
#         )
#         self.distance_emb = nn.Embedding(16, pair_dim)
#         self.state_init_norm = nn.LayerNorm(self.config.hidden_size)
#         self.state_node = nn.Linear(self.config.hidden_size, pair_dim, bias=False)
#         self.state_init = nn.Linear(pair_dim * 4, pair_dim)
#         self.steps = nn.ModuleList([
#             PairStateStep(
#                 self.config.hidden_size, pair_dim, pair_heads, dropout,
#                 residual_cap, pair_mass_cap,
#             )
#             for _ in range(routed_layers)
#         ])
#         self.coupling_head = nn.Linear(pair_dim, 16)
#         self.structure_head = nn.Linear(pair_dim, 1)

#     def _initialize_state(self, hidden: torch.Tensor, indices: torch.Tensor,
#                           mask: torch.Tensor) -> torch.Tensor:
#         node = self.state_node(self.state_init_norm(hidden))
#         anchor = node[:, :, None, :].expand(-1, -1, indices.shape[-1], -1)
#         partner = _gather(node, indices)
#         positions = torch.arange(hidden.shape[1], device=hidden.device)[None, :, None]
#         distance = (positions - indices.clamp(min=0)).abs().clamp_min(1)
#         buckets = distance.float().log2().floor().long().clamp(0, 15)
#         state = self.state_init(torch.cat((
#             anchor + partner, (anchor - partner).abs(), anchor * partner,
#             self.distance_emb(buckets),
#         ), dim=-1))
#         return state * mask[:, :, :, None].to(state.dtype)

#     def forward(self, input_ids: torch.Tensor,
#                 attention_mask: Optional[torch.Tensor] = None,
#                 forced_indices: Optional[torch.Tensor] = None,
#                 forced_mask: Optional[torch.Tensor] = None,
#                 edge_row: Optional[torch.Tensor] = None,
#                 edge_i: Optional[torch.Tensor] = None,
#                 edge_j: Optional[torch.Tensor] = None,
#                 edge_slot_i: Optional[torch.Tensor] = None,
#                 edge_slot_j: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
#         encoder = self.base.encoder
#         hidden = encoder.dropout(encoder.embeddings(input_ids))
#         state = candidates = mask = None
#         layer_metrics = []
#         for layer_index, layer in enumerate(encoder.layers):
#             if layer_index == self.pair_layer_start:
#                 retrieval = self.retriever(
#                     hidden, input_ids, attention_mask
#                 )
#                 candidates = retrieval["candidate_indices"]
#                 mask = retrieval["candidate_mask"]
#                 state = self._initialize_state(hidden, candidates, mask)
#             if layer_index >= self.pair_layer_start:
#                 routed = self.steps[layer_index - self.pair_layer_start](
#                     hidden, state, candidates, mask
#                 )
#                 hidden, state = routed["hidden_states"], routed["pair_state"]
#                 layer_metrics.append(routed)
#             hidden = layer(hidden, attention_mask=attention_mask)
#         hidden = encoder.ln_f(hidden)
#         logits = self.base.lm_head(hidden)
#         result = {
#             "logits": logits,
#             "hidden_states": hidden,
#             "pair_state": state,
#             "candidate_indices": candidates,
#             "candidate_mask": mask,
#             "layer_metrics": layer_metrics,
#         }
#         if edge_row is not None:
#             state_i = state[edge_row, edge_i, edge_slot_i]
#             state_j = state[edge_row, edge_j, edge_slot_j]
#             result["edge_state"] = 0.5 * (state_i + state_j)
#             result["coupling"] = self.coupling_head(result["edge_state"]).view(-1, 4, 4)
#             result["pair_logit"] = self.structure_head(result["edge_state"]).squeeze(-1)
#             retrieval_q = retrieval["retrieval_q"]
#             retrieval_k = retrieval["retrieval_k"]
#             score_ij = (
#                 retrieval_q[edge_row, edge_i] * retrieval_k[edge_row, edge_j]
#             ).sum(-1)
#             score_ji = (
#                 retrieval_q[edge_row, edge_j] * retrieval_k[edge_row, edge_i]
#             ).sum(-1)
#             result["retrieval_logit"] = 0.5 * (score_ij + score_ji) * math.sqrt(
#                 retrieval_q.shape[-1]
#             )
#         return result
