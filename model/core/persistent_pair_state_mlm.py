"""Persistent pair-state RNA masked-language model.

This is the scalable, checkpointable form of the pair-state module validated in
``model/experiments/dependency_graph_feasibility``.  A sparse pair state is
created once and updated through the upper Transformer layers. Native learned
relations affect token representations through a bounded residual. Training-only
objective relations use a shared-parameter shadow state that never writes into
token hidden states.
"""
from __future__ import annotations

import json
import os
from dataclasses import fields
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from model.core.models import RNAModelConfig, RNATransformerForMaskedLM
from model.core.pair_state_attention import (
    CandidateRetriever,
    PairStateStep,
    _gather,
)


class PersistentPairStateForMaskedLM(nn.Module):
    """Transformer + MLM with a persistent sparse pair state in upper layers."""

    model_type = "rna_rope_persistent_pair_state_mlm"

    def __init__(
        self,
        base_config: RNAModelConfig,
        *,
        pair_layer_start: int,
        pair_dim: int = 128,
        pair_heads: int = 4,
        max_candidates: int = 8,
        forced_slots: int = 2,
        min_distance: int = 4,
        pair_dropout: float = 0.1,
        residual_cap: float = 0.006,
        pair_mass_cap: float = 0.25,
        gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        if not 0 <= pair_layer_start < base_config.num_hidden_layers:
            raise ValueError("pair_layer_start must be inside the encoder")
        if forced_slots > max_candidates:
            raise ValueError("forced_slots cannot exceed max_candidates")
        if pair_dim % pair_heads:
            raise ValueError("pair_dim must be divisible by pair_heads")
        self.base = RNATransformerForMaskedLM(base_config)
        self.config = base_config
        self.pair_layer_start = int(pair_layer_start)
        self.pair_dim = int(pair_dim)
        self.pair_heads = int(pair_heads)
        self.max_candidates = int(max_candidates)
        self.forced_slots = int(forced_slots)
        self.min_distance = int(min_distance)
        self.pair_dropout = float(pair_dropout)
        self.default_residual_cap = float(residual_cap)
        self.pair_mass_cap = float(pair_mass_cap)
        self.gradient_checkpointing = bool(gradient_checkpointing)

        self.retriever = CandidateRetriever(
            base_config.hidden_size,
            pair_dim,
            max_candidates,
            forced_slots,
            min_distance,
        )
        self.distance_emb = nn.Embedding(16, pair_dim)
        self.state_init_norm = nn.LayerNorm(base_config.hidden_size)
        self.state_node = nn.Linear(base_config.hidden_size, pair_dim, bias=False)
        self.state_init = nn.Linear(pair_dim * 4, pair_dim)
        routed_layers = base_config.num_hidden_layers - pair_layer_start
        self.steps = nn.ModuleList(
            [
                PairStateStep(
                    base_config.hidden_size,
                    pair_dim,
                    pair_heads,
                    pair_dropout,
                    residual_cap,
                    pair_mass_cap,
                )
                for _ in range(routed_layers)
            ]
        )
        self.coupling_head = nn.Linear(pair_dim, 16)

    @property
    def pair_modules(self):
        return (
            self.retriever,
            self.distance_emb,
            self.state_init_norm,
            self.state_node,
            self.state_init,
            self.steps,
            self.coupling_head,
        )

    def set_residual_cap(self, value: float) -> None:
        value = float(value)
        if not 0.0 <= value <= 1.0:
            raise ValueError("residual cap must be in [0, 1]")
        for pair_step in self.steps:
            pair_step.residual_cap = value

    def current_residual_cap(self) -> float:
        return float(self.steps[0].residual_cap)

    def _initialize_state(
        self,
        hidden: torch.Tensor,
        indices: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        node = self.state_node(self.state_init_norm(hidden))
        anchor = node[:, :, None, :].expand(-1, -1, indices.shape[-1], -1)
        partner = _gather(node, indices)
        positions = torch.arange(hidden.shape[1], device=hidden.device)[None, :, None]
        distance = (positions - indices.clamp(min=0)).abs().clamp_min(1)
        buckets = distance.float().log2().floor().long().clamp(0, 15)
        state = self.state_init(
            torch.cat(
                (
                    anchor + partner,
                    (anchor - partner).abs(),
                    anchor * partner,
                    self.distance_emb(buckets),
                ),
                dim=-1,
            )
        )
        return state * mask[:, :, :, None].to(state.dtype)

    def _initialize_objective_state(
        self,
        hidden: torch.Tensor,
        edge_row: torch.Tensor,
        edge_i: torch.Tensor,
        edge_j: torch.Tensor,
    ) -> torch.Tensor:
        """Initialize one symmetric shadow state per canonical objective edge."""
        node = self.state_node(self.state_init_norm(hidden))
        anchor = node[edge_row, edge_i]
        partner = node[edge_row, edge_j]
        distance = (edge_i - edge_j).abs().clamp_min(1)
        buckets = distance.float().log2().floor().long().clamp(0, 15)
        return self.state_init(
            torch.cat(
                (
                    anchor + partner,
                    (anchor - partner).abs(),
                    anchor * partner,
                    self.distance_emb(buckets),
                ),
                dim=-1,
            )
        )

    def _transformer_layer(
        self,
        layer: nn.Module,
        hidden: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
            if attention_mask is None:
                return checkpoint(
                    lambda value: layer(value, attention_mask=None),
                    hidden,
                    use_reentrant=False,
                )
            return checkpoint(
                lambda value, mask: layer(value, attention_mask=mask),
                hidden,
                attention_mask,
                use_reentrant=False,
            )
        return layer(hidden, attention_mask=attention_mask)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        objective_edge_row: Optional[torch.Tensor] = None,
        objective_edge_i: Optional[torch.Tensor] = None,
        objective_edge_j: Optional[torch.Tensor] = None,
        retrieval_edge_row: Optional[torch.Tensor] = None,
        retrieval_edge_i: Optional[torch.Tensor] = None,
        retrieval_edge_j: Optional[torch.Tensor] = None,
        residual_cap: Optional[float] = None,
        return_hidden_states: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if residual_cap is not None:
            self.set_residual_cap(residual_cap)
        encoder = self.base.encoder
        hidden = encoder.dropout(encoder.embeddings(input_ids))
        state = objective_state = candidates = candidate_mask = retrieval = None
        layer_metrics = []
        for layer_index, layer in enumerate(encoder.layers):
            if layer_index == self.pair_layer_start:
                retrieval = self.retriever(
                    hidden,
                    input_ids,
                    attention_mask,
                )
                candidates = retrieval["candidate_indices"]
                candidate_mask = retrieval["candidate_mask"]
                state = self._initialize_state(hidden, candidates, candidate_mask)
                if objective_edge_row is not None:
                    required = (objective_edge_i, objective_edge_j)
                    if any(value is None for value in required):
                        raise ValueError("all objective edge tensors are required together")
                    objective_state = self._initialize_objective_state(
                        hidden,
                        objective_edge_row,
                        objective_edge_i,
                        objective_edge_j,
                    )
            if layer_index >= self.pair_layer_start:
                pair_step = self.steps[layer_index - self.pair_layer_start]
                state = pair_step.update_pair_state(
                    hidden, state, candidates, candidate_mask
                )
                if objective_state is not None:
                    objective_state = pair_step.update_objective_state(
                        hidden,
                        objective_state,
                        objective_edge_row,
                        objective_edge_i,
                        objective_edge_j,
                    )
                routed = pair_step.apply_pair_to_hidden(
                    hidden, state, candidates, candidate_mask
                )
                hidden = routed["hidden_states"]
                layer_metrics.append(routed)
            hidden = self._transformer_layer(layer, hidden, attention_mask)
        hidden = encoder.ln_f(hidden)
        logits = self.base.lm_head(hidden)
        result: Dict[str, torch.Tensor] = {
            "logits": logits,
            "pair_state": state,
            "objective_edge_state": objective_state,
            "candidate_indices": candidates,
            "candidate_mask": candidate_mask,
            "retrieval_q": retrieval["retrieval_q"],
            "retrieval_k": retrieval["retrieval_k"],
            "layer_metrics": layer_metrics,
        }
        if return_hidden_states:
            result["hidden_states"] = hidden
        if objective_state is not None:
            result["coupling"] = self.coupling_head(objective_state).view(-1, 4, 4)
        if retrieval_edge_row is not None:
            required = (retrieval_edge_i, retrieval_edge_j)
            if any(item is None for item in required):
                raise ValueError("all retrieval edge tensors are required together")
            retrieval_q = retrieval["retrieval_q"]
            retrieval_k = retrieval["retrieval_k"]
            result["retrieval_logit"] = (
                retrieval_q[retrieval_edge_row, retrieval_edge_i]
                * retrieval_k[retrieval_edge_row, retrieval_edge_j]
            ).sum(-1) * retrieval_q.shape[-1] ** 0.5
        return result

    def checkpoint_config(self) -> Dict[str, object]:
        return {
            **self.config.to_dict(),
            "model_type": self.model_type,
            "input_channels": ["base"],
            "pretraining_objective": "mlm_persistent_pair_state_joint16_retrieval",
            "has_structure_channel": False,
            "pair_state_persistent": True,
            "native_graph_shape": "LxK",
            "objective_edges_shadow_only": True,
            "pair_state_enters_token_attention": True,
            "inference_residual_cap": self.default_residual_cap,
            "pair_layer_start": self.pair_layer_start,
            "pair_dim": self.pair_dim,
            "pair_heads": self.pair_heads,
            "max_candidates": self.max_candidates,
            "forced_slots": self.forced_slots,
            "pair_min_distance": self.min_distance,
            "pair_dropout": self.pair_dropout,
            "training_residual_cap": self.default_residual_cap,
            "pair_mass_cap": self.pair_mass_cap,
            "gradient_checkpointing": self.gradient_checkpointing,
        }

    def save_pretrained(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "config.json"), "w", encoding="utf-8") as handle:
            json.dump(self.checkpoint_config(), handle, indent=2)
        temporary = os.path.join(path, "pytorch_model.bin.tmp")
        torch.save(self.state_dict(), temporary)
        os.replace(temporary, os.path.join(path, "pytorch_model.bin"))

    @classmethod
    def from_pretrained(
        cls,
        path: str,
        *,
        map_location: str | torch.device = "cpu",
        strict: bool = True,
    ) -> "PersistentPairStateForMaskedLM":
        with open(os.path.join(path, "config.json"), encoding="utf-8") as handle:
            value = json.load(handle)
        if value.get("model_type") != cls.model_type:
            raise ValueError(f"not a {cls.model_type} checkpoint: {path}")
        base_names = {field.name for field in fields(RNAModelConfig)}
        base_config = RNAModelConfig(
            **{key: item for key, item in value.items() if key in base_names}
        )
        model = cls(
            base_config,
            pair_layer_start=int(value["pair_layer_start"]),
            pair_dim=int(value["pair_dim"]),
            pair_heads=int(value["pair_heads"]),
            max_candidates=int(value["max_candidates"]),
            forced_slots=int(value["forced_slots"]),
            min_distance=int(value.get("pair_min_distance", 4)),
            pair_dropout=float(value.get("pair_dropout", 0.1)),
            residual_cap=float(value.get("training_residual_cap", 0.006)),
            pair_mass_cap=float(value.get("pair_mass_cap", 0.25)),
            gradient_checkpointing=bool(value.get("gradient_checkpointing", True)),
        )
        state = torch.load(
            os.path.join(path, "pytorch_model.bin"),
            map_location=map_location,
            weights_only=True,
        )
        model.load_state_dict(state, strict=strict)
        return model


class PersistentPairStateEncoderAdapter(nn.Module):
    """Sequence-only downstream view with learned pair feedback enabled."""

    def __init__(
        self,
        foundation: PersistentPairStateForMaskedLM,
        residual_cap: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.foundation = foundation
        self.residual_cap = float(
            foundation.default_residual_cap if residual_cap is None else residual_cap
        )

    @property
    def config(self):
        return self.foundation.config

    @property
    def layers(self):
        return self.foundation.base.encoder.layers

    @property
    def ln_f(self):
        return self.foundation.base.encoder.ln_f

    def forward(
        self,
        input_ids: torch.Tensor,
        struct_input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del struct_input_ids
        return self.foundation(
            input_ids=input_ids,
            attention_mask=attention_mask,
            residual_cap=self.residual_cap,
            return_hidden_states=True,
        )["hidden_states"]


class PersistentPairStateFoundationView(nn.Module):
    """Compatibility shell expected by the repository's downstream benchmark."""

    def __init__(self, foundation: PersistentPairStateForMaskedLM) -> None:
        super().__init__()
        self.encoder = PersistentPairStateEncoderAdapter(foundation)
        self.config = foundation.config
