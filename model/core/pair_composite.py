"""Joint endpoint probabilities for the persistent pair state."""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

AUCG_IDS = (4, 5, 6, 7)
ID_TO_AUCG = {4: 0, 5: 1, 6: 2, 7: 3}


def aucg_index_from_token(token_id: int) -> Optional[int]:
    return ID_TO_AUCG.get(int(token_id))


def pair_class_from_tokens(left_id: int, right_id: int) -> Optional[int]:
    a = aucg_index_from_token(left_id)
    b = aucg_index_from_token(right_id)
    if a is None or b is None:
        return None
    return 4 * a + b


def distance_bucket(distance: int) -> int:
    """Log2-style buckets for relative distance (j-i >= 1)."""
    d = max(1, int(distance))
    return min(15, int(math.floor(math.log2(d))))


def double_center_coupling(coupling: torch.Tensor) -> torch.Tensor:
    """Double-center 4x4 coupling so zero coupling preserves independent joint."""
    return (
        coupling
        - coupling.mean(dim=-1, keepdim=True)
        - coupling.mean(dim=-2, keepdim=True)
        + coupling.mean(dim=(-1, -2), keepdim=True)
    )


class PersistentPairBranch(nn.Module):
    """Explicit edge state z_ij updated over several residual blocks."""

    def __init__(
        self,
        hidden_size: int,
        pair_dim: int = 256,
        pair_blocks: int = 3,
        pair_dropout: float = 0.1,
        num_distance_buckets: int = 16,
    ):
        super().__init__()
        self.pair_dim = pair_dim
        self.distance_emb = nn.Embedding(num_distance_buckets, 32)
        init_in = hidden_size * 4 + 32
        self.input_proj = nn.Linear(init_in, pair_dim)
        self.blocks = nn.ModuleList()
        for _ in range(pair_blocks):
            block_in = pair_dim + hidden_size * 2 + 32
            self.blocks.append(
                nn.Sequential(
                    nn.LayerNorm(block_in),
                    nn.Linear(block_in, pair_dim * 2),
                    nn.GELU(),
                    nn.Dropout(pair_dropout),
                    nn.Linear(pair_dim * 2, pair_dim),
                    nn.Dropout(pair_dropout),
                )
            )
        self.coupling_head = nn.Linear(pair_dim, 16)

    def forward(
        self,
        h_i: torch.Tensor,
        h_j: torch.Tensor,
        distance: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        h_i, h_j: [E, H]
        distance: [E] long (j-i)
        returns z [E, D], raw coupling [E, 4, 4]
        """
        buckets = distance.clamp_min(1).float().log2().floor().long().clamp(0, 15)
        dist = self.distance_emb(buckets)
        feat = torch.cat([h_i, h_j, h_i * h_j, (h_i - h_j).abs(), dist], dim=-1)
        z = self.input_proj(feat)
        for block in self.blocks:
            update_in = torch.cat([z, h_i, h_j, dist], dim=-1)
            z = z + block(update_in)
        coupling = self.coupling_head(z).view(-1, 4, 4)
        return z, coupling


class UnaryCalibrationHead(nn.Module):
    """Shared endpoint calibrator used by both the null and pair models."""

    def __init__(self, hidden_size: int, calibration_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, calibration_dim),
            nn.GELU(),
            nn.Linear(calibration_dim, 4),
        )

    def forward(self, hidden: torch.Tensor, frozen_logp: torch.Tensor) -> torch.Tensor:
        return F.log_softmax(frozen_logp.float() + self.net(hidden.float()), dim=-1)


def joint_log_probs_from_unary_and_coupling(
    logp_i: torch.Tensor,
    logp_j: torch.Tensor,
    coupling: torch.Tensor,
) -> torch.Tensor:
    """
    logp_i/j: [E, 4] float32 log-softmax over AUCG
    coupling: [E, 4, 4]
    returns joint_logp [E, 16]
    """
    centered = double_center_coupling(coupling.float())
    base = logp_i.float().unsqueeze(-1) + logp_j.float().unsqueeze(-2)
    joint_logits = (base + centered).reshape(logp_i.shape[0], 16)
    return F.log_softmax(joint_logits, dim=-1)


def fixed_marginal_joint_log_probs(
    logp_i: torch.Tensor,
    logp_j: torch.Tensor,
    coupling: torch.Tensor,
    iterations: int = 32,
) -> torch.Tensor:
    """Create an interaction joint while preserving both supplied marginals.

    Iterative proportional fitting prevents the pair coupling from winning by
    merely recalibrating either endpoint's unary distribution.
    """
    target_log_i = F.log_softmax(logp_i.float(), dim=-1)
    target_log_j = F.log_softmax(logp_j.float(), dim=-1)
    centered = double_center_coupling(coupling.float()).clamp(-8.0, 8.0)
    log_joint = target_log_i.unsqueeze(-1) + target_log_j.unsqueeze(-2) + centered
    for _ in range(iterations):
        log_joint = log_joint + target_log_i.unsqueeze(-1) - torch.logsumexp(
            log_joint, dim=-1, keepdim=True
        )
        log_joint = log_joint + target_log_j.unsqueeze(-2) - torch.logsumexp(
            log_joint, dim=-2, keepdim=True
        )
    log_joint = log_joint - torch.logsumexp(
        log_joint.reshape(log_joint.shape[0], -1), dim=-1
    ).view(-1, 1, 1)
    return log_joint.reshape(logp_i.shape[0], 16)


def marginal_max_abs_error(
    joint_logp: torch.Tensor, logp_i: torch.Tensor, logp_j: torch.Tensor
) -> torch.Tensor:
    joint = joint_logp.float().exp().view(-1, 4, 4)
    err_i = (joint.sum(dim=-1) - logp_i.float().exp()).abs().amax(dim=-1)
    err_j = (joint.sum(dim=-2) - logp_j.float().exp()).abs().amax(dim=-1)
    return torch.maximum(err_i, err_j)


def independent_joint_log_probs(logp_i: torch.Tensor, logp_j: torch.Tensor) -> torch.Tensor:
    base = logp_i.float().unsqueeze(-1) + logp_j.float().unsqueeze(-2)
    return F.log_softmax(base.reshape(logp_i.shape[0], 16), dim=-1)


def nll_from_joint_logp(joint_logp: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return -joint_logp.gather(1, targets.view(-1, 1)).squeeze(1)
