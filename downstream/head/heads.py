"""Downstream prediction heads for the RNA foundation benchmark.

Token embeddings are nucleotide states only. Padding positions are marked by
``padding_mask`` (True = pad). Sequence tasks pool with ``masked_mean`` before
the MLP. Pair tasks consume the full token embedding.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


def masked_mean(embeddings: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
    """Mean over non-padding nucleotide tokens. embeddings: [B, L, D]."""
    keep = (~padding_mask).unsqueeze(-1).to(embeddings.dtype)
    return (embeddings * keep).sum(dim=1) / keep.sum(dim=1).clamp(min=1.0)


class MLPHead(nn.Module):
    """Masked mean pool is applied outside. Linear -> GELU -> Linear.

    ncRNA family uses hidden=256 and 88 classes.
    Splice site uses hidden=128 and 1 logit.
    """

    def __init__(self, c_in: int, hidden: int, n_out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(c_in, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_out),
        )

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.net(pooled)


class ResNet1DBlock(nn.Module):
    def __init__(self, embed_dim: int, kernel_size: int = 3):
        super().__init__()
        self.conv_net = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, kernel_size, padding="same", bias=False),
            nn.InstanceNorm1d(embed_dim),
            nn.ELU(inplace=True),
            nn.Conv1d(embed_dim, embed_dim, kernel_size, padding="same", bias=False),
            nn.InstanceNorm1d(embed_dim),
            nn.ELU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv_net(x)


class MRLResNetHead(nn.Module):
    """Token embeddings -> 6 x 1D ResNet -> masked global pool -> linear.

    Width 32 and dropout 0.2 follow the RiNALMo ribosome-loading head.
    """

    def __init__(self, c_in: int, embed_dim: int = 32, num_blocks: int = 6, dropout: float = 0.2):
        super().__init__()
        self.linear_in = nn.Linear(c_in, embed_dim)
        self.blocks = nn.ModuleList(ResNet1DBlock(embed_dim) for _ in range(num_blocks))
        self.dropout = nn.Dropout(dropout)
        self.linear_out = nn.Linear(embed_dim, 1)

    def forward(self, embeddings: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        x = self.linear_in(embeddings)
        x = x.transpose(1, 2)
        for block in self.blocks:
            x = block(x)
        x = x.transpose(1, 2)
        x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        x = x.sum(dim=1) / (~padding_mask).sum(dim=1).clamp(min=1).unsqueeze(-1)
        return self.linear_out(self.dropout(x)).squeeze(-1)


class ResNet2DBlock(nn.Module):
    def __init__(self, embed_dim: int, kernel_size: int = 3):
        super().__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=False),
            nn.InstanceNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=kernel_size, padding="same", bias=False),
            nn.InstanceNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=False),
            nn.InstanceNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv_net(x)


def _symmetrize(logits: torch.Tensor) -> torch.Tensor:
    upper = torch.triu(logits, diagonal=1)
    return upper + upper.transpose(-1, -2)


class PairResNetHead(nn.Module):
    """Pair concatenation -> 2D ResNet -> pair or contact logits.

    Used by secondary structure (TS0, bpRNA-new, RIVAS, ArchiveII) and by the
    strong RNA 3D contact head. Default width 64, two residual blocks.
    """

    def __init__(self, c_in: int, conv_dim: int = 64, num_blocks: int = 2, kernel_size: int = 3):
        super().__init__()
        self.linear_in = nn.Linear(c_in * 2, conv_dim)
        self.blocks = nn.ModuleList(ResNet2DBlock(conv_dim, kernel_size) for _ in range(num_blocks))
        self.conv_out = nn.Conv2d(conv_dim, 1, kernel_size=kernel_size, padding="same")

    def forward(self, embeddings: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        del padding_mask
        # Linear(cat(left, right)) == Linear(left) + Linear(right). The split form
        # does not materialize [B, L, L, 2D] and matches the concatenated path.
        weight = self.linear_in.weight
        hidden = embeddings.shape[-1]
        left = nn.functional.linear(embeddings, weight[:, :hidden])
        right = nn.functional.linear(embeddings, weight[:, hidden:])
        paired = left.unsqueeze(2) + right.unsqueeze(1)
        if self.linear_in.bias is not None:
            paired = paired + self.linear_in.bias
        x = paired.permute(0, 3, 1, 2).contiguous()
        for block in self.blocks:
            x = block(x)
        return _symmetrize(self.conv_out(x).squeeze(1))


class ContactProbeHead(nn.Module):
    """Shared 64-d projection -> bilinear dot product -> distance embedding -> 3x3 conv."""

    def __init__(self, c_in: int, proj_dim: int = 64, max_distance: int = 32):
        super().__init__()
        self.max_distance = max_distance
        self.proj = nn.Linear(c_in, proj_dim)
        self.dist_emb = nn.Embedding(max_distance + 1, 1)
        self.smooth = nn.Conv2d(1, 1, kernel_size=3, padding=1)

    def forward(self, embeddings: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        del padding_mask
        content = self.proj(embeddings)
        score = torch.matmul(content, content.transpose(-1, -2))
        length = content.shape[1]
        positions = torch.arange(length, device=content.device)
        distance = (positions[:, None] - positions[None, :]).abs().clamp(max=self.max_distance)
        score = score + self.dist_emb(distance).squeeze(-1)
        logits = self.smooth(score.unsqueeze(1)).squeeze(1)
        return _symmetrize(logits)
