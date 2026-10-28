"""
Transformer config (12 layers, 12 heads, 768 dims) + RoPE-Transformer-Encoder + MLM。
"""
import json
import os
import warnings
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Optional, Tuple, Union, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RNAModelConfig:
    vocab_size: int = 9  # [PAD],[CLS],[SEP],[MASK], A, U, C, G, N
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072  # 4 * hidden_size
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 512
    type_vocab_size: int = 2
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    pad_token_id: int = 0
    # auto: CUDA FP16/BF16 -> packed varlen Flash; otherwise -> SDPA.
    # Other supported values: manual, cudnn_sdpa, varlen_flash.
    attention_backend: str = "auto"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "intermediate_size": self.intermediate_size,
            "hidden_act": self.hidden_act,
            "hidden_dropout_prob": self.hidden_dropout_prob,
            "attention_probs_dropout_prob": self.attention_probs_dropout_prob,
            "max_position_embeddings": self.max_position_embeddings,
            "type_vocab_size": self.type_vocab_size,
            "initializer_range": self.initializer_range,
            "layer_norm_eps": self.layer_norm_eps,
            "pad_token_id": self.pad_token_id,
            "attention_backend": self.attention_backend,
        }


# ---------------- RoPE 本地实现（不依赖 transformers 具体版本） ----------------


class RotaryEmbedding(nn.Module):
    """
    简单 RoPE：给定 head_dim 和 max_position_embeddings，生成 cos/sin。
    调用方式：cos, sin = rotary_emb(x, seq_len) 其中 x 形状为 [B, H, L, D]。
    """

    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))  # [D/2]
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B, H, L, D]
        if seq_len is None:
            seq_len = x.size(-2)
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)  # [L]
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)  # [L, D/2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [L, D]
        cos = emb.cos()[None, None, :, :]  # [1,1,L,D]
        sin = emb.sin()[None, None, :, :]
        return cos.to(x.dtype), sin.to(x.dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    d = x.size(-1)
    x1, x2 = x[..., : d // 2], x[..., d // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """对 q,k 应用 RoPE。cos/sin 会 broadcast 到 [B,H,L,D]。"""
    q_rot = (q * cos) + (_rotate_half(q) * sin)
    k_rot = (k * cos) + (_rotate_half(k) * sin)
    return q_rot, k_rot


class RNAAttentionWithRoPE(nn.Module):
    """多头自注意力 + RoPE，支持 manual/SDPA/packed-varlen 后端。"""

    def __init__(self, config: RNAModelConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.all_head_dim = self.head_dim * self.num_heads
        self.attention_backend = config.attention_backend.lower()
        valid_backends = {"auto", "manual", "cudnn_sdpa", "varlen_flash"}
        if self.attention_backend not in valid_backends:
            raise ValueError(
                f"Unsupported attention_backend={config.attention_backend!r}; "
                f"expected one of {sorted(valid_backends)}"
            )
        # Runtime-only fallback flag; intentionally absent from state_dict.
        self._auto_varlen_disabled = False

        self.qkv = nn.Linear(config.hidden_size, 3 * self.all_head_dim)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.rotary_emb = RotaryEmbedding(
            self.head_dim, max_position_embeddings=config.max_position_embeddings
        )

    def _shape(self, x: torch.Tensor, seq_len: int, bsz: int) -> torch.Tensor:
        return x.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def _manual_attention(self, q, k, v, attention_mask):
        """Reference implementation. Returns [B, L, H, D]."""
        scores = torch.matmul(q, k.transpose(-1, -2)) / (self.head_dim ** 0.5)
        if attention_mask is not None:
            # masked_fill semantics: True means masked out.
            pad_mask = (attention_mask == 0)[:, None, None, :]
            scores = scores.masked_fill(pad_mask, torch.finfo(scores.dtype).min)
        attn_probs = torch.softmax(scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        return torch.matmul(attn_probs, v).transpose(1, 2).contiguous()

    def _sdpa_attention(self, q, k, v, attention_mask, force_cudnn: bool):
        """PyTorch SDPA path. Returns [B, L, H, D]."""
        # SDPA bool mask: True means the key participates in attention.
        valid_key_mask = None
        if attention_mask is not None:
            valid_key_mask = attention_mask.bool()[:, None, None, :]
        dropout_p = self.dropout.p if self.training else 0.0

        if force_cudnn and q.is_cuda:
            from torch.nn.attention import SDPBackend, sdpa_kernel
            context_manager = sdpa_kernel(SDPBackend.CUDNN_ATTENTION)
        else:
            context_manager = nullcontext()

        with context_manager:
            context = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=valid_key_mask,
                dropout_p=dropout_p,
                is_causal=False,
            )
        return context.transpose(1, 2).contiguous()

    def _varlen_flash_attention(self, q, k, v, attention_mask):
        """Packed FlashAttention using cu_seqlens. Returns [B, L, H, D]."""
        if not q.is_cuda or q.dtype not in (torch.float16, torch.bfloat16):
            raise RuntimeError("varlen_flash requires CUDA FP16 or BF16 Q/K/V tensors")

        bsz, _, seq_len, _ = q.shape
        if attention_mask is None:
            valid = torch.ones((bsz, seq_len), dtype=torch.bool, device=q.device)
        else:
            if attention_mask.ndim != 2 or attention_mask.shape != (bsz, seq_len):
                raise ValueError(
                    "attention_mask must have shape [batch_size, sequence_length], "
                    f"got {tuple(attention_mask.shape)}"
                )
            valid = attention_mask.bool()

        lengths = valid.sum(dim=1, dtype=torch.int32)
        if torch.any(lengths <= 0):
            raise ValueError("varlen_flash does not support an all-padding sequence")

        cu_seqlens = torch.zeros(bsz + 1, dtype=torch.int32, device=q.device)
        cu_seqlens[1:] = torch.cumsum(lengths, dim=0)
        max_seqlen = int(lengths.max().item())

        # Pack only after Q/K/V projection and RoPE to retain per-RNA positions.
        q_blhd = q.transpose(1, 2).contiguous()
        k_blhd = k.transpose(1, 2).contiguous()
        v_blhd = v.transpose(1, 2).contiguous()
        flat_valid_idx = valid.reshape(-1).nonzero(as_tuple=False).squeeze(1)

        def pack(x: torch.Tensor) -> torch.Tensor:
            flat = x.reshape(bsz * seq_len, self.num_heads, self.head_dim)
            return flat.index_select(0, flat_valid_idx)

        # Keep strong references alive until the asynchronous private op returns.
        q_packed = pack(q_blhd)
        k_packed = pack(k_blhd)
        v_packed = pack(v_blhd)
        packed_out = torch.ops.aten._flash_attention_forward(
            q_packed, k_packed, v_packed,
            cu_seqlens, cu_seqlens,
            max_seqlen, max_seqlen,
            self.dropout.p if self.training else 0.0,
            False, False,
            scale=None,
            window_size_left=-1,
            window_size_right=-1,
        )[0]

        # Padded queries get zero context; valid-token outputs remain unchanged.
        flat_context = q_blhd.new_zeros(
            (bsz * seq_len, self.num_heads, self.head_dim)
        )
        flat_context = flat_context.index_copy(0, flat_valid_idx, packed_out)
        return flat_context.view(bsz, seq_len, self.num_heads, self.head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.size()
        qkv = self.qkv(hidden_states)
        q, k, v = qkv.split(self.all_head_dim, dim=-1)
        q = self._shape(q, seq_len, bsz)
        k = self._shape(k, seq_len, bsz)
        v = self._shape(v, seq_len, bsz)

        cos, sin = self.rotary_emb(v, seq_len=seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        backend = self.attention_backend
        if backend == "manual":
            context = self._manual_attention(q, k, v, attention_mask)
        elif backend == "cudnn_sdpa":
            context = self._sdpa_attention(q, k, v, attention_mask, force_cudnn=True)
        elif backend == "varlen_flash":
            context = self._varlen_flash_attention(q, k, v, attention_mask)
        else:
            can_use_varlen = (
                q.is_cuda
                and q.dtype in (torch.float16, torch.bfloat16)
                and not self._auto_varlen_disabled
            )
            if can_use_varlen:
                try:
                    context = self._varlen_flash_attention(q, k, v, attention_mask)
                except RuntimeError as exc:
                    if "out of memory" in str(exc).lower():
                        raise
                    self._auto_varlen_disabled = True
                    warnings.warn(
                        f"Packed varlen FlashAttention unavailable ({exc}); "
                        "falling back to PyTorch SDPA for this module.",
                        RuntimeWarning,
                    )
                    context = self._sdpa_attention(
                        q,
                        k,
                        v,
                        attention_mask,
                        force_cudnn=(
                            q.is_cuda
                            and q.dtype in (torch.float16, torch.bfloat16)
                        ),
                    )
            else:
                context = self._sdpa_attention(
                    q,
                    k,
                    v,
                    attention_mask,
                    force_cudnn=(
                        q.is_cuda
                        and q.dtype in (torch.float16, torch.bfloat16)
                    ),
                )

        context = context.reshape(bsz, seq_len, self.all_head_dim)
        return self.out_proj(context)


class RNATransformerLayer(nn.Module):
    def __init__(self, config: RNAModelConfig):
        super().__init__()
        self.self_attn = RNAAttentionWithRoPE(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.linear1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.linear2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.act = nn.GELU() if config.hidden_act == "gelu" else nn.ReLU()

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_out = self.self_attn(hidden_states, attention_mask=attention_mask)
        hidden_states = self.norm1(hidden_states + self.dropout(attn_out))
        ffn_out = self.linear2(self.act(self.linear1(hidden_states)))
        hidden_states = self.norm2(hidden_states + self.dropout(ffn_out))
        return hidden_states


class RNARopeEncoder(nn.Module):
    """仅 encoder：token embedding + RoPE-transformer layers。"""

    def __init__(self, config: RNAModelConfig):
        super().__init__()
        self.config = config
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.layers = nn.ModuleList([RNATransformerLayer(config) for _ in range(config.num_hidden_layers)])
        self.ln_f = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.dropout(self.embeddings(input_ids))
        for layer in self.layers:
            x = layer(x, attention_mask=attention_mask)
        return self.ln_f(x)


class RNATransformerForMaskedLM(nn.Module):
    """RoPE-Transformer encoder + MLM head."""

    def __init__(self, config: Optional[RNAModelConfig] = None):
        super().__init__()
        config = config or RNAModelConfig()
        self.config = config
        self.encoder = RNARopeEncoder(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Dict[str, torch.Tensor]:
        hidden = self.encoder(input_ids, attention_mask=attention_mask)
        logits = self.lm_head(hidden)
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss(ignore_index=-100)(
                logits.reshape(-1, self.config.vocab_size), labels.reshape(-1)
            )
        if return_dict is False:
            return (loss, logits)
        return {"loss": loss, "logits": logits}

    @classmethod
    def from_pretrained(cls, path: str, **kwargs) -> "RNATransformerForMaskedLM":
        """从保存的 config.json + pytorch_model.bin 加载模型。"""
        config_path = os.path.join(path, "config.json")
        if os.path.isfile(config_path):
            with open(config_path) as f:
                d = json.load(f)
            config = RNAModelConfig(
                vocab_size=d.get("vocab_size", 9),
                hidden_size=d.get("hidden_size", 768),
                num_hidden_layers=d.get("num_hidden_layers", 12),
                num_attention_heads=d.get("num_attention_heads", 12),
                intermediate_size=d.get("intermediate_size", 3072),
                hidden_act=d.get("hidden_act", "gelu"),
                hidden_dropout_prob=d.get("hidden_dropout_prob", 0.1),
                attention_probs_dropout_prob=d.get("attention_probs_dropout_prob", 0.1),
                max_position_embeddings=d.get("max_position_embeddings", 512),
                type_vocab_size=d.get("type_vocab_size", 2),
                initializer_range=d.get("initializer_range", 0.02),
                layer_norm_eps=d.get("layer_norm_eps", 1e-12),
                pad_token_id=d.get("pad_token_id", 0),
                attention_backend=d.get("attention_backend", "auto"),
            )
        else:
            config = RNAModelConfig()
        model = cls(config)
        model_file = os.path.join(path, "pytorch_model.bin")
        if os.path.isfile(model_file):
            state = torch.load(model_file, map_location="cpu")
            model.load_state_dict(state, strict=False)
        return model

    def save_pretrained(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)
        torch.save(self.state_dict(), os.path.join(path, "pytorch_model.bin"))
