"""
Hierarchical Transformer Language Model

Architecture per layer:
  x → RMSNorm → MultiHeadHierarchicalAttention → residual
    → RMSNorm → SwiGLU FFN                      → residual

Design choices:
  - No bias in linear layers (GPT-3 / LLaMA style)
  - SwiGLU instead of GELU (better empirically at same param count)
  - RMSNorm instead of LayerNorm (faster, no mean subtraction)
  - Weight tying: embedding and lm_head share weights
  - Residual projection scaled down by 1/sqrt(2*n_layers)
  - No absolute positional embeddings: ALiBi handled inside attention
"""

from __future__ import annotations
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from kernel.attention import MultiHeadHierarchicalAttention


# ──────────────────────────────────────────
# Config
# ──────────────────────────────────────────

@dataclass
class ModelConfig:
    vocab_size: int = 32000
    d_model: int    = 512
    n_heads: int    = 8
    n_layers: int   = 6
    d_ff: int       = 1536      # SwiGLU: 2 gates, so effective = d_ff * 2/3 * 2
    local_W: int    = 128       # local window (level 0)
    chunk_B: int    = 8         # branching factor per level
    n_levels: int   = 3         # compressed levels above local
    gamma_init: float = 0.1     # initial ALiBi slope at level 0
    dropout: float  = 0.0
    grad_checkpoint: bool = False  # recompute activations in backward to save memory

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


# ──────────────────────────────────────────
# Primitives
# ──────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class SwiGLU(nn.Module):
    """
    SwiGLU FFN: output = (W1·x ⊙ swish(W2·x)) · W3
    Two input projections, one output projection.
    d_ff is the inner dimension of each gate.
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)   # gate
        self.w2 = nn.Linear(d_model, d_ff, bias=False)   # value
        self.w3 = nn.Linear(d_ff, d_model, bias=False)   # output
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w3(F.silu(self.w1(x)) * self.w2(x)))


# ──────────────────────────────────────────
# Transformer Block
# ──────────────────────────────────────────

class HierarchicalBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.norm1 = RMSNorm(config.d_model)
        self.norm2 = RMSNorm(config.d_model)
        self.attn  = MultiHeadHierarchicalAttention(
            d_model    = config.d_model,
            n_heads    = config.n_heads,
            local_W    = config.local_W,
            chunk_B    = config.chunk_B,
            n_levels   = config.n_levels,
            gamma_init = config.gamma_init,
        )
        self.ff = SwiGLU(config.d_model, config.d_ff, config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


# ──────────────────────────────────────────
# Full Model
# ──────────────────────────────────────────

class HierarchicalLM(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks     = nn.ModuleList([
            HierarchicalBlock(config) for _ in range(config.n_layers)
        ])
        self.norm       = RMSNorm(config.d_model)
        self.lm_head    = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.embedding.weight

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        residual_std = 0.02 / math.sqrt(2 * self.config.n_layers)
        for block in self.blocks:
            # Scale down residual projections to keep residual stream stable
            for proj_name in ("attn.out_proj", "ff.w3"):
                parts = proj_name.split(".")
                module = block
                for p in parts:
                    module = getattr(module, p)
                nn.init.normal_(module.weight, mean=0.0, std=residual_std)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
          input_ids : (batch, seq)
        Returns:
          logits    : (batch, seq, vocab_size)
        """
        x = self.embedding(input_ids)   # (batch, seq, d_model)
        for block in self.blocks:
            if self.config.grad_checkpoint and self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        x = self.norm(x)
        return self.lm_head(x)          # (batch, seq, vocab_size)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(config: ModelConfig) -> HierarchicalLM:
    return HierarchicalLM(config)
