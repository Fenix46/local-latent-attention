"""
Hierarchical Causal Attention Kernel

Architecture:
  Level 0: sliding-window exact attention over the last W tokens
           — implemented via F.scaled_dot_product_attention with block tiling
           — memory: O(N·W), NOT O(N²)
  Level l: compressed attention over chunks of B^l tokens fully in the past
           — mean-pooled K,V per chunk
           — memory: O(N/B^l) per level, O(N) total

Positional encoding: ALiBi-style bias per level with learned slopes gamma_l.
  Level 0: score(i,j) = Q_i·K_j/√d  - γ_0·(i-j)
  Level l: score(i,c) = Q_i·K^l_c/√d - γ_l·|i - centroid(c)|

Causality: enforced by construction at every level.

Memory: O(N·W) for level 0, O(N) total — safe for seq=8192 on H200.
Compute: O(N·W·d) local + O(N·log_B(N/W)·d) compressed.
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn

from kernel.triton_local import _sw_pytorch_accum
from kernel.triton_hier import _hier_pytorch


# ──────────────────────────────────────────────────────────────────
# Merge utilities — inline for speed
# ──────────────────────────────────────────────────────────────────

def _merge(
    m_a: torch.Tensor, l_a: torch.Tensor, o_a: torch.Tensor,
    m_b: torch.Tensor, l_b: torch.Tensor, o_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Merge two attention accumulators. All shapes: (B, H, N) for m/l, (B,H,N,d) for o."""
    m  = torch.maximum(m_a, m_b)
    ea = torch.exp(m_a - m)
    eb = torch.exp(m_b - m)
    # o may be bfloat16 while ea/eb are float32 — cast to o dtype to preserve input dtype
    ea_o = ea.to(o_a.dtype)
    eb_o = eb.to(o_b.dtype)
    return m, l_a * ea + l_b * eb, o_a * ea_o.unsqueeze(-1) + o_b * eb_o.unsqueeze(-1)


# ──────────────────────────────────────────────────────────────────
# Helper: build K/V hierarchy (mean-pool per level)
# ──────────────────────────────────────────────────────────────────

def _build_hierarchy(
    k: torch.Tensor,
    v: torch.Tensor,
    chunk_B: int,
    n_levels: int,
) -> list:
    """Mean-pool K,V into compressed levels. Returns list indexed by level.
    Level 0 is None (uses raw k, v). Levels 1..n_levels are (k_l, v_l, n_chunks).
    """
    hierarchy: list = [None]
    k_cur, v_cur = k, v
    for _ in range(1, n_levels + 1):
        _, _, N_cur, d = k_cur.shape
        n_chunks = N_cur // chunk_B
        if n_chunks == 0:
            hierarchy.append(None)
            continue
        usable  = n_chunks * chunk_B
        k_level = k_cur[:, :, :usable, :].view(*k_cur.shape[:2], n_chunks, chunk_B, d).mean(dim=3)
        v_level = v_cur[:, :, :usable, :].view(*v_cur.shape[:2], n_chunks, chunk_B, d).mean(dim=3)
        hierarchy.append((k_level, v_level, n_chunks))
        k_cur, v_cur = k_level, v_level
    return hierarchy


# ──────────────────────────────────────────────────────────────────
# Differentiable forward — autograd handles the backward exactly
# ──────────────────────────────────────────────────────────────────

def _hierarchical_forward_pytorch(q, k, v, gammas, local_W, chunk_B, n_levels, scale):
    """
    Pure-PyTorch differentiable forward. Used for backward via autograd.
    The Triton kernels accelerate the forward pass but their custom backwards
    are removed — autograd differentiates through this instead.
    """
    gammas_c = gammas.to(q.dtype)

    # Level 0: sliding window
    m, l, o = _sw_pytorch_accum(q, k, v, local_W, gammas_c[0], scale)
    m = m.to(q.dtype); l = l.to(q.dtype)

    # Levels 1..n_levels
    hierarchy = _build_hierarchy(k, v, chunk_B, n_levels)
    for lvl in range(1, n_levels + 1):
        entry = hierarchy[lvl]
        if entry is None:
            continue
        k_l, v_l, _ = entry
        chunk_size = chunk_B ** lvl
        m_l, l_l, o_l = _hier_pytorch(q, k_l, v_l, chunk_size, local_W, gammas_c[lvl], scale)
        m_l = m_l.to(q.dtype); l_l = l_l.to(q.dtype)
        m, l, o = _merge(m, l, o, m_l, l_l, o_l)

    l_safe = l.clamp(min=1e-8)
    return o / l_safe.unsqueeze(-1)


# ──────────────────────────────────────────────────────────────────
# Main module
# ──────────────────────────────────────────────────────────────────

class HierarchicalAttention(nn.Module):
    """
    Hierarchical causal attention — memory-efficient for long sequences.

    Level 0 : sliding window, exact, O(N·W) memory
    Level l  : compressed chunks, O(N·C_l) memory where C_l = N/B^l

    Args:
      d          : head dimension
      local_W    : sliding window size (number of recent tokens)
      chunk_B    : branching factor per compressed level
      n_levels   : number of compressed levels (0 = local-only)
      gamma_init : initial ALiBi slope at level 0
    """

    def __init__(
        self,
        d: int,
        local_W: int   = 128,
        chunk_B: int   = 8,
        n_levels: int  = 3,
        gamma_init: float = 0.1,
    ) -> None:
        super().__init__()
        self.d        = d
        self.local_W  = local_W
        self.chunk_B  = chunk_B
        self.n_levels = n_levels
        self.scale    = 1.0 / math.sqrt(d)

        gammas = torch.tensor(
            [gamma_init / (chunk_B ** l) for l in range(n_levels + 1)],
            dtype=torch.float32,
        )
        self.gammas = nn.Parameter(gammas)   # (n_levels+1,)

    def forward(
        self,
        q: torch.Tensor,   # (B, H, N, d)
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:     # (B, H, N, d)
        return _hierarchical_forward_pytorch(
            q, k, v,
            self.gammas,
            self.local_W,
            self.chunk_B,
            self.n_levels,
            self.scale,
        )


# ──────────────────────────────────────────────────────────────────
# Multi-head wrapper
# ──────────────────────────────────────────────────────────────────

class MultiHeadHierarchicalAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        local_W: int      = 128,
        chunk_B: int      = 8,
        n_levels: int     = 3,
        gamma_init: float = 0.1,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads  = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj   = nn.Linear(d_model, d_model, bias=False)
        self.k_proj   = nn.Linear(d_model, d_model, bias=False)
        self.v_proj   = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.attn = HierarchicalAttention(
            d          = self.head_dim,
            local_W    = local_W,
            chunk_B    = chunk_B,
            n_levels   = n_levels,
            gamma_init = gamma_init,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        H, hd   = self.n_heads, self.head_dim

        q = self.q_proj(x).view(B, N, H, hd).transpose(1, 2)   # (B, H, N, hd)
        k = self.k_proj(x).view(B, N, H, hd).transpose(1, 2)
        v = self.v_proj(x).view(B, N, H, hd).transpose(1, 2)

        out = self.attn(q, k, v)                                  # (B, H, N, hd)
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.out_proj(out)
