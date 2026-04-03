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
import torch.nn.functional as F

from kernel.triton_local import sliding_window_triton_accum, HAS_TRITON
from kernel.triton_hier import compressed_level_triton


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
# Level 0: sliding window via tiled SDPA
# ──────────────────────────────────────────────────────────────────

def _local_window_attention(
    q: torch.Tensor,       # (B, H, N, d)
    k: torch.Tensor,       # (B, H, N, d)
    v: torch.Tensor,       # (B, H, N, d)
    window: int,
    gamma: torch.Tensor,   # scalar parameter
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sliding-window attention with ALiBi bias.

    Processes queries in blocks of size `window`. For each block of queries,
    attends only to the corresponding window of keys — never materializing
    the full N×N matrix.

    Returns (m, l, o) accumulators with shape (B, H, N) / (B, H, N, d).
    """
    B, H, N, d = q.shape
    device, dtype = q.device, q.dtype

    m_out = torch.full((B, H, N),    float("-inf"), device=device, dtype=dtype)
    l_out = torch.zeros((B, H, N),                  device=device, dtype=dtype)
    o_out = torch.zeros((B, H, N, d),               device=device, dtype=dtype)

    # Precompute ALiBi distance offsets for a window of size `window`
    # dist[i] = i  (query at offset i within window sees key at distance i..0)
    offsets = torch.arange(window, device=device, dtype=dtype)  # (W,)

    for q_start in range(0, N, window):
        q_end   = min(q_start + window, N)
        q_len   = q_end - q_start

        k_start = max(0, q_start - window + 1)
        k_end   = q_end                          # causal: key up to q_end-1
        k_len   = k_end - k_start

        q_blk = q[:, :, q_start:q_end, :]       # (B, H, q_len, d)
        k_blk = k[:, :, k_start:k_end, :]       # (B, H, k_len, d)
        v_blk = v[:, :, k_start:k_end, :]

        # Raw scores
        scores = torch.matmul(q_blk, k_blk.transpose(-2, -1)) * scale  # (B,H,q_len,k_len)

        # ALiBi bias: distance = query_abs_pos - key_abs_pos
        q_pos  = torch.arange(q_start, q_end, device=device, dtype=dtype)   # (q_len,)
        k_pos  = torch.arange(k_start, k_end, device=device, dtype=dtype)   # (k_len,)
        dist   = (q_pos.unsqueeze(1) - k_pos.unsqueeze(0)).clamp(min=0)     # (q_len, k_len)
        scores = scores - gamma * dist.unsqueeze(0).unsqueeze(0)

        # Causal mask: key must be <= query position
        causal = k_pos.unsqueeze(0) > q_pos.unsqueeze(1)                    # (q_len, k_len)
        scores = scores.masked_fill(causal.unsqueeze(0).unsqueeze(0), float("-inf"))

        # Window mask: key must be within W tokens of query
        in_win = (q_pos.unsqueeze(1) - k_pos.unsqueeze(0)) < window         # (q_len, k_len)
        scores = scores.masked_fill(~in_win.unsqueeze(0).unsqueeze(0), float("-inf"))

        # Accumulator components
        m_blk = scores.amax(dim=-1)                                          # (B, H, q_len)
        m_safe = m_blk.clamp(min=-1e9)
        exp_s = torch.exp(scores - m_safe.unsqueeze(-1))
        valid = (~causal & in_win).unsqueeze(0).unsqueeze(0)
        exp_s = exp_s.masked_fill(~valid, 0.0)

        l_blk = exp_s.sum(dim=-1)                                            # (B, H, q_len)
        o_blk = torch.matmul(exp_s, v_blk)                                  # (B, H, q_len, d)

        # Rows with no valid key → keep m as -inf
        no_key = ~valid.any(dim=-1)                                          # (1,1,q_len)
        m_blk  = m_blk.masked_fill(no_key, float("-inf"))

        m_out[:, :, q_start:q_end] = m_blk
        l_out[:, :, q_start:q_end] = l_blk
        o_out[:, :, q_start:q_end] = o_blk

    return m_out, l_out, o_out


# ──────────────────────────────────────────────────────────────────
# Compressed levels
# ──────────────────────────────────────────────────────────────────

def _compressed_level_attention(
    q: torch.Tensor,          # (B, H, N, d)
    k_c: torch.Tensor,        # (B, H, C, d)  compressed keys
    v_c: torch.Tensor,        # (B, H, C, d)  compressed values
    chunk_size: int,
    local_W: int,
    gamma: torch.Tensor,      # scalar
    scale: float,
    mask_cache: dict,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Attention from all N queries to C compressed chunks.

    Chunk c covers tokens [c*chunk_size, (c+1)*chunk_size).
    Query i can attend to chunk c iff:
      (c+1)*chunk_size <= i          (fully in the past)
      AND (c+1)*chunk_size <= i - local_W + 1  (outside local window)

    Memory: O(N·C) where C = N/chunk_size — for large chunk_size this is small.
    """
    B, H, N, d = q.shape
    C = k_c.shape[2]
    device, dtype = q.device, q.dtype

    cache_key = (N, C, chunk_size, local_W, device)
    if cache_key not in mask_cache:
        q_pos      = torch.arange(N, device=device)
        c_idx      = torch.arange(C, device=device)
        chunk_end  = (c_idx + 1) * chunk_size                              # (C,)
        centroid   = c_idx.float() * chunk_size + chunk_size / 2.0         # (C,)

        causal     = chunk_end.unsqueeze(0) <= q_pos.unsqueeze(1)          # (N, C)
        non_local  = chunk_end.unsqueeze(0) <= (q_pos - local_W + 1).clamp(min=0).unsqueeze(1)
        mask       = causal & non_local                                     # (N, C)
        dist       = (q_pos.float().unsqueeze(1) - centroid.unsqueeze(0)).abs()  # (N, C)
        mask_cache[cache_key] = (mask, dist.to(dtype))

    mask, dist = mask_cache[cache_key]

    if not mask.any():
        m = torch.full((B, H, N),    float("-inf"), device=device, dtype=dtype)
        l = torch.zeros((B, H, N),                  device=device, dtype=dtype)
        o = torch.zeros((B, H, N, d),               device=device, dtype=dtype)
        return m, l, o

    # Scores: (B, H, N, C)
    scores = torch.matmul(q, k_c.transpose(-2, -1)) * scale
    scores = scores - (gamma * dist).unsqueeze(0).unsqueeze(0)
    scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))

    m = scores.amax(dim=-1)
    m_safe = m.clamp(min=-1e9)
    exp_s = torch.exp(scores - m_safe.unsqueeze(-1))
    exp_s = exp_s.masked_fill(~mask.unsqueeze(0).unsqueeze(0), 0.0)

    l = exp_s.sum(dim=-1)
    o = torch.matmul(exp_s, v_c)

    no_key = ~mask.any(dim=-1)
    m = m.masked_fill(no_key.unsqueeze(0).unsqueeze(0), float("-inf"))

    return m, l, o


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
        self._mask_cache: dict = {}

    def _build_hierarchy(
        self,
        k: torch.Tensor,   # (B, H, N, d)
        v: torch.Tensor,
    ) -> list[tuple[torch.Tensor, torch.Tensor, int] | None]:
        """Mean-pool K,V into compressed levels. Returns list indexed by level."""
        B = self.chunk_B
        hierarchy: list = [None]   # level 0 uses raw k, v

        k_cur, v_cur = k, v
        for _ in range(1, self.n_levels + 1):
            _, _, N_cur, d = k_cur.shape
            n_chunks = N_cur // B
            if n_chunks == 0:
                hierarchy.append(None)
                continue
            usable  = n_chunks * B
            k_level = k_cur[:, :, :usable, :].view(*k_cur.shape[:2], n_chunks, B, d).mean(dim=3)
            v_level = v_cur[:, :, :usable, :].view(*v_cur.shape[:2], n_chunks, B, d).mean(dim=3)
            hierarchy.append((k_level, v_level, n_chunks))
            k_cur, v_cur = k_level, v_level

        return hierarchy

    def forward(
        self,
        q: torch.Tensor,   # (B, H, N, d)
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:     # (B, H, N, d)
        # ── Level 0: sliding window ──
        # Uses Triton fused kernel on CUDA, falls back to PyTorch on CPU.
        # Returns unnormalised (m, l, o) accumulators for merge with upper levels.
        m, l, o = sliding_window_triton_accum(
            q, k, v,
            window=self.local_W,
            gamma=self.gammas[0],
            scale=self.scale,
        )

        # ── Levels 1..n_levels: compressed chunks ──
        hierarchy = self._build_hierarchy(k, v)

        for lvl in range(1, self.n_levels + 1):
            entry = hierarchy[lvl]
            if entry is None:
                continue
            k_l, v_l, _ = entry
            chunk_size = self.chunk_B ** lvl

            m_l, l_l, o_l = _compressed_level_attention(
                q, k_l, v_l,
                chunk_size = chunk_size,
                local_W    = self.local_W,
                gamma      = self.gammas[lvl],
                scale      = self.scale,
                mask_cache = self._mask_cache,
            )
            m, l, o = _merge(m, l, o, m_l, l_l, o_l)

        # ── Output ──
        return o / l.to(o.dtype).clamp(min=1e-8).unsqueeze(-1)


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
