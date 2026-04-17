"""
LocalLatentAttentionTriton — drop-in nn.Module backed by fused Triton kernels.

Replaces LocalLatentAttention in models.py when use_triton_kernel=True.
Identical parameter names and shapes; same forward signature.

The module holds the same parameters as the PyTorch reference:
  q_proj, k_proj, v_proj       — local QKV projections
  gq_proj, gk_proj, gv_proj    — latent QKV projections
  out_proj                     — output projection
  gate_local, gate_remote      — per-head gate linears [head_dim → 1]
  r_out                        — latent-to-head projection

In the Triton kernel the gate linears are passed as weight matrices
[n_heads, head_dim], and r_out as [n_heads*head_dim, latent_heads*latent_head_dim].
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .kernels.llattn_op import LLAttnFunction
except ImportError:
    from kernels.llattn_op import LLAttnFunction


class LocalLatentAttentionTriton(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        latent_heads: int,
        latent_dim: int,       # total latent dimension = latent_heads * latent_head_dim
        n_latents: int,
        window: int,
        bias: bool = False,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        assert n_heads % latent_heads == 0, "n_heads must be divisible by latent_heads"
        assert latent_dim % latent_heads == 0, "latent_dim must be divisible by latent_heads"

        self.d_model       = d_model
        self.n_heads       = n_heads
        self.latent_heads  = latent_heads
        self.head_dim      = d_model // n_heads
        self.latent_dim    = latent_dim
        self.latent_head_dim = latent_dim // latent_heads
        self.n_latents     = n_latents
        self.window        = window

        # Local attention projections
        self.q_proj = nn.Linear(d_model, d_model,      bias=bias)
        self.k_proj = nn.Linear(d_model, d_model,      bias=bias)
        self.v_proj = nn.Linear(d_model, d_model,      bias=bias)

        # Latent attention projections
        self.gq_proj = nn.Linear(d_model, latent_dim,  bias=bias)
        self.gk_proj = nn.Linear(d_model, latent_dim,  bias=bias)
        self.gv_proj = nn.Linear(d_model, latent_dim,  bias=bias)

        # Output projection (local head space → d_model)
        self.out_proj = nn.Linear(d_model, d_model,    bias=bias)

        # Latent → head projection (r_out): maps latent output back to head space
        # Weight shape: [d_model, latent_dim]  (out_features, in_features)
        self.r_out = nn.Linear(latent_dim, d_model, bias=False)

        # Per-head gate: one weight vector per head [Nh, Dh]
        # Stored as Parameter directly (avoids the [1, Dh] shape of nn.Linear)
        self.gate_local_w  = nn.Parameter(torch.empty(n_heads, self.head_dim))
        self.gate_remote_w = nn.Parameter(torch.empty(n_heads, self.head_dim))
        nn.init.normal_(self.gate_local_w,  std=0.02)
        nn.init.normal_(self.gate_remote_w, std=0.02)
        self.profile_enabled = False

    def set_profile(self, enabled: bool) -> None:
        # Keep the same attention-module API as the PyTorch implementation so
        # model-level profiling toggles do not crash when Triton is selected.
        self.profile_enabled = enabled

    def get_profile_stats(self) -> dict[str, float]:
        # Fine-grained timing is not implemented for the fused Triton path yet.
        return {}

    # ── helpers ──────────────────────────────────────────────────────────────

    def _chunk_size(self, seq_len: int) -> int:
        return max(1, seq_len // self.n_latents)

    def _pool_to_latents(self, x: torch.Tensor) -> torch.Tensor:
        """Mean-pool x [B, S, latent_dim] into [B, n_latents, latent_dim]."""
        B, S, C = x.shape
        chunk = self._chunk_size(S)
        full  = (S // chunk) * chunk
        parts = []
        if full > 0:
            parts.append(x[:, :full, :].reshape(B, full // chunk, chunk, C).mean(dim=2))
        if full < S:
            parts.append(x[:, full:, :].mean(dim=1, keepdim=True))
        pooled = torch.cat(parts, dim=1)
        # Pad or trim to exactly n_latents
        nl = self.n_latents
        if pooled.shape[1] < nl:
            pad = pooled[:, -1:, :].expand(B, nl - pooled.shape[1], C)
            pooled = torch.cat([pooled, pad], dim=1)
        else:
            pooled = pooled[:, :nl, :]
        return pooled  # [B, nl, latent_dim]

    # ── forward ──────────────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,                          # [B, S, d_model]
        _positions: Optional[torch.Tensor] = None,  # unused (kept for API compat)
    ) -> torch.Tensor:
        B, S, _ = x.shape
        Nh  = self.n_heads
        Lh  = self.latent_heads
        Dh  = self.head_dim
        Ld  = self.latent_head_dim
        Nl  = self.n_latents
        chunk_size = self._chunk_size(S)

        # ── Projections ───────────────────────────────────────────────────────
        Q = self.q_proj(x).view(B, S, Nh, Dh).permute(0, 2, 1, 3).contiguous()  # [B,Nh,S,Dh]
        K = self.k_proj(x).view(B, S, Nh, Dh).permute(0, 2, 1, 3).contiguous()
        V = self.v_proj(x).view(B, S, Nh, Dh).permute(0, 2, 1, 3).contiguous()

        gq_flat = self.gq_proj(x)                                  # [B, S, latent_dim]
        gk_flat = self.gk_proj(x)                                  # [B, S, latent_dim]
        gv_flat = self.gv_proj(x)

        GQ = gq_flat.view(B, S, Lh, Ld).permute(0, 2, 1, 3).contiguous()  # [B,Lh,S,Ld]

        # Pool GK, GV to latents
        gk_pooled = self._pool_to_latents(gk_flat)                 # [B, Nl, latent_dim]
        gv_pooled = self._pool_to_latents(gv_flat)

        GK = gk_pooled.view(B, Nl, Lh, Ld).permute(0, 2, 1, 3).contiguous()  # [B,Lh,Nl,Ld]
        GV = gv_pooled.view(B, Nl, Lh, Ld).permute(0, 2, 1, 3).contiguous()

        # ── Gate weight matrices — [Nh, Dh] ──────────────────────────────────
        gate_local_w  = self.gate_local_w.contiguous()              # [Nh, Dh]
        gate_remote_w = self.gate_remote_w.contiguous()             # [Nh, Dh]

        # r_out.weight shape: [d_model, latent_dim] = [Nh*Dh, Lh*Ld]
        r_out_w = self.r_out.weight.contiguous()                    # [Nh*Dh, Lh*Ld]

        # ── Ensure bf16 for kernel ────────────────────────────────────────────
        def _bf16(t):
            return t.to(torch.bfloat16) if t.dtype != torch.bfloat16 else t

        Q, K, V       = _bf16(Q), _bf16(K), _bf16(V)
        GQ, GK, GV    = _bf16(GQ), _bf16(GK), _bf16(GV)
        gate_local_w  = _bf16(gate_local_w)
        gate_remote_w = _bf16(gate_remote_w)
        r_out_w       = _bf16(r_out_w)

        # ── Fused kernel ──────────────────────────────────────────────────────
        out = LLAttnFunction.apply(
            Q, K, V, GQ, GK, GV,
            gate_local_w, gate_remote_w, r_out_w,
            self.window, chunk_size,
        )                                                           # [B, Nh, S, Dh]

        # ── Merge heads and output projection ─────────────────────────────────
        out = out.permute(0, 2, 1, 3).reshape(B, S, self.d_model)
        return self.out_proj(out)                                   # [B, S, d_model]
