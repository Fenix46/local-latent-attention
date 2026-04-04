"""
Triton kernel for Telescopic Attention — Compressed levels (1..L).

For each compressed level l, chunk c covers tokens [c*S, (c+1)*S) where S = B^l.
Query i can attend to chunk c iff:
  (c+1)*S <= i                  (chunk fully in the past — causal)
  AND (c+1)*S <= i - W + 1      (chunk outside the local window)

Combined: chunk_end <= i - W + 1  (since i - W + 1 <= i always)

ALiBi bias: score -= gamma * |i - centroid(c)|
  where centroid(c) = c * S + S/2

Returns (m, l, o) accumulators — unnormalised — same interface as triton_local.py.

Design:
  Grid: (ceil(N / BLOCK_Q), B*H)
  Each program handles one (bh, q_block) pair.
  Inner loop: iterate over ALL C chunks, skip invalid ones via masking.
  C is small (N/S), so the inner loop is short even without tiling.

Memory: O(N * C) reads of K_c, V_c — already small.
Compute: O(N * C * d) — dominated by the matmul.
"""

from __future__ import annotations
import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


# ─────────────────────────────────────────────────────────────────────────────
# Forward kernel
# ─────────────────────────────────────────────────────────────────────────────

if HAS_TRITON:

    @triton.jit
    def _hier_fwd_kernel(
        # Q: (BH, N, d),  K_c/V_c: (BH, C, d)
        Q_ptr, Kc_ptr, Vc_ptr,
        M_ptr, L_ptr, O_ptr,      # outputs: (BH, N), (BH, N), (BH, N, d)
        # Strides for Q / O  — layout (BH, N, d)
        stride_qn, stride_qd,
        # Strides for K_c / V_c — layout (BH, C, d)
        stride_kc, stride_kd,
        # Strides for M / L — layout (BH, N)
        stride_ln,
        # Scalars
        N: tl.constexpr,
        C_real,                    # actual number of chunks (runtime bound for loop)
        d: tl.constexpr,
        S,                         # chunk_size = B^level  (runtime, not constexpr)
        W,                         # local window size     (runtime)
        scale,                     # 1/sqrt(d)
        gamma,                     # ALiBi slope for this level
        BLOCK_Q: tl.constexpr,
        BLOCK_C: tl.constexpr,    # tile size for tl.dot (>= 16, power of 2)
    ):
        """
        Grid: (ceil(N/BLOCK_Q), BH)
        """
        q_blk = tl.program_id(0)
        bh    = tl.program_id(1)

        q_start = q_blk * BLOCK_Q
        q_offs  = q_start + tl.arange(0, BLOCK_Q)   # (BLOCK_Q,)
        q_valid = q_offs < N
        d_offs  = tl.arange(0, d)

        base_q  = bh * N
        base_kv = bh * C_real

        # Load Q block: (BLOCK_Q, d)
        q = tl.load(
            Q_ptr + base_q + q_offs[:, None] * stride_qn + d_offs[None, :] * stride_qd,
            mask=q_valid[:, None],
            other=0.0,
        )

        # Online softmax state
        m     = tl.full((BLOCK_Q,),    -1e38,  dtype=tl.float32)
        acc_l = tl.zeros((BLOCK_Q,),           dtype=tl.float32)
        acc_o = tl.zeros((BLOCK_Q, d),         dtype=tl.float32)

        # Threshold: query i can see chunk c iff chunk_end <= i - W + 1
        # i.e. (c+1)*S <= q_offs - W + 1  →  c+1 <= (q_offs - W + 1) / S
        # Equivalently: chunk_end = (c+1)*S <= q_offs - W + 1
        threshold = q_offs.to(tl.float32) - tl.cast(W, tl.float32) + 1.0  # (BLOCK_Q,)

        # Iterate over chunk tiles
        c_start = 0
        while c_start < C_real:
            c_offs  = c_start + tl.arange(0, BLOCK_C)   # (BLOCK_C,)
            c_valid = c_offs < C_real

            # chunk_end = (c+1) * S,  centroid = c*S + S/2
            chunk_end = (c_offs.to(tl.float32) + 1.0) * tl.cast(S, tl.float32)   # (BLOCK_C,)
            centroid  = c_offs.to(tl.float32) * tl.cast(S, tl.float32) + tl.cast(S, tl.float32) * 0.5

            # Load K_c tile: (BLOCK_C, d)
            kc = tl.load(
                Kc_ptr + base_kv + c_offs[:, None] * stride_kc + d_offs[None, :] * stride_kd,
                mask=c_valid[:, None],
                other=0.0,
            )

            # Scores: (BLOCK_Q, BLOCK_C)
            s = tl.dot(q.to(tl.float32), tl.trans(kc).to(tl.float32))
            s = s * tl.cast(scale, tl.float32)

            # ALiBi: |q_pos - centroid|
            dist = tl.abs(
                q_offs[:, None].to(tl.float32) - centroid[None, :].to(tl.float32)
            )
            s = s - tl.cast(gamma, tl.float32) * dist

            # Validity mask: chunk_end <= threshold  AND  c_valid
            valid = (chunk_end[None, :] <= threshold[:, None]) & c_valid[None, :]
            neg_inf = tl.full(s.shape, -1e38, dtype=tl.float32)
            s = tl.where(valid, s, neg_inf)

            # Online softmax update
            m_new  = tl.maximum(m, tl.max(s, axis=1).to(tl.float32))
            alpha  = tl.exp(m - m_new)
            beta   = tl.exp(s - m_new[:, None])
            beta   = tl.where(valid, beta, tl.zeros_like(beta))

            # Load V_c tile: (BLOCK_C, d)
            vc = tl.load(
                Vc_ptr + base_kv + c_offs[:, None] * stride_kc + d_offs[None, :] * stride_kd,
                mask=c_valid[:, None],
                other=0.0,
            )

            acc_l = acc_l * alpha + tl.sum(beta, axis=1)
            acc_o = acc_o * alpha[:, None] + tl.dot(beta.to(vc.dtype), vc).to(tl.float32)
            m     = m_new

            c_start += BLOCK_C

        # Rows with no valid chunk → m stays -1e38, acc_l stays 0

        # Store
        tl.store(
            M_ptr + bh * N + q_offs,
            m,
            mask=q_valid,
        )
        tl.store(
            L_ptr + bh * N + q_offs,
            acc_l,
            mask=q_valid,
        )
        tl.store(
            O_ptr + base_q + q_offs[:, None] * stride_qn + d_offs[None, :] * stride_qd,
            acc_o.to(Q_ptr.dtype.element_ty),
            mask=q_valid[:, None],
        )


# ─────────────────────────────────────────────────────────────────────────────
# Backward kernel
# ─────────────────────────────────────────────────────────────────────────────

if HAS_TRITON:

    @triton.jit
    def _hier_bwd_kernel(
        Q_ptr, Kc_ptr, Vc_ptr,
        dO_ptr,
        M_ptr, L_ptr,
        dQ_ptr, dKc_ptr, dVc_ptr,
        stride_qn, stride_qd,
        stride_kc, stride_kd,
        stride_ln,
        N: tl.constexpr,
        C_real,
        d: tl.constexpr,
        S, W, scale, gamma,
        BLOCK_Q: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        q_blk = tl.program_id(0)
        bh    = tl.program_id(1)

        q_start = q_blk * BLOCK_Q
        q_offs  = q_start + tl.arange(0, BLOCK_Q)
        q_valid = q_offs < N
        d_offs  = tl.arange(0, d)

        base_q  = bh * N
        base_kv = bh * C_real

        q  = tl.load(Q_ptr  + base_q + q_offs[:, None] * stride_qn + d_offs[None, :] * stride_qd,
                     mask=q_valid[:, None], other=0.0)
        do = tl.load(dO_ptr + base_q + q_offs[:, None] * stride_qn + d_offs[None, :] * stride_qd,
                     mask=q_valid[:, None], other=0.0)
        m  = tl.load(M_ptr  + bh * N + q_offs, mask=q_valid, other=-1e38)
        l  = tl.load(L_ptr  + bh * N + q_offs, mask=q_valid, other=1.0)
        l  = tl.maximum(l, 1e-8)

        dq = tl.zeros((BLOCK_Q, d), dtype=tl.float32)

        threshold = q_offs.to(tl.float32) - tl.cast(W, tl.float32) + 1.0

        c_start = 0
        while c_start < C_real:
            c_offs  = c_start + tl.arange(0, BLOCK_C)
            c_valid = c_offs < C_real

            chunk_end = (c_offs.to(tl.float32) + 1.0) * tl.cast(S, tl.float32)
            centroid  = c_offs.to(tl.float32) * tl.cast(S, tl.float32) + tl.cast(S, tl.float32) * 0.5

            kc = tl.load(Kc_ptr + base_kv + c_offs[:, None] * stride_kc + d_offs[None, :] * stride_kd,
                         mask=c_valid[:, None], other=0.0)
            vc = tl.load(Vc_ptr + base_kv + c_offs[:, None] * stride_kc + d_offs[None, :] * stride_kd,
                         mask=c_valid[:, None], other=0.0)

            # Recompute s and p
            s = tl.dot(q.to(tl.float32), tl.trans(kc).to(tl.float32)) * tl.cast(scale, tl.float32)
            dist = tl.abs(q_offs[:, None].to(tl.float32) - centroid[None, :].to(tl.float32))
            s = s - tl.cast(gamma, tl.float32) * dist

            valid = (chunk_end[None, :] <= threshold[:, None]) & c_valid[None, :]
            s = tl.where(valid, s, tl.full(s.shape, -1e38, dtype=tl.float32))

            p = tl.exp(s - m[:, None]) / l[:, None]
            p = tl.where(valid, p, tl.zeros_like(p))

            # dV_c += P^T · dO
            dvc = tl.dot(tl.trans(p.to(do.dtype)), do)
            tl.atomic_add(
                dVc_ptr + base_kv + c_offs[:, None] * stride_kc + d_offs[None, :] * stride_kd,
                dvc.to(dVc_ptr.dtype.element_ty),
                mask=c_valid[:, None],
            )

            # dP = dO · V^T;  D = rowsum(P * dP)
            dp = tl.dot(do.to(tl.float32), tl.trans(vc).to(tl.float32))
            D  = tl.sum(p * dp, axis=1)

            # dS = P * (dP - D)
            ds = p * (dp - D[:, None])
            ds = tl.where(valid, ds, tl.zeros_like(ds))

            # dQ += scale * dS · K_c
            dq = dq + tl.dot(ds.to(kc.dtype), kc).to(tl.float32) * tl.cast(scale, tl.float32)

            # dK_c += scale * dS^T · Q
            dkc = tl.dot(tl.trans(ds.to(q.dtype)), q).to(tl.float32) * tl.cast(scale, tl.float32)
            tl.atomic_add(
                dKc_ptr + base_kv + c_offs[:, None] * stride_kc + d_offs[None, :] * stride_kd,
                dkc.to(dKc_ptr.dtype.element_ty),
                mask=c_valid[:, None],
            )

            c_start += BLOCK_C

        tl.store(
            dQ_ptr + base_q + q_offs[:, None] * stride_qn + d_offs[None, :] * stride_qd,
            dq.to(dQ_ptr.dtype.element_ty),
            mask=q_valid[:, None],
        )


# ─────────────────────────────────────────────────────────────────────────────
# autograd.Function
# ─────────────────────────────────────────────────────────────────────────────

if HAS_TRITON:

    class _HierAccumFn(torch.autograd.Function):

        @staticmethod
        def forward(ctx, q, k_c, v_c, chunk_size: int, local_W: int,
                    gamma: torch.Tensor, scale: float):
            """
            q   : (B, H, N, d)
            k_c : (B, H, C, d)
            v_c : (B, H, C, d)
            Returns (m, l, o) unnormalised accumulators.
            """
            B, H, N, d = q.shape
            C = k_c.shape[2]
            assert q.is_cuda
            assert d in (16, 32, 64, 128)

            BH = B * H
            q2  = q.contiguous().view(BH, N, d)
            kc2 = k_c.contiguous().view(BH, C, d)
            vc2 = v_c.contiguous().view(BH, C, d)

            m  = torch.full((BH, N),    -1e38, device=q.device, dtype=torch.float32)
            l  = torch.zeros((BH, N),          device=q.device, dtype=torch.float32)
            o2 = torch.zeros((BH, N, d),       device=q.device, dtype=q.dtype)

            # BLOCK_C must be power of 2 and >= C, capped at 128
            BLOCK_C = min(128, triton.next_power_of_2(max(C, 16)))
            BLOCK_Q = 64
            grid = (triton.cdiv(N, BLOCK_Q), BH)

            _hier_fwd_kernel[grid](
                q2, kc2, vc2, m, l, o2,
                stride_qn=q2.stride(1),  stride_qd=q2.stride(2),
                stride_kc=kc2.stride(1), stride_kd=kc2.stride(2),
                stride_ln=1,
                N=N, C_real=C, d=d,
                S=chunk_size, W=local_W,
                scale=scale, gamma=gamma.item(),
                BLOCK_Q=BLOCK_Q, BLOCK_C=BLOCK_C,
            )

            m = m.view(B, H, N)
            l = l.view(B, H, N)
            o = o2.view(B, H, N, d)

            ctx.save_for_backward(q, k_c, v_c, m, l, gamma)
            ctx.chunk_size = chunk_size
            ctx.local_W    = local_W
            ctx.scale      = scale
            ctx.BLOCK_C    = BLOCK_C
            return m, l, o

        @staticmethod
        def backward(ctx, dm, dl, do_unnorm):
            q, k_c, v_c, m, l, gamma = ctx.saved_tensors
            chunk_size = ctx.chunk_size
            local_W    = ctx.local_W
            scale      = ctx.scale
            BLOCK_C    = ctx.BLOCK_C

            B, H, N, d = q.shape
            C = k_c.shape[2]
            BH = B * H

            l_safe   = l.clamp(min=1e-8)
            do_norm  = (do_unnorm / l_safe.unsqueeze(-1)).contiguous()

            q2   = q.contiguous().view(BH, N, d)
            kc2  = k_c.contiguous().view(BH, C, d)
            vc2  = v_c.contiguous().view(BH, C, d)
            do2  = do_norm.view(BH, N, d)
            m2   = m.view(BH, N).contiguous()
            l2   = l.view(BH, N).contiguous()

            dq  = torch.zeros_like(q2)
            dkc = torch.zeros_like(kc2)
            dvc = torch.zeros_like(vc2)

            BLOCK_Q = 64
            grid = (triton.cdiv(N, BLOCK_Q), BH)

            _hier_bwd_kernel[grid](
                q2, kc2, vc2, do2, m2, l2,
                dq, dkc, dvc,
                stride_qn=q2.stride(1),   stride_qd=q2.stride(2),
                stride_kc=kc2.stride(1),  stride_kd=kc2.stride(2),
                stride_ln=1,
                N=N, C_real=C, d=d,
                S=chunk_size, W=local_W,
                scale=scale, gamma=gamma.item(),
                BLOCK_Q=BLOCK_Q, BLOCK_C=BLOCK_C,
            )

            return (dq.view(B, H, N, d), dkc.view(B, H, C, d), dvc.view(B, H, C, d),
                    None, None, None, None)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

_HIER_TRITON_OK: bool | None = None


def _check_hier_triton() -> bool:
    global _HIER_TRITON_OK
    if _HIER_TRITON_OK is not None:
        return _HIER_TRITON_OK
    from kernel.triton_local import _check_triton_runtime
    _HIER_TRITON_OK = _check_triton_runtime()
    return _HIER_TRITON_OK


def compressed_level_triton(
    q: torch.Tensor,        # (B, H, N, d)
    k_c: torch.Tensor,      # (B, H, C, d)
    v_c: torch.Tensor,      # (B, H, C, d)
    chunk_size: int,
    local_W: int,
    gamma: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fused compressed-level attention kernel.
    Returns (m, l, o) unnormalised accumulators.
    Falls back to PyTorch if Triton unavailable or tensors on CPU.
    """
    if not q.is_cuda or not _check_hier_triton():
        return _hier_pytorch(q, k_c, v_c, chunk_size, local_W, gamma, scale)
    C = k_c.shape[2]
    if C == 0:
        B, H, N, d = q.shape
        m = torch.full((B, H, N),    -1e38, device=q.device, dtype=torch.float32)
        l = torch.zeros((B, H, N),          device=q.device, dtype=torch.float32)
        o = torch.zeros((B, H, N, d),       device=q.device, dtype=q.dtype)
        return m, l, o
    return _HierAccumFn.apply(
        q.contiguous(), k_c.contiguous(), v_c.contiguous(),
        chunk_size, local_W, gamma, scale,
    )


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch fallback — same (m, l, o) interface, no mask_cache
# ─────────────────────────────────────────────────────────────────────────────

def _hier_pytorch(
    q: torch.Tensor,
    k_c: torch.Tensor,
    v_c: torch.Tensor,
    chunk_size: int,
    local_W: int,
    gamma: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, H, N, d = q.shape
    C = k_c.shape[2]
    device, dtype = q.device, q.dtype

    if C == 0:
        m = torch.full((B, H, N),    float("-inf"), device=device, dtype=dtype)
        l = torch.zeros((B, H, N),                  device=device, dtype=dtype)
        o = torch.zeros((B, H, N, d),               device=device, dtype=dtype)
        return m, l, o

    acc_dtype = dtype if dtype == torch.float64 else torch.float32

    q_pos     = torch.arange(N, device=device, dtype=acc_dtype)
    c_idx     = torch.arange(C, device=device, dtype=acc_dtype)
    chunk_end = (c_idx + 1) * chunk_size
    centroid  = c_idx * chunk_size + chunk_size / 2.0

    threshold = (q_pos - local_W + 1).clamp(min=0)
    mask      = chunk_end.unsqueeze(0) <= threshold.unsqueeze(1)   # (N, C)
    dist      = (q_pos.unsqueeze(1) - centroid.unsqueeze(0)).abs()

    if not mask.any():
        m = torch.full((B, H, N),    float("-inf"), device=device, dtype=acc_dtype)
        l = torch.zeros((B, H, N),                  device=device, dtype=acc_dtype)
        o = torch.zeros((B, H, N, d),               device=device, dtype=dtype)
        return m, l, o

    scores = torch.matmul(q.to(acc_dtype), k_c.to(acc_dtype).transpose(-2, -1)) * scale
    scores = scores - (gamma.to(acc_dtype) * dist).unsqueeze(0).unsqueeze(0)
    scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))

    m_     = scores.amax(dim=-1)
    m_safe = m_.clamp(min=-1e9)
    exp_s  = torch.exp(scores - m_safe.unsqueeze(-1))
    exp_s  = exp_s.masked_fill(~mask.unsqueeze(0).unsqueeze(0), 0.0)

    l_ = exp_s.sum(dim=-1)
    o_ = torch.matmul(exp_s, v_c.to(acc_dtype)).to(dtype)

    no_key = ~mask.any(dim=-1)
    m_     = m_.masked_fill(no_key.unsqueeze(0).unsqueeze(0), float("-inf"))

    return m_, l_, o_
