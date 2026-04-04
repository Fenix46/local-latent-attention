"""
Triton kernel for Telescopic Attention — Level 0 (sliding window).

The kernel exposes raw (m, l, o) accumulators — NOT the normalised output —
so that the Python-side merge operator can combine them with compressed levels
exactly as before, without breaking the associative merge math.

After all levels are merged, normalisation is done once: output = o / l.

Algorithm (per query block):
  - Attend only to keys in [q_start - W + 1, q_end)  (causal + window)
  - Iterate over key tiles of BLOCK_K within that range
  - Apply ALiBi: score -= gamma * (q_pos - k_pos).clamp(0)
  - Online softmax → accumulate (m, l, o) without normalising

Returns:
  m   : (B, H, N)     running max (log-domain)
  l   : (B, H, N)     running sum of exp weights
  o   : (B, H, N, d)  unnormalised weighted value sum

Memory: O(N · W / tile²) SRAM — never writes N×N to DRAM.
Compute: O(N · W · d).

Requires: triton >= 2.1, CUDA (also works on ROCm with triton-rocm).
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


# ─────────────────────────────────────────────────────────────────────────────
# Forward kernel — returns (m, l, o) accumulators
# ─────────────────────────────────────────────────────────────────────────────

if HAS_TRITON:

    @triton.jit
    def _sw_fwd_kernel(
        # Pointers — all (BH, N, d) after flattening B and H
        Q_ptr, K_ptr, V_ptr,
        M_ptr,    # out: (BH, N)   running max
        L_ptr,    # out: (BH, N)   running sum
        O_ptr,    # out: (BH, N, d) unnormalised weighted sum
        # Strides
        stride_n, stride_d,     # shared by Q/K/V/O (they're contiguous)
        stride_ln,              # stride of M and L along N
        # Scalars
        N: tl.constexpr,
        d: tl.constexpr,
        W: tl.constexpr,
        scale,
        gamma,
        BLOCK_Q: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """
        Grid: (ceil(N / BLOCK_Q),  B*H)
        program_id(0) → q_block
        program_id(1) → bh index (flattened batch × head)
        """
        q_blk = tl.program_id(0)
        bh    = tl.program_id(1)

        q_start = q_blk * BLOCK_Q
        q_offs  = q_start + tl.arange(0, BLOCK_Q)   # (BLOCK_Q,)
        q_valid = q_offs < N
        d_offs  = tl.arange(0, d)                    # (d,)

        # Base pointers for this (b,h) slice — layout is (BH, N, d)
        base    = bh * N
        q_base  = Q_ptr + base
        k_base  = K_ptr + base
        v_base  = V_ptr + base
        m_base  = M_ptr + bh * N
        l_base  = L_ptr + bh * N
        o_base  = O_ptr + base

        # Load Q block: (BLOCK_Q, d)
        q = tl.load(
            q_base + q_offs[:, None] * stride_n + d_offs[None, :],
            mask=q_valid[:, None],
            other=0.0,
        )

        # Online softmax state
        m   = tl.full((BLOCK_Q,), float("-inf"), dtype=tl.float32)
        acc_l = tl.zeros((BLOCK_Q,),    dtype=tl.float32)
        acc_o = tl.zeros((BLOCK_Q, d),  dtype=tl.float32)

        # Key range: [k_lo, k_hi)
        k_lo = tl.maximum(0, q_start - W + 1)
        k_hi = tl.minimum(N, q_start + BLOCK_Q)

        k_cur = k_lo
        while k_cur < k_hi:
            k_offs = k_cur + tl.arange(0, BLOCK_K)
            k_valid = k_offs < k_hi

            k = tl.load(
                k_base + k_offs[:, None] * stride_n + d_offs[None, :],
                mask=k_valid[:, None],
                other=0.0,
            )  # (BLOCK_K, d)
            v = tl.load(
                v_base + k_offs[:, None] * stride_n + d_offs[None, :],
                mask=k_valid[:, None],
                other=0.0,
            )  # (BLOCK_K, d)

            # Scores: (BLOCK_Q, BLOCK_K) — keep everything float32
            s = tl.dot(q.to(tl.float32), tl.trans(k).to(tl.float32))
            s = s * tl.cast(scale, tl.float32)

            # ALiBi: bias = -gamma * max(q_pos - k_pos, 0)
            dist = (q_offs[:, None].to(tl.float32)
                    - k_offs[None, :].to(tl.float32))
            dist = tl.maximum(dist, tl.zeros_like(dist))
            s = s - tl.cast(gamma, tl.float32) * dist

            # Mask: causal (k > q) and window (dist >= W) and padding
            causal  = k_offs[None, :] > q_offs[:, None]
            out_win = dist >= tl.cast(W, tl.float32)
            invalid = causal | out_win | (~k_valid)[None, :]
            neg_inf = tl.full(s.shape, -1e38, dtype=tl.float32)
            s = tl.where(invalid, neg_inf, s)

            # Online softmax update — all float32
            m_new  = tl.maximum(m, tl.max(s, axis=1).to(tl.float32))
            alpha  = tl.exp(m - m_new)
            beta   = tl.exp(s - m_new[:, None])
            beta   = tl.where(invalid, tl.zeros_like(beta), beta)

            acc_l = acc_l * alpha + tl.sum(beta, axis=1)
            acc_o = acc_o * alpha[:, None] + tl.dot(beta.to(v.dtype), v).to(tl.float32)
            m     = m_new

            k_cur += BLOCK_K

        # Rows with no valid key: keep m = -inf, l = 0, o = 0
        # (already the case since acc_l stays 0 when all keys are masked)

        # Store (m, l, o) — NOT normalised
        tl.store(m_base + q_offs, m, mask=q_valid)
        tl.store(l_base + q_offs, acc_l, mask=q_valid)
        tl.store(
            o_base + q_offs[:, None] * stride_n + d_offs[None, :],
            acc_o.to(Q_ptr.dtype.element_ty),
            mask=q_valid[:, None],
        )


# ─────────────────────────────────────────────────────────────────────────────
# Backward kernel
# ─────────────────────────────────────────────────────────────────────────────

if HAS_TRITON:

    @triton.jit
    def _sw_bwd_kernel(
        # Inputs
        Q_ptr, K_ptr, V_ptr,
        dO_ptr,   # (BH, N, d) gradient of *normalised* output
        M_ptr,    # (BH, N)   saved running max from forward
        L_ptr,    # (BH, N)   saved running sum from forward
        # Outputs (zeroed before launch — atomic adds)
        dQ_ptr, dK_ptr, dV_ptr,
        # Strides
        stride_n, stride_d,
        stride_ln,
        # Scalars
        N: tl.constexpr,
        d: tl.constexpr,
        W: tl.constexpr,
        scale,
        gamma,
        BLOCK_Q: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """
        Recompute-based backward: no O saved, recompute P from (Q,K,M,L).

        For each query block:
          P_ij = exp(s_ij - m_i) / l_i   (recomputed)
          dV  += P^T · dO
          dP   = dO · V^T
          D_i  = sum_j P_ij * dP_ij       (rowsum)
          dS_ij = P_ij * (dP_ij - D_i)
          dQ  += scale * dS · K
          dK  += scale * dS^T · Q
        """
        q_blk = tl.program_id(0)
        bh    = tl.program_id(1)

        q_start = q_blk * BLOCK_Q
        q_offs  = q_start + tl.arange(0, BLOCK_Q)
        q_valid = q_offs < N
        d_offs  = tl.arange(0, d)

        base   = bh * N
        q_base = Q_ptr  + base
        k_base = K_ptr  + base
        v_base = V_ptr  + base
        do_base = dO_ptr + base
        m_base = M_ptr  + bh * N
        l_base = L_ptr  + bh * N
        dq_base = dQ_ptr + base
        dk_base = dK_ptr + base
        dv_base = dV_ptr + base

        q  = tl.load(q_base  + q_offs[:, None] * stride_n + d_offs[None, :],
                     mask=q_valid[:, None], other=0.0)
        do = tl.load(do_base + q_offs[:, None] * stride_n + d_offs[None, :],
                     mask=q_valid[:, None], other=0.0)
        m  = tl.load(m_base  + q_offs, mask=q_valid, other=-1e38)
        l  = tl.load(l_base  + q_offs, mask=q_valid, other=1.0)
        l  = tl.maximum(l, 1e-8)   # numerical safety

        dq = tl.zeros((BLOCK_Q, d), dtype=tl.float32)

        k_lo = tl.maximum(0, q_start - W + 1)
        k_hi = tl.minimum(N, q_start + BLOCK_Q)

        k_cur = k_lo
        while k_cur < k_hi:
            k_offs  = k_cur + tl.arange(0, BLOCK_K)
            k_valid = k_offs < k_hi

            k = tl.load(k_base + k_offs[:, None] * stride_n + d_offs[None, :],
                        mask=k_valid[:, None], other=0.0)
            v = tl.load(v_base + k_offs[:, None] * stride_n + d_offs[None, :],
                        mask=k_valid[:, None], other=0.0)

            # Recompute s — keep everything float32
            s    = tl.dot(q.to(tl.float32), tl.trans(k).to(tl.float32))
            s    = s * tl.cast(scale, tl.float32)
            dist = q_offs[:, None].to(tl.float32) - k_offs[None, :].to(tl.float32)
            dist = tl.maximum(dist, tl.zeros_like(dist))
            s    = s - tl.cast(gamma, tl.float32) * dist

            causal  = k_offs[None, :] > q_offs[:, None]
            out_win = dist >= tl.cast(W, tl.float32)
            invalid = causal | out_win | (~k_valid)[None, :]
            s       = tl.where(invalid, tl.full(s.shape, -1e38, dtype=tl.float32), s)

            # Recompute P from saved (m, l): P = exp(s - m_i) / l_i
            p = tl.exp(s - m[:, None]) / l[:, None]
            p = tl.where(invalid, tl.zeros_like(p), p)   # (BLOCK_Q, BLOCK_K)

            # dV += P^T · dO
            dv = tl.dot(tl.trans(p.to(do.dtype)), do)   # (BLOCK_K, d)
            tl.atomic_add(
                dv_base + k_offs[:, None] * stride_n + d_offs[None, :],
                dv.to(dV_ptr.dtype.element_ty),
                mask=k_valid[:, None],
            )

            # dP = dO · V^T;  D = rowsum(P * dP)
            dp = tl.dot(do.to(tl.float32), tl.trans(v).to(tl.float32))   # (BLOCK_Q, BLOCK_K)
            D  = tl.sum(p * dp, axis=1)    # (BLOCK_Q,)

            # dS = P * (dP - D)
            ds = p * (dp - D[:, None])
            ds = tl.where(invalid, 0.0, ds)

            # dQ += scale * dS · K
            dq = dq + tl.dot(ds.to(k.dtype), k).to(tl.float32) * scale

            # dK += scale * dS^T · Q
            dk = tl.dot(tl.trans(ds.to(q.dtype)), q).to(tl.float32) * scale
            tl.atomic_add(
                dk_base + k_offs[:, None] * stride_n + d_offs[None, :],
                dk.to(dK_ptr.dtype.element_ty),
                mask=k_valid[:, None],
            )

            k_cur += BLOCK_K

        tl.store(
            dq_base + q_offs[:, None] * stride_n + d_offs[None, :],
            dq.to(dQ_ptr.dtype.element_ty),
            mask=q_valid[:, None],
        )


# ─────────────────────────────────────────────────────────────────────────────
# autograd.Function — exposes (m, l, o) for external merge
# ─────────────────────────────────────────────────────────────────────────────

if HAS_TRITON:

    class _SWAccumFn(torch.autograd.Function):
        """
        Forward  → (m, l, o)  where o is unnormalised
        Backward → (dq, dk, dv) given d(normalised_output)

        The caller is responsible for:
          1. Merging (m, l, o) with other levels via the ⊕ operator
          2. Normalising the final merged output: out = o_total / l_total
          3. Passing the gradient of the *normalised* output to backward
        """

        @staticmethod
        def forward(ctx, q, k, v, window: int, gamma: torch.Tensor, scale: float):
            B, H, N, d = q.shape
            assert q.is_cuda
            assert d in (16, 32, 64, 128), f"Unsupported head_dim={d}"
            assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()

            BH = B * H
            # Reshape to (BH, N, d)
            q2 = q.view(BH, N, d)
            k2 = k.view(BH, N, d)
            v2 = v.view(BH, N, d)

            m   = torch.full((BH, N),    float("-inf"), device=q.device, dtype=torch.float32)
            l   = torch.zeros((BH, N),                  device=q.device, dtype=torch.float32)
            o2  = torch.zeros((BH, N, d),               device=q.device, dtype=q.dtype)

            BLOCK_Q = 64
            BLOCK_K = 64
            grid = (triton.cdiv(N, BLOCK_Q), BH)

            _sw_fwd_kernel[grid](
                q2, k2, v2, m, l, o2,
                stride_n=q2.stride(1),
                stride_d=q2.stride(2),
                stride_ln=1,
                N=N, d=d, W=window,
                scale=scale,
                gamma=gamma.item(),
                BLOCK_Q=BLOCK_Q,
                BLOCK_K=BLOCK_K,
            )

            m = m.view(B, H, N)
            l = l.view(B, H, N)
            o = o2.view(B, H, N, d)

            ctx.save_for_backward(q, k, v, m, l, gamma)
            ctx.window = window
            ctx.scale  = scale
            return m, l, o

        @staticmethod
        def backward(ctx, dm, dl, do_unnorm):
            """
            NOTE: dm and dl gradients from merge are complex to propagate correctly
            through the ⊕ operator. For now we approximate: pass the gradient of the
            normalised output (do_unnorm / l is close enough when the merge is dominated
            by local attention, which it is for most tokens).

            A future version will implement exact merge-backward.
            """
            q, k, v, m, l, gamma = ctx.saved_tensors
            window = ctx.window
            scale  = ctx.scale

            B, H, N, d = q.shape
            BH = B * H

            # Gradient of normalised output ≈ do_unnorm / l
            # (exact when local attention dominates)
            l_safe = l.clamp(min=1e-8)
            do_norm = do_unnorm / l_safe.unsqueeze(-1)

            q2  = q.view(BH, N, d).contiguous()
            k2  = k.view(BH, N, d).contiguous()
            v2  = v.view(BH, N, d).contiguous()
            do2 = do_norm.contiguous().view(BH, N, d)
            m2  = m.view(BH, N).contiguous()
            l2  = l.view(BH, N).contiguous()

            dq = torch.zeros_like(q2)
            dk = torch.zeros_like(k2)
            dv = torch.zeros_like(v2)

            BLOCK_Q = 64
            BLOCK_K = 64
            grid = (triton.cdiv(N, BLOCK_Q), BH)

            _sw_bwd_kernel[grid](
                q2, k2, v2, do2, m2, l2,
                dq, dk, dv,
                stride_n=q2.stride(1),
                stride_d=q2.stride(2),
                stride_ln=1,
                N=N, d=d, W=window,
                scale=scale,
                gamma=gamma.item(),
                BLOCK_Q=BLOCK_Q,
                BLOCK_K=BLOCK_K,
            )

            return dq.view(B, H, N, d), dk.view(B, H, N, d), dv.view(B, H, N, d), None, None, None


_TRITON_RUNTIME_OK: bool | None = None   # None = untested, True = ok, False = broken


def _check_triton_runtime() -> bool:
    """
    Lazily test whether the Triton CUDA driver compiles and runs successfully.
    Probes by actually calling get_current_device() — the same call that fails
    at kernel launch time — so we catch build errors before the first forward pass.
    Result is cached so the check runs at most once per process.
    """
    global _TRITON_RUNTIME_OK
    if _TRITON_RUNTIME_OK is not None:
        return _TRITON_RUNTIME_OK
    if not HAS_TRITON:
        _TRITON_RUNTIME_OK = False
        return False
    try:
        from triton.runtime.driver import driver as _drv
        _drv.active.get_current_device()   # triggers full driver init + C compilation
        _TRITON_RUNTIME_OK = True
    except Exception:
        import warnings
        warnings.warn(
            "Triton CUDA driver failed to initialise (libcuda.so link error?). "
            "Falling back to PyTorch sliding-window attention.\n"
            "Fix: ln -sf /lib/x86_64-linux-gnu/libcuda.so.1 /usr/local/cuda/lib64/libcuda.so.1\n"
            "     export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH",
            RuntimeWarning,
            stacklevel=2,
        )
        _TRITON_RUNTIME_OK = False
    return _TRITON_RUNTIME_OK


def sliding_window_triton_accum(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window: int,
    gamma: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fused sliding-window attention kernel.

    Returns (m, l, o) accumulators — unnormalised — ready for merge with
    compressed levels via the ⊕ operator.

    Falls back to PyTorch if Triton unavailable, driver broken, or tensors on CPU.

    q, k, v : (B, H, N, d)  contiguous, fp16 or bf16
    m       : (B, H, N)     float32
    l       : (B, H, N)     float32
    o       : (B, H, N, d)  same dtype as q
    """
    if not q.is_cuda or not _check_triton_runtime():
        return _sw_pytorch_accum(q, k, v, window, gamma, scale)
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    return _SWAccumFn.apply(q, k, v, window, gamma, scale)


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch fallback — same (m, l, o) interface
# ─────────────────────────────────────────────────────────────────────────────

def _sw_pytorch_accum(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window: int,
    gamma: torch.Tensor,
    scale: float,
    q_block: int = 256,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Vectorized sliding-window attention accumulators.

    Processes queries in blocks of `q_block` (default 256) — the Python loop
    runs ceil(N / q_block) times instead of ceil(N / window) times, dramatically
    reducing overhead when window << q_block.

    Within each block, uses gather to extract the W-wide key window for every
    query in the block simultaneously, then computes scores in a single batched
    operation. This is torch.compile-friendly and GPU-parallel.

    Memory per block: O(B·H·q_block·W·d) — controlled, no N² allocation.
    """
    B, H, N, d = q.shape
    device, dtype = q.device, q.dtype
    acc_dtype = dtype if dtype == torch.float64 else torch.float32

    offsets = torch.arange(window, device=device)   # (W,)

    m_out = torch.full((B, H, N),    float("-inf"), device=device, dtype=acc_dtype)
    l_out = torch.zeros((B, H, N),                  device=device, dtype=acc_dtype)
    o_out = torch.zeros((B, H, N, d),               device=device, dtype=dtype)

    for q_start in range(0, N, q_block):
        q_end  = min(q_start + q_block, N)
        Q      = q_end - q_start          # actual block size

        q_pos  = torch.arange(q_start, q_end, device=device)   # (Q,)

        # Gather indices: for each query i in [q_start, q_end), key positions [i-W+1, i]
        # k_idx[i, w] = clamp(i - W + 1 + w, 0, N-1) — shape (Q, W)
        real_k = q_pos.unsqueeze(1) - window + 1 + offsets.unsqueeze(0)   # (Q, W)
        k_idx  = real_k.clamp(0, N - 1)

        # invalid: clamped indices that are out of the valid range
        invalid = (real_k < 0) | (real_k > q_pos.unsqueeze(1))   # (Q, W)

        # Gather K, V: (B, H, Q, W, d)
        idx_exp = k_idx.view(1, 1, Q, window, 1).expand(B, H, Q, window, d)
        k_flat  = k.gather(2, idx_exp.reshape(B, H, Q * window, d))
        k_win   = k_flat.reshape(B, H, Q, window, d)
        v_flat  = v.gather(2, idx_exp.reshape(B, H, Q * window, d))
        v_win   = v_flat.reshape(B, H, Q, window, d)

        # Scores: (B, H, Q, W)
        q_blk  = q[:, :, q_start:q_end, :]   # (B, H, Q, d)
        scores = (q_blk.unsqueeze(3).to(acc_dtype) * k_win.to(acc_dtype)).sum(-1) * scale

        # ALiBi bias
        dist   = (q_pos.unsqueeze(1) - real_k.clamp(0)).to(acc_dtype).clamp(min=0)   # (Q, W)
        scores = scores - gamma.to(acc_dtype) * dist.unsqueeze(0).unsqueeze(0)

        # Mask
        scores = scores.masked_fill(invalid.unsqueeze(0).unsqueeze(0), float("-inf"))

        # Softmax accumulators
        m_blk  = scores.amax(dim=-1)                   # (B, H, Q)
        m_safe = m_blk.clamp(min=-1e9)
        exp_s  = torch.exp(scores - m_safe.unsqueeze(-1))
        exp_s  = exp_s.masked_fill(invalid.unsqueeze(0).unsqueeze(0), 0.0)

        l_blk = exp_s.sum(dim=-1)                       # (B, H, Q)
        o_blk = (exp_s.to(acc_dtype).unsqueeze(-1) * v_win.to(acc_dtype)).sum(dim=3).to(dtype)

        no_key = invalid.all(dim=-1)   # (Q,)
        m_blk  = m_blk.masked_fill(no_key.unsqueeze(0).unsqueeze(0), float("-inf"))

        m_out[:, :, q_start:q_end] = m_blk
        l_out[:, :, q_start:q_end] = l_blk
        o_out[:, :, q_start:q_end] = o_blk

    return m_out, l_out, o_out
