"""
LocalLatentAttention — torch.autograd.Function wrapper.

Combines the forward and backward Triton kernels into a differentiable op
that PyTorch's autograd engine can use transparently.

Usage
-----
    out = LLAttnFunction.apply(
        Q, K, V, GQ, GK, GV,
        gate_local_w, gate_remote_w, r_out_w,
        window, chunk_size,
    )
"""

import math

import triton
import torch
from torch.autograd import Function

try:
    from .llattn_fwd import llattn_fwd_kernel
    from .llattn_bwd import llattn_bwd_dq_kernel, llattn_bwd_dkv_kernel
except ImportError:
    from llattn_fwd import llattn_fwd_kernel
    from llattn_bwd import llattn_bwd_dq_kernel, llattn_bwd_dkv_kernel


def _stride(t: torch.Tensor, dim: int) -> int:
    return t.stride(dim)


def _call_fwd(Q, K, V, GQ, GK, GV,
              gate_local_w, gate_remote_w, r_out_w,
              window: int, chunk_size: int):
    B, Nh, S, Dh = Q.shape
    Lh = GK.shape[1]
    Nl = GK.shape[2]
    Ld = GK.shape[3]

    scale        = 1.0 / math.sqrt(Dh)
    latent_scale = 1.0 / math.sqrt(Ld)

    Out        = torch.empty_like(Q)
    LSE_local  = torch.empty (B, Nh, S, device=Q.device, dtype=torch.float32)
    LSE_remote = torch.empty (B, Lh, S, device=Q.device, dtype=torch.float32)
    W_local    = torch.empty (B, Nh, S, device=Q.device, dtype=torch.float32)

    BLOCK_M = 64
    grid = (B * Nh, triton.cdiv(S, BLOCK_M))

    llattn_fwd_kernel[grid](
        Q, K, V,
        GQ, GK, GV,
        gate_local_w, gate_remote_w, r_out_w,
        Out, LSE_local, LSE_remote, W_local,
        # Q strides
        _stride(Q,0), _stride(Q,1), _stride(Q,2), _stride(Q,3),
        # K strides
        _stride(K,0), _stride(K,1), _stride(K,2), _stride(K,3),
        # V strides
        _stride(V,0), _stride(V,1), _stride(V,2), _stride(V,3),
        # GQ strides
        _stride(GQ,0), _stride(GQ,1), _stride(GQ,2), _stride(GQ,3),
        # GK strides
        _stride(GK,0), _stride(GK,1), _stride(GK,2), _stride(GK,3),
        # GV strides
        _stride(GV,0), _stride(GV,1), _stride(GV,2), _stride(GV,3),
        # Out strides
        _stride(Out,0), _stride(Out,1), _stride(Out,2), _stride(Out,3),
        # LSE_local strides
        _stride(LSE_local,0), _stride(LSE_local,1), _stride(LSE_local,2),
        # LSE_remote strides
        _stride(LSE_remote,0), _stride(LSE_remote,1), _stride(LSE_remote,2),
        # W_local strides
        _stride(W_local,0), _stride(W_local,1), _stride(W_local,2),
        # Scalars
        SEQ_LEN=S, N_HEADS=Nh, LATENT_HEADS=Lh, N_LATENTS=Nl,
        HEAD_DIM=Dh, LATENT_HEAD_DIM=Ld,
        WINDOW=window, CHUNK_SIZE=chunk_size,
        scale=scale, latent_scale=latent_scale,
    )
    return Out, LSE_local, LSE_remote, W_local


def _call_bwd(Q, K, V, GQ, GK, GV,
              gate_local_w, gate_remote_w, r_out_w,
              Out, LSE_local, LSE_remote, W_local,
              dOut,
              window: int, chunk_size: int):
    B, Nh, S, Dh = Q.shape
    Lh = GK.shape[1]
    Nl = GK.shape[2]
    Ld = GK.shape[3]

    scale        = 1.0 / math.sqrt(Dh)
    latent_scale = 1.0 / math.sqrt(Ld)

    # Allocate gradient tensors in float32 — tl.atomic_add does NOT support bf16.
    # We cast back to the input dtype at the end.
    dQ            = torch.zeros_like(Q,            dtype=torch.float32)
    dK            = torch.zeros_like(K,            dtype=torch.float32)
    dV            = torch.zeros_like(V,            dtype=torch.float32)
    dGQ           = torch.zeros_like(GQ,           dtype=torch.float32)
    dGK           = torch.zeros_like(GK,           dtype=torch.float32)
    dGV           = torch.zeros_like(GV,           dtype=torch.float32)
    dgate_local_w = torch.zeros_like(gate_local_w, dtype=torch.float32)
    dgate_remote_w= torch.zeros_like(gate_remote_w,dtype=torch.float32)
    dr_out_w      = torch.zeros_like(r_out_w,      dtype=torch.float32)

    common_strides = (
        # Q
        _stride(Q,0), _stride(Q,1), _stride(Q,2), _stride(Q,3),
        # K
        _stride(K,0), _stride(K,1), _stride(K,2), _stride(K,3),
        # V
        _stride(V,0), _stride(V,1), _stride(V,2), _stride(V,3),
        # GQ
        _stride(GQ,0), _stride(GQ,1), _stride(GQ,2), _stride(GQ,3),
        # GK
        _stride(GK,0), _stride(GK,1), _stride(GK,2), _stride(GK,3),
        # GV
        _stride(GV,0), _stride(GV,1), _stride(GV,2), _stride(GV,3),
        # dOut strides (same layout as Q/Out: [B, Nh, S, Dh])
        _stride(dOut,0), _stride(dOut,1), _stride(dOut,2), _stride(dOut,3),
        # LSE_local
        _stride(LSE_local,0), _stride(LSE_local,1), _stride(LSE_local,2),
        # LSE_remote
        _stride(LSE_remote,0), _stride(LSE_remote,1), _stride(LSE_remote,2),
        # W_local
        _stride(W_local,0), _stride(W_local,1), _stride(W_local,2),
    )
    # dkv kernel needs Out strides too (prepend before common_strides)
    out_strides = (
        _stride(Out,0), _stride(Out,1), _stride(Out,2), _stride(Out,3),
    )
    common_scalars = dict(
        SEQ_LEN=S, N_HEADS=Nh, LATENT_HEADS=Lh, N_LATENTS=Nl,
        HEAD_DIM=Dh, LATENT_HEAD_DIM=Ld,
        WINDOW=window, CHUNK_SIZE=chunk_size,
        scale=scale, latent_scale=latent_scale,
    )

    # Pass 1: dQ, dGQ, dgate_local_w, dgate_remote_w, dr_out_w
    BLOCK_M_BWD = 64
    grid1 = (B * Nh, triton.cdiv(S, BLOCK_M_BWD))
    llattn_bwd_dq_kernel[grid1](
        Q, K, V, GQ, GK, GV,
        gate_local_w, gate_remote_w, r_out_w,
        LSE_local, LSE_remote, W_local, dOut,
        dQ, dGQ,
        dgate_local_w, dgate_remote_w, dr_out_w,
        *common_strides,
        **common_scalars,
    )

    # Pass 2: dK, dV
    BLOCK_N_BWD = 64
    grid2 = (B * Nh, triton.cdiv(S, BLOCK_N_BWD))
    # common_strides order: Q,K,V,GQ,GK,GV strides, dOut strides, LSE/W strides
    # dkv_kernel expects: Q,K,V,GQ,GK,GV strides, Out strides, dOut strides, LSE/W strides
    # Split common_strides and insert out_strides in the right position
    # Q(4)+K(4)+V(4)+GQ(4)+GK(4)+GV(4) = 24 strides before dOut
    dkv_strides = common_strides[:24] + out_strides + common_strides[24:]
    llattn_bwd_dkv_kernel[grid2](
        Q, K, V, GQ, GK, GV,
        gate_local_w, gate_remote_w, r_out_w,
        Out, LSE_local, LSE_remote, W_local, dOut,
        dK, dV, dGK, dGV,
        *dkv_strides,
        **common_scalars,
    )

    # Cast gradients back to the input dtype (inputs may be bf16)
    src_dtype = Q.dtype
    def _cast(t): return t.to(src_dtype) if t.dtype != src_dtype else t

    return (_cast(dQ), _cast(dK), _cast(dV),
            _cast(dGQ), _cast(dGK), _cast(dGV),
            _cast(dgate_local_w), _cast(dgate_remote_w), _cast(dr_out_w))


class LLAttnFunction(Function):
    """
    Differentiable wrapper around the fused LocalLatentAttention kernels.

    forward inputs
    --------------
    Q, K, V       : [B, Nh, S, Dh]   bf16
    GQ, GK, GV    : latent tensors    bf16
    gate_local_w  : [Nh, Dh]          bf16
    gate_remote_w : [Nh, Dh]          bf16
    r_out_w       : [Nh*Dh, Lh*Ld]   bf16
    window        : int
    chunk_size    : int
    """

    @staticmethod
    def forward(ctx, Q, K, V, GQ, GK, GV,
                gate_local_w, gate_remote_w, r_out_w,
                window, chunk_size):

        Out, LSE_local, LSE_remote, W_local = _call_fwd(
            Q, K, V, GQ, GK, GV,
            gate_local_w, gate_remote_w, r_out_w,
            window, chunk_size,
        )
        ctx.save_for_backward(
            Q, K, V, GQ, GK, GV,
            gate_local_w, gate_remote_w, r_out_w,
            Out, LSE_local, LSE_remote, W_local,
        )
        ctx.window      = window
        ctx.chunk_size  = chunk_size
        return Out

    @staticmethod
    def backward(ctx, dOut):
        (Q, K, V, GQ, GK, GV,
         gate_local_w, gate_remote_w, r_out_w,
         Out, LSE_local, LSE_remote, W_local) = ctx.saved_tensors

        dOut = dOut.contiguous()

        (dQ, dK, dV, dGQ, dGK, dGV,
         dgate_local_w, dgate_remote_w, dr_out_w) = _call_bwd(
            Q, K, V, GQ, GK, GV,
            gate_local_w, gate_remote_w, r_out_w,
            Out, LSE_local, LSE_remote, W_local,
            dOut,
            ctx.window, ctx.chunk_size,
        )

        # Gradients for non-tensor args (window, chunk_size) → None
        return (dQ, dK, dV, dGQ, dGK, dGV,
                dgate_local_w, dgate_remote_w, dr_out_w,
                None, None)
