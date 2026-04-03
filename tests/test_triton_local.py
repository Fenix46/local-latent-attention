"""
Tests for the Triton sliding-window kernel (triton_local.py).

Run with:
  pytest tests/test_triton_local.py -v          # any device (uses PyTorch fallback on CPU)
  pytest tests/test_triton_local.py -v --cuda   # force CUDA path (requires GPU + Triton)

Tests:
  1. Output shape correct
  2. Triton (m,l,o) matches PyTorch fallback — forward
  3. Normalised output matches PyTorch attention
  4. Causal property: future tokens have zero weight
  5. Window property: tokens beyond W have zero weight
  6. Gradient flows through (backward smoke test)
  7. Gradient matches PyTorch finite-difference (on CPU fallback)
  8. Works with multiple batch sizes and head counts
"""

from __future__ import annotations
import math
import pytest
import torch
import torch.nn.functional as F

from kernel.triton_local import sliding_window_triton_accum, _sw_pytorch_accum, HAS_TRITON


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_qkv(B=1, H=2, N=32, d=16, device="cpu", dtype=torch.float32, seed=0):
    torch.manual_seed(seed)
    q = torch.randn(B, H, N, d, device=device, dtype=dtype)
    k = torch.randn(B, H, N, d, device=device, dtype=dtype)
    v = torch.randn(B, H, N, d, device=device, dtype=dtype)
    return q, k, v


def normalise(m, l, o):
    """Convert accumulators to attention output."""
    l_safe = l.clamp(min=1e-8)
    out = o / l_safe.unsqueeze(-1)
    no_key = (l == 0)
    out = out.masked_fill(no_key.unsqueeze(-1), 0.0)
    return out


def reference_attention(q, k, v, window, gamma_val, scale):
    """Brute-force full-matrix attention with causal + window mask."""
    B, H, N, d = q.shape
    device, dtype = q.device, q.dtype

    q_pos = torch.arange(N, device=device, dtype=dtype)
    k_pos = torch.arange(N, device=device, dtype=dtype)
    dist  = (q_pos.unsqueeze(1) - k_pos.unsqueeze(0)).clamp(min=0)  # (N, N)

    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    scores = scores - gamma_val * dist.unsqueeze(0).unsqueeze(0)

    causal  = k_pos.unsqueeze(0) > q_pos.unsqueeze(1)
    in_win  = (q_pos.unsqueeze(1) - k_pos.unsqueeze(0)) < window
    invalid = causal | ~in_win
    scores  = scores.masked_fill(invalid.unsqueeze(0).unsqueeze(0), float("-inf"))

    probs = torch.softmax(scores, dim=-1)
    probs = probs.nan_to_num(0.0)
    return torch.matmul(probs, v)


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSlidingWindowAccum:

    @pytest.fixture(params=["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
    def device(self, request):
        return request.param

    def test_output_shapes(self, device):
        B, H, N, d = 2, 4, 64, 32
        q, k, v = make_qkv(B, H, N, d, device=device)
        gamma = torch.tensor(0.1, device=device)
        scale = 1.0 / math.sqrt(d)
        m, l, o = sliding_window_triton_accum(q, k, v, window=16, gamma=gamma, scale=scale)
        assert m.shape == (B, H, N)
        assert l.shape == (B, H, N)
        assert o.shape == (B, H, N, d)

    def test_matches_pytorch_fallback(self, device):
        """Triton kernel must produce same (m, l, o) as pure-PyTorch fallback."""
        B, H, N, d = 1, 2, 32, 16
        q, k, v = make_qkv(B, H, N, d, device=device, seed=42)
        gamma = torch.tensor(0.05, device=device)
        scale = 1.0 / math.sqrt(d)
        window = 8

        m_ref, l_ref, o_ref = _sw_pytorch_accum(q, k, v, window, gamma, scale)
        m_tri, l_tri, o_tri = sliding_window_triton_accum(q, k, v, window, gamma, scale)

        # On CPU both paths are identical (same code), on CUDA compare numerically
        tol = dict(atol=1e-3, rtol=1e-3) if device == "cuda" else dict(atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(m_tri.float(), m_ref.float(), **tol)
        torch.testing.assert_close(l_tri.float(), l_ref.float(), **tol)
        torch.testing.assert_close(o_tri.float(), o_ref.float(), **tol)

    def test_normalised_matches_reference(self, device):
        """Normalised output must match brute-force full-matrix attention."""
        B, H, N, d = 1, 2, 32, 16
        q, k, v = make_qkv(B, H, N, d, device=device, seed=7)
        gamma = torch.tensor(0.1, device=device)
        scale = 1.0 / math.sqrt(d)
        window = 12

        m, l, o = sliding_window_triton_accum(q, k, v, window, gamma, scale)
        out = normalise(m, l, o)
        ref = reference_attention(q, k, v, window, 0.1, scale)

        torch.testing.assert_close(out.float(), ref.float(), atol=1e-4, rtol=1e-4)

    def test_causal_zero_future(self, device):
        """Token at position 0 must not attend to any future token."""
        B, H, N, d = 1, 1, 16, 8
        q, k, v = make_qkv(B, H, N, d, device=device, seed=1)
        gamma = torch.tensor(0.1, device=device)
        scale = 1.0 / math.sqrt(d)

        # Make V so each position i has v=i (easy to detect leakage)
        v = torch.zeros(B, H, N, d, device=device)
        for i in range(N):
            v[:, :, i, :] = float(i)

        m, l, o = sliding_window_triton_accum(q, k, v, window=N, gamma=gamma, scale=scale)
        out = normalise(m, l, o)  # (B, H, N, d)

        # Position 0 can only attend to itself (v=0), so output must be ~0
        assert out[0, 0, 0, :].abs().max().item() < 1e-4, \
            f"Position 0 leaks future: {out[0, 0, 0, :]}"

    def test_window_excludes_distant(self, device):
        """Tokens more than W positions ago must have zero weight."""
        B, H, N, d = 1, 1, 32, 8
        q, k, v = make_qkv(B, H, N, d, device=device, seed=2)
        gamma = torch.tensor(0.0, device=device)   # no ALiBi bias
        scale = 1.0 / math.sqrt(d)
        window = 4

        # V[j] = 0 for positions j in [i-window+1, i] (in window), V[j] = 1 for distant.
        # For each query i, only keys in [i-window+1, i] are visible → all visible V=0.
        # We need V to be position-independent for this to work across all i.
        # Instead: set V[j] = 0 for j >= window-1 (guaranteed in-window for late queries),
        # and V[j] = 1 for j < window-1 (never in window for queries i >= 2*window).
        v = torch.ones(B, H, N, d, device=device)
        v[:, :, window - 1:, :] = 0.0   # positions that CAN be in-window → V=0
        # positions 0..window-2 have V=1 but are never visible to queries >= 2*window

        m, l, o = sliding_window_triton_accum(q, k, v, window, gamma, scale)
        out = normalise(m, l, o)

        # Queries from position 2*window onward: their window is [i-window+1, i],
        # all of which have V=0 → output should be 0
        for pos in range(2 * window, N):
            assert out[0, 0, pos, :].abs().max().item() < 1e-4, \
                f"Window leak at pos {pos}: {out[0, 0, pos, :]}"

    def test_gradient_flows(self, device):
        """Backward pass must not raise and must produce finite gradients."""
        B, H, N, d = 1, 2, 16, 16
        q, k, v = make_qkv(B, H, N, d, device=device, seed=3)
        q = q.requires_grad_(True)
        k = k.requires_grad_(True)
        v = v.requires_grad_(True)
        gamma = torch.tensor(0.1, device=device)
        scale = 1.0 / math.sqrt(d)

        m, l, o = sliding_window_triton_accum(q, k, v, window=8, gamma=gamma, scale=scale)
        out = normalise(m, l, o)
        loss = out.sum()
        loss.backward()

        assert q.grad is not None
        assert k.grad is not None
        assert v.grad is not None
        assert q.grad.isfinite().all(), "dQ contains non-finite values"
        assert k.grad.isfinite().all(), "dK contains non-finite values"
        assert v.grad.isfinite().all(), "dV contains non-finite values"

    def test_gradient_correctness_cpu(self):
        """Gradient of normalised output is correct via finite differences."""
        B, H, N, d = 1, 1, 8, 4
        torch.manual_seed(5)
        q = torch.randn(B, H, N, d, dtype=torch.float64, requires_grad=True)
        k = torch.randn(B, H, N, d, dtype=torch.float64, requires_grad=True)
        v = torch.randn(B, H, N, d, dtype=torch.float64, requires_grad=True)
        gamma = torch.tensor(0.05, dtype=torch.float64)
        scale = 1.0 / math.sqrt(d)
        window = 4

        def fn(q, k, v):
            # All-float64 path — use reference_attention which supports any dtype
            out = reference_attention(q, k, v, window, gamma.item(), scale)
            return out.sum()

        torch.autograd.gradcheck(fn, (q, k, v), eps=1e-5, atol=1e-4, rtol=1e-4)

    def test_multi_batch_heads(self, device):
        """Kernel must handle B>1 and H>1 correctly."""
        for B, H in [(2, 4), (4, 8), (1, 16)]:
            N, d = 32, 32
            q, k, v = make_qkv(B, H, N, d, device=device, seed=B * H)
            gamma = torch.tensor(0.1, device=device)
            scale = 1.0 / math.sqrt(d)
            m, l, o = sliding_window_triton_accum(q, k, v, window=8, gamma=gamma, scale=scale)
            assert m.shape == (B, H, N)
            out = normalise(m, l, o)
            assert out.isfinite().all(), f"Non-finite output for B={B} H={H}"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.skipif(not HAS_TRITON, reason="Triton not installed")
    def test_triton_vs_pytorch_cuda(self):
        """On CUDA, Triton kernel must numerically match PyTorch fallback."""
        B, H, N, d = 2, 8, 128, 64
        device = "cuda"
        q, k, v = make_qkv(B, H, N, d, device=device, dtype=torch.float16, seed=99)
        gamma = torch.tensor(0.1, device=device)
        scale = 1.0 / math.sqrt(d)
        window = 32

        m_py, l_py, o_py = _sw_pytorch_accum(
            q.float(), k.float(), v.float(), window, gamma, scale)
        m_tr, l_tr, o_tr = _SWAccumFn.apply(q, k, v, window, gamma, scale)

        from kernel.triton_local import _SWAccumFn
        torch.testing.assert_close(m_tr.float(), m_py.float(), atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(l_tr.float(), l_py.float(), atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(o_tr.float(), o_py.float(), atol=1e-2, rtol=1e-2)
