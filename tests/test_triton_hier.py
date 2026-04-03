"""
Tests for the Triton compressed-level kernel (triton_hier.py).

Tests:
  1. Output shapes correct
  2. Triton matches PyTorch fallback — forward
  3. Normalised output matches brute-force reference
  4. Causal: queries cannot see chunks in the future
  5. Local window: queries cannot see chunks inside the local window
  6. Gradient flows (backward smoke test)
  7. Multi batch/head
  8. C=0 edge case (no chunks)
"""

from __future__ import annotations
import math
import pytest
import torch

from kernel.triton_hier import compressed_level_triton, _hier_pytorch, HAS_TRITON


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_inputs(B=1, H=2, N=64, d=16, C=8, device="cpu", dtype=torch.float32, seed=0):
    torch.manual_seed(seed)
    q   = torch.randn(B, H, N, d, device=device, dtype=dtype)
    k_c = torch.randn(B, H, C, d, device=device, dtype=dtype)
    v_c = torch.randn(B, H, C, d, device=device, dtype=dtype)
    return q, k_c, v_c


def normalise(m, l, o):
    l_safe = l.clamp(min=1e-8)
    out = o / l_safe.unsqueeze(-1)
    out = out.masked_fill((l == 0).unsqueeze(-1), 0.0)
    return out


def reference_hier(q, k_c, v_c, chunk_size, local_W, gamma_val, scale):
    """Brute-force reference using full (N, C) mask."""
    B, H, N, d = q.shape
    C = k_c.shape[2]
    device, dtype = q.device, q.dtype

    q_pos     = torch.arange(N, device=device)
    c_idx     = torch.arange(C, device=device)
    chunk_end = (c_idx + 1) * chunk_size
    centroid  = c_idx.float() * chunk_size + chunk_size / 2.0

    threshold = (q_pos - local_W + 1).clamp(min=0)
    mask      = chunk_end.unsqueeze(0) <= threshold.unsqueeze(1)   # (N, C)
    dist      = (q_pos.float().unsqueeze(1) - centroid.unsqueeze(0)).abs().to(dtype)

    scores = torch.matmul(q, k_c.transpose(-2, -1)) * scale
    scores = scores - gamma_val * dist.unsqueeze(0).unsqueeze(0)
    scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))

    probs = torch.softmax(scores, dim=-1).nan_to_num(0.0)
    return torch.matmul(probs, v_c)


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCompressedLevel:

    @pytest.fixture(params=["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
    def device(self, request):
        return request.param

    def test_output_shapes(self, device):
        B, H, N, d, C = 2, 4, 64, 32, 8
        q, k_c, v_c = make_inputs(B, H, N, d, C, device=device)
        gamma = torch.tensor(0.1, device=device)
        scale = 1.0 / math.sqrt(d)
        m, l, o = compressed_level_triton(q, k_c, v_c, chunk_size=8, local_W=16,
                                          gamma=gamma, scale=scale)
        assert m.shape == (B, H, N)
        assert l.shape == (B, H, N)
        assert o.shape == (B, H, N, d)

    def test_matches_pytorch_fallback(self, device):
        B, H, N, d, C = 1, 2, 64, 16, 8
        q, k_c, v_c = make_inputs(B, H, N, d, C, device=device, seed=42)
        gamma = torch.tensor(0.05, device=device)
        scale = 1.0 / math.sqrt(d)
        chunk_size, local_W = 8, 16

        m_ref, l_ref, o_ref = _hier_pytorch(q, k_c, v_c, chunk_size, local_W, gamma, scale)
        m_tri, l_tri, o_tri = compressed_level_triton(q, k_c, v_c, chunk_size, local_W, gamma, scale)

        tol = dict(atol=1e-3, rtol=1e-3) if device == "cuda" else dict(atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(m_tri.float(), m_ref.float(), **tol)
        torch.testing.assert_close(l_tri.float(), l_ref.float(), **tol)
        torch.testing.assert_close(o_tri.float(), o_ref.float(), **tol)

    def test_normalised_matches_reference(self, device):
        B, H, N, d, C = 1, 2, 64, 16, 8
        q, k_c, v_c = make_inputs(B, H, N, d, C, device=device, seed=7)
        gamma = torch.tensor(0.1, device=device)
        scale = 1.0 / math.sqrt(d)
        chunk_size, local_W = 8, 16

        m, l, o = compressed_level_triton(q, k_c, v_c, chunk_size, local_W, gamma, scale)
        out = normalise(m, l, o)
        ref = reference_hier(q, k_c, v_c, chunk_size, local_W, 0.1, scale)

        torch.testing.assert_close(out.float(), ref.float(), atol=1e-4, rtol=1e-4)

    def test_causal_no_future_chunks(self, device):
        """Query at position 0 must attend to no chunk (all chunks are in the future)."""
        B, H, N, d, C = 1, 1, 64, 8, 8
        q, k_c, v_c = make_inputs(B, H, N, d, C, device=device, seed=1)
        gamma = torch.tensor(0.0, device=device)
        scale = 1.0 / math.sqrt(d)
        chunk_size, local_W = 8, 8

        # Set V_c = 1 everywhere — if any chunk leaks, output will be nonzero
        v_c = torch.ones(B, H, C, d, device=device)

        m, l, o = compressed_level_triton(q, k_c, v_c, chunk_size, local_W, gamma, scale)
        out = normalise(m, l, o)

        # Position 0: threshold = max(0 - 8 + 1, 0) = 0
        # chunk_end[0] = 8 > 0 → no valid chunk
        assert out[0, 0, 0, :].abs().max().item() < 1e-4, \
            f"Position 0 should have no valid chunk: {out[0, 0, 0, :]}"

    def test_local_window_excluded(self, device):
        """Chunks inside the local window must be masked out."""
        B, H, N, d = 1, 1, 128, 8
        chunk_size, local_W = 8, 32
        C = N // chunk_size   # 16 chunks
        q, k_c, _ = make_inputs(B, H, N, d, C, device=device, seed=2)
        gamma = torch.tensor(0.0, device=device)
        scale = 1.0 / math.sqrt(d)

        # V_c[c] = c+1 so we can detect which chunks contribute
        v_c = torch.zeros(B, H, C, d, device=device)
        for c in range(C):
            v_c[:, :, c, :] = float(c + 1)

        m, l, o = compressed_level_triton(q, k_c, v_c, chunk_size, local_W, gamma, scale)
        out = normalise(m, l, o)

        # For query at position N-1=127:
        # threshold = 127 - 32 + 1 = 96
        # valid chunks: chunk_end <= 96 → c+1 <= 12 → c <= 11 (chunks 0..11)
        # chunks 12..15 are inside the window or future → must be masked
        # So valid V values are 1..12, output mean ≈ 6.5
        # Chunks 13..16 (V=13..16) must NOT contribute
        # Check: output at pos 127 should be <= 12 (not contaminated by 13-16)
        out_val = out[0, 0, N - 1, 0].item()
        assert out_val <= 12.5, f"Window leak: output={out_val:.2f} > 12.5"

    def test_gradient_flows(self, device):
        B, H, N, d, C = 1, 2, 32, 16, 4
        q, k_c, v_c = make_inputs(B, H, N, d, C, device=device, seed=3)
        q   = q.requires_grad_(True)
        k_c = k_c.requires_grad_(True)
        v_c = v_c.requires_grad_(True)
        gamma = torch.tensor(0.1, device=device)
        scale = 1.0 / math.sqrt(d)

        m, l, o = compressed_level_triton(q, k_c, v_c, chunk_size=8, local_W=8,
                                          gamma=gamma, scale=scale)
        out = normalise(m, l, o)
        out.sum().backward()

        assert q.grad is not None and q.grad.isfinite().all()
        assert k_c.grad is not None and k_c.grad.isfinite().all()
        assert v_c.grad is not None and v_c.grad.isfinite().all()

    def test_multi_batch_heads(self, device):
        for B, H in [(2, 4), (4, 8)]:
            N, d, C = 64, 32, 8
            q, k_c, v_c = make_inputs(B, H, N, d, C, device=device, seed=B * H)
            gamma = torch.tensor(0.1, device=device)
            scale = 1.0 / math.sqrt(d)
            m, l, o = compressed_level_triton(q, k_c, v_c, chunk_size=8, local_W=16,
                                              gamma=gamma, scale=scale)
            assert m.shape == (B, H, N)
            assert normalise(m, l, o).isfinite().all()

    def test_empty_chunks(self, device):
        """C=0 must return zero output without crashing."""
        B, H, N, d = 1, 2, 32, 16
        q = torch.randn(B, H, N, d, device=device)
        k_c = torch.zeros(B, H, 0, d, device=device)
        v_c = torch.zeros(B, H, 0, d, device=device)
        gamma = torch.tensor(0.1, device=device)
        scale = 1.0 / math.sqrt(d)
        m, l, o = compressed_level_triton(q, k_c, v_c, chunk_size=8, local_W=8,
                                          gamma=gamma, scale=scale)
        assert o.abs().max().item() == 0.0
