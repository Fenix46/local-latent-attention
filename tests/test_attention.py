"""
Tests for HierarchicalAttention kernel.

  1. Equivalence with standard causal softmax (short sequence, L=0)
  2. Causal guarantee: output at i does not depend on tokens j > i
  3. Shape correctness
  4. Gradient flow through gammas
"""

import sys
import os
import math
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from kernel.attention import HierarchicalAttention, MultiHeadHierarchicalAttention


DTYPE = torch.float32
DEVICE = torch.device("cpu")


# ──────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────

def standard_causal_attention(q, k, v):
    """Reference: standard causal softmax attention. (batch, seq, d)"""
    batch, seq, d = q.shape
    scale = 1.0 / math.sqrt(d)
    scores = torch.einsum("bqd,bkd->bqk", q, k) * scale   # (B, seq, seq)
    mask = torch.triu(torch.ones(seq, seq, device=q.device), diagonal=1).bool()
    scores = scores.masked_fill(mask, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    return torch.einsum("bqk,bkd->bqd", attn, v)


# ──────────────────────────────────────────
# Test 1: Equivalence (local only, W >= seq)
# ──────────────────────────────────────────

def test_equivalence_local_only():
    """
    With n_levels=0 and local_W >= seq and gamma=0 (no bias),
    output must match standard causal softmax exactly.
    """
    batch, heads, seq, d = 2, 4, 16, 32

    attn = HierarchicalAttention(
        d=d,
        local_W=seq,
        chunk_B=4,
        n_levels=0,
        gamma_init=0.0,
    )
    with torch.no_grad():
        attn.gammas.fill_(0.0)

    # HierarchicalAttention expects (batch, heads, seq, d)
    q = torch.randn(batch, heads, seq, d)
    k = torch.randn(batch, heads, seq, d)
    v = torch.randn(batch, heads, seq, d)

    # Reference: apply standard causal attention per head
    ref_heads = []
    for h in range(heads):
        ref_heads.append(standard_causal_attention(q[:, h], k[:, h], v[:, h]))
    ref = torch.stack(ref_heads, dim=1)   # (batch, heads, seq, d)

    out = attn(q, k, v)

    max_diff = (out - ref).abs().max().item()
    assert max_diff < 1e-4, f"Equivalence failed: max diff = {max_diff:.2e}"
    print(f"  PASS  test_equivalence_local_only  (max_diff={max_diff:.2e})")


# ──────────────────────────────────────────
# Test 2: Causal guarantee
# ──────────────────────────────────────────

def test_causal_local_only():
    """
    Perturbing token at position j must not affect output at positions i < j.
    Tested with n_levels=0 (local only).
    """
    batch, heads, seq, d = 1, 2, 32, 16

    attn = HierarchicalAttention(d=d, local_W=seq, chunk_B=4, n_levels=0, gamma_init=0.0)
    with torch.no_grad():
        attn.gammas.fill_(0.0)

    q = torch.randn(batch, heads, seq, d)
    k = torch.randn(batch, heads, seq, d)
    v = torch.randn(batch, heads, seq, d)

    out_orig = attn(q, k, v).detach()

    perturb_pos = seq // 2
    k2 = k.clone(); v2 = v.clone()
    k2[:, :, perturb_pos, :] += 100.0
    v2[:, :, perturb_pos, :] += 100.0

    out_perturbed = attn(q, k2, v2).detach()

    diff_before = (out_orig[:, :, :perturb_pos, :] - out_perturbed[:, :, :perturb_pos, :]).abs().max().item()
    assert diff_before < 1e-5, f"Causal violation: positions before {perturb_pos} changed by {diff_before:.2e}"

    diff_after = (out_orig[:, :, perturb_pos:, :] - out_perturbed[:, :, perturb_pos:, :]).abs().max().item()
    assert diff_after > 1e-3, f"Perturbation had no effect after pos {perturb_pos} — suspicious"

    print(f"  PASS  test_causal_local_only  (before_diff={diff_before:.2e}, after_diff={diff_after:.2e})")


def test_causal_with_levels():
    """
    Causal guarantee must hold even with compressed levels.
    """
    batch, heads, seq, d = 1, 2, 64, 16

    attn = HierarchicalAttention(d=d, local_W=8, chunk_B=4, n_levels=2, gamma_init=0.05)

    q = torch.randn(batch, heads, seq, d)
    k = torch.randn(batch, heads, seq, d)
    v = torch.randn(batch, heads, seq, d)

    out_orig = attn(q, k, v).detach()

    perturb_pos = seq // 2
    k2 = k.clone(); v2 = v.clone()
    k2[:, :, perturb_pos, :] += 100.0
    v2[:, :, perturb_pos, :] += 100.0

    out_perturbed = attn(q, k2, v2).detach()

    diff_before = (out_orig[:, :, :perturb_pos, :] - out_perturbed[:, :, :perturb_pos, :]).abs().max().item()
    assert diff_before < 1e-5, f"Causal violation with levels: positions before {perturb_pos} changed by {diff_before:.2e}"

    print(f"  PASS  test_causal_with_levels  (before_diff={diff_before:.2e})")


# ──────────────────────────────────────────
# Test 3: Shape correctness
# ──────────────────────────────────────────

def test_output_shape_single_head():
    batch, heads, seq, d = 3, 4, 48, 32
    attn = HierarchicalAttention(d=d, local_W=16, chunk_B=4, n_levels=2)
    q = torch.randn(batch, heads, seq, d)
    k = torch.randn(batch, heads, seq, d)
    v = torch.randn(batch, heads, seq, d)
    out = attn(q, k, v)
    assert out.shape == (batch, heads, seq, d), f"Wrong shape: {out.shape}"
    print(f"  PASS  test_output_shape_single_head  {out.shape}")


def test_output_shape_multihead():
    batch, seq, d_model, n_heads = 2, 32, 64, 4
    attn = MultiHeadHierarchicalAttention(
        d_model=d_model, n_heads=n_heads, local_W=8, chunk_B=4, n_levels=2
    )
    x = torch.randn(batch, seq, d_model)
    out = attn(x)
    assert out.shape == (batch, seq, d_model), f"Wrong shape: {out.shape}"
    print(f"  PASS  test_output_shape_multihead  {out.shape}")


# ──────────────────────────────────────────
# Test 4: Gradient flow
# ──────────────────────────────────────────

def test_gradient_flow_gammas():
    """
    Gradients must flow back to gamma parameters.
    """
    batch, heads, seq, d = 1, 2, 32, 16
    attn = HierarchicalAttention(d=d, local_W=8, chunk_B=4, n_levels=2, gamma_init=0.1)

    q = torch.randn(batch, heads, seq, d)
    k = torch.randn(batch, heads, seq, d)
    v = torch.randn(batch, heads, seq, d)

    out = attn(q, k, v)
    loss = out.sum()
    loss.backward()

    assert attn.gammas.grad is not None, "No gradient on gammas"
    assert not torch.isnan(attn.gammas.grad).any(), "NaN gradient on gammas"
    print(f"  PASS  test_gradient_flow_gammas  (gamma_grad={attn.gammas.grad.tolist()})")


def test_gradient_flow_multihead():
    """
    Gradients must flow to all parameters in MultiHeadHierarchicalAttention.
    """
    batch, seq, d_model, n_heads = 1, 24, 32, 2
    attn = MultiHeadHierarchicalAttention(
        d_model=d_model, n_heads=n_heads, local_W=8, chunk_B=4, n_levels=2
    )
    x = torch.randn(batch, seq, d_model)
    out = attn(x)
    loss = out.sum()
    loss.backward()

    for name, param in attn.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"

    print(f"  PASS  test_gradient_flow_multihead")


# ──────────────────────────────────────────
# Run
# ──────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_equivalence_local_only,
        test_causal_local_only,
        test_causal_with_levels,
        test_output_shape_single_head,
        test_output_shape_multihead,
        test_gradient_flow_gammas,
        test_gradient_flow_multihead,
    ]
    for t in tests:
        try:
            t()
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
        except Exception as e:
            import traceback
            print(f"  ERROR {t.__name__}: {e}")
            traceback.print_exc()
