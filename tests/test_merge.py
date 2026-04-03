"""
Tests for the ⊕ merge operator.

Three properties must hold exactly:
  1. Associativity:  (A ⊕ B) ⊕ C  ==  A ⊕ (B ⊕ C)
  2. Identity:       A ⊕ empty()   ==  A
  3. Equivalence:    merging states token-by-token == standard softmax
"""

import math
import torch
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from kernel.merge import AttnState, merge, state_from_scores, state_from_single


BATCH = 2
HEADS = 4
D = 32
DTYPE = torch.float32
DEVICE = torch.device("cpu")


def random_state(seq: int) -> tuple[AttnState, torch.Tensor, torch.Tensor]:
    """Returns (state, scores, values) for a random sequence of `seq` tokens."""
    scores = torch.randn(BATCH, HEADS, seq, dtype=DTYPE, device=DEVICE)
    values = torch.randn(BATCH, HEADS, seq, D, dtype=DTYPE, device=DEVICE)
    state = state_from_scores(scores, values)
    return state, scores, values


# ─────────────────────────────────────────────
# Test 1: Associativity
# ─────────────────────────────────────────────

def test_associativity_m():
    """m* is associative because max() is associative."""
    s_a, _, _ = random_state(8)
    s_b, _, _ = random_state(8)
    s_c, _, _ = random_state(8)

    left  = merge(merge(s_a, s_b), s_c)
    right = merge(s_a, merge(s_b, s_c))

    assert torch.allclose(left.m, right.m, atol=1e-6), \
        f"m not associative: max diff {(left.m - right.m).abs().max()}"


def test_associativity_l():
    """l* is associative — the telescoping cancellation of intermediate m."""
    s_a, _, _ = random_state(8)
    s_b, _, _ = random_state(8)
    s_c, _, _ = random_state(8)

    left  = merge(merge(s_a, s_b), s_c)
    right = merge(s_a, merge(s_b, s_c))

    assert torch.allclose(left.l, right.l, atol=1e-5), \
        f"l not associative: max diff {(left.l - right.l).abs().max()}"


def test_associativity_output():
    """Final output o/l is associative."""
    s_a, _, _ = random_state(8)
    s_b, _, _ = random_state(8)
    s_c, _, _ = random_state(8)

    left  = merge(merge(s_a, s_b), s_c).output()
    right = merge(s_a, merge(s_b, s_c)).output()

    assert torch.allclose(left, right, atol=1e-5), \
        f"output not associative: max diff {(left - right).abs().max()}"


# ─────────────────────────────────────────────
# Test 2: Identity element
# ─────────────────────────────────────────────

def test_identity_right():
    """A ⊕ empty == A (up to numerical precision)."""
    s_a, _, _ = random_state(16)
    empty = AttnState.empty(BATCH, HEADS, D, DEVICE, DTYPE)

    result = merge(s_a, empty)

    assert torch.allclose(result.output(), s_a.output(), atol=1e-6), \
        f"right identity failed: max diff {(result.output() - s_a.output()).abs().max()}"


def test_identity_left():
    """empty ⊕ A == A."""
    s_a, _, _ = random_state(16)
    empty = AttnState.empty(BATCH, HEADS, D, DEVICE, DTYPE)

    result = merge(empty, s_a)

    assert torch.allclose(result.output(), s_a.output(), atol=1e-6), \
        f"left identity failed: max diff {(result.output() - s_a.output()).abs().max()}"


# ─────────────────────────────────────────────
# Test 3: Equivalence with standard softmax
# ─────────────────────────────────────────────

def test_equivalence_standard_softmax():
    """
    Merging states token-by-token must give the same result as
    computing softmax over all tokens at once.
    """
    seq = 32
    scores = torch.randn(BATCH, HEADS, seq, dtype=DTYPE, device=DEVICE)
    values = torch.randn(BATCH, HEADS, seq, D, dtype=DTYPE, device=DEVICE)

    # Standard softmax reference
    attn = torch.softmax(scores, dim=-1)                          # (B, H, seq)
    ref  = torch.einsum("bhs,bhsd->bhd", attn, values)           # (B, H, D)

    # Our merge: accumulate token by token
    state = AttnState.empty(BATCH, HEADS, D, DEVICE, DTYPE)
    for t in range(seq):
        s_t = state_from_single(scores[:, :, t], values[:, :, t, :])
        state = merge(state, s_t)

    out = state.output()

    assert torch.allclose(out, ref, atol=1e-5), \
        f"token-by-token merge != softmax: max diff {(out - ref).abs().max()}"


def test_equivalence_block_merge():
    """
    Splitting sequence into blocks and merging states must give
    the same result as computing softmax over the full sequence.
    """
    seq = 64
    block = 16
    scores = torch.randn(BATCH, HEADS, seq, dtype=DTYPE, device=DEVICE)
    values = torch.randn(BATCH, HEADS, seq, D, dtype=DTYPE, device=DEVICE)

    # Reference
    attn = torch.softmax(scores, dim=-1)
    ref  = torch.einsum("bhs,bhsd->bhd", attn, values)

    # Block merge
    state = AttnState.empty(BATCH, HEADS, D, DEVICE, DTYPE)
    for start in range(0, seq, block):
        end = start + block
        s_block = state_from_scores(scores[:, :, start:end], values[:, :, start:end, :])
        state = merge(state, s_block)

    out = state.output()

    assert torch.allclose(out, ref, atol=1e-5), \
        f"block merge != softmax: max diff {(out - ref).abs().max()}"


def test_equivalence_tree_merge():
    """
    Tree-structured merge (parallel reduction) must give the same
    result as linear merge and standard softmax.
    """
    seq = 64
    block = 8
    scores = torch.randn(BATCH, HEADS, seq, dtype=DTYPE, device=DEVICE)
    values = torch.randn(BATCH, HEADS, seq, D, dtype=DTYPE, device=DEVICE)

    # Reference
    attn = torch.softmax(scores, dim=-1)
    ref  = torch.einsum("bhs,bhsd->bhd", attn, values)

    # Build leaf states
    states = []
    for start in range(0, seq, block):
        end = start + block
        s = state_from_scores(scores[:, :, start:end], values[:, :, start:end, :])
        states.append(s)

    # Tree reduction
    while len(states) > 1:
        next_level = []
        for i in range(0, len(states), 2):
            if i + 1 < len(states):
                next_level.append(merge(states[i], states[i + 1]))
            else:
                next_level.append(states[i])
        states = next_level

    out = states[0].output()

    assert torch.allclose(out, ref, atol=1e-5), \
        f"tree merge != softmax: max diff {(out - ref).abs().max()}"


# ─────────────────────────────────────────────
# Test 4: Numerical stability
# ─────────────────────────────────────────────

def test_stability_large_scores():
    """Merge should not overflow with large scores."""
    seq = 16
    scores = torch.randn(BATCH, HEADS, seq) * 100  # very large scores
    values = torch.randn(BATCH, HEADS, seq, D)

    state = state_from_scores(scores, values)
    out = state.output()

    assert not torch.isnan(out).any(), "NaN in output with large scores"
    assert not torch.isinf(out).any(), "Inf in output with large scores"


def test_stability_merge_extreme_diff():
    """Merging states with very different m values should be stable."""
    s_a, _, _ = random_state(8)
    s_b, _, _ = random_state(8)

    # force extreme difference in m
    s_a.m = s_a.m + 500.0
    s_b.m = s_b.m - 500.0

    result = merge(s_a, s_b)
    out = result.output()

    assert not torch.isnan(out).any(), "NaN when merging extreme m values"
    assert not torch.isinf(out).any(), "Inf when merging extreme m values"


if __name__ == "__main__":
    tests = [
        test_associativity_m,
        test_associativity_l,
        test_associativity_output,
        test_identity_right,
        test_identity_left,
        test_equivalence_standard_softmax,
        test_equivalence_block_merge,
        test_equivalence_tree_merge,
        test_stability_large_scores,
        test_stability_merge_extreme_diff,
    ]
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
        except Exception as e:
            print(f"  ERROR {t.__name__}: {e}")
