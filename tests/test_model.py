"""
Tests for the full HierarchicalLM model.

  1. Forward pass shape
  2. Causal guarantee end-to-end
  3. Loss computable and backward works
  4. Parameter count sanity
  5. Weight tying (embedding == lm_head)
"""

import sys
import os
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model import ModelConfig, HierarchicalLM, build_model


# Small config for fast tests
def small_config(**overrides) -> ModelConfig:
    cfg = ModelConfig(
        vocab_size = 256,
        d_model    = 64,
        n_heads    = 4,
        n_layers   = 2,
        d_ff       = 128,
        local_W    = 16,
        chunk_B    = 4,
        n_levels   = 2,
        gamma_init = 0.1,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


# ──────────────────────────────────────────
# Test 1: Forward shape
# ──────────────────────────────────────────

def test_forward_shape():
    cfg = small_config()
    model = build_model(cfg)
    batch, seq = 2, 32
    ids = torch.randint(0, cfg.vocab_size, (batch, seq))
    logits = model(ids)
    assert logits.shape == (batch, seq, cfg.vocab_size), \
        f"Wrong shape: {logits.shape}"
    print(f"  PASS  test_forward_shape  {logits.shape}")


# ──────────────────────────────────────────
# Test 2: Causal guarantee end-to-end
# ──────────────────────────────────────────

def test_causal_end_to_end():
    """
    Perturbing token at position j must not affect logits at i < j.
    """
    cfg = small_config()
    model = build_model(cfg)
    model.eval()

    batch, seq = 1, 32
    ids = torch.randint(0, cfg.vocab_size, (batch, seq))

    with torch.no_grad():
        logits_orig = model(ids)

    perturb_pos = seq // 2
    ids2 = ids.clone()
    ids2[0, perturb_pos] = (ids[0, perturb_pos] + 1) % cfg.vocab_size

    with torch.no_grad():
        logits_pert = model(ids2)

    diff_before = (logits_orig[:, :perturb_pos, :] - logits_pert[:, :perturb_pos, :]).abs().max().item()
    diff_after  = (logits_orig[:, perturb_pos:, :] - logits_pert[:, perturb_pos:, :]).abs().max().item()

    assert diff_before < 1e-5, f"Causal violation: diff before perturb_pos = {diff_before:.2e}"
    assert diff_after  > 1e-4, f"Perturbation had no downstream effect — suspicious"

    print(f"  PASS  test_causal_end_to_end  (before={diff_before:.2e}, after={diff_after:.2e})")


# ──────────────────────────────────────────
# Test 3: Loss and backward
# ──────────────────────────────────────────

def test_loss_and_backward():
    cfg = small_config()
    model = build_model(cfg)

    batch, seq = 2, 24
    ids = torch.randint(0, cfg.vocab_size, (batch, seq))

    # Standard LM loss: predict next token
    inputs  = ids[:, :-1]   # (batch, seq-1)
    targets = ids[:, 1:]    # (batch, seq-1)

    logits = model(inputs)  # (batch, seq-1, vocab_size)
    loss = F.cross_entropy(
        logits.reshape(-1, cfg.vocab_size),
        targets.reshape(-1),
    )

    assert not torch.isnan(loss), "Loss is NaN"
    assert not torch.isinf(loss), "Loss is Inf"

    loss.backward()

    nan_grads = [(n, p) for n, p in model.named_parameters() if p.grad is not None and torch.isnan(p.grad).any()]
    assert len(nan_grads) == 0, f"NaN gradients in: {[n for n,_ in nan_grads]}"

    print(f"  PASS  test_loss_and_backward  (loss={loss.item():.4f})")


# ──────────────────────────────────────────
# Test 4: Parameter count
# ──────────────────────────────────────────

def test_parameter_count():
    cfg = small_config()
    model = build_model(cfg)
    n_params = model.count_parameters()
    # Sanity: should be > 0 and < 100M for small config
    assert n_params > 0
    assert n_params < 100_000_000
    print(f"  PASS  test_parameter_count  ({n_params:,} params)")


# ──────────────────────────────────────────
# Test 5: Weight tying
# ──────────────────────────────────────────

def test_weight_tying():
    cfg = small_config()
    model = build_model(cfg)
    assert model.embedding.weight is model.lm_head.weight, \
        "Embedding and lm_head weights are not tied"
    print(f"  PASS  test_weight_tying")


# ──────────────────────────────────────────
# Run
# ──────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_forward_shape,
        test_causal_end_to_end,
        test_loss_and_backward,
        test_parameter_count,
        test_weight_tying,
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
