"""
Hierarchical Attention Merge Operator

Core mathematical primitive: the ⊕ operator that merges two attention states.

State: (m, l, o) where
  m  : scalar  - running maximum of attention scores
  l  : scalar  - sum of exp(score - m), i.e. the softmax denominator
  o  : (d,)    - weighted sum of values, i.e. the softmax numerator

The merge is exact (not approximate) and associative:
  (s_A ⊕ s_B) ⊕ s_C  ==  s_A ⊕ (s_B ⊕ s_C)

This is the foundation of the hierarchical kernel.
"""

from __future__ import annotations
from dataclasses import dataclass
import torch


@dataclass
class AttnState:
    """
    Attention accumulator state for a set of (key, value) pairs.

    m : (batch, heads)        running max of scores
    l : (batch, heads)        softmax denominator
    o : (batch, heads, d)     weighted value sum (numerator)
    """
    m: torch.Tensor
    l: torch.Tensor
    o: torch.Tensor

    @staticmethod
    def empty(batch: int, heads: int, d: int, device: torch.device, dtype: torch.dtype) -> "AttnState":
        """Identity element for ⊕ — represents 'no tokens seen'."""
        return AttnState(
            m=torch.full((batch, heads), float("-inf"), device=device, dtype=dtype),
            l=torch.zeros((batch, heads), device=device, dtype=dtype),
            o=torch.zeros((batch, heads, d), device=device, dtype=dtype),
        )

    def output(self) -> torch.Tensor:
        """
        Recover the final attention output: o / l

        Returns: (batch, heads, d)
        """
        # l can be zero if no valid tokens were seen (all -inf scores)
        # safe_l avoids division by zero
        safe_l = self.l.clamp(min=1e-8).unsqueeze(-1)
        return self.o / safe_l


def merge(s_a: AttnState, s_b: AttnState) -> AttnState:
    """
    Merge two attention states: s_a ⊕ s_b

    Mathematically exact — equivalent to having computed attention
    over the union of tokens in s_a and s_b jointly.

    Derivation:
      m* = max(m_a, m_b)
      l* = l_a * exp(m_a - m*) + l_b * exp(m_b - m*)
      o* = o_a * exp(m_a - m*) + o_b * exp(m_b - m*)

    The exp(m_x - m*) terms are the rescaling factors that correct
    for the fact that m_a and m_b were computed independently.
    """
    m_star = torch.maximum(s_a.m, s_b.m)

    # (batch, heads) -> (batch, heads, 1) for broadcasting with o
    alpha_a = torch.exp(s_a.m - m_star)
    alpha_b = torch.exp(s_b.m - m_star)

    l_star = s_a.l * alpha_a + s_b.l * alpha_b
    o_star = s_a.o * alpha_a.unsqueeze(-1) + s_b.o * alpha_b.unsqueeze(-1)

    return AttnState(m=m_star, l=l_star, o=o_star)


def state_from_scores(
    scores: torch.Tensor,
    values: torch.Tensor,
) -> AttnState:
    """
    Build an AttnState from raw (pre-softmax) scores and values.

    Args:
      scores : (batch, heads, seq)     raw attention scores Q·Kᵀ/√d
      values : (batch, heads, seq, d)  value vectors

    Returns: AttnState with m, l, o computed over the seq dimension.
    """
    m = scores.max(dim=-1).values                        # (batch, heads)
    exp_s = torch.exp(scores - m.unsqueeze(-1))          # (batch, heads, seq)
    l = exp_s.sum(dim=-1)                                # (batch, heads)
    o = torch.einsum("bhs,bhsd->bhd", exp_s, values)    # (batch, heads, d)
    return AttnState(m=m, l=l, o=o)


def state_from_single(
    score: torch.Tensor,
    value: torch.Tensor,
) -> AttnState:
    """
    Build an AttnState from a single score and value.

    Args:
      score : (batch, heads)     single attention score
      value : (batch, heads, d)  single value vector

    Returns: AttnState representing one token.
    """
    m = score                                   # (batch, heads)
    l = torch.ones_like(score)                  # exp(score - score) = 1
    o = value.clone()                           # 1 * value
    return AttnState(m=m, l=l, o=o)
