import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn


@dataclass
class IncrementalCacheState:
    local_keys: list[torch.Tensor] = field(default_factory=list)
    local_values: list[torch.Tensor] = field(default_factory=list)
    remote_keys: list[torch.Tensor] = field(default_factory=list)
    remote_values: list[torch.Tensor] = field(default_factory=list)
    pending_keys: list[torch.Tensor] = field(default_factory=list)
    pending_values: list[torch.Tensor] = field(default_factory=list)

    def cache_tokens(self) -> int:
        return len(self.local_keys) + len(self.remote_keys)


class IncrementalAttentionBlock(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        local_window: int = 64,
        latent_tokens: int = 16,
        remote_chunk_size: int = 32,
        mode: str = "local_latent",
        gate_mode: str = "simple",
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if mode not in {"baseline", "local_latent"}:
            raise ValueError(f"Unsupported mode: {mode}")
        if gate_mode not in {"simple", "improved"}:
            raise ValueError(f"Unsupported gate_mode: {gate_mode}")

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.local_window = local_window
        self.latent_tokens = latent_tokens
        self.remote_chunk_size = remote_chunk_size
        self.mode = mode
        self.gate_mode = gate_mode

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.gate = nn.Linear(d_model, n_heads)
        self.head_gate = nn.Linear(self.head_dim * 3, 1)
        self.last_gate_mean = 0.0

    def init_state(self) -> IncrementalCacheState:
        return IncrementalCacheState()

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(self.n_heads, self.head_dim)

    def _compress_pending(self, state: IncrementalCacheState) -> None:
        if not state.pending_keys:
            return
        key = torch.stack(state.pending_keys, dim=0).mean(dim=0)
        value = torch.stack(state.pending_values, dim=0).mean(dim=0)
        if len(state.remote_keys) == self.latent_tokens:
            state.remote_keys.pop(0)
            state.remote_values.pop(0)
        state.remote_keys.append(key)
        state.remote_values.append(value)
        state.pending_keys.clear()
        state.pending_values.clear()

    def _append_kv(self, state: IncrementalCacheState, k: torch.Tensor, v: torch.Tensor) -> None:
        if self.mode == "baseline":
            state.local_keys.append(k)
            state.local_values.append(v)
            return

        state.local_keys.append(k)
        state.local_values.append(v)
        if len(state.local_keys) > self.local_window:
            state.pending_keys.append(state.local_keys.pop(0))
            state.pending_values.append(state.local_values.pop(0))
            if len(state.pending_keys) >= self.remote_chunk_size:
                self._compress_pending(state)

    def prefill(self, x: torch.Tensor, state: IncrementalCacheState | None = None) -> IncrementalCacheState:
        state = self.init_state() if state is None else state
        for token in x.unbind(dim=0):
            _, state = self.forward_step(token, state)
        return state

    def _attend(self, q: torch.Tensor, keys: list[torch.Tensor], values: list[torch.Tensor]) -> tuple[torch.Tensor, int]:
        if not keys:
            return torch.zeros_like(q), 0
        k = torch.stack(keys, dim=1)
        v = torch.stack(values, dim=1)
        scores = torch.einsum("hd,hnd->hn", q, k) / math.sqrt(self.head_dim)
        attn = torch.softmax(scores, dim=-1)
        out = torch.einsum("hn,hnd->hd", attn, v)
        return out, scores.numel()

    def _compute_gate(self, x_t: torch.Tensor, q: torch.Tensor, local_out: torch.Tensor, remote_out: torch.Tensor) -> torch.Tensor:
        if self.gate_mode == "simple":
            gate = torch.sigmoid(self.gate(x_t)).unsqueeze(-1)
        else:
            gate_input = torch.cat([q, local_out, remote_out], dim=-1)
            gate = torch.sigmoid(self.head_gate(gate_input))
        self.last_gate_mean = gate.mean().item()
        return gate

    def forward_step(
        self,
        x_t: torch.Tensor,
        state: IncrementalCacheState | None = None,
    ) -> tuple[torch.Tensor, IncrementalCacheState]:
        state = self.init_state() if state is None else state
        q = self._split_heads(self.q_proj(x_t))
        k = self._split_heads(self.k_proj(x_t))
        v = self._split_heads(self.v_proj(x_t))

        local_out, _ = self._attend(q, state.local_keys, state.local_values)
        if self.mode == "baseline":
            output = local_out
        else:
            remote_out, _ = self._attend(q, state.remote_keys, state.remote_values)
            gate = self._compute_gate(x_t, q, local_out, remote_out)
            output = gate * local_out + (1.0 - gate) * remote_out
        if self.mode == "baseline":
            self.last_gate_mean = 1.0

        self._append_kv(state, k, v)
        return self.out_proj(output.reshape(self.d_model)), state
