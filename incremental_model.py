from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from prototype.incremental_block import IncrementalAttentionBlock, IncrementalCacheState
from prototype.models import RMSNorm


@dataclass
class IncrementalModelConfig:
    vocab_size: int = 128
    max_seq_len: int = 1024
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 256
    local_window: int = 64
    latent_tokens: int = 16
    remote_chunk_size: int = 32
    mode: str = "local_latent"
    gate_mode: str = "simple"


class IncrementalFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.gelu(self.w1(x)))


class IncrementalDecoderLayer(nn.Module):
    def __init__(self, config: IncrementalModelConfig) -> None:
        super().__init__()
        self.norm1 = RMSNorm(config.d_model)
        self.norm2 = RMSNorm(config.d_model)
        self.attn = IncrementalAttentionBlock(
            d_model=config.d_model,
            n_heads=config.n_heads,
            local_window=config.local_window,
            latent_tokens=config.latent_tokens,
            remote_chunk_size=config.remote_chunk_size,
            mode=config.mode,
            gate_mode=config.gate_mode,
        )
        self.ff = IncrementalFeedForward(config.d_model, config.d_ff)

    def init_state(self) -> IncrementalCacheState:
        return self.attn.init_state()

    def forward_step(
        self,
        x_t: torch.Tensor,
        state: IncrementalCacheState | None = None,
    ) -> tuple[torch.Tensor, IncrementalCacheState]:
        attn_out, state = self.attn.forward_step(self.norm1(x_t), state)
        x_t = x_t + attn_out
        x_t = x_t + self.ff(self.norm2(x_t))
        return x_t, state


class IncrementalDecoderLM(nn.Module):
    def __init__(self, config: IncrementalModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position = nn.Embedding(config.max_seq_len, config.d_model)
        self.layers = nn.ModuleList(
            [IncrementalDecoderLayer(config) for _ in range(config.n_layers)]
        )
        self.norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight
        self.last_gate_mean = 1.0

    def init_state(self) -> list[IncrementalCacheState]:
        return [layer.init_state() for layer in self.layers]

    def forward_step(
        self,
        token_id: torch.Tensor,
        position: int,
        state: list[IncrementalCacheState] | None = None,
    ) -> tuple[torch.Tensor, list[IncrementalCacheState]]:
        if state is None:
            state = self.init_state()

        x_t = self.embedding(token_id) + self.position.weight[position]
        next_state: list[IncrementalCacheState] = []
        gate_means = []
        for layer, layer_state in zip(self.layers, state):
            x_t, updated = layer.forward_step(x_t, layer_state)
            next_state.append(updated)
            gate_means.append(layer.attn.last_gate_mean)
        x_t = self.norm(x_t)
        logits = self.lm_head(x_t)
        if gate_means:
            self.last_gate_mean = sum(gate_means) / len(gate_means)
        return logits, next_state

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        logits = []
        for batch_idx in range(input_ids.size(0)):
            state = self.init_state()
            sample_logits = []
            for pos, token_id in enumerate(input_ids[batch_idx]):
                step_logits, state = self.forward_step(token_id, pos, state)
                sample_logits.append(step_logits)
            logits.append(torch.stack(sample_logits, dim=0))
        return torch.stack(logits, dim=0)


def build_incremental_model(**kwargs: int | str) -> IncrementalDecoderLM:
    config = IncrementalModelConfig(**kwargs)
    return IncrementalDecoderLM(config)
