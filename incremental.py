import argparse
import json
import math
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class IncrementalConfig:
    d_model: int = 128
    n_heads: int = 4
    local_window: int = 64
    latent_tokens: int = 16
    remote_chunk_size: int = 32

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


class IncrementalBaselineAttention:
    def __init__(self, config: IncrementalConfig) -> None:
        self.config = config
        self.keys: list[torch.Tensor] = []
        self.values: list[torch.Tensor] = []

    def prefill(self, x: torch.Tensor) -> None:
        for token in x.unbind(dim=0):
            self.append(token)

    def append(self, token: torch.Tensor) -> None:
        heads = token.view(self.config.n_heads, self.config.head_dim)
        self.keys.append(heads)
        self.values.append(heads)

    def step(self, query: torch.Tensor) -> tuple[torch.Tensor, int]:
        q = query.view(self.config.n_heads, self.config.head_dim)
        k = torch.stack(self.keys, dim=1)
        v = torch.stack(self.values, dim=1)
        scores = torch.einsum("hd,hnd->hn", q, k) / math.sqrt(self.config.head_dim)
        attn = torch.softmax(scores, dim=-1)
        out = torch.einsum("hn,hnd->hd", attn, v).reshape(self.config.d_model)
        return out, scores.numel()

    def cache_tokens(self) -> int:
        return len(self.keys)


class IncrementalLocalLatentAttention:
    def __init__(self, config: IncrementalConfig) -> None:
        self.config = config
        self.local_keys: list[torch.Tensor] = []
        self.local_values: list[torch.Tensor] = []
        self.remote_keys: list[torch.Tensor] = []
        self.remote_values: list[torch.Tensor] = []
        self.pending: list[torch.Tensor] = []

    def prefill(self, x: torch.Tensor) -> None:
        for token in x.unbind(dim=0):
            self.append(token)

    def _compress_pending(self) -> None:
        if not self.pending:
            return
        stacked = torch.stack(self.pending, dim=0)
        summary = stacked.mean(dim=0)
        if len(self.remote_keys) == self.config.latent_tokens:
            self.remote_keys.pop(0)
            self.remote_values.pop(0)
        self.remote_keys.append(summary)
        self.remote_values.append(summary)
        self.pending.clear()

    def append(self, token: torch.Tensor) -> None:
        heads = token.view(self.config.n_heads, self.config.head_dim)
        self.local_keys.append(heads)
        self.local_values.append(heads)
        if len(self.local_keys) > self.config.local_window:
            self.pending.append(self.local_keys.pop(0))
            self.local_values.pop(0)
            if len(self.pending) >= self.config.remote_chunk_size:
                self._compress_pending()

    def step(self, query: torch.Tensor) -> tuple[torch.Tensor, int]:
        q = query.view(self.config.n_heads, self.config.head_dim)

        local_k = torch.stack(self.local_keys, dim=1)
        local_v = torch.stack(self.local_values, dim=1)
        local_scores = torch.einsum("hd,hnd->hn", q, local_k) / math.sqrt(self.config.head_dim)
        local_attn = torch.softmax(local_scores, dim=-1)
        local_out = torch.einsum("hn,hnd->hd", local_attn, local_v)

        score_count = local_scores.numel()
        if self.remote_keys:
            remote_k = torch.stack(self.remote_keys, dim=1)
            remote_v = torch.stack(self.remote_values, dim=1)
            remote_scores = torch.einsum("hd,hnd->hn", q, remote_k) / math.sqrt(self.config.head_dim)
            remote_attn = torch.softmax(remote_scores, dim=-1)
            remote_out = torch.einsum("hn,hnd->hd", remote_attn, remote_v)
            gate = torch.sigmoid(query.mean()).item()
            out = gate * local_out + (1.0 - gate) * remote_out
            score_count += remote_scores.numel()
        else:
            out = local_out

        return out.reshape(self.config.d_model), score_count

    def cache_tokens(self) -> int:
        return len(self.local_keys) + len(self.remote_keys)

    def remote_tokens(self) -> int:
        return len(self.remote_keys)

    def pending_tokens(self) -> int:
        return len(self.pending)


@torch.no_grad()
def benchmark_incremental(
    model_name: str,
    prompt_len: int,
    generate_tokens: int,
    config: IncrementalConfig,
) -> dict:
    if model_name == "baseline":
        runner = IncrementalBaselineAttention(config)
    elif model_name == "local_latent":
        runner = IncrementalLocalLatentAttention(config)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    prompt = torch.randn(prompt_len, config.d_model)
    runner.prefill(prompt)

    total_scores = 0
    peak_cache = runner.cache_tokens()
    start = time.perf_counter()
    current = torch.randn(config.d_model)
    for _ in range(generate_tokens):
        current, scores = runner.step(current)
        total_scores += scores
        runner.append(current)
        peak_cache = max(peak_cache, runner.cache_tokens())
    elapsed = time.perf_counter() - start

    row = {
        "model": model_name,
        "prompt_len": prompt_len,
        "generate_tokens": generate_tokens,
        "avg_decode_step_ms": elapsed * 1000.0 / max(generate_tokens, 1),
        "peak_cache_tokens": peak_cache,
        "total_score_elements": total_scores,
    }
    if model_name == "local_latent":
        row["remote_tokens"] = runner.remote_tokens()
        row["pending_tokens"] = runner.pending_tokens()
    return row


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-lengths", type=int, nargs="+", default=[128, 512, 1024])
    parser.add_argument("--generate-tokens", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--local-window", type=int, default=64)
    parser.add_argument("--latent-tokens", type=int, default=16)
    parser.add_argument("--remote-chunk-size", type=int, default=32)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = IncrementalConfig(
        d_model=args.d_model,
        n_heads=args.n_heads,
        local_window=args.local_window,
        latent_tokens=args.latent_tokens,
        remote_chunk_size=args.remote_chunk_size,
    )
    for model_name in ("baseline", "local_latent"):
        for prompt_len in args.prompt_lengths:
            row = benchmark_incremental(
                model_name=model_name,
                prompt_len=prompt_len,
                generate_tokens=args.generate_tokens,
                config=config,
            )
            print(json.dumps(row))


if __name__ == "__main__":
    main()
