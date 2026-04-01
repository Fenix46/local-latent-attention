import argparse
import json
import math
import time
from dataclasses import dataclass

import torch

from prototype.models import build_model, estimate_attention_bytes, estimate_kv_cache_bytes


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-lengths", type=int, nargs="+", default=[128, 256, 512, 1024])
    parser.add_argument("--generate-tokens", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=256)
    parser.add_argument("--local-window", type=int, default=64)
    parser.add_argument("--latent-tokens", type=int, default=16)
    parser.add_argument("--remote-chunk-size", type=int, default=32)
    parser.add_argument("--timing-iters", type=int, default=3)
    return parser.parse_args(argv)


def bytes_to_mib(value: int) -> float:
    return value / (1024 * 1024)


@dataclass
class DecodeSimulationSummary:
    peak_kv_bytes: int
    peak_score_bytes: int
    total_score_elements: int
    final_cache_tokens: int
    compressed_remote_tokens: int
    evicted_raw_tokens: int


class BaselineDecodeCache:
    def __init__(self) -> None:
        self.tokens = 0

    def seed(self, prompt_len: int) -> None:
        self.tokens = prompt_len

    def score_elements(self, n_heads: int) -> int:
        return n_heads * self.tokens

    def cache_tokens(self) -> int:
        return self.tokens

    def append(self) -> None:
        self.tokens += 1


class LocalLatentDecodeCache:
    def __init__(self, local_window: int, latent_tokens: int, remote_chunk_size: int) -> None:
        self.local_window = local_window
        self.latent_tokens = latent_tokens
        self.remote_chunk_size = remote_chunk_size
        self.local_tokens = 0
        self.remote_chunks = 0
        self.pending_remote_tokens = 0
        self.evicted_raw_tokens = 0

    def _fold_remote_token(self) -> None:
        self.pending_remote_tokens += 1
        self.evicted_raw_tokens += 1
        if self.pending_remote_tokens >= self.remote_chunk_size:
            self.pending_remote_tokens = 0
            if self.latent_tokens > 0:
                self.remote_chunks = min(self.remote_chunks + 1, self.latent_tokens)

    def seed(self, prompt_len: int) -> None:
        self.local_tokens = min(prompt_len, self.local_window)
        raw_remote_tokens = max(prompt_len - self.local_window, 0)
        self.evicted_raw_tokens = raw_remote_tokens
        if self.latent_tokens > 0 and self.remote_chunk_size > 0:
            self.remote_chunks = min(
                math.ceil(raw_remote_tokens / self.remote_chunk_size),
                self.latent_tokens,
            )
            self.pending_remote_tokens = raw_remote_tokens % self.remote_chunk_size
        else:
            self.remote_chunks = 0
            self.pending_remote_tokens = 0

    def score_elements(self, n_heads: int) -> int:
        return n_heads * (self.local_tokens + self.remote_chunks)

    def cache_tokens(self) -> int:
        return self.local_tokens + self.remote_chunks

    def append(self) -> None:
        if self.local_tokens == self.local_window:
            self._fold_remote_token()
        else:
            self.local_tokens += 1


def simulate_decode(
    model_name: str,
    prompt_len: int,
    generate_tokens: int,
    batch_size: int,
    n_layers: int,
    n_heads: int,
    head_dim: int,
    local_window: int,
    latent_tokens: int,
    remote_chunk_size: int,
    bytes_per_scalar: int = 4,
) -> DecodeSimulationSummary:
    if model_name == "baseline":
        cache = BaselineDecodeCache()
    elif model_name == "local_latent":
        cache = LocalLatentDecodeCache(
            local_window=local_window,
            latent_tokens=latent_tokens,
            remote_chunk_size=remote_chunk_size,
        )
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    cache.seed(prompt_len)
    peak_kv_bytes = 0
    peak_score_bytes = 0
    total_score_elements = 0

    for _ in range(generate_tokens):
        step_score_elements = batch_size * cache.score_elements(n_heads=n_heads)
        total_score_elements += step_score_elements
        peak_score_bytes = max(peak_score_bytes, step_score_elements * bytes_per_scalar)

        step_kv_bytes = (
            batch_size
            * n_layers
            * n_heads
            * cache.cache_tokens()
            * head_dim
            * 2
            * bytes_per_scalar
        )
        peak_kv_bytes = max(peak_kv_bytes, step_kv_bytes)
        cache.append()

    return DecodeSimulationSummary(
        peak_kv_bytes=peak_kv_bytes,
        peak_score_bytes=peak_score_bytes,
        total_score_elements=total_score_elements,
        final_cache_tokens=cache.cache_tokens(),
        compressed_remote_tokens=getattr(cache, "remote_chunks", 0),
        evicted_raw_tokens=getattr(cache, "evicted_raw_tokens", 0),
    )


@torch.no_grad()
def time_decode_prefix(model: torch.nn.Module, prompt_len: int, timing_iters: int) -> float:
    model.eval()
    x = torch.randint(0, 128, (1, prompt_len), dtype=torch.long)

    start = time.perf_counter()
    for _ in range(timing_iters):
        _ = model(x)
    duration = time.perf_counter() - start
    return duration * 1000.0 / timing_iters


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    max_len = max(args.prompt_lengths) + args.generate_tokens
    head_dim = args.d_model // args.n_heads

    for model_name in ("baseline", "local_latent"):
        model = build_model(
            model_name,
            vocab_size=128,
            max_seq_len=max_len,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            d_ff=args.d_ff,
            local_window=args.local_window,
            latent_tokens=args.latent_tokens,
        )

        for prompt_len in args.prompt_lengths:
            total_context = prompt_len + args.generate_tokens
            prefill_ms = time_decode_prefix(
                model=model,
                prompt_len=prompt_len,
                timing_iters=args.timing_iters,
            )
            kv_bytes = estimate_kv_cache_bytes(
                model_name=model_name,
                batch_size=args.batch_size,
                seq_len=total_context,
                n_layers=args.n_layers,
                n_heads=args.n_heads,
                head_dim=head_dim,
                local_window=args.local_window,
                latent_tokens=args.latent_tokens,
            )
            score_bytes = estimate_attention_bytes(
                model_name=model_name,
                batch_size=args.batch_size,
                seq_len=total_context,
                n_heads=args.n_heads,
                local_window=args.local_window,
                latent_tokens=args.latent_tokens,
            )
            decode = simulate_decode(
                model_name=model_name,
                prompt_len=prompt_len,
                generate_tokens=args.generate_tokens,
                batch_size=args.batch_size,
                n_layers=args.n_layers,
                n_heads=args.n_heads,
                head_dim=head_dim,
                local_window=args.local_window,
                latent_tokens=args.latent_tokens,
                remote_chunk_size=args.remote_chunk_size,
            )
            row = {
                "model": model_name,
                "prompt_len": prompt_len,
                "generate_tokens": args.generate_tokens,
                "total_context": total_context,
                "prefill_forward_ms": prefill_ms,
                "estimated_kv_cache_mib": round(bytes_to_mib(kv_bytes), 4),
                "estimated_attention_score_mib": round(bytes_to_mib(score_bytes), 4),
                "simulated_decode_peak_kv_mib": round(bytes_to_mib(decode.peak_kv_bytes), 4),
                "simulated_decode_peak_score_mib": round(bytes_to_mib(decode.peak_score_bytes), 6),
                "simulated_decode_total_score_million": round(decode.total_score_elements / 1_000_000, 6),
                "simulated_final_cache_tokens": decode.final_cache_tokens,
                "simulated_compressed_remote_tokens": decode.compressed_remote_tokens,
                "simulated_evicted_raw_tokens": decode.evicted_raw_tokens,
                "remote_chunk_size": args.remote_chunk_size,
            }
            print(json.dumps(row))


if __name__ == "__main__":
    main()
