import argparse
import json
import time

import torch

from prototype.models import (
    attention_workload,
    build_model,
    estimate_attention_bytes,
    estimate_kv_cache_bytes,
    estimate_parameter_bytes,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context-lengths", type=int, nargs="+", default=[128, 256, 512, 1024])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=256)
    parser.add_argument("--local-window", type=int, default=64)
    parser.add_argument("--latent-tokens", type=int, default=16)
    return parser.parse_args(argv)


@torch.no_grad()
def benchmark_model(model: torch.nn.Module, seq_len: int, batch_size: int, warmup: int, iters: int) -> dict:
    x = torch.randint(0, 128, (batch_size, seq_len), dtype=torch.long)

    for _ in range(warmup):
        _ = model(x)

    start = time.perf_counter()
    for _ in range(iters):
        _ = model(x)
    duration = time.perf_counter() - start

    tokens = batch_size * seq_len * iters
    return {
        "seq_len": seq_len,
        "avg_forward_ms": duration * 1000.0 / iters,
        "tokens_per_sec": tokens / max(duration, 1e-8),
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    head_dim = args.d_model // args.n_heads

    rows = []
    for model_name in ("baseline", "flash_dense", "local_latent"):
        model = build_model(
            model_name,
            vocab_size=128,
            max_seq_len=max(args.context_lengths),
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            d_ff=args.d_ff,
            local_window=args.local_window,
            latent_tokens=args.latent_tokens,
        )
        model.eval()
        params = sum(p.numel() for p in model.parameters())
        parameter_bytes = estimate_parameter_bytes(model)

        for seq_len in args.context_lengths:
            row = {
                "model": model_name,
                "params": params,
                "parameter_mib": round(parameter_bytes / (1024 * 1024), 4),
            }
            row.update(
                benchmark_model(
                    model,
                    seq_len=seq_len,
                    batch_size=args.batch_size,
                    warmup=args.warmup,
                    iters=args.iters,
                )
            )
            row["attention_workload"] = attention_workload(
                model_name,
                seq_len=seq_len,
                local_window=args.local_window,
                latent_tokens=args.latent_tokens,
            )
            row["estimated_attention_score_mib"] = round(
                estimate_attention_bytes(
                    model_name=model_name,
                    batch_size=args.batch_size,
                    seq_len=seq_len,
                    n_heads=args.n_heads,
                    local_window=args.local_window,
                    latent_tokens=args.latent_tokens,
                )
                / (1024 * 1024),
                4,
            )
            row["estimated_kv_cache_mib"] = round(
                estimate_kv_cache_bytes(
                    model_name=model_name,
                    batch_size=args.batch_size,
                    seq_len=seq_len,
                    n_layers=args.n_layers,
                    n_heads=args.n_heads,
                    head_dim=head_dim,
                    local_window=args.local_window,
                    latent_tokens=args.latent_tokens,
                )
                / (1024 * 1024),
                4,
            )
            rows.append(row)
            print(json.dumps(row))


if __name__ == "__main__":
    main()
