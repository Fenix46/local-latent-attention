import argparse
import json

import torch

from prototype.incremental_block import IncrementalAttentionBlock


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-length", type=int, default=512)
    parser.add_argument("--generate-tokens", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--local-window", type=int, default=64)
    parser.add_argument("--latent-tokens", type=int, default=16)
    parser.add_argument("--remote-chunk-size", type=int, default=32)
    return parser.parse_args(argv)


@torch.no_grad()
def run_one(mode: str, args: argparse.Namespace) -> dict:
    block = IncrementalAttentionBlock(
        d_model=args.d_model,
        n_heads=args.n_heads,
        local_window=args.local_window,
        latent_tokens=args.latent_tokens,
        remote_chunk_size=args.remote_chunk_size,
        mode=mode,
    )
    state = block.init_state()
    prompt = torch.randn(args.prompt_length, args.d_model)
    state = block.prefill(prompt, state)

    current = torch.randn(args.d_model)
    for _ in range(args.generate_tokens):
        current, state = block.forward_step(current, state)

    return {
        "mode": mode,
        "prompt_length": args.prompt_length,
        "generate_tokens": args.generate_tokens,
        "cache_tokens": state.cache_tokens(),
        "local_tokens": len(state.local_keys),
        "remote_tokens": len(state.remote_keys),
        "pending_tokens": len(state.pending_keys),
        "final_norm": round(current.norm().item(), 6),
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    for mode in ("baseline", "local_latent"):
        print(json.dumps(run_one(mode, args)))


if __name__ == "__main__":
    main()
