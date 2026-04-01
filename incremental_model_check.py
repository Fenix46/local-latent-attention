import argparse
import json

import torch

from prototype.incremental_model import build_incremental_model


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "local_latent"], default="baseline")
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--d-ff", type=int, default=128)
    parser.add_argument("--local-window", type=int, default=16)
    parser.add_argument("--latent-tokens", type=int, default=8)
    parser.add_argument("--remote-chunk-size", type=int, default=8)
    parser.add_argument("--gate-mode", choices=["simple", "improved"], default="simple")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


@torch.no_grad()
def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    torch.manual_seed(args.seed)

    model = build_incremental_model(
        vocab_size=128,
        max_seq_len=args.seq_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        local_window=args.local_window,
        latent_tokens=args.latent_tokens,
        remote_chunk_size=args.remote_chunk_size,
        mode=args.mode,
        gate_mode=args.gate_mode,
    )
    model.eval()

    input_ids = torch.randint(0, 128, (1, args.seq_len), dtype=torch.long)
    full_logits = model(input_ids)

    state = model.init_state()
    step_logits = []
    for pos, token_id in enumerate(input_ids[0]):
        logits_t, state = model.forward_step(token_id, pos, state)
        step_logits.append(logits_t)
    step_logits = torch.stack(step_logits, dim=0).unsqueeze(0)

    max_diff = (full_logits - step_logits).abs().max().item()
    cache_tokens = [layer_state.cache_tokens() for layer_state in state]
    print(
        json.dumps(
            {
                "mode": args.mode,
                "gate_mode": args.gate_mode,
                "seq_len": args.seq_len,
                "max_abs_diff": round(max_diff, 8),
                "final_layer_cache_tokens": cache_tokens,
            }
        )
    )


if __name__ == "__main__":
    main()
