import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from prototype.incremental_model import build_incremental_model
from prototype.runtime import get_peak_memory_stats, reset_peak_memory, resolve_device
from prototype.tasks import RetrievalDataset, RetrievalTaskConfig


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "local_latent"], default="baseline")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--d-ff", type=int, default=256)
    parser.add_argument("--local-window", type=int, default=64)
    parser.add_argument("--latent-tokens", type=int, default=16)
    parser.add_argument("--remote-chunk-size", type=int, default=32)
    parser.add_argument("--gate-mode", choices=["simple", "improved"], default="simple")
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


@torch.no_grad()
def evaluate_full(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor) -> dict:
    model.eval()
    logits = model(x)
    final_logits = logits[:, -1, :]
    loss = F.cross_entropy(final_logits, y)
    preds = final_logits.argmax(dim=-1)
    return {
        "eval_full_loss": loss.item(),
        "eval_full_acc": (preds == y).float().mean().item(),
        "eval_full_gate_mean": getattr(model, "last_gate_mean", 1.0),
    }


@torch.no_grad()
def evaluate_step(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor) -> dict:
    model.eval()
    logits = []
    for batch_idx in range(x.size(0)):
        state = model.init_state()
        sample_logits = []
        for pos, token_id in enumerate(x[batch_idx]):
            logits_t, state = model.forward_step(token_id, pos, state)
            sample_logits.append(logits_t)
        logits.append(torch.stack(sample_logits, dim=0))
    logits = torch.stack(logits, dim=0)
    final_logits = logits[:, -1, :]
    loss = F.cross_entropy(final_logits, y)
    preds = final_logits.argmax(dim=-1)
    return {
        "eval_step_loss": loss.item(),
        "eval_step_acc": (preds == y).float().mean().item(),
        "eval_step_gate_mean": getattr(model, "last_gate_mean", 1.0),
    }


def append_memory_stats(row: dict, device: torch.device) -> None:
    row.update(get_peak_memory_stats(device))


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)

    dataset = RetrievalDataset(RetrievalTaskConfig(seq_len=args.seq_len))
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
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    metrics = []
    start = time.perf_counter()
    reset_peak_memory(device)
    for step in range(1, args.steps + 1):
        model.train()
        x, y = dataset.sample_batch(batch_size=args.batch_size, device=device)
        logits = model(x)
        final_logits = logits[:, -1, :]
        loss = F.cross_entropy(final_logits, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        row = {
            "step": step,
            "train_loss": loss.item(),
            "train_gate_mean": getattr(model, "last_gate_mean", 1.0),
        }
        if step % args.eval_every == 0 or step == args.steps:
            eval_x, eval_y = dataset.sample_batch(batch_size=args.batch_size, device=device)
            row.update(evaluate_full(model, eval_x, eval_y))
            row.update(evaluate_step(model, eval_x, eval_y))
        append_memory_stats(row, device)
        metrics.append(row)
        if step % args.eval_every == 0 or step == 1 or step == args.steps:
            print(json.dumps(row))

    duration = time.perf_counter() - start
    summary = {
        "mode": args.mode,
        "device": str(device),
        "gate_mode": args.gate_mode,
        "steps": args.steps,
        "seq_len": args.seq_len,
        "params": sum(p.numel() for p in model.parameters()),
        "duration_sec": duration,
        "steps_per_sec": args.steps / max(duration, 1e-8),
        "last": metrics[-1],
    }
    summary.update(get_peak_memory_stats(device))
    print(json.dumps(summary))

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps({"summary": summary, "metrics": metrics}, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
