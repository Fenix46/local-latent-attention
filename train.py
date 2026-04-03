"""
Training loop for HierarchicalLM.

Usage:
  python train.py --data path/to/text.txt --steps 10000

Features:
  - Cosine LR schedule with linear warmup
  - Gradient clipping
  - Checkpoint save/resume
  - BF16 mixed precision (optional)
  - JSON log per step
"""

from __future__ import annotations
import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from model import ModelConfig, HierarchicalLM, build_model
from data import BinaryDataset


# ──────────────────────────────────────────
# Data
# ──────────────────────────────────────────

class TextDataset:
    """
    Simple character/byte-level dataset from a raw text file.
    Encodes as UTF-8 bytes (vocab_size=256).
    """
    def __init__(self, path: Path, seq_len: int, split: str = "train", train_fraction: float = 0.9) -> None:
        data = path.read_bytes()
        n = int(len(data) * train_fraction)
        raw = data[:n] if split == "train" else data[n:]
        self.data = torch.tensor(list(raw), dtype=torch.long)
        self.seq_len = seq_len
        self.vocab_size = 256

    def sample_batch(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        ix = torch.randint(0, len(self.data) - self.seq_len - 1, (batch_size,))
        x = torch.stack([self.data[i : i + self.seq_len] for i in ix]).to(device)
        y = torch.stack([self.data[i + 1 : i + self.seq_len + 1] for i in ix]).to(device)
        return x, y


class TokenizedDataset:
    """
    Pre-tokenized dataset from a .pt file containing a 1D LongTensor of token ids.
    """
    def __init__(self, path: Path, seq_len: int, split: str = "train", train_fraction: float = 0.9, vocab_size: int = 32000) -> None:
        data = torch.load(path, weights_only=True)
        n = int(len(data) * train_fraction)
        self.data = data[:n] if split == "train" else data[n:]
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def sample_batch(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        ix = torch.randint(0, len(self.data) - self.seq_len - 1, (batch_size,))
        x = torch.stack([self.data[i : i + self.seq_len] for i in ix]).to(device)
        y = torch.stack([self.data[i + 1 : i + self.seq_len + 1] for i in ix]).to(device)
        return x, y


# ──────────────────────────────────────────
# LR Schedule
# ──────────────────────────────────────────

def get_lr(step: int, warmup: int, total: int, lr_max: float, lr_min: float) -> float:
    if step < warmup:
        return lr_max * step / max(warmup, 1)
    if step >= total:
        return lr_min
    progress = (step - warmup) / (total - warmup)
    return lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos(math.pi * progress))


# ──────────────────────────────────────────
# Checkpoint
# ──────────────────────────────────────────

def save_checkpoint(path: Path, model: HierarchicalLM, optimizer: torch.optim.Optimizer, step: int, config: ModelConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": config.__dict__,
    }, path)


def load_checkpoint(path: Path, model: HierarchicalLM, optimizer: torch.optim.Optimizer) -> int:
    ckpt = torch.load(path, weights_only=False)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt["step"]


# ──────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────

@torch.no_grad()
def evaluate(model: HierarchicalLM, dataset: TextDataset | TokenizedDataset, batch_size: int, batches: int, device: torch.device, amp_ctx) -> dict:
    model.eval()
    total_loss = 0.0
    for _ in range(batches):
        x, y = dataset.sample_batch(batch_size, device)
        with amp_ctx:
            logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        total_loss += loss.item()
    model.train()
    avg_loss = total_loss / batches
    return {
        "eval_loss": avg_loss,
        "eval_bpb": avg_loss / math.log(2),
        "eval_ppl": math.exp(min(avg_loss, 20.0)),
    }


# ──────────────────────────────────────────
# Args
# ──────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    # Data
    p.add_argument("--data",           type=Path, required=True)
    p.add_argument("--data-format",    choices=["text", "tokenized", "bin"], default="text")
    p.add_argument("--bin-dtype",      choices=["int32", "int16", "uint16"], default="int32")
    p.add_argument("--vocab-size",     type=int, default=256)
    p.add_argument("--train-fraction", type=float, default=0.9)

    # Model
    p.add_argument("--d-model",     type=int,   default=512)
    p.add_argument("--n-heads",     type=int,   default=8)
    p.add_argument("--n-layers",    type=int,   default=6)
    p.add_argument("--d-ff",        type=int,   default=1536)
    p.add_argument("--local-w",     type=int,   default=128)
    p.add_argument("--chunk-b",     type=int,   default=8)
    p.add_argument("--n-levels",    type=int,   default=3)
    p.add_argument("--gamma-init",  type=float, default=0.1)
    p.add_argument("--dropout",     type=float, default=0.0)
    p.add_argument("--grad-checkpoint", action="store_true")
    p.add_argument("--compile",         action="store_true")

    # Training
    p.add_argument("--steps",        type=int,   default=10000)
    p.add_argument("--batch-size",   type=int,   default=8)
    p.add_argument("--seq-len",      type=int,   default=512)
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--lr-min",       type=float, default=3e-5)
    p.add_argument("--warmup",       type=int,   default=200)
    p.add_argument("--grad-clip",    type=float, default=1.0)
    p.add_argument("--bf16",         action="store_true")
    p.add_argument("--device",       default="auto")

    # Eval / Logging
    p.add_argument("--eval-every",   type=int, default=500)
    p.add_argument("--eval-batches", type=int, default=4)
    p.add_argument("--log-every",    type=int, default=50)

    # Checkpointing
    p.add_argument("--save-dir",     type=Path, default=None)
    p.add_argument("--save-every",   type=int,  default=1000)
    p.add_argument("--resume",       type=Path, default=None)

    # Output
    p.add_argument("--output",       type=Path, default=None)

    return p.parse_args()


# ──────────────────────────────────────────
# Main
# ──────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else
                              "mps"  if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(json.dumps({"event": "start", "device": str(device)}))

    # Dataset
    if args.data_format == "text":
        train_ds = TextDataset(args.data, args.seq_len, "train", args.train_fraction)
        eval_ds  = TextDataset(args.data, args.seq_len, "eval",  args.train_fraction)
        vocab_size = 256
    elif args.data_format == "bin":
        train_ds = BinaryDataset(args.data, args.seq_len, args.bin_dtype, "train", args.train_fraction, args.vocab_size)
        eval_ds  = BinaryDataset(args.data, args.seq_len, args.bin_dtype, "eval",  args.train_fraction, args.vocab_size)
        vocab_size = args.vocab_size
    else:
        train_ds = TokenizedDataset(args.data, args.seq_len, "train", args.train_fraction, args.vocab_size)
        eval_ds  = TokenizedDataset(args.data, args.seq_len, "eval",  args.train_fraction, args.vocab_size)
        vocab_size = args.vocab_size

    # Model
    config = ModelConfig(
        vocab_size  = vocab_size,
        d_model     = args.d_model,
        n_heads     = args.n_heads,
        n_layers    = args.n_layers,
        d_ff        = args.d_ff,
        local_W     = args.local_w,
        chunk_B     = args.chunk_b,
        n_levels    = args.n_levels,
        gamma_init      = args.gamma_init,
        dropout         = args.dropout,
        grad_checkpoint = args.grad_checkpoint,
    )
    model = build_model(config).to(device)
    if args.bf16:
        model = model.to(torch.bfloat16)

    n_params = model.count_parameters()
    print(json.dumps({"event": "model", "params": n_params, "config": config.__dict__}))

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=0.1,
        fused=device.type == "cuda",
    )

    # AMP
    amp_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        if args.bf16 and device.type == "cuda"
        else torch.amp.autocast(device_type=device.type, enabled=False)
    )

    # Resume — must happen BEFORE torch.compile so key names match
    start_step = 0
    if args.resume is not None:
        start_step = load_checkpoint(args.resume, model, optimizer)
        print(json.dumps({"event": "resume", "step": start_step}))

    # torch.compile — after resume so checkpoint keys match uncompiled model
    if args.compile and device.type == "cuda":
        model = torch.compile(model, mode="reduce-overhead")
        print(json.dumps({"event": "compile", "mode": "reduce-overhead"}))

    # Training
    metrics: list[dict] = []
    t0 = time.perf_counter()

    for step in range(start_step + 1, args.steps + 1):
        model.train()

        # LR
        lr = get_lr(step, args.warmup, args.steps, args.lr, args.lr_min)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        x, y = train_ds.sample_batch(args.batch_size, device)

        with amp_ctx:
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if args.grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        optimizer.step()

        row: dict = {
            "step": step,
            "train_loss": loss.item(),
            "train_bpb": loss.item() / math.log(2),
            "lr": lr,
        }

        # Eval
        if step % args.eval_every == 0 or step == args.steps:
            row.update(evaluate(model, eval_ds, args.batch_size, args.eval_batches, device, amp_ctx))

        # Log
        if step % args.log_every == 0 or step == 1 or step == args.steps:
            elapsed = time.perf_counter() - t0
            row["elapsed_sec"] = round(elapsed, 2)
            row["steps_per_sec"] = round(step / max(elapsed, 1e-8), 2)
            print(json.dumps(row))

        metrics.append(row)

        # Checkpoint
        if args.save_dir is not None and args.save_every > 0 and step % args.save_every == 0:
            ckpt_path = args.save_dir / f"step_{step}.pt"
            save_checkpoint(ckpt_path, model, optimizer, step, config)
            print(json.dumps({"event": "checkpoint", "path": str(ckpt_path), "step": step}))

    # Final checkpoint
    if args.save_dir is not None:
        save_checkpoint(args.save_dir / "final.pt", model, optimizer, args.steps, config)

    # Summary
    last = metrics[-1]
    summary = {
        "event":        "done",
        "steps":        args.steps,
        "params":       n_params,
        "final_loss":   last.get("train_loss"),
        "final_bpb":    last.get("train_bpb"),
        "eval_loss":    last.get("eval_loss"),
        "eval_bpb":     last.get("eval_bpb"),
        "eval_ppl":     last.get("eval_ppl"),
        "elapsed_sec":  round(time.perf_counter() - t0, 2),
    }
    print(json.dumps(summary))

    # Save full metrics
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps({"config": config.__dict__, "summary": summary, "metrics": metrics}, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
