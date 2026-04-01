import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from prototype.models import build_model
from prototype.runtime import get_peak_memory_stats, reset_peak_memory, resolve_device
from prototype.tasks import (
    ByteTextDataset,
    ByteTextTaskConfig,
    RetrievalDataset,
    RetrievalTaskConfig,
    TokenizedTextDataset,
    TokenizedTextTaskConfig,
)
from prototype.tokenizers import load_text_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["retrieval", "text"], default="retrieval")
    parser.add_argument(
        "--model",
        choices=["baseline", "flash_dense", "local_latent"],
        default="baseline",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lr-warmup-steps", type=int, default=0)
    parser.add_argument("--lr-min", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=0.0)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=256)
    parser.add_argument("--local-window", type=int, default=64)
    parser.add_argument("--local-block-size", type=int, default=256)
    parser.add_argument("--latent-tokens", type=int, default=16)
    parser.add_argument("--latent-d-model", type=int, default=64)
    parser.add_argument("--latent-heads", type=int, default=2)
    parser.add_argument("--latent-query-block-size", type=int, default=0)
    parser.add_argument("--checkpoint-blocks", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--eval-batches", type=int, default=1)
    parser.add_argument("--profile-attention", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--text-path", type=Path, default=None)
    parser.add_argument("--tokenizer-model", type=Path, default=None)
    parser.add_argument("--train-fraction", type=float, default=0.9)
    parser.add_argument("--save-dir", type=Path, default=None)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--save-final", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataset,
    batch_size: int,
    device: torch.device,
    batches: int,
    amp_ctx=None,
) -> dict:
    model.eval()
    totals: dict[str, float] = {}
    ctx = amp_ctx if amp_ctx is not None else torch.amp.autocast(device_type="cuda", enabled=False)
    for _ in range(batches):
        x, y = dataset.sample_batch(batch_size=batch_size, device=device)
        with ctx:
            logits = model(x)
        _, metrics = compute_loss_and_metrics(
            logits,
            y,
            bits_metric_name=getattr(dataset, "bits_metric_name", "eval_bpb"),
        )
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + value
    return {key: value / batches for key, value in totals.items()}


def compute_loss_and_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    bits_metric_name: str = "eval_bpb",
) -> tuple[torch.Tensor, dict[str, float]]:
    if targets.ndim == 1:
        final_logits = logits[:, -1, :]
        loss = F.cross_entropy(final_logits, targets)
        preds = final_logits.argmax(dim=-1)
        acc = (preds == targets).float().mean().item()
        return loss, {"eval_loss": loss.item(), "eval_acc": acc}

    flat_logits = logits.reshape(-1, logits.size(-1))
    flat_targets = targets.reshape(-1)
    loss = F.cross_entropy(flat_logits, flat_targets)
    preds = logits.argmax(dim=-1)
    acc = (preds == targets).float().mean().item()
    perplexity = math.exp(min(loss.item(), 20.0))
    bpb = loss.item() / math.log(2.0)
    return loss, {
        "eval_loss": loss.item(),
        "eval_acc": acc,
        "eval_perplexity": perplexity,
        bits_metric_name: bpb,
    }


def get_lr(step: int, warmup_steps: int, total_steps: int, max_lr: float, min_lr: float) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return max_lr * step / warmup_steps
    if total_steps <= warmup_steps:
        return max_lr
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def build_datasets(args: argparse.Namespace):
    if args.task == "retrieval":
        dataset = RetrievalDataset(RetrievalTaskConfig(seq_len=args.seq_len))
        return dataset, dataset

    if args.text_path is None:
        raise ValueError("--text-path is required when --task text")

    if args.tokenizer_model is None:
        train_dataset = ByteTextDataset(
            ByteTextTaskConfig(
                path=args.text_path,
                seq_len=args.seq_len,
                train_fraction=args.train_fraction,
                split="train",
                seed=args.seed,
            )
        )
        eval_dataset = ByteTextDataset(
            ByteTextTaskConfig(
                path=args.text_path,
                seq_len=args.seq_len,
                train_fraction=args.train_fraction,
                split="eval",
                seed=args.seed + 1,
            )
        )
        return train_dataset, eval_dataset

    tokenizer = load_text_tokenizer(args.tokenizer_model)
    train_dataset = TokenizedTextDataset(
        TokenizedTextTaskConfig(
            path=args.text_path,
            seq_len=args.seq_len,
            train_fraction=args.train_fraction,
            split="train",
            seed=args.seed,
        ),
        tokenizer=tokenizer,
    )
    eval_dataset = TokenizedTextDataset(
        TokenizedTextTaskConfig(
            path=args.text_path,
            seq_len=args.seq_len,
            train_fraction=args.train_fraction,
            split="eval",
            seed=args.seed + 1,
        ),
        tokenizer=tokenizer,
    )
    return train_dataset, eval_dataset


def append_memory_stats(row: dict, device: torch.device) -> None:
    row.update(get_peak_memory_stats(device))


def append_profile_stats(row: dict, model: torch.nn.Module, enabled: bool) -> None:
    if enabled and hasattr(model, "get_profile_stats"):
        row.update(model.get_profile_stats())


def build_run_config(
    args: argparse.Namespace,
    vocab_size: int,
    bits_metric_name: str,
    tokenizer_kind: str,
) -> dict:
    config = {
        "task": args.task,
        "model": args.model,
        "device": args.device,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "lr": args.lr,
        "d_model": args.d_model,
        "n_heads": args.n_heads,
        "n_layers": args.n_layers,
        "d_ff": args.d_ff,
        "local_window": args.local_window,
        "local_block_size": args.local_block_size,
        "latent_tokens": args.latent_tokens,
        "latent_d_model": args.latent_d_model,
        "latent_heads": args.latent_heads,
        "latent_query_block_size": args.latent_query_block_size,
        "checkpoint_blocks": args.checkpoint_blocks,
        "lr_warmup_steps": args.lr_warmup_steps,
        "lr_min": args.lr_min,
        "grad_clip": args.grad_clip,
        "eval_every": args.eval_every,
        "eval_batches": args.eval_batches,
        "profile_attention": args.profile_attention,
        "seed": args.seed,
        "train_fraction": args.train_fraction,
        "vocab_size": vocab_size,
        "bits_metric_name": bits_metric_name,
        "tokenizer_kind": tokenizer_kind,
    }
    if args.text_path is not None:
        config["text_path"] = str(args.text_path)
    if args.tokenizer_model is not None:
        config["tokenizer_model"] = str(args.tokenizer_model)
    return config


def save_checkpoint(
    save_dir: Path,
    filename: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    run_config: dict,
    step: int,
    summary: dict | None,
    metrics: list[dict],
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "config": run_config,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "summary": summary,
        "metrics": metrics,
    }
    torch.save(payload, save_dir / filename)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    device = resolve_device(args.device)
    train_dataset, eval_dataset = build_datasets(args)
    run_config = build_run_config(
        args,
        vocab_size=train_dataset.vocab_size,
        bits_metric_name=getattr(train_dataset, "bits_metric_name", "eval_bpb"),
        tokenizer_kind=getattr(train_dataset, "tokenizer_kind", "byte"),
    )
    model = build_model(
        args.model,
        vocab_size=train_dataset.vocab_size,
        max_seq_len=args.seq_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        local_window=args.local_window,
        local_block_size=args.local_block_size,
        latent_tokens=args.latent_tokens,
        latent_d_model=args.latent_d_model,
        latent_heads=args.latent_heads,
        latent_query_block_size=args.latent_query_block_size,
        checkpoint_blocks=args.checkpoint_blocks,
    ).to(device)
    if args.bf16:
        model = model.to(torch.bfloat16)
    if args.profile_attention and hasattr(model, "set_profile"):
        model.set_profile(True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    amp_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        if args.bf16 and device.type == "cuda"
        else torch.amp.autocast(device_type="cuda", enabled=False)
    )
    metrics = []
    start = time.perf_counter()
    reset_peak_memory(device)

    for step in range(1, args.steps + 1):
        model.train()
        x, y = train_dataset.sample_batch(batch_size=args.batch_size, device=device)
        with amp_ctx:
            logits = model(x)
            loss, _ = compute_loss_and_metrics(
                logits,
                y,
                bits_metric_name=getattr(train_dataset, "bits_metric_name", "eval_bpb"),
            )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        if args.lr_warmup_steps > 0 or args.lr_min != args.lr:
            lr = get_lr(step, args.lr_warmup_steps, args.steps, args.lr, args.lr_min)
            for pg in optimizer.param_groups:
                pg["lr"] = lr
        optimizer.step()

        current_lr = optimizer.param_groups[0]["lr"]
        row = {"step": step, "train_loss": loss.item(), "lr": current_lr}
        if step % args.eval_every == 0 or step == args.steps:
            row.update(evaluate(model, eval_dataset, args.batch_size, device, batches=args.eval_batches, amp_ctx=amp_ctx))
        append_memory_stats(row, device)
        append_profile_stats(row, model, args.profile_attention)
        metrics.append(row)

        if step % args.eval_every == 0 or step == 1 or step == args.steps:
            print(json.dumps(row))

        if args.save_dir is not None and args.save_every > 0 and step % args.save_every == 0:
            save_checkpoint(
                save_dir=args.save_dir,
                filename=f"checkpoint_step_{step}.pt",
                model=model,
                optimizer=optimizer,
                run_config=run_config,
                step=step,
                summary=None,
                metrics=metrics,
            )

    duration = time.perf_counter() - start
    total_params = sum(p.numel() for p in model.parameters())
    summary = {
        "model": args.model,
        "task": args.task,
        "device": str(device),
        "steps": args.steps,
        "seq_len": args.seq_len,
        "params": total_params,
        "duration_sec": duration,
        "steps_per_sec": args.steps / max(duration, 1e-8),
        "last": metrics[-1],
    }
    if args.task == "text" and args.text_path is not None:
        summary["text_path"] = str(args.text_path)
        summary["bits_metric_name"] = getattr(train_dataset, "bits_metric_name", "eval_bpb")
        summary["tokenizer_kind"] = getattr(train_dataset, "tokenizer_kind", "byte")
    if args.tokenizer_model is not None:
        summary["tokenizer_model"] = str(args.tokenizer_model)
    summary.update(get_peak_memory_stats(device))
    append_profile_stats(summary, model, args.profile_attention)
    print(json.dumps(summary))

    if args.save_dir is not None and args.save_final:
        save_checkpoint(
            save_dir=args.save_dir,
            filename="final.pt",
            model=model,
            optimizer=optimizer,
            run_config=run_config,
            step=args.steps,
            summary=summary,
            metrics=metrics,
        )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps({"summary": summary, "metrics": metrics}, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
