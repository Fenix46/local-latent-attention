import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

try:
    from prototype.models import build_model
    from prototype.runtime import get_peak_memory_stats, reset_peak_memory, resolve_device
    from prototype.tasks import (
        BinTextDataset,
        BinTextTaskConfig,
        ByteTextDataset,
        ByteTextTaskConfig,
        RetrievalDataset,
        RetrievalTaskConfig,
        TokenizedTextDataset,
        TokenizedTextTaskConfig,
    )
    from prototype.tokenizers import load_text_tokenizer
except ImportError:
    from models import build_model
    from runtime import get_peak_memory_stats, reset_peak_memory, resolve_device
    from tasks import (
        BinTextDataset,
        BinTextTaskConfig,
        ByteTextDataset,
        ByteTextTaskConfig,
        RetrievalDataset,
        RetrievalTaskConfig,
        TokenizedTextDataset,
        TokenizedTextTaskConfig,
    )
    from tokenizers import load_text_tokenizer


MODEL_PRESETS: dict[str, dict[str, int | float | str]] = {
    # Approximate parameter counts assume vocab_size=32000 and seq_len=4096.
    "0.55b": {
        "approx_params_b": 0.543,
        "description": "Balanced 4k-context preset around 0.54B parameters",
        "d_model": 1408,
        "n_heads": 16,
        "n_layers": 22,
        "d_ff": 4096,
        "latent_d_model": 512,
        "latent_heads": 8,
    },
    "0.60b": {
        "approx_params_b": 0.596,
        "description": "Slightly wider 4k-context preset around 0.60B parameters",
        "d_model": 1536,
        "n_heads": 16,
        "n_layers": 22,
        "d_ff": 4096,
        "latent_d_model": 384,
        "latent_heads": 8,
    },
}

PRESET_ARCH_KEYS = (
    "d_model",
    "n_heads",
    "n_layers",
    "d_ff",
    "latent_d_model",
    "latent_heads",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a LocalLatentAttention language model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Task / data ────────────────────────────────────────────────────────
    parser.add_argument("--task", choices=["retrieval", "text", "bin"], default="retrieval",
                        help="'retrieval': synthetic task; 'text': raw text / SentencePiece; "
                             "'bin': pre-tokenized binary file")
    parser.add_argument("--text-path", type=Path, default=None,
                        help="Path to .txt corpus (task=text) or pre-tokenized .bin file (task=bin)")
    parser.add_argument("--tokenizer-model", type=Path, default=None,
                        help="SentencePiece .model file (task=text only)")
    parser.add_argument("--vocab-size", type=int, default=None,
                        help="Vocabulary size for task=bin (must match the tokenizer used to build the file)")
    parser.add_argument("--train-fraction", type=float, default=0.9)

    # ── Model architecture ─────────────────────────────────────────────────
    parser.add_argument("--model-preset", choices=sorted(MODEL_PRESETS), default=None,
                        help="Override the core architecture with a known-good size preset")
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
    parser.add_argument("--checkpoint-blocks", action="store_true",
                        help="Activation checkpointing to reduce VRAM (slower forward, no memory for activations)")
    parser.add_argument("--use-triton-kernel", action="store_true",
                        help="Use fused Triton kernel for LocalLatentAttention (requires Triton + CUDA)")
    parser.add_argument("--allow-unsafe-triton-training", action="store_true",
                        help="Override the safety check and allow the known-incomplete Triton training path")
    parser.add_argument("--torch-compile", action="store_true",
                        help="Wrap the model with torch.compile for the PyTorch attention path")
    parser.add_argument("--compile-mode",
                        choices=["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"],
                        default="default",
                        help="torch.compile mode")
    parser.add_argument("--compile-cudagraphs", action="store_true",
                        help="Enable Inductor cudagraphs for torch.compile. Disabled by default because this model hits overwrite issues.")

    # ── Training ───────────────────────────────────────────────────────────
    parser.add_argument("--steps", type=int, default=200,
                        help="Total optimiser steps (after gradient accumulation)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Micro-batch size per accumulation step")
    parser.add_argument("--accum-steps", type=int, default=1,
                        help="Gradient accumulation steps. Effective batch = batch_size × accum_steps")
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4, help="Peak learning rate")
    parser.add_argument("--lr-min", type=float, default=3e-5,
                        help="Minimum LR at end of cosine decay (default: lr/10)")
    parser.add_argument("--warmup-steps", type=int, default=0,
                        help="Linear warmup steps before cosine decay begins")
    parser.add_argument("--grad-clip", type=float, default=1.0,
                        help="Max gradient norm (0 = disabled)")
    parser.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="float32",
                        help="Training dtype. Use bfloat16 on H100/H200/A100 for 2-3× speedup")
    parser.add_argument("--seed", type=int, default=0)

    # ── Evaluation ─────────────────────────────────────────────────────────
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--eval-batches", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=1,
                        help="Print one training JSON row every N optimisation steps")
    parser.add_argument("--profile-attention", action="store_true")
    parser.add_argument("--debug-first-step", action="store_true",
                        help="Log batch/forward/backward/optimizer timings for the first optimisation step")

    # ── Checkpointing ──────────────────────────────────────────────────────
    parser.add_argument("--save-dir", type=Path, default=None,
                        help="Directory for checkpoint files")
    parser.add_argument("--save-every", type=int, default=0,
                        help="Save a rolling checkpoint every N optimiser steps (0 = disabled)")
    parser.add_argument("--save-final", action="store_true",
                        help="Save final.pt after training completes")
    parser.add_argument("--resume", type=Path, default=None,
                        help="Resume from this checkpoint file (model + optimiser + step)")

    # ── Output ─────────────────────────────────────────────────────────────
    parser.add_argument("--output", type=Path, default=None,
                        help="Write JSON metrics to this file at the end")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")

    args = parser.parse_args()
    apply_model_preset(args, parser)
    return args


def apply_model_preset(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.model_preset is None:
        return

    preset = MODEL_PRESETS[args.model_preset]
    defaults = parser.parse_args([])
    conflicting = [
        key for key in PRESET_ARCH_KEYS
        if getattr(args, key) != getattr(defaults, key)
    ]
    if conflicting:
        raise ValueError(
            "--model-preset cannot be combined with explicit architecture flags for the same fields: "
            + ", ".join("--" + key.replace("_", "-") for key in conflicting)
        )
    for key in PRESET_ARCH_KEYS:
        setattr(args, key, int(preset[key]))


# ── LR schedule ────────────────────────────────────────────────────────────

def cosine_lr(step: int, total_steps: int, warmup_steps: int, lr_max: float, lr_min: float) -> float:
    """Linear warmup then cosine decay.

    Returns the LR scalar for the given step (1-indexed).
    """
    if step <= warmup_steps:
        # Linear warmup from lr_min → lr_max
        return lr_min + (lr_max - lr_min) * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    progress = min(progress, 1.0)
    return lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos(math.pi * progress))


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def mark_compile_step_begin() -> None:
    compiler = getattr(torch, "compiler", None)
    if compiler is not None and hasattr(compiler, "cudagraph_mark_step_begin"):
        compiler.cudagraph_mark_step_begin()


def emit_event(payload: dict) -> None:
    print(json.dumps(payload), flush=True)


# ── Evaluation ─────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataset,
    batch_size: int,
    device: torch.device,
    batches: int,
    autocast_ctx,
    torch_compile: bool = False,
) -> dict:
    model.eval()
    totals: dict[str, float] = {}
    for _ in range(batches):
        x, y = dataset.sample_batch(batch_size=batch_size, device=device)
        if torch_compile:
            mark_compile_step_begin()
        with autocast_ctx:
            logits = model(x)
        _, metrics = compute_loss_and_metrics(
            logits,
            y,
            bits_metric_name=getattr(dataset, "bits_metric_name", "eval_bpt"),
        )
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + value
    return {key: value / batches for key, value in totals.items()}


def compute_loss_and_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    bits_metric_name: str = "eval_bpt",
) -> tuple[torch.Tensor, dict[str, float]]:
    if targets.ndim == 1:
        # Retrieval task: predict at the last position only
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
    bpt = loss.item() / math.log(2.0)
    return loss, {
        "eval_loss": loss.item(),
        "eval_acc": acc,
        "eval_perplexity": perplexity,
        bits_metric_name: bpt,
    }


# ── Dataset factory ────────────────────────────────────────────────────────

def build_datasets(args: argparse.Namespace):
    if args.task == "retrieval":
        dataset = RetrievalDataset(RetrievalTaskConfig(seq_len=args.seq_len))
        return dataset, dataset

    if args.text_path is None:
        raise ValueError("--text-path is required for task=text/bin")

    if args.task == "bin":
        if args.vocab_size is None:
            raise ValueError("--vocab-size is required for task=bin")
        train_ds = BinTextDataset(BinTextTaskConfig(
            path=args.text_path,
            seq_len=args.seq_len,
            vocab_size=args.vocab_size,
            train_fraction=args.train_fraction,
            split="train",
            seed=args.seed,
        ))
        eval_ds = BinTextDataset(BinTextTaskConfig(
            path=args.text_path,
            seq_len=args.seq_len,
            vocab_size=args.vocab_size,
            train_fraction=args.train_fraction,
            split="eval",
            seed=args.seed + 1,
        ))
        return train_ds, eval_ds

    # task == "text"
    if args.tokenizer_model is None:
        train_ds = ByteTextDataset(ByteTextTaskConfig(
            path=args.text_path, seq_len=args.seq_len,
            train_fraction=args.train_fraction, split="train", seed=args.seed,
        ))
        eval_ds = ByteTextDataset(ByteTextTaskConfig(
            path=args.text_path, seq_len=args.seq_len,
            train_fraction=args.train_fraction, split="eval", seed=args.seed + 1,
        ))
        return train_ds, eval_ds

    tokenizer = load_text_tokenizer(args.tokenizer_model)
    train_ds = TokenizedTextDataset(
        TokenizedTextTaskConfig(
            path=args.text_path, seq_len=args.seq_len,
            train_fraction=args.train_fraction, split="train", seed=args.seed,
        ),
        tokenizer=tokenizer,
    )
    eval_ds = TokenizedTextDataset(
        TokenizedTextTaskConfig(
            path=args.text_path, seq_len=args.seq_len,
            train_fraction=args.train_fraction, split="eval", seed=args.seed + 1,
        ),
        tokenizer=tokenizer,
    )
    return train_ds, eval_ds


# ── Checkpoint helpers ─────────────────────────────────────────────────────

def save_checkpoint(
    save_dir: Path,
    filename: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler | None,
    run_config: dict,
    step: int,
    metrics: list[dict],
    summary: dict | None = None,
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "config": run_config,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "metrics": metrics,
        "summary": summary,
    }
    tmp = save_dir / f"{filename}.tmp"
    torch.save(payload, tmp)
    tmp.replace(save_dir / filename)   # atomic on POSIX — never leaves a corrupt file


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler | None,
) -> tuple[int, list[dict]]:
    """Load model + optimiser state. Returns (resumed_step, metrics_so_far)."""
    payload = torch.load(path, map_location="cpu")
    model.load_state_dict(payload["model_state_dict"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    if scaler is not None and payload.get("scaler_state_dict") is not None:
        scaler.load_state_dict(payload["scaler_state_dict"])
    step = int(payload.get("step", 0))
    metrics = payload.get("metrics", [])
    print(json.dumps({"event": "resumed", "from_step": step, "checkpoint": str(path)}))
    return step, metrics


def build_run_config(args: argparse.Namespace, vocab_size: int, bits_metric_name: str, tokenizer_kind: str) -> dict:
    return {
        "task": args.task,
        "device": args.device,
        "dtype": args.dtype,
        "model_preset": args.model_preset,
        "use_triton_kernel": args.use_triton_kernel,
        "attention_impl": "local_latent_triton" if args.use_triton_kernel else "local_latent_pytorch",
        "torch_compile": args.torch_compile,
        "compile_mode": args.compile_mode,
        "compile_cudagraphs": args.compile_cudagraphs,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "accum_steps": args.accum_steps,
        "seq_len": args.seq_len,
        "lr": args.lr,
        "lr_min": args.lr_min,
        "warmup_steps": args.warmup_steps,
        "grad_clip": args.grad_clip,
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
        "eval_every": args.eval_every,
        "eval_batches": args.eval_batches,
        "seed": args.seed,
        "train_fraction": args.train_fraction,
        "vocab_size": vocab_size,
        "bits_metric_name": bits_metric_name,
        "tokenizer_kind": tokenizer_kind,
        **({"text_path": str(args.text_path)} if args.text_path else {}),
        **({"tokenizer_model": str(args.tokenizer_model)} if args.tokenizer_model else {}),
    }


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    device = resolve_device(args.device)
    if args.use_triton_kernel and device.type != "cuda":
        raise RuntimeError("--use-triton-kernel requires --device cuda (or auto resolving to CUDA)")
    if args.use_triton_kernel and not args.allow_unsafe_triton_training:
        raise RuntimeError(
            "--use-triton-kernel is disabled for training by default: the current Triton backward is incomplete "
            "and training results are not reliable. Pass --allow-unsafe-triton-training only for technical benchmarking."
        )
    if args.use_triton_kernel and args.torch_compile:
        raise RuntimeError("--torch-compile is only supported on the PyTorch attention path")
    if args.compile_mode == "max-autotune-no-cudagraphs" and args.compile_cudagraphs:
        raise RuntimeError("--compile-cudagraphs conflicts with --compile-mode max-autotune-no-cudagraphs")

    # ── dtype & autocast ──────────────────────────────────────────────────
    _dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    train_dtype = _dtype_map[args.dtype]
    use_amp = train_dtype in {torch.bfloat16, torch.float16} and device.type == "cuda"

    if use_amp:
        autocast_ctx = torch.autocast(device_type="cuda", dtype=train_dtype)
        # GradScaler only needed for float16; bfloat16 has sufficient range
        scaler = torch.amp.GradScaler("cuda", enabled=(train_dtype == torch.float16))
    else:
        import contextlib
        autocast_ctx = contextlib.nullcontext()
        scaler = None

    # ── Datasets ──────────────────────────────────────────────────────────
    train_dataset, eval_dataset = build_datasets(args)
    vocab_size = train_dataset.vocab_size
    bits_metric_name = getattr(train_dataset, "bits_metric_name", "eval_bpt")
    tokenizer_kind = getattr(train_dataset, "tokenizer_kind", "byte")

    run_config = build_run_config(args, vocab_size, bits_metric_name, tokenizer_kind)

    # ── Model ─────────────────────────────────────────────────────────────
    model = build_model(
        vocab_size=vocab_size,
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
        use_triton_kernel=args.use_triton_kernel,
    ).to(device)

    if not use_amp and train_dtype != torch.float32:
        model = model.to(dtype=train_dtype)

    if args.torch_compile:
        compile_cudagraphs = args.compile_cudagraphs
        if args.compile_mode == "max-autotune-no-cudagraphs":
            compile_cudagraphs = False
        try:
            import torch._inductor.config as inductor_config
            inductor_config.triton.cudagraphs = compile_cudagraphs
        except Exception:
            pass
        emit_event({
            "event": "compile_notice",
            "message": "torch.compile enabled; the first few steps may include graph capture and autotuning overhead",
            "compile_mode": args.compile_mode,
            "compile_cudagraphs": compile_cudagraphs,
        })
        compile_mode = None if args.compile_mode == "default" else args.compile_mode
        model = torch.compile(model, mode=compile_mode)

    if args.profile_attention:
        model.set_profile(True)

    total_params = sum(p.numel() for p in model.parameters())
    preset = MODEL_PRESETS.get(args.model_preset) if args.model_preset is not None else None
    emit_event({
        "event": "model_ready",
        "params": total_params,
        "params_M": round(total_params / 1e6, 2),
        "dtype": args.dtype,
        "model_preset": args.model_preset,
        "model_preset_approx_params_b": preset["approx_params_b"] if preset is not None else None,
        "amp": use_amp,
        "use_triton_kernel": args.use_triton_kernel,
        "attention_impl": "local_latent_triton" if args.use_triton_kernel else "local_latent_pytorch",
        "torch_compile": args.torch_compile,
        "compile_mode": args.compile_mode,
        "compile_cudagraphs": args.compile_cudagraphs,
    })

    if (
        not args.use_triton_kernel
        and args.seq_len >= 2048
        and args.latent_query_block_size <= 1
    ):
        emit_event({
            "event": "performance_hint",
            "message": "latent_query_block_size=128 or 256 usually reduces remote-branch projection cost at long context",
            "seq_len": args.seq_len,
            "latent_query_block_size": args.latent_query_block_size,
        })

    if args.use_triton_kernel:
        emit_event({
            "event": "triton_notice",
            "message": "first step includes Triton JIT compile; forward/backward may take a while before step 1 finishes",
            "debug_first_step": bool(args.debug_first_step),
        })

    # ── Optimiser ─────────────────────────────────────────────────────────
    # Separate weight-decay groups: no decay on biases and norms (standard practice)
    decay_params = [p for n, p in model.named_parameters() if p.ndim >= 2]
    no_decay_params = [p for n, p in model.named_parameters() if p.ndim < 2]
    optimizer = torch.optim.AdamW(
        [{"params": decay_params, "weight_decay": 0.1},
         {"params": no_decay_params, "weight_decay": 0.0}],
        lr=args.lr,
        betas=(0.9, 0.95),
        fused=device.type == "cuda",   # fused AdamW is faster on CUDA
    )

    # ── Resume ────────────────────────────────────────────────────────────
    start_step = 0
    metrics: list[dict] = []
    if args.resume is not None:
        start_step, metrics = load_checkpoint(args.resume, model, optimizer, scaler)

    # ── Training loop ─────────────────────────────────────────────────────
    reset_peak_memory(device)
    wall_start = time.perf_counter()

    optimizer.zero_grad(set_to_none=True)
    accum_loss = 0.0
    accum_count = 0

    for step in range(start_step + 1, args.steps + 1):
        model.train()
        debug_first_step = args.debug_first_step or (args.use_triton_kernel and step == start_step + 1)
        step_wall_start = time.perf_counter()

        # LR schedule (applied at the start of each optimiser step)
        if accum_count == 0:
            lr = cosine_lr(step, args.steps, args.warmup_steps, args.lr, args.lr_min)
            set_lr(optimizer, lr)

        # ── Micro-step (gradient accumulation) ────────────────────────────
        for micro in range(args.accum_steps):
            x, y = train_dataset.sample_batch(batch_size=args.batch_size, device=device)
            if debug_first_step:
                sync_device(device)
                emit_event({
                    "event": "first_step_phase",
                    "step": step,
                    "micro": micro,
                    "phase": "batch_ready",
                    "elapsed_sec": round(time.perf_counter() - step_wall_start, 4),
                    "batch_shape": list(x.shape),
                })
            if args.torch_compile:
                mark_compile_step_begin()
            with autocast_ctx:
                logits = model(x)
                loss, _ = compute_loss_and_metrics(logits, y, bits_metric_name=bits_metric_name)
                # Scale loss so gradients are averaged across accumulation steps
                loss_scaled = loss / args.accum_steps
            if debug_first_step:
                sync_device(device)
                emit_event({
                    "event": "first_step_phase",
                    "step": step,
                    "micro": micro,
                    "phase": "forward_done",
                    "elapsed_sec": round(time.perf_counter() - step_wall_start, 4),
                    "loss": round(loss.item(), 6),
                })

            if scaler is not None:
                scaler.scale(loss_scaled).backward()
            else:
                loss_scaled.backward()
            if debug_first_step:
                sync_device(device)
                emit_event({
                    "event": "first_step_phase",
                    "step": step,
                    "micro": micro,
                    "phase": "backward_done",
                    "elapsed_sec": round(time.perf_counter() - step_wall_start, 4),
                })

            accum_loss += loss.item()
            accum_count += 1

        # ── Optimiser step ────────────────────────────────────────────────
        if args.grad_clip > 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        if debug_first_step:
            sync_device(device)
            emit_event({
                "event": "first_step_phase",
                "step": step,
                "phase": "optimizer_done",
                "elapsed_sec": round(time.perf_counter() - step_wall_start, 4),
            })

        optimizer.zero_grad(set_to_none=True)
        train_loss = accum_loss / accum_count
        accum_loss = 0.0
        accum_count = 0

        # ── Logging ───────────────────────────────────────────────────────
        step_duration = time.perf_counter() - step_wall_start
        effective_tokens = args.batch_size * args.accum_steps * args.seq_len
        row: dict = {
            "step": step,
            "lr": lr,
            "train_loss": train_loss,
            "step_time_sec": round(step_duration, 4),
            "train_tokens_per_sec": round(effective_tokens / max(step_duration, 1e-8), 0),
        }

        if step % args.eval_every == 0 or step == args.steps:
            row.update(
                evaluate(
                    model,
                    eval_dataset,
                    args.batch_size,
                    device,
                    args.eval_batches,
                    autocast_ctx,
                    torch_compile=args.torch_compile,
                )
            )

        row.update(get_peak_memory_stats(device))
        if args.profile_attention:
            row.update(model.get_profile_stats())
        metrics.append(row)

        if step % args.log_every == 0 or step == 1 or step == args.steps:
            emit_event(row)

        # ── Periodic checkpoint ───────────────────────────────────────────
        if args.save_dir is not None and args.save_every > 0 and step % args.save_every == 0:
            save_checkpoint(
                save_dir=args.save_dir,
                filename="checkpoint_latest.pt",
                model=model, optimizer=optimizer, scaler=scaler,
                run_config=run_config, step=step, metrics=metrics,
            )

    # ── End of training ───────────────────────────────────────────────────
    duration = time.perf_counter() - wall_start
    summary = {
        "task": args.task,
        "device": str(device),
        "dtype": args.dtype,
        "use_triton_kernel": args.use_triton_kernel,
        "attention_impl": "local_latent_triton" if args.use_triton_kernel else "local_latent_pytorch",
        "torch_compile": args.torch_compile,
        "compile_mode": args.compile_mode,
        "compile_cudagraphs": args.compile_cudagraphs,
        "steps": args.steps,
        "seq_len": args.seq_len,
        "params": total_params,
        "params_M": round(total_params / 1e6, 2),
        "effective_batch_tokens": args.batch_size * args.accum_steps * args.seq_len,
        "duration_sec": round(duration, 2),
        "steps_per_sec": round(args.steps / max(duration, 1e-8), 4),
        "tokens_per_sec": round(
            args.steps * args.batch_size * args.accum_steps * args.seq_len / max(duration, 1e-8), 0
        ),
        "last": metrics[-1],
    }
    summary.update(get_peak_memory_stats(device))
    if args.profile_attention:
        summary.update(model.get_profile_stats())
    emit_event(summary)

    if args.save_dir is not None and args.save_final:
        save_checkpoint(
            save_dir=args.save_dir,
            filename="final.pt",
            model=model, optimizer=optimizer, scaler=scaler,
            run_config=run_config, step=args.steps, metrics=metrics, summary=summary,
        )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps({"summary": summary, "metrics": metrics}, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
