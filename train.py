import argparse
import itertools
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler

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
    from prototype.dist_utils import (
        all_reduce_mean,
        barrier,
        cleanup_distributed,
        get_local_rank,
        get_rank,
        get_world_size,
        is_distributed,
        is_main_process,
        launched_under_torchrun,
        maybe_no_sync,
        setup_distributed,
        unwrap,
    )
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
    from dist_utils import (
        all_reduce_mean,
        barrier,
        cleanup_distributed,
        get_local_rank,
        get_rank,
        get_world_size,
        is_distributed,
        is_main_process,
        launched_under_torchrun,
        maybe_no_sync,
        setup_distributed,
        unwrap,
    )


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
    parser.add_argument("--dtype", choices=["auto", "float32", "bfloat16", "float16"], default="auto",
                        help="Training dtype. 'auto' (default) picks bfloat16 on CUDA Ampere+ "
                             "(compute capability >= 8.0), float32 otherwise. "
                             "Use bfloat16 on H100/H200/A100 for 2-3× speedup.")
    parser.add_argument("--seed", type=int, default=0)

    # ── Evaluation ─────────────────────────────────────────────────────────
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--eval-batches", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=1,
                        help="Print one training JSON row every N optimisation steps")
    parser.add_argument("--profile-attention", action="store_true")
    parser.add_argument("--torch-profile", action="store_true",
                        help="Enable torch.profiler with schedule wait=5/warmup=5/active=10. "
                             "Exports Chrome trace + op table. Use for real bottleneck analysis.")
    parser.add_argument("--torch-profile-dir", type=Path, default=Path("runs/profiles"),
                        help="Output directory for torch.profiler traces")
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

    # ── Data loading ───────────────────────────────────────────────────────
    parser.add_argument("--num-workers", type=int, default=2,
                        help="DataLoader worker processes per rank (only used for map-style datasets)")
    parser.add_argument("--prefetch-factor", type=int, default=4,
                        help="DataLoader prefetch factor (ignored when num_workers=0)")

    # ── Distributed ────────────────────────────────────────────────────────
    parser.add_argument("--ddp-bucket-cap-mb", type=int, default=25,
                        help="DDP gradient bucket size (MB). 25 is the PyTorch default; "
                             "increase to 50-100 on fast interconnects for fewer all-reduces.")
    parser.add_argument("--ddp-find-unused-parameters", action="store_true",
                        help="Enable only if DDP raises the 'unused parameters' error. "
                             "Adds overhead; keep off by default.")

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


def emit_event(payload: dict, *, all_ranks: bool = False) -> None:
    """Emit a JSON event to stdout.

    By default only rank 0 prints — multi-rank interleaved JSON is unparseable
    and triples disk-log size.  Use `all_ranks=True` for genuinely
    rank-specific diagnostics (e.g. OOM location).
    """
    if all_ranks or is_main_process():
        if all_ranks and is_distributed():
            payload = {"_rank": get_rank(), **payload}
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
    """Accumulate eval metrics as device tensors, all-reduce across ranks.

    Under DDP, each rank samples its own `batches` windows and contributes
    to the mean.  The effective eval sample size is therefore
    `batches * world_size * batch_size` tokens per call.  This is not
    epoch-aware evaluation — it's a quick health-check signal; drive a
    full eval loop outside the training loop if you need strict numbers.
    """
    model.eval()
    totals: dict[str, torch.Tensor] = {}
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
            value_t = value if isinstance(value, torch.Tensor) else torch.as_tensor(
                value, device=device
            )
            if key in totals:
                totals[key] = totals[key] + value_t
            else:
                totals[key] = value_t.clone().float()

    # Per-rank mean first, then all-reduce mean across ranks — avoids
    # precision loss from summing many terms before dividing.
    out: dict[str, float] = {}
    for key, value in totals.items():
        local_mean = value / max(batches, 1)
        reduced = all_reduce_mean(local_mean.detach().clone())
        out[key] = reduced.item()
    return out


def compute_loss_and_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    bits_metric_name: str = "eval_bpt",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute loss and metrics without host synchronisation.

    Returns tensors (all on the input device) instead of Python floats.  The
    caller is responsible for materialising values via `.item()` only at
    logging boundaries.  This avoids a GPU→CPU sync on every micro-step.
    """
    if targets.ndim == 1:
        # Retrieval task: predict at the last position only
        final_logits = logits[:, -1, :]
        loss = F.cross_entropy(final_logits, targets)
        preds = final_logits.argmax(dim=-1)
        acc = (preds == targets).float().mean()
        return loss, {"eval_loss": loss.detach(), "eval_acc": acc}

    flat_logits = logits.reshape(-1, logits.size(-1))
    flat_targets = targets.reshape(-1)
    loss = F.cross_entropy(flat_logits, flat_targets)
    preds = logits.argmax(dim=-1)
    acc = (preds == targets).float().mean()
    # perplexity/bpt are monotonic transforms of loss — compute as tensors,
    # materialise only at logging time.
    loss_det = loss.detach()
    perplexity = torch.exp(torch.clamp(loss_det, max=20.0))
    bpt = loss_det / math.log(2.0)
    return loss, {
        "eval_loss": loss_det,
        "eval_acc": acc,
        "eval_perplexity": perplexity,
        bits_metric_name: bpt,
    }


def _tensors_to_floats(d: dict) -> dict:
    """Materialise any tensor values in `d` to Python floats (single sync)."""
    out: dict = {}
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.item()
        else:
            out[k] = v
    return out


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


def _is_map_style(dataset) -> bool:
    """True iff the dataset is a torch.utils.data.Dataset (has __len__/__getitem__)."""
    return isinstance(dataset, torch.utils.data.Dataset)


def build_train_loader(
    dataset,
    batch_size: int,
    seed: int,
    num_workers: int,
    prefetch_factor: int,
    pin_memory: bool,
) -> DataLoader | None:
    """Return a DataLoader for map-style datasets, None for legacy ones.

    We keep a dual-path design: the new `BinTextDataset` is map-style and
    flows through DataLoader + DistributedSampler; legacy datasets
    (retrieval / byte / tokenized) still use their in-process sample_batch
    method.  This avoids a bigger refactor we don't need for the weekend.
    """
    if not _is_map_style(dataset):
        return None

    if is_distributed():
        sampler = DistributedSampler(
            dataset,
            num_replicas=get_world_size(),
            rank=get_rank(),
            shuffle=True,
            seed=seed,
            drop_last=True,
        )
        shuffle = False
    else:
        # Even in single-process mode, use a RandomSampler so the DataLoader
        # reshuffles deterministically per epoch.
        generator = torch.Generator()
        generator.manual_seed(seed)
        sampler = RandomSampler(dataset, generator=generator)
        shuffle = False

    loader_kwargs = dict(
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle,
        drop_last=True,
        pin_memory=pin_memory,
        num_workers=num_workers,
    )
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = prefetch_factor

    return DataLoader(dataset, **loader_kwargs)


def _infinite_loader(loader: DataLoader, sampler) -> "itertools.Iterator":
    """Yield batches forever, calling `sampler.set_epoch` on each new epoch.

    DistributedSampler needs `set_epoch(epoch)` for its shuffle to change
    across epochs; without it every epoch uses the same shuffle, which
    silently kills training.
    """
    epoch = 0
    while True:
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(epoch)
        for batch in loader:
            yield batch
        epoch += 1


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
    sampler_epoch: int | None = None,
) -> None:
    """Save a training checkpoint atomically.

    Rank safety
    -----------
    Call sites must gate this on `is_main_process()`.  Writing from every
    rank produces either interleaved bytes (same path) or N copies (rank-
    suffixed paths); neither is what we want.

    `model` should already be unwrapped (no DDP / torch.compile wrapper) so
    the state_dict keys don't carry a `module.` / `_orig_mod.` prefix that
    would confuse a resume on a different parallelism layout.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    # Strip DDP (`module.`) / torch.compile (`_orig_mod.`) wrappers so the
    # checkpoint is portable across parallelism and compile layouts.
    inner = unwrap(model)
    payload = {
        "step": step,
        "config": run_config,
        "model_state_dict": inner.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "metrics": metrics,
        "summary": summary,
        "sampler_epoch": sampler_epoch,
        "world_size": get_world_size(),
    }
    tmp = save_dir / f"{filename}.tmp"
    torch.save(payload, tmp)
    tmp.replace(save_dir / filename)   # atomic on POSIX — never leaves a corrupt file


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler | None,
    device: torch.device,
) -> tuple[int, list[dict], int | None]:
    """Load model + optimiser state.

    Returns (resumed_step, metrics_so_far, sampler_epoch).

    `map_location` points to the current rank's device so the loaded tensors
    live where we need them — loading everything onto CPU first and then
    moving would double peak CPU RAM on large checkpoints.

    `weights_only=False` is required because we also load optimizer state
    and config dicts; this checkpoint format is trusted (produced by us).
    """
    # Map to the current rank's device; avoids CPU→GPU copy after load.
    map_location = (
        f"cuda:{get_local_rank()}" if device.type == "cuda" else "cpu"
    )
    payload = torch.load(path, map_location=map_location, weights_only=False)
    # Strip any `_orig_mod.` / `module.` prefixes left over from pre-fix
    # checkpoints (or from a mismatched wrapper order) so load is robust.
    state = payload["model_state_dict"]
    cleaned = {}
    for k, v in state.items():
        nk = k
        while nk.startswith("_orig_mod.") or nk.startswith("module."):
            if nk.startswith("_orig_mod."):
                nk = nk[len("_orig_mod."):]
            elif nk.startswith("module."):
                nk = nk[len("module."):]
        cleaned[nk] = v
    unwrap(model).load_state_dict(cleaned)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    if scaler is not None and payload.get("scaler_state_dict") is not None:
        scaler.load_state_dict(payload["scaler_state_dict"])
    step = int(payload.get("step", 0))
    metrics = payload.get("metrics", [])
    sampler_epoch = payload.get("sampler_epoch")
    saved_world_size = payload.get("world_size")
    if saved_world_size is not None and saved_world_size != get_world_size():
        # Not fatal — optimizer state is compatible — but loss-curve
        # reproducibility degrades if you resume on a different world size
        # because the effective batch size changes.
        emit_event({
            "event": "resume_world_size_mismatch",
            "checkpoint_world_size": saved_world_size,
            "current_world_size": get_world_size(),
            "warning": "Effective batch size differs; loss curve will not match.",
        })
    emit_event({"event": "resumed", "from_step": step, "checkpoint": str(path)})
    return step, metrics, sampler_epoch


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

    # ── Distributed bootstrap ─────────────────────────────────────────────
    # If launched under torchrun, setup_distributed picks the device for
    # this rank (cuda:LOCAL_RANK) and initialises NCCL.  Otherwise we fall
    # back to the plain resolve_device policy.
    if launched_under_torchrun():
        device = setup_distributed()
        emit_event({
            "event": "distributed_ready",
            "backend": "nccl",
            "world_size": get_world_size(),
            "rank": get_rank(),
            "local_rank": get_local_rank(),
        })
    else:
        device = resolve_device(args.device)

    # Per-rank seed so every process draws distinct dropout/init noise while
    # still being deterministic given (--seed, world_size, rank).  The model
    # broadcast through DDP keeps parameters in sync regardless.
    torch.manual_seed(args.seed + get_rank())

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

    if args.dtype == "auto":
        # bf16 on Ampere+ (CC>=8.0): H100/H200, A100, RTX 30/40/PRO 6000.
        # float32 elsewhere (older GPUs, CPU, MPS) to avoid silent numerical issues.
        if device.type == "cuda":
            cc_major, _ = torch.cuda.get_device_capability(device)
            resolved = "bfloat16" if cc_major >= 8 else "float32"
        else:
            resolved = "float32"
        args.dtype = resolved
        emit_event({
            "event": "dtype_resolved",
            "device": str(device),
            "resolved_dtype": resolved,
            "reason": "auto policy: bfloat16 on CUDA CC>=8.0, float32 otherwise",
        })

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

    # ── DDP wrap (before torch.compile: compile sees the DDP graph so
    # bucket view / gradient hooks interact correctly with compiled
    # forward/backward). static_graph=False because activation
    # checkpointing (--checkpoint-blocks) creates a non-static autograd
    # graph which breaks static_graph=True. ────────────────────────────
    if is_distributed():
        if device.type != "cuda":
            raise RuntimeError("DDP currently requires CUDA in this codebase")
        model = DDP(
            model,
            device_ids=[get_local_rank()],
            output_device=get_local_rank(),
            gradient_as_bucket_view=True,
            static_graph=False,
            find_unused_parameters=args.ddp_find_unused_parameters,
            bucket_cap_mb=args.ddp_bucket_cap_mb,
        )
        emit_event({
            "event": "ddp_ready",
            "bucket_cap_mb": args.ddp_bucket_cap_mb,
            "find_unused_parameters": args.ddp_find_unused_parameters,
        })

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
        # set_profile lives on the underlying LM, not on the DDP/compile wrapper
        unwrap(model).set_profile(True)
        emit_event({
            "event": "profile_attention_warning",
            "message": (
                "--profile-attention forces torch.cuda.synchronize() inside every "
                "attention forward.  Throughput numbers reported under this flag are "
                "NOT representative of real training speed.  Use it only for "
                "component-level breakdown, not for tokens/sec benchmarking."
            ),
        })

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
    resumed_sampler_epoch: int | None = None
    if args.resume is not None:
        start_step, metrics, resumed_sampler_epoch = load_checkpoint(
            args.resume, model, optimizer, scaler, device
        )

    # ── DataLoader (map-style datasets only) ──────────────────────────────
    pin_memory = device.type == "cuda"
    train_loader = build_train_loader(
        train_dataset,
        batch_size=args.batch_size,
        seed=args.seed,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        pin_memory=pin_memory,
    )
    if train_loader is not None:
        # Iterator that auto-advances epoch and calls set_epoch on
        # DistributedSampler so shuffling stays non-trivial across epochs.
        train_sampler = train_loader.sampler
        if resumed_sampler_epoch is not None and isinstance(train_sampler, DistributedSampler):
            train_sampler.set_epoch(resumed_sampler_epoch)
        train_iter = _infinite_loader(train_loader, train_sampler)
        emit_event({
            "event": "train_loader_ready",
            "num_workers": args.num_workers,
            "prefetch_factor": args.prefetch_factor if args.num_workers > 0 else None,
            "pin_memory": pin_memory,
            "sampler": type(train_sampler).__name__,
            "dataset_len": len(train_dataset),
        })
    else:
        train_iter = None
        train_sampler = None
        if is_distributed():
            raise RuntimeError(
                "Distributed training requires a map-style dataset (e.g. --task bin). "
                f"Got legacy dataset type: {type(train_dataset).__name__}"
            )

    def _next_batch() -> tuple[torch.Tensor, torch.Tensor]:
        if train_iter is not None:
            x, y = next(train_iter)
            # DataLoader returns CPU tensors when pin_memory=True; move async.
            x = x.to(device=device, non_blocking=pin_memory)
            y = y.to(device=device, non_blocking=pin_memory)
            return x, y
        # Legacy datasets keep their own device-side sample_batch.
        return train_dataset.sample_batch(batch_size=args.batch_size, device=device)

    # ── Training loop ─────────────────────────────────────────────────────
    reset_peak_memory(device)
    wall_start = time.perf_counter()

    optimizer.zero_grad(set_to_none=True)
    # Accumulate loss on-device to avoid a GPU→CPU sync every micro-step.
    # We materialise to Python float only when we actually emit a log row.
    accum_loss = torch.zeros((), device=device, dtype=torch.float32)
    accum_count = 0

    # ── torch.profiler (optional, heavy) ──────────────────────────────────
    # Schedule: skip 5 warmup-warmup steps, 5 warmup-traced steps (no record),
    # record 10 active steps, then stop.  First real measurement at step 11.
    profiler = None
    if args.torch_profile:
        args.torch_profile_dir.mkdir(parents=True, exist_ok=True)
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        profiler = torch.profiler.profile(
            activities=activities,
            schedule=torch.profiler.schedule(wait=5, warmup=5, active=10, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                str(args.torch_profile_dir)
            ),
            record_shapes=True,
            profile_memory=True,
            with_stack=False,   # stacks are expensive; keep shapes instead
        )
        profiler.start()
        emit_event({
            "event": "torch_profile_started",
            "output_dir": str(args.torch_profile_dir),
            "schedule": {"wait": 5, "warmup": 5, "active": 10, "repeat": 1},
            "note": "active window is steps 11-20 (1-indexed from start_step+1)",
        })

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
            x, y = _next_batch()
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

            # Only trigger DDP all-reduce on the final micro-step.  Without
            # this, grad accumulation would issue (accum_steps - 1) needless
            # all-reduces; on 8 GPUs with accum=8 that's a 7× waste of
            # interconnect bandwidth.
            is_final_micro = (micro == args.accum_steps - 1)
            with maybe_no_sync(model, sync=is_final_micro), autocast_ctx:
                logits = model(x)
                loss, _ = compute_loss_and_metrics(logits, y, bits_metric_name=bits_metric_name)
                # Scale loss so gradients are averaged across accumulation steps
                loss_scaled = loss / args.accum_steps
                if debug_first_step:
                    sync_device(device)
                    # .item() here is intentional: debug-first-step is off in real
                    # training runs, and we want the concrete number for the log.
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

            accum_loss = accum_loss + loss.detach().float()
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

        # ── Logging ───────────────────────────────────────────────────────
        # Only materialise the accumulated loss when we actually emit a log
        # row.  On non-logging steps we keep accum_loss as a device tensor
        # and avoid the GPU→CPU sync entirely.
        step_duration = time.perf_counter() - step_wall_start
        effective_tokens = args.batch_size * args.accum_steps * args.seq_len
        will_log = (step % args.log_every == 0) or step == 1 or step == args.steps
        will_eval = (step % args.eval_every == 0) or step == args.steps

        if will_log or will_eval:
            train_loss_val = (accum_loss / max(accum_count, 1)).item()
        else:
            train_loss_val = None

        accum_loss = torch.zeros((), device=device, dtype=torch.float32)
        accum_count = 0

        row: dict = {
            "step": step,
            "lr": lr,
            "step_time_sec": round(step_duration, 4),
            "train_tokens_per_sec": round(effective_tokens / max(step_duration, 1e-8), 0),
        }
        if train_loss_val is not None:
            row["train_loss"] = train_loss_val

        if will_eval:
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

        if will_log:
            emit_event(row)

        # Advance profiler schedule (wait → warmup → active → done)
        if profiler is not None:
            profiler.step()

        # ── Periodic checkpoint ───────────────────────────────────────────
        if args.save_dir is not None and args.save_every > 0 and step % args.save_every == 0:
            save_checkpoint(
                save_dir=args.save_dir,
                filename="checkpoint_latest.pt",
                model=model, optimizer=optimizer, scaler=scaler,
                run_config=run_config, step=step, metrics=metrics,
            )

    # ── End of training ───────────────────────────────────────────────────
    if profiler is not None:
        profiler.stop()
        # Dump a human-readable top-ops table alongside the Chrome trace.
        try:
            table = profiler.key_averages().table(
                sort_by="self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total",
                row_limit=30,
            )
            (args.torch_profile_dir / "top_ops.txt").write_text(table, encoding="utf-8")
            emit_event({
                "event": "torch_profile_finished",
                "output_dir": str(args.torch_profile_dir),
                "top_ops_file": str(args.torch_profile_dir / "top_ops.txt"),
            })
        except Exception as exc:  # pragma: no cover — profiler output is best-effort
            emit_event({"event": "torch_profile_table_failed", "error": repr(exc)})

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

    # Teardown del process group DDP — NCCL lamenta un warning a shutdown
    # se il gruppo non viene distrutto esplicitamente.
    cleanup_distributed()


if __name__ == "__main__":
    main()
