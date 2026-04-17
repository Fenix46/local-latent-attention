"""Distributed training helpers.

The module is a thin wrapper around `torch.distributed` that:
  * falls back cleanly to single-process mode when no `RANK` env var is set,
    so the same `train.py` works both with `python train.py ...` and with
    `torchrun --nproc_per_node=N train.py ...`;
  * centralises the rank/world-size queries so the training script never
    has to check `dist.is_initialized()` itself;
  * exposes a `maybe_no_sync` context manager that turns DDP gradient
    synchronisation off for non-final micro-steps in gradient accumulation
    (skipping this optimisation wastes `accum_steps - 1` all-reduces per
    optimiser step, which is the biggest DDP footgun for this codebase).
"""

from __future__ import annotations

import contextlib
import os
from typing import Iterator

import torch
import torch.distributed as dist


# ── Environment / state ──────────────────────────────────────────────────


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def get_local_rank() -> int:
    """Local rank (one value per process on the same node).

    `torchrun` sets LOCAL_RANK; we honour it even before init_process_group
    so that device selection can happen first.
    """
    return int(os.environ.get("LOCAL_RANK", "0"))


def is_main_process() -> bool:
    return get_rank() == 0


def launched_under_torchrun() -> bool:
    """True iff the process was launched by torchrun / torch.distributed.run."""
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


# ── Setup / teardown ─────────────────────────────────────────────────────


def setup_distributed(backend: str = "nccl") -> torch.device:
    """Initialise the process group from torchrun env vars.

    Returns the device this rank should use.  Safe to call in single-process
    mode: if no RANK env is set, we skip init and just resolve the device.
    """
    if not launched_under_torchrun():
        # Single-process fallback — caller picks device with runtime.resolve_device.
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "torchrun launched but CUDA is unavailable.  Distributed training "
            "currently requires CUDA + NCCL."
        )

    local_rank = get_local_rank()
    torch.cuda.set_device(local_rank)
    # init_method=env:// reads MASTER_ADDR, MASTER_PORT, RANK, WORLD_SIZE
    # from the environment (all set by torchrun).
    dist.init_process_group(backend=backend, init_method="env://")
    device = torch.device(f"cuda:{local_rank}")
    # Barrier here so we don't interleave rank-specific init with other ranks'
    # stdout.  device_ids avoids a deprecation warning in NCCL builds.
    dist.barrier(device_ids=[local_rank])
    return device


def cleanup_distributed() -> None:
    if is_distributed():
        dist.destroy_process_group()


# ── Reductions ───────────────────────────────────────────────────────────


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """In-place SUM all-reduce, then divide by world size.

    Returns the tensor so callers can chain.  No-op when not distributed.
    Tensor must already be on the correct CUDA device.
    """
    if not is_distributed():
        return tensor
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor.div_(get_world_size())
    return tensor


def all_reduce_sum(tensor: torch.Tensor) -> torch.Tensor:
    if not is_distributed():
        return tensor
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def barrier() -> None:
    if is_distributed():
        dist.barrier()


# ── DDP helpers ──────────────────────────────────────────────────────────


@contextlib.contextmanager
def maybe_no_sync(model: torch.nn.Module, sync: bool) -> Iterator[None]:
    """Disable DDP gradient all-reduce when `sync` is False.

    Use this during gradient accumulation: only the final micro-step should
    trigger the all-reduce.  Without this, an 8-step accumulation produces
    8× the network traffic for no benefit — the intermediate gradients are
    summed locally into .grad anyway and then overwritten by the final
    all-reduce.

    Falls back to nullcontext for non-DDP models or single-process runs.
    """
    no_sync = getattr(model, "no_sync", None)
    if not sync and no_sync is not None and is_distributed():
        with no_sync():
            yield
    else:
        yield


def unwrap(model: torch.nn.Module) -> torch.nn.Module:
    """Return the underlying module, stripping DDP / compile wrappers.

    Call sites:
      * `save_checkpoint` wants the real state_dict (DDP adds a `module.`
        prefix that would poison resume).
      * profiling / introspection should see the real module tree.
    """
    # Unwrap torch.compile (_orig_mod) first, then DDP (module), because
    # compile can wrap a DDP module or vice-versa depending on order.
    inner = model
    while True:
        if hasattr(inner, "_orig_mod"):
            inner = inner._orig_mod
            continue
        if hasattr(inner, "module") and isinstance(inner, torch.nn.parallel.DistributedDataParallel):
            inner = inner.module
            continue
        return inner
