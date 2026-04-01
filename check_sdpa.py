import argparse
import json
import time

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from prototype.runtime import resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--dropout-p", type=float, default=0.0)
    parser.add_argument("--non-causal", dest="is_causal", action="store_false")
    parser.set_defaults(is_causal=True)
    return parser.parse_args()


def parse_dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def try_backend(
    backend: SDPBackend,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float,
    is_causal: bool,
) -> dict:
    try:
        sync(q.device)
        start = time.perf_counter()
        with sdpa_kernel(backends=[backend]):
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=is_causal,
                dropout_p=dropout_p,
            )
        sync(q.device)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return {
            "ok": True,
            "time_ms": elapsed_ms,
            "output_shape": list(out.shape),
        }
    except Exception as exc:  # pragma: no cover - runtime-dependent probe
        return {
            "ok": False,
            "error": str(exc),
        }


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    dtype = parse_dtype(args.dtype)

    if args.d_model % args.n_heads != 0:
        raise SystemExit("--d-model must be divisible by --n-heads")

    head_dim = args.d_model // args.n_heads
    q = torch.randn(args.batch_size, args.n_heads, args.seq_len, head_dim, device=device, dtype=dtype)
    k = torch.randn(args.batch_size, args.n_heads, args.seq_len, head_dim, device=device, dtype=dtype)
    v = torch.randn(args.batch_size, args.n_heads, args.seq_len, head_dim, device=device, dtype=dtype)

    report = {
        "device": str(device),
        "torch_version": torch.__version__,
        "dtype": args.dtype,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "d_model": args.d_model,
        "n_heads": args.n_heads,
        "head_dim": head_dim,
        "dropout_p": args.dropout_p,
        "is_causal": args.is_causal,
    }

    if device.type != "cuda":
        report["note"] = "CUDA is required to inspect Flash / memory-efficient SDPA backends."
        print(json.dumps(report, indent=2))
        return

    params = torch.backends.cuda.SDPAParams(
        q,
        k,
        v,
        None,
        args.dropout_p,
        args.is_causal,
        False,
    )
    report["flash_attention_available"] = torch.backends.cuda.is_flash_attention_available()
    report["flash_sdp_enabled"] = torch.backends.cuda.flash_sdp_enabled()
    report["mem_efficient_sdp_enabled"] = torch.backends.cuda.mem_efficient_sdp_enabled()
    report["math_sdp_enabled"] = torch.backends.cuda.math_sdp_enabled()
    report["flash_eligible"] = torch.backends.cuda.can_use_flash_attention(params, debug=False)
    report["mem_efficient_eligible"] = torch.backends.cuda.can_use_efficient_attention(params, debug=False)
    report["forced_backend_checks"] = {
        "flash_attention": try_backend(
            SDPBackend.FLASH_ATTENTION,
            q,
            k,
            v,
            dropout_p=args.dropout_p,
            is_causal=args.is_causal,
        ),
        "efficient_attention": try_backend(
            SDPBackend.EFFICIENT_ATTENTION,
            q,
            k,
            v,
            dropout_p=args.dropout_p,
            is_causal=args.is_causal,
        ),
        "math": try_backend(
            SDPBackend.MATH,
            q,
            k,
            v,
            dropout_p=args.dropout_p,
            is_causal=args.is_causal,
        ),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
