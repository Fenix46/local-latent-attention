import argparse
import io
import json
from contextlib import redirect_stdout
from pathlib import Path

from prototype import incremental_train as incremental_train_module


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[64, 128])
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--d-ff", type=int, default=256)
    parser.add_argument("--local-window", type=int, default=64)
    parser.add_argument("--latent-tokens", type=int, default=16)
    parser.add_argument("--remote-chunk-size", type=int, default=32)
    parser.add_argument("--json-output", type=Path, default=None)
    return parser.parse_args(argv)


def parse_last_json(text: str) -> dict:
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            rows.append(json.loads(line))
    if not rows:
        raise ValueError("No JSON rows found in training output")
    return rows[-1]


def run_one(
    mode: str,
    gate_mode: str,
    seed: int,
    seq_len: int,
    args: argparse.Namespace,
) -> dict:
    argv = [
        "--mode",
        mode,
        "--gate-mode",
        gate_mode,
        "--steps",
        str(args.steps),
        "--eval-every",
        str(args.eval_every),
        "--batch-size",
        str(args.batch_size),
        "--seq-len",
        str(seq_len),
        "--seed",
        str(seed),
        "--d-model",
        str(args.d_model),
        "--n-heads",
        str(args.n_heads),
        "--n-layers",
        str(args.n_layers),
        "--d-ff",
        str(args.d_ff),
        "--local-window",
        str(args.local_window),
        "--latent-tokens",
        str(args.latent_tokens),
        "--remote-chunk-size",
        str(args.remote_chunk_size),
    ]
    buf = io.StringIO()
    with redirect_stdout(buf):
        incremental_train_module.main(argv)
    summary = parse_last_json(buf.getvalue())
    summary["seed"] = seed
    summary["seq_len"] = seq_len
    return summary


def mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def format_table(rows: list[dict]) -> str:
    headers = ["seq", "model", "gate", "loss", "acc", "gate_mean", "steps_per_sec"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["seq_len"]),
                    row["mode"],
                    row["gate_mode"],
                    f"{row['eval_loss_mean']:.4f}",
                    f"{row['eval_acc_mean']:.4f}",
                    f"{row['gate_mean']:.4f}",
                    f"{row['steps_per_sec_mean']:.4f}",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    runs = []
    configs = [
        ("baseline", "simple"),
        ("local_latent", "improved"),
    ]
    for seq_len in args.seq_lens:
        for seed in args.seeds:
            for mode, gate_mode in configs:
                run = run_one(mode=mode, gate_mode=gate_mode, seed=seed, seq_len=seq_len, args=args)
                runs.append(run)
                print(json.dumps(run))

    aggregates = []
    for seq_len in args.seq_lens:
        for mode, gate_mode in configs:
            subset = [
                row for row in runs if row["seq_len"] == seq_len and row["mode"] == mode and row["gate_mode"] == gate_mode
            ]
            aggregates.append(
                {
                    "seq_len": seq_len,
                    "mode": mode,
                    "gate_mode": gate_mode,
                    "eval_loss_mean": mean([row["last"]["eval_full_loss"] for row in subset]),
                    "eval_acc_mean": mean([row["last"]["eval_full_acc"] for row in subset]),
                    "gate_mean": mean([row["last"].get("eval_full_gate_mean", 1.0) for row in subset]),
                    "steps_per_sec_mean": mean([row["steps_per_sec"] for row in subset]),
                }
            )

    report = {
        "config": {
            "seeds": args.seeds,
            "seq_lens": args.seq_lens,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "local_window": args.local_window,
            "latent_tokens": args.latent_tokens,
            "remote_chunk_size": args.remote_chunk_size,
        },
        "aggregates": aggregates,
        "runs": runs,
    }

    print(format_table(aggregates))
    print()
    print(json.dumps(report, indent=2))

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
