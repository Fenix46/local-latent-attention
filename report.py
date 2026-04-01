import argparse
import io
import json
from contextlib import redirect_stdout
from pathlib import Path

from prototype import benchmark as benchmark_module
from prototype import inference as inference_module
from prototype import incremental as incremental_module


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context-lengths", type=int, nargs="+", default=[128, 256, 512, 1024, 2048])
    parser.add_argument("--generate-tokens", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=2)
    parser.add_argument("--timing-iters", type=int, default=1)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=256)
    parser.add_argument("--local-window", type=int, default=64)
    parser.add_argument("--latent-tokens", type=int, default=16)
    parser.add_argument("--remote-chunk-size", type=int, default=32)
    parser.add_argument("--gate-mode", choices=["simple", "improved"], default="simple")
    parser.add_argument("--json-output", type=Path, default=None)
    return parser.parse_args(argv)


def parse_json_lines(text: str) -> list[dict]:
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            rows.append(json.loads(line))
    return rows


def run_benchmark(args: argparse.Namespace) -> list[dict]:
    argv = [
        "--context-lengths",
        *[str(x) for x in args.context_lengths],
        "--batch-size",
        str(args.batch_size),
        "--warmup",
        str(args.warmup),
        "--iters",
        str(args.iters),
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
    ]
    buf = io.StringIO()
    with redirect_stdout(buf):
        benchmark_module.main(argv)
    return parse_json_lines(buf.getvalue())


def run_inference(args: argparse.Namespace) -> list[dict]:
    argv = [
        "--prompt-lengths",
        *[str(x) for x in args.context_lengths],
        "--generate-tokens",
        str(args.generate_tokens),
        "--batch-size",
        str(args.batch_size),
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
        "--timing-iters",
        str(args.timing_iters),
    ]
    buf = io.StringIO()
    with redirect_stdout(buf):
        inference_module.main(argv)
    return parse_json_lines(buf.getvalue())


def run_incremental(args: argparse.Namespace) -> list[dict]:
    argv = [
        "--prompt-lengths",
        *[str(x) for x in args.context_lengths],
        "--generate-tokens",
        str(args.generate_tokens),
        "--d-model",
        str(args.d_model),
        "--n-heads",
        str(args.n_heads),
        "--local-window",
        str(args.local_window),
        "--latent-tokens",
        str(args.latent_tokens),
        "--remote-chunk-size",
        str(args.remote_chunk_size),
    ]
    buf = io.StringIO()
    with redirect_stdout(buf):
        incremental_module.main(argv)
    return parse_json_lines(buf.getvalue())


def index_rows(rows: list[dict], key_name: str) -> dict[tuple[str, int], dict]:
    return {(row["model"], row[key_name]): row for row in rows}


def ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def format_table(rows: list[dict]) -> str:
    headers = [
        "ctx",
        "attn_mem_x",
        "kv_mem_x",
        "decode_kv_x",
        "decode_score_x",
        "explicit_cache_x",
        "explicit_score_x",
        "explicit_step_ms",
        "raw->compressed",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["context"]),
                    f"{row['attention_mem_ratio']:.2f}",
                    f"{row['kv_mem_ratio']:.2f}",
                    f"{row['decode_kv_ratio']:.2f}",
                    f"{row['decode_score_ratio']:.2f}",
                    f"{row['explicit_cache_ratio']:.2f}",
                    f"{row['explicit_score_ratio']:.2f}",
                    f"{row['explicit_step_ms_ratio']:.2f}",
                    f"{row['evicted_raw_tokens']}->{row['compressed_remote_tokens']}",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    benchmark_rows = run_benchmark(args)
    inference_rows = run_inference(args)
    incremental_rows = run_incremental(args)
    benchmark_index = index_rows(benchmark_rows, "seq_len")
    inference_index = index_rows(inference_rows, "prompt_len")
    incremental_index = index_rows(incremental_rows, "prompt_len")

    summary_rows = []
    for context in args.context_lengths:
        base_bench = benchmark_index[("baseline", context)]
        alt_bench = benchmark_index[("local_latent", context)]
        base_inf = inference_index[("baseline", context)]
        alt_inf = inference_index[("local_latent", context)]
        base_inc = incremental_index[("baseline", context)]
        alt_inc = incremental_index[("local_latent", context)]

        summary_rows.append(
            {
                "context": context,
                "attention_mem_ratio": ratio(
                    base_bench["estimated_attention_score_mib"],
                    alt_bench["estimated_attention_score_mib"],
                ),
                "kv_mem_ratio": ratio(
                    base_bench["estimated_kv_cache_mib"],
                    alt_bench["estimated_kv_cache_mib"],
                ),
                "decode_kv_ratio": ratio(
                    base_inf["simulated_decode_peak_kv_mib"],
                    alt_inf["simulated_decode_peak_kv_mib"],
                ),
                "decode_score_ratio": ratio(
                    base_inf["simulated_decode_total_score_million"],
                    alt_inf["simulated_decode_total_score_million"],
                ),
                "explicit_cache_ratio": ratio(
                    base_inc["peak_cache_tokens"],
                    alt_inc["peak_cache_tokens"],
                ),
                "explicit_score_ratio": ratio(
                    base_inc["total_score_elements"],
                    alt_inc["total_score_elements"],
                ),
                "explicit_step_ms_ratio": ratio(
                    base_inc["avg_decode_step_ms"],
                    alt_inc["avg_decode_step_ms"],
                ),
                "evicted_raw_tokens": alt_inf["simulated_evicted_raw_tokens"],
                "compressed_remote_tokens": alt_inf["simulated_compressed_remote_tokens"],
            }
        )

    report = {
        "config": {
            "context_lengths": args.context_lengths,
            "generate_tokens": args.generate_tokens,
            "batch_size": args.batch_size,
            "local_window": args.local_window,
            "latent_tokens": args.latent_tokens,
            "remote_chunk_size": args.remote_chunk_size,
            "gate_mode": args.gate_mode,
        },
        "summary_rows": summary_rows,
        "benchmark_rows": benchmark_rows,
        "inference_rows": inference_rows,
        "incremental_rows": incremental_rows,
    }

    print(format_table(summary_rows))
    print()
    print(json.dumps(report, indent=2))

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
