import argparse
import json
from pathlib import Path

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path, help="JSON run files produced by prototype.train")
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["eval_loss", "eval_bpb", "eval_acc", "train_loss", "peak_cuda_allocated_mib"],
    )
    parser.add_argument("--labels", nargs="*", default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/plots"))
    parser.add_argument("--title", type=str, default=None)
    return parser.parse_args()


def load_run(path: Path) -> dict:
    with path.open() as handle:
        data = json.load(handle)
    if "metrics" not in data:
        raise ValueError(f"{path} does not contain a metrics array")
    return data


def build_label(path: Path, run: dict, explicit_label: str | None) -> str:
    if explicit_label:
        return explicit_label
    summary = run.get("summary", {})
    model = summary.get("model")
    seq_len = summary.get("seq_len")
    seed = None
    config = run.get("config", {})
    if isinstance(config, dict):
        seed = config.get("seed")
    parts = [part for part in [model, f"L{seq_len}" if seq_len else None, f"s{seed}" if seed is not None else None] if part]
    return " ".join(parts) if parts else path.stem


def metric_points(metrics: list[dict], metric_name: str) -> tuple[list[int], list[float]]:
    xs: list[int] = []
    ys: list[float] = []
    for row in metrics:
        if metric_name not in row:
            continue
        xs.append(int(row["step"]))
        ys.append(float(row[metric_name]))
    return xs, ys


def main() -> None:
    args = parse_args()
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "matplotlib is required for prototype.plot_runs. Install it in your venv with: pip install matplotlib"
        ) from exc

    if args.labels is not None and len(args.labels) not in {0, len(args.runs)}:
        raise ValueError("--labels must be omitted or have the same length as the run list")

    runs = [load_run(path) for path in args.runs]
    labels = [
        build_label(path, run, args.labels[idx] if args.labels else None)
        for idx, (path, run) in enumerate(zip(args.runs, runs))
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    for metric_name in args.metrics:
        fig, ax = plt.subplots(figsize=(8, 5))
        plotted = False
        for path, run, label in zip(args.runs, runs, labels):
            xs, ys = metric_points(run["metrics"], metric_name)
            if not xs:
                continue
            plotted = True
            ax.plot(xs, ys, label=label, linewidth=2)
            ax.scatter(xs[-1], ys[-1], s=24)

        if not plotted:
            plt.close(fig)
            continue

        ax.set_xlabel("Step")
        ax.set_ylabel(metric_name)
        chart_title = metric_name.replace("_", " ")
        if args.title:
            chart_title = f"{args.title} - {chart_title}"
        ax.set_title(chart_title)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()

        output_path = args.output_dir / f"{metric_name}.png"
        fig.savefig(output_path, dpi=180)
        generated.append(output_path)
        plt.close(fig)

    print(
        json.dumps(
            {
                "runs": [str(path) for path in args.runs],
                "metrics": args.metrics,
                "output_dir": str(args.output_dir),
                "generated": [str(path) for path in generated],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
