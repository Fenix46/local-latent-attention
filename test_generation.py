#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from generate import (
    build_model_from_checkpoint,
    generate,
    resolve_eos_token_id,
    resolve_generation_tokenizer,
    resolve_inference_dtype,
)
from runtime import resolve_device


DEFAULT_PROMPT = "Once upon a time"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load a .pt checkpoint and run a quick generation test.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "checkpoint",
        nargs="?",
        type=Path,
        help="Checkpoint .pt file. If omitted, the script picks the newest checkpoint under --search-root.",
    )
    parser.add_argument(
        "--search-root",
        type=Path,
        default=Path("runs"),
        help="Directory searched recursively when checkpoint is omitted.",
    )
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--prompt-file", type=Path, default=None)
    parser.add_argument(
        "--tokenizer-model",
        type=Path,
        default=None,
        help="Optional SentencePiece .model override. Required for --task bin checkpoints unless it was saved in the checkpoint config.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=3)
    parser.add_argument("--disable-eos-stop", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "float32", "bfloat16", "float16"], default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-file", type=Path, default=None)
    parser.add_argument(
        "--show-config",
        action="store_true",
        help="Print the full checkpoint config before generating text.",
    )
    return parser.parse_args()


def load_prompt(args: argparse.Namespace) -> str:
    if args.prompt_file is not None:
        prompt = args.prompt_file.read_text(encoding="utf-8")
        return prompt if prompt else DEFAULT_PROMPT
    return args.prompt if args.prompt else DEFAULT_PROMPT


def resolve_checkpoint_path(checkpoint: Path | None, search_root: Path) -> Path:
    if checkpoint is not None:
        if not checkpoint.exists():
            raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
        return checkpoint

    if not search_root.exists():
        raise FileNotFoundError(
            f"search root does not exist: {search_root}. Pass a checkpoint path explicitly."
        )

    candidates = [path for path in search_root.rglob("*.pt") if path.is_file()]
    if not candidates:
        raise FileNotFoundError(
            f"no .pt checkpoint found under {search_root}. Pass a checkpoint path explicitly."
        )

    final_candidates = [path for path in candidates if path.name == "final.pt"]
    pool = final_candidates or candidates
    return max(pool, key=lambda path: path.stat().st_mtime)


def print_header(
    checkpoint_path: Path,
    checkpoint: dict,
    tokenizer_model: Path | None,
    prompt: str,
    prompt_token_ids: list[int],
    device: torch.device,
    inference_dtype: torch.dtype,
) -> None:
    config = checkpoint["config"]
    payload = {
        "checkpoint": str(checkpoint_path),
        "step": checkpoint.get("step"),
        "task": config.get("task"),
        "tokenizer_kind": config.get("tokenizer_kind"),
        "tokenizer_model": str(tokenizer_model) if tokenizer_model is not None else None,
        "seq_len": config.get("seq_len"),
        "vocab_size": config.get("vocab_size"),
        "d_model": config.get("d_model"),
        "n_layers": config.get("n_layers"),
        "n_heads": config.get("n_heads"),
        "device": str(device),
        "dtype": str(inference_dtype).replace("torch.", ""),
        "prompt_chars": len(prompt),
        "prompt_tokens": len(prompt_token_ids),
    }
    print(json.dumps(payload), flush=True)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    checkpoint_path = resolve_checkpoint_path(args.checkpoint, args.search_root)
    device = resolve_device(args.device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    inference_dtype = resolve_inference_dtype(args, device=device, checkpoint=checkpoint)
    model = build_model_from_checkpoint(checkpoint, device=device, dtype=inference_dtype)

    config = checkpoint["config"]
    resolved_tokenizer_model = args.tokenizer_model
    if resolved_tokenizer_model is None and config.get("tokenizer_model"):
        resolved_tokenizer_model = Path(config["tokenizer_model"])
    tokenizer = resolve_generation_tokenizer(config, tokenizer_model_override=args.tokenizer_model)
    eos_token_id = resolve_eos_token_id(tokenizer)
    prompt = load_prompt(args)
    prompt_token_ids = tokenizer.encode_text(prompt)
    if not prompt_token_ids:
        raise ValueError("the prompt produced zero token ids; pass a non-empty prompt or prompt file")

    print_header(
        checkpoint_path=checkpoint_path,
        checkpoint=checkpoint,
        tokenizer_model=resolved_tokenizer_model,
        prompt=prompt,
        prompt_token_ids=prompt_token_ids,
        device=device,
        inference_dtype=inference_dtype,
    )
    if args.show_config:
        print(json.dumps(config, indent=2, sort_keys=True), flush=True)

    generated_ids = generate(
        model=model,
        prompt_token_ids=prompt_token_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        eos_token_id=eos_token_id,
        stop_on_eos=not args.disable_eos_stop,
        device=device,
    )
    text = tokenizer.decode_ids(generated_ids)
    print()
    print(text)

    if args.output_file is not None:
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        args.output_file.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
