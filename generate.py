import argparse
import json
from pathlib import Path

import torch

from models import build_model
from runtime import resolve_device
from tokenizers import load_text_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--prompt-file", type=Path, default=None)
    parser.add_argument(
        "--tokenizer-model",
        type=Path,
        default=None,
        help="Optional SentencePiece .model override. Required for --task bin checkpoints unless it was saved in the checkpoint config.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=3)
    parser.add_argument("--disable-eos-stop", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "float32", "bfloat16", "float16"], default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-file", type=Path, default=None)
    return parser.parse_args()


def load_prompt(args: argparse.Namespace) -> str:
    if args.prompt_file is not None:
        return args.prompt_file.read_text(encoding="utf-8")
    return args.prompt


def filter_logits(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    if top_k <= 0 or top_k >= logits.size(-1):
        return logits
    values, _ = torch.topk(logits, k=top_k, dim=-1)
    threshold = values[..., -1, None]
    return logits.masked_fill(logits < threshold, float("-inf"))


def apply_repetition_penalty(
    logits: torch.Tensor,
    generated_ids: torch.Tensor,
    repetition_penalty: float,
) -> torch.Tensor:
    if repetition_penalty <= 1.0 or generated_ids.numel() == 0:
        return logits

    adjusted = logits.clone()
    seen_tokens = torch.unique(generated_ids)
    seen_logits = adjusted[..., seen_tokens]
    adjusted[..., seen_tokens] = torch.where(
        seen_logits < 0,
        seen_logits * repetition_penalty,
        seen_logits / repetition_penalty,
    )
    return adjusted


def banned_ngram_tokens(token_ids: list[int], ngram_size: int) -> set[int]:
    if ngram_size <= 0 or len(token_ids) < ngram_size - 1:
        return set()

    prefix = tuple(token_ids[-(ngram_size - 1) :]) if ngram_size > 1 else tuple()
    banned: set[int] = set()
    limit = len(token_ids) - ngram_size + 1
    for idx in range(max(0, limit)):
        ngram = token_ids[idx : idx + ngram_size]
        if tuple(ngram[:-1]) == prefix:
            banned.add(ngram[-1])
    return banned


def sample_next_token(
    logits: torch.Tensor,
    temperature: float,
    top_k: int,
    generated_ids: torch.Tensor,
    repetition_penalty: float,
    banned_tokens: set[int],
) -> torch.Tensor:
    adjusted = apply_repetition_penalty(
        logits,
        generated_ids=generated_ids,
        repetition_penalty=repetition_penalty,
    )
    if banned_tokens:
        banned = torch.tensor(sorted(banned_tokens), device=adjusted.device, dtype=torch.long)
        adjusted[..., banned] = float("-inf")

    if temperature <= 0.0:
        return adjusted.argmax(dim=-1, keepdim=True)

    scaled = adjusted / temperature
    filtered = filter_logits(scaled, top_k=top_k)
    probs = torch.softmax(filtered, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def parse_dtype(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[name]


def infer_checkpoint_dtype(checkpoint: dict) -> torch.dtype:
    state_dict = checkpoint["model_state_dict"]
    for tensor in state_dict.values():
        if isinstance(tensor, torch.Tensor) and tensor.is_floating_point():
            return tensor.dtype
    return torch.float32


def resolve_inference_dtype(args: argparse.Namespace, device: torch.device, checkpoint: dict) -> torch.dtype:
    if args.dtype != "auto":
        return parse_dtype(args.dtype)

    checkpoint_dtype = infer_checkpoint_dtype(checkpoint)
    if device.type == "cuda" and checkpoint_dtype in {torch.float16, torch.bfloat16}:
        return checkpoint_dtype
    return torch.float32


def build_model_from_checkpoint(
    checkpoint: dict,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.nn.Module:
    config = checkpoint["config"]
    task = config.get("task")
    if task not in {"text", "bin"}:
        raise ValueError(
            "prototype.generate currently supports only checkpoints trained with --task text or --task bin"
        )

    model = build_model(
        vocab_size=config.get("vocab_size", 256),
        max_seq_len=config["seq_len"],
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        d_ff=config["d_ff"],
        local_window=config["local_window"],
        local_block_size=config["local_block_size"],
        latent_tokens=config["latent_tokens"],
        latent_d_model=config["latent_d_model"],
        latent_heads=config["latent_heads"],
        latent_query_block_size=config.get("latent_query_block_size", 0),
        checkpoint_blocks=config.get("checkpoint_blocks", False),
        use_triton_kernel=config.get("use_triton_kernel", False),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device=device, dtype=dtype)
    model.eval()
    return model


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    prompt_token_ids: list[int],
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
    eos_token_id: int | None,
    stop_on_eos: bool,
    device: torch.device,
) -> list[int]:
    if len(prompt_token_ids) == 0:
        raise ValueError("prompt_token_ids must not be empty")

    input_ids = torch.tensor(prompt_token_ids, dtype=torch.long, device=device).unsqueeze(0)
    max_seq_len = model.config.max_seq_len

    for _ in range(max_new_tokens):
        context = input_ids[:, -max_seq_len:]
        logits = model(context)
        banned_tokens = banned_ngram_tokens(input_ids[0].tolist(), no_repeat_ngram_size)
        next_token = sample_next_token(
            logits[:, -1, :],
            temperature=temperature,
            top_k=top_k,
            generated_ids=input_ids[0],
            repetition_penalty=repetition_penalty,
            banned_tokens=banned_tokens,
        )
        input_ids = torch.cat([input_ids, next_token], dim=1)
        if stop_on_eos and eos_token_id is not None and next_token.item() == eos_token_id:
            break

    return input_ids[0].tolist()


def resolve_eos_token_id(tokenizer) -> int | None:
    processor = getattr(tokenizer, "processor", None)
    if processor is None:
        return None
    eos_id = int(processor.eos_id())
    return eos_id if eos_id >= 0 else None


def resolve_generation_tokenizer(config: dict, tokenizer_model_override: Path | None = None):
    tokenizer_model = tokenizer_model_override
    if tokenizer_model is None and config.get("tokenizer_model"):
        tokenizer_model = Path(config["tokenizer_model"])

    if tokenizer_model is not None:
        return load_text_tokenizer(tokenizer_model)

    if config.get("task") == "text":
        return load_text_tokenizer(None)

    if config.get("task") == "bin":
        raise ValueError(
            "this checkpoint was trained with --task bin, but no tokenizer model is stored in the checkpoint. "
            "Pass --tokenizer-model with the same SentencePiece .model used to create the .bin dataset."
        )

    raise ValueError(f"unsupported checkpoint task for tokenizer resolution: {config.get('task')!r}")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    inference_dtype = resolve_inference_dtype(args, device=device, checkpoint=checkpoint)
    model = build_model_from_checkpoint(checkpoint, device=device, dtype=inference_dtype)
    config = checkpoint["config"]
    tokenizer = resolve_generation_tokenizer(config, tokenizer_model_override=args.tokenizer_model)
    eos_token_id = resolve_eos_token_id(tokenizer)
    checkpoint.pop("optimizer_state_dict", None)
    checkpoint.pop("metrics", None)
    prompt = load_prompt(args)
    prompt_token_ids = tokenizer.encode_text(prompt if prompt else "Once upon a time")
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

    print(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "prompt_chars": len(prompt),
                "generated_tokens": len(generated_ids),
                "temperature": args.temperature,
                "top_k": args.top_k,
                "repetition_penalty": args.repetition_penalty,
                "no_repeat_ngram_size": args.no_repeat_ngram_size,
                "eos_token_id": eos_token_id,
                "stop_on_eos": not args.disable_eos_stop,
                "device": str(device),
                "dtype": str(inference_dtype).replace("torch.", ""),
                "tokenizer": tokenizer.kind,
            }
        )
    )
    print()
    print(text)

    if args.output_file is not None:
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        args.output_file.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
