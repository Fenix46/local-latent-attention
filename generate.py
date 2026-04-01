import argparse
import json
from pathlib import Path

import torch

from prototype.models import build_model
from prototype.runtime import resolve_device
from prototype.tokenizers import load_text_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--prompt-file", type=Path, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
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


def sample_next_token(logits: torch.Tensor, temperature: float, top_k: int) -> torch.Tensor:
    if temperature <= 0.0:
        return logits.argmax(dim=-1, keepdim=True)

    scaled = logits / temperature
    filtered = filter_logits(scaled, top_k=top_k)
    probs = torch.softmax(filtered, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def build_model_from_checkpoint(checkpoint: dict, device: torch.device) -> torch.nn.Module:
    config = checkpoint["config"]
    if config.get("task") != "text":
        raise ValueError("prototype.generate currently supports only checkpoints trained with --task text")

    model = build_model(
        config["model"],
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
        latent_query_block_size=config["latent_query_block_size"],
        checkpoint_blocks=config["checkpoint_blocks"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    prompt_token_ids: list[int],
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    device: torch.device,
) -> list[int]:
    if len(prompt_token_ids) == 0:
        raise ValueError("prompt_token_ids must not be empty")

    input_ids = torch.tensor(prompt_token_ids, dtype=torch.long, device=device).unsqueeze(0)
    max_seq_len = model.config.max_seq_len

    for _ in range(max_new_tokens):
        context = input_ids[:, -max_seq_len:]
        logits = model(context)
        next_token = sample_next_token(
            logits[:, -1, :],
            temperature=temperature,
            top_k=top_k,
        )
        input_ids = torch.cat([input_ids, next_token], dim=1)

    return input_ids[0].tolist()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = build_model_from_checkpoint(checkpoint, device=device)
    config = checkpoint["config"]
    tokenizer = load_text_tokenizer(Path(config["tokenizer_model"]) if config.get("tokenizer_model") else None)
    prompt = load_prompt(args)
    prompt_token_ids = tokenizer.encode_text(prompt if prompt else "Once upon a time")
    generated_ids = generate(
        model=model,
        prompt_token_ids=prompt_token_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
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
                "device": str(device),
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
