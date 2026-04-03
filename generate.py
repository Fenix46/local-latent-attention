"""
Inference / text generation for TelescopicAttention LM.

Usage:
  python generate.py \
    --checkpoint checkpoints/step_20000.pt \
    --tokenizer /opt/algoritmo/runs/tokenizers/fineweb_spm_32k.model \
    --prompt "The history of artificial intelligence" \
    --max-tokens 200 \
    --temperature 0.8 \
    --top-p 0.9
"""

from __future__ import annotations
import argparse
import json
import torch
import torch.nn.functional as F
from pathlib import Path

from model import ModelConfig, HierarchicalLM


# ──────────────────────────────────────────
# Tokenizer wrapper
# ──────────────────────────────────────────

class SPTokenizer:
    def __init__(self, model_path: str | Path) -> None:
        import sentencepiece as spm
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(str(model_path))

    def encode(self, text: str) -> list[int]:
        return self.sp.EncodeAsIds(text)

    def decode(self, ids: list[int]) -> str:
        return self.sp.DecodeIds(ids)

    @property
    def vocab_size(self) -> int:
        return self.sp.GetPieceSize()


# ──────────────────────────────────────────
# Sampling
# ──────────────────────────────────────────

def sample_token(
    logits: torch.Tensor,   # (vocab_size,)
    temperature: float,
    top_p: float,
    top_k: int,
) -> int:
    if temperature == 0.0:
        return logits.argmax().item()

    logits = logits / temperature

    # Top-k
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        kth = torch.topk(logits, top_k).values[-1]
        logits = logits.masked_fill(logits < kth, float("-inf"))

    # Top-p (nucleus)
    if top_p < 1.0:
        probs = torch.softmax(logits, dim=-1)
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumprobs = torch.cumsum(sorted_probs, dim=-1)
        # Remove tokens where cumulative prob exceeds top_p
        remove = cumprobs - sorted_probs > top_p
        sorted_probs[remove] = 0.0
        sorted_probs /= sorted_probs.sum()
        token = sorted_idx[torch.multinomial(sorted_probs, 1)].item()
    else:
        probs = torch.softmax(logits, dim=-1)
        token = torch.multinomial(probs, 1).item()

    return token


# ──────────────────────────────────────────
# Load model from checkpoint
# ──────────────────────────────────────────

def load_model(checkpoint_path: Path, device: torch.device) -> tuple[HierarchicalLM, ModelConfig]:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ModelConfig(**ckpt["config"])
    model = HierarchicalLM(config)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model, config


# ──────────────────────────────────────────
# Generation loop
# ──────────────────────────────────────────

@torch.no_grad()
def generate(
    model: HierarchicalLM,
    tokenizer: SPTokenizer,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    device: torch.device,
    bf16: bool,
) -> str:
    input_ids = tokenizer.encode(prompt)
    if not input_ids:
        raise ValueError("Prompt encodes to empty sequence")

    ids = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)  # (1, seq)

    generated = []
    for _ in range(max_tokens):
        # Truncate context to avoid OOM on very long sequences
        # Use last local_W * 4 tokens as context (safe heuristic)
        max_ctx = model.config.local_W * 4
        ctx = ids[:, -max_ctx:] if ids.size(1) > max_ctx else ids

        if bf16:
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                logits = model(ctx)
        else:
            logits = model(ctx)

        next_logits = logits[0, -1, :]   # (vocab_size,)
        next_token = sample_token(next_logits, temperature, top_p, top_k)

        generated.append(next_token)
        ids = torch.cat([ids, torch.tensor([[next_token]], device=device)], dim=1)

    return tokenizer.decode(input_ids + generated)


# ──────────────────────────────────────────
# Args
# ──────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",   type=Path, required=True)
    p.add_argument("--tokenizer",    type=Path, required=True)
    p.add_argument("--prompt",       type=str,  default="The")
    p.add_argument("--max-tokens",   type=int,  default=200)
    p.add_argument("--temperature",  type=float, default=0.8)
    p.add_argument("--top-p",        type=float, default=0.9)
    p.add_argument("--top-k",        type=int,  default=50)
    p.add_argument("--bf16",         action="store_true")
    p.add_argument("--device",       default="auto")
    p.add_argument("--n-samples",    type=int,  default=1)
    return p.parse_args()


# ──────────────────────────────────────────
# Main
# ──────────────────────────────────────────

def main() -> None:
    args = parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else
                              "mps"  if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Loading model from {args.checkpoint}...")
    model, config = load_model(args.checkpoint, device)
    n_params = model.count_parameters()
    print(f"Model loaded: {n_params:,} params, device={device}")
    print(json.dumps(config.__dict__, indent=2))

    print(f"\nLoading tokenizer from {args.tokenizer}...")
    tokenizer = SPTokenizer(args.tokenizer)
    print(f"Vocab size: {tokenizer.vocab_size}")

    print(f"\nPrompt: {args.prompt!r}")
    print(f"Generating {args.max_tokens} tokens × {args.n_samples} sample(s)...\n")
    print("─" * 60)

    for i in range(args.n_samples):
        text = generate(
            model       = model,
            tokenizer   = tokenizer,
            prompt      = args.prompt,
            max_tokens  = args.max_tokens,
            temperature = args.temperature,
            top_p       = args.top_p,
            top_k       = args.top_k,
            device      = device,
            bf16        = args.bf16,
        )
        if args.n_samples > 1:
            print(f"[Sample {i+1}]")
        print(text)
        print("─" * 60)


if __name__ == "__main__":
    main()
