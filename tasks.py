import random
from dataclasses import dataclass
from pathlib import Path

import torch

from prototype.tokenizers import TextTokenizer


@dataclass
class RetrievalTaskConfig:
    vocab_size: int = 128
    seq_len: int = 256
    marker_token: int = 1
    query_token: int = 2
    pad_low: int = 3
    key_low: int = 16
    value_offset: int = 32
    num_pairs: int = 16


class RetrievalDataset:
    def __init__(self, config: RetrievalTaskConfig) -> None:
        self.config = config
        self.vocab_size = config.vocab_size

    def sample_batch(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = []
        targets = []
        for _ in range(batch_size):
            tokens, target = self._sample_sequence()
            inputs.append(tokens)
            targets.append(target)

        x = torch.tensor(inputs, dtype=torch.long, device=device)
        y = torch.tensor(targets, dtype=torch.long, device=device)
        return x, y

    def _sample_sequence(self) -> tuple[list[int], int]:
        cfg = self.config
        tokens = [random.randint(cfg.pad_low, cfg.vocab_size - 1) for _ in range(cfg.seq_len)]

        keys = random.sample(range(cfg.key_low, cfg.key_low + cfg.num_pairs), k=cfg.num_pairs)
        values = [key + cfg.value_offset for key in keys]
        mapping = dict(zip(keys, values))

        gap = max(8, cfg.seq_len // (cfg.num_pairs + 2))
        positions = [2 + i * gap for i in range(cfg.num_pairs)]
        for pos, key, value in zip(positions, keys, values):
            if pos + 1 >= cfg.seq_len - 2:
                break
            tokens[pos] = cfg.marker_token
            tokens[pos + 1] = key
            if pos + 2 < cfg.seq_len - 2:
                tokens[pos + 2] = value

        chosen_key = random.choice(keys)
        tokens[-2] = cfg.query_token
        tokens[-1] = chosen_key
        target = mapping[chosen_key]
        return tokens, target


@dataclass
class ByteTextTaskConfig:
    path: Path
    seq_len: int = 256
    train_fraction: float = 0.9
    split: str = "train"
    seed: int = 0
    vocab_size: int = 256


class ByteTextDataset:
    def __init__(self, config: ByteTextTaskConfig) -> None:
        self.config = config
        self.vocab_size = config.vocab_size
        self.bits_metric_name = "eval_bpb"
        self.tokenizer_kind = "byte"
        self.rng = random.Random(config.seed)

        raw = config.path.read_bytes()
        min_tokens = config.seq_len + 1
        if len(raw) < min_tokens * 2:
            raise ValueError(
                f"text corpus is too small for seq_len={config.seq_len}; "
                f"need at least {min_tokens * 2} bytes, got {len(raw)}"
            )

        split_idx = int(len(raw) * config.train_fraction)
        split_idx = max(min_tokens, min(split_idx, len(raw) - min_tokens))
        if config.split == "train":
            split_bytes = raw[:split_idx]
        elif config.split == "eval":
            split_bytes = raw[split_idx:]
        else:
            raise ValueError(f"unsupported text split: {config.split}")

        if len(split_bytes) < min_tokens:
            raise ValueError(
                f"text split '{config.split}' is too small for seq_len={config.seq_len}; "
                f"got {len(split_bytes)} bytes"
            )

        self.tokens = torch.tensor(list(split_bytes), dtype=torch.long)

    def sample_batch(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        max_start = self.tokens.numel() - self.config.seq_len - 1
        starts = [self.rng.randint(0, max_start) for _ in range(batch_size)]
        inputs = [self.tokens[start : start + self.config.seq_len] for start in starts]
        targets = [self.tokens[start + 1 : start + self.config.seq_len + 1] for start in starts]
        x = torch.stack(inputs).to(device=device, dtype=torch.long)
        y = torch.stack(targets).to(device=device, dtype=torch.long)
        return x, y


@dataclass
class TokenizedTextTaskConfig:
    path: Path
    seq_len: int = 256
    train_fraction: float = 0.9
    split: str = "train"
    seed: int = 0


class TokenizedTextDataset:
    def __init__(self, config: TokenizedTextTaskConfig, tokenizer: TextTokenizer) -> None:
        self.config = config
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size
        self.bits_metric_name = tokenizer.bits_metric_name
        self.tokenizer_kind = tokenizer.kind
        self.rng = random.Random(config.seed)

        token_ids = tokenizer.load_token_tensor(config.path)
        min_tokens = config.seq_len + 1
        if token_ids.numel() < min_tokens * 2:
            raise ValueError(
                f"text corpus is too small for seq_len={config.seq_len}; "
                f"need at least {min_tokens * 2} tokens, got {token_ids.numel()}"
            )

        split_idx = int(token_ids.numel() * config.train_fraction)
        split_idx = max(min_tokens, min(split_idx, token_ids.numel() - min_tokens))
        if config.split == "train":
            split_tokens = token_ids[:split_idx]
        elif config.split == "eval":
            split_tokens = token_ids[split_idx:]
        else:
            raise ValueError(f"unsupported text split: {config.split}")

        if split_tokens.numel() < min_tokens:
            raise ValueError(
                f"text split '{config.split}' is too small for seq_len={config.seq_len}; "
                f"got {split_tokens.numel()} tokens"
            )

        self.tokens = split_tokens

    def sample_batch(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        max_start = self.tokens.numel() - self.config.seq_len - 1
        starts = [self.rng.randint(0, max_start) for _ in range(batch_size)]
        inputs = [self.tokens[start : start + self.config.seq_len] for start in starts]
        targets = [self.tokens[start + 1 : start + self.config.seq_len + 1] for start in starts]
        x = torch.stack(inputs).to(device=device, dtype=torch.long)
        y = torch.stack(targets).to(device=device, dtype=torch.long)
        return x, y
