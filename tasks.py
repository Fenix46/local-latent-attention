import random
from dataclasses import dataclass
from pathlib import Path

import torch

from tokenizers import TextTokenizer


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


@dataclass
class BinTextTaskConfig:
    """Dataset for pre-tokenized binary files.

    Expects a flat binary file of token ids produced by an external
    tokenizer pipeline (e.g. the SentencePiece .bin files).  Token ids
    are stored as uint16 (vocab ≤ 65535) or uint32, detected automatically
    from the vocab_size argument.

    The file is memory-mapped so even a 100 GB corpus uses negligible RAM.
    """
    path: Path
    seq_len: int = 4096
    vocab_size: int = 32000
    train_fraction: float = 0.9
    split: str = "train"
    seed: int = 0


class BinTextDataset:
    """Memory-mapped dataset over a pre-tokenized .bin file.

    Supports vocab sizes up to 2^32-1.  Uses uint16 storage for vocab ≤ 65535,
    uint32 otherwise — must match how the file was written.
    """

    def __init__(self, config: BinTextTaskConfig) -> None:
        self.config = config
        self.vocab_size = config.vocab_size
        self.bits_metric_name = "eval_bpt"
        self.tokenizer_kind = "pretokenized"
        self.rng = random.Random(config.seed)

        self._dtype = torch.uint16 if config.vocab_size <= 65535 else torch.uint32
        bytes_per_token = 2 if self._dtype == torch.uint16 else 4
        # Memory-map the file — zero RAM copy, OS handles paging.
        num_tokens = config.path.stat().st_size // bytes_per_token
        all_tokens = torch.from_file(
            str(config.path),
            shared=False,
            size=num_tokens,
            dtype=self._dtype,
        )
        # NOTE: no .long() here — we stay in uint16/uint32 to avoid copying
        # the entire corpus into RAM. The cast to long happens per-batch in
        # sample_batch(), only on the small slices we actually need.

        min_tokens = config.seq_len + 1
        if all_tokens.numel() < min_tokens * 2:
            raise ValueError(
                f"bin corpus too small for seq_len={config.seq_len}: "
                f"need ≥ {min_tokens * 2} tokens, got {all_tokens.numel()}"
            )

        split_idx = int(all_tokens.numel() * config.train_fraction)
        split_idx = max(min_tokens, min(split_idx, all_tokens.numel() - min_tokens))

        if config.split == "train":
            self.tokens = all_tokens[:split_idx]
        elif config.split == "eval":
            self.tokens = all_tokens[split_idx:]
        else:
            raise ValueError(f"unsupported split: {config.split!r}")

        if self.tokens.numel() < min_tokens:
            raise ValueError(
                f"split '{config.split}' too small: got {self.tokens.numel()} tokens"
            )

    def sample_batch(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        max_start = self.tokens.numel() - self.config.seq_len - 1
        starts = [self.rng.randint(0, max_start) for _ in range(batch_size)]
        # Cast to long only on the small slices — no full-corpus copy.
        inputs  = [self.tokens[s : s + self.config.seq_len].long()     for s in starts]
        targets = [self.tokens[s + 1 : s + self.config.seq_len + 1].long() for s in starts]
        x = torch.stack(inputs).to(device=device)
        y = torch.stack(targets).to(device=device)
        return x, y
