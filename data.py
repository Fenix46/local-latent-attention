"""
Dataset loader for large pre-tokenized binary files.

Format: flat binary file of int32 or int16 token ids (SentencePiece output).
        Supports files larger than RAM via memory-mapped I/O.

Usage:
  dataset = BinaryDataset("data/train.bin", seq_len=8192, dtype="int32")
  x, y = dataset.sample_batch(batch_size=8, device=device)
"""

from __future__ import annotations
import os
import numpy as np
import torch
from pathlib import Path


class BinaryDataset:
    """
    Memory-mapped dataset for large pre-tokenized .bin files.

    The file is never fully loaded into RAM — numpy memmap reads
    only the pages actually accessed, letting the OS handle caching.

    Args:
      path          : path to the .bin file
      seq_len       : context length per sample
      dtype         : numpy dtype of stored tokens ("int32", "int16", "uint16")
      split         : "train" or "eval"
      train_fraction: fraction of tokens used for training
      vocab_size    : used for validation only
    """

    def __init__(
        self,
        path: Path | str,
        seq_len: int,
        dtype: str = "int32",
        split: str = "train",
        train_fraction: float = 0.98,
        vocab_size: int = 32000,
    ) -> None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")

        self.seq_len    = seq_len
        self.vocab_size = vocab_size

        # Memory-map the full file
        data = np.memmap(path, dtype=np.dtype(dtype), mode="r")
        n_tokens = len(data)
        split_idx = int(n_tokens * train_fraction)

        if split == "train":
            self.data = data[:split_idx]
        else:
            self.data = data[split_idx:]

        self.n_tokens = len(self.data)
        if self.n_tokens < seq_len + 1:
            raise ValueError(
                f"Split '{split}' has only {self.n_tokens} tokens, "
                f"need at least {seq_len + 1}."
            )

        file_gb = os.path.getsize(path) / 1e9
        print(f"[BinaryDataset] {path.name}  split={split}  "
              f"tokens={self.n_tokens:,}  file={file_gb:.1f}GB  dtype={dtype}")

    def sample_batch(
        self,
        batch_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample a random batch of (input, target) sequences.

        Returns:
          x : (batch_size, seq_len)   LongTensor
          y : (batch_size, seq_len)   LongTensor  (x shifted by 1)
        """
        max_start = self.n_tokens - self.seq_len - 1
        starts = np.random.randint(0, max_start, size=batch_size)

        x_list = []
        y_list = []
        for s in starts:
            chunk = self.data[s : s + self.seq_len + 1].astype(np.int64)
            x_list.append(chunk[:-1])
            y_list.append(chunk[1:])

        x = torch.from_numpy(np.stack(x_list)).to(device)
        y = torch.from_numpy(np.stack(y_list)).to(device)
        return x, y

    def __len__(self) -> int:
        return self.n_tokens
