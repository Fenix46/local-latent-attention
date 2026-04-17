from __future__ import annotations

import hashlib
import json
from array import array
from dataclasses import dataclass
from pathlib import Path

import torch


class TextTokenizer:
    kind: str
    vocab_size: int
    bits_metric_name: str

    def encode_file(self, path: Path) -> list[int]:
        raise NotImplementedError

    def encode_text(self, text: str) -> list[int]:
        raise NotImplementedError

    def decode_ids(self, token_ids: list[int]) -> str:
        raise NotImplementedError

    def load_token_tensor(self, path: Path) -> torch.Tensor:
        return torch.tensor(self.encode_file(path), dtype=torch.long)


@dataclass
class ByteTokenizer(TextTokenizer):
    kind: str = "byte"
    vocab_size: int = 256
    bits_metric_name: str = "eval_bpb"

    def encode_file(self, path: Path) -> list[int]:
        return list(path.read_bytes())

    def encode_text(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode_ids(self, token_ids: list[int]) -> str:
        return bytes(token_ids).decode("utf-8", errors="replace")


class SentencePieceTokenizer(TextTokenizer):
    def __init__(self, model_path: Path) -> None:
        try:
            import sentencepiece as spm
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "sentencepiece is required for tokenized text runs. Install it in your venv with: pip install sentencepiece"
            ) from exc

        self.model_path = model_path
        self.kind = "sentencepiece"
        self.bits_metric_name = "eval_bpt"
        self.processor = spm.SentencePieceProcessor(model_file=str(model_path))
        self.vocab_size = int(self.processor.get_piece_size())

    def encode_file(self, path: Path) -> list[int]:
        token_ids: list[int] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                token_ids.extend(self.processor.encode(line, out_type=int))
        return token_ids

    def encode_text(self, text: str) -> list[int]:
        return list(self.processor.encode(text, out_type=int))

    def decode_ids(self, token_ids: list[int]) -> str:
        return self.processor.decode(token_ids)

    def load_token_tensor(self, path: Path) -> torch.Tensor:
        cache = self._ensure_token_cache(path)
        return torch.from_file(
            str(cache.data_path),
            shared=False,
            size=cache.num_tokens,
            dtype=cache.storage_dtype,
        )

    def _ensure_token_cache(self, text_path: Path) -> "TokenCache":
        cache = self._cache_paths(text_path)
        if self._cache_is_fresh(cache, text_path):
            return self._load_token_cache(cache)

        typecode, storage_dtype, storage_name = _storage_spec(self.vocab_size)
        tmp_path = cache.data_path.with_suffix(cache.data_path.suffix + ".tmp")
        num_tokens = 0

        cache.data_path.parent.mkdir(parents=True, exist_ok=True)
        with text_path.open("r", encoding="utf-8") as source, tmp_path.open("wb") as sink:
            for line in source:
                token_ids = self.processor.encode(line, out_type=int)
                if not token_ids:
                    continue
                chunk = array(typecode, token_ids)
                sink.write(chunk.tobytes())
                num_tokens += len(token_ids)

        tmp_path.replace(cache.data_path)
        metadata = {
            "text_path": str(text_path),
            "text_size": text_path.stat().st_size,
            "text_mtime_ns": text_path.stat().st_mtime_ns,
            "tokenizer_model": str(self.model_path),
            "tokenizer_model_size": self.model_path.stat().st_size,
            "tokenizer_model_mtime_ns": self.model_path.stat().st_mtime_ns,
            "vocab_size": self.vocab_size,
            "storage_dtype": storage_name,
            "num_tokens": num_tokens,
        }
        cache.meta_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        return TokenCache(
            data_path=cache.data_path,
            meta_path=cache.meta_path,
            num_tokens=num_tokens,
            storage_dtype=storage_dtype,
        )

    def _cache_paths(self, text_path: Path) -> "TokenCachePaths":
        fingerprint = hashlib.sha1(
            "|".join(
                [
                    str(text_path.resolve()),
                    str(text_path.stat().st_size),
                    str(text_path.stat().st_mtime_ns),
                    str(self.model_path.resolve()),
                    str(self.model_path.stat().st_size),
                    str(self.model_path.stat().st_mtime_ns),
                    str(self.vocab_size),
                ]
            ).encode("utf-8")
        ).hexdigest()[:16]
        cache_root = Path("runs") / "token_cache"
        stem = f"{text_path.stem}.{self.model_path.stem}.{fingerprint}"
        return TokenCachePaths(
            data_path=cache_root / f"{stem}.bin",
            meta_path=cache_root / f"{stem}.json",
        )

    def _cache_is_fresh(self, cache: "TokenCachePaths", text_path: Path) -> bool:
        if not cache.data_path.exists() or not cache.meta_path.exists():
            return False
        try:
            metadata = json.loads(cache.meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return False

        text_stat = text_path.stat()
        model_stat = self.model_path.stat()
        return (
            metadata.get("text_path") == str(text_path)
            and metadata.get("text_size") == text_stat.st_size
            and metadata.get("text_mtime_ns") == text_stat.st_mtime_ns
            and metadata.get("tokenizer_model") == str(self.model_path)
            and metadata.get("tokenizer_model_size") == model_stat.st_size
            and metadata.get("tokenizer_model_mtime_ns") == model_stat.st_mtime_ns
            and metadata.get("vocab_size") == self.vocab_size
            and "num_tokens" in metadata
            and "storage_dtype" in metadata
        )

    def _load_token_cache(self, cache: "TokenCachePaths") -> "TokenCache":
        metadata = json.loads(cache.meta_path.read_text(encoding="utf-8"))
        _, storage_dtype, storage_name = _storage_spec(self.vocab_size)
        if metadata["storage_dtype"] != storage_name:
            raise ValueError(
                f"token cache dtype mismatch for {cache.data_path}: "
                f"expected {storage_name}, found {metadata['storage_dtype']}"
            )
        return TokenCache(
            data_path=cache.data_path,
            meta_path=cache.meta_path,
            num_tokens=int(metadata["num_tokens"]),
            storage_dtype=storage_dtype,
        )


@dataclass
class TokenCachePaths:
    data_path: Path
    meta_path: Path


@dataclass
class TokenCache(TokenCachePaths):
    num_tokens: int
    storage_dtype: torch.dtype


def _storage_spec(vocab_size: int) -> tuple[str, torch.dtype, str]:
    if vocab_size <= 2**16 - 1:
        return "H", torch.uint16, "uint16"
    if vocab_size <= 2**32 - 1:
        return "I", torch.uint32, "uint32"
    raise ValueError(f"unsupported vocab_size for on-disk token cache: {vocab_size}")


def load_text_tokenizer(tokenizer_model: Path | None) -> TextTokenizer:
    if tokenizer_model is None:
        return ByteTokenizer()
    return SentencePieceTokenizer(tokenizer_model)
