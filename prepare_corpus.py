import argparse
import gzip
import hashlib
import html
import json
import re
import sqlite3
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator


DEFAULT_JUNK_SUBSTRINGS = [
    "cookie",
    "cookies",
    "javascript",
    "<html",
    "</html",
    "privacy policy",
    "terms of service",
    "terms of use",
    "all rights reserved",
    "enable javascript",
]

URL_RE = re.compile(r"https?://|www\.")
WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
WHITESPACE_RE = re.compile(r"\s+")
CONTROL_RE = re.compile(r"[\x00-\x08\x0B-\x1F\x7F]")
SIZE_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([kmgt]?i?b)?\s*$", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream a large text corpus, clean it aggressively, and write one normalized document per line."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-path", type=Path, default=None)
    source.add_argument("--hf-dataset", type=str, default=None)

    parser.add_argument("--hf-config", type=str, default=None)
    parser.add_argument("--hf-split", type=str, default="train")
    parser.add_argument("--hf-cache-dir", type=Path, default=None)
    parser.add_argument("--text-column", type=str, default="text")
    parser.add_argument("--input-format", choices=["auto", "jsonl", "txt", "parquet"], default="auto")
    parser.add_argument("--language", type=str, default="en")
    parser.add_argument("--language-mode", choices=["auto", "metadata", "langdetect", "none"], default="auto")
    parser.add_argument("--language-column", type=str, default="language")
    parser.add_argument("--language-score-column", type=str, default="language_score")
    parser.add_argument("--min-language-score", type=float, default=0.8)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stats-output", type=Path, default=None)
    parser.add_argument("--dedupe-db", type=Path, default=None)
    parser.add_argument("--normalize", choices=["none", "nfc", "nfkc"], default="nfkc")
    parser.add_argument("--min-chars", type=int, default=400)
    parser.add_argument("--max-chars", type=int, default=20000)
    parser.add_argument("--min-letter-ratio", type=float, default=0.7)
    parser.add_argument("--max-equals", type=int, default=20)
    parser.add_argument("--max-braces", type=int, default=20)
    parser.add_argument("--max-urls", type=int, default=3)
    parser.add_argument("--min-unique-word-ratio", type=float, default=0.38)
    parser.add_argument("--min-words-for-unique-ratio", type=int, default=40)
    parser.add_argument("--junk-substring", action="append", default=None)
    parser.add_argument("--prefix-dedupe-chars", type=int, default=256)
    parser.add_argument("--prefix-dedupe-min-chars", type=int, default=256)
    parser.add_argument("--max-output-bytes", type=str, default=None)
    parser.add_argument("--max-docs", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=10000)
    parser.add_argument("--sqlite-commit-every", type=int, default=1000)
    return parser.parse_args()


def parse_byte_size(value: str | None) -> int | None:
    if value is None:
        return None
    match = SIZE_RE.match(value)
    if match is None:
        raise ValueError(f"invalid byte size: {value}")
    number = float(match.group(1))
    unit = (match.group(2) or "b").lower()
    multipliers = {
        "b": 1,
        "kb": 10**3,
        "mb": 10**6,
        "gb": 10**9,
        "tb": 10**12,
        "kib": 2**10,
        "mib": 2**20,
        "gib": 2**30,
        "tib": 2**40,
    }
    if unit not in multipliers:
        raise ValueError(f"unsupported byte size unit: {unit}")
    return int(number * multipliers[unit])


def normalize_text(text: str, mode: str) -> str:
    text = html.unescape(text)
    if mode != "none":
        text = unicodedata.normalize(mode.upper(), text)
    text = CONTROL_RE.sub(" ", text)
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text


def collect_input_files(path: Path, input_format: str) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise ValueError(f"input path does not exist: {path}")

    suffixes = {
        "jsonl": {".jsonl", ".gz"},
        "txt": {".txt"},
        "parquet": {".parquet"},
    }
    if input_format == "auto":
        candidates = []
        for candidate in path.rglob("*"):
            if candidate.is_file() and candidate.suffix.lower() in {".jsonl", ".gz", ".txt", ".parquet"}:
                candidates.append(candidate)
        return sorted(candidates)

    allowed = suffixes[input_format]
    return sorted(candidate for candidate in path.rglob("*") if candidate.is_file() and candidate.suffix.lower() in allowed)


def detect_input_format(path: Path, explicit_format: str) -> str:
    if explicit_format != "auto":
        return explicit_format
    if path.suffix.lower() == ".parquet":
        return "parquet"
    if path.suffix.lower() == ".txt":
        return "txt"
    if path.suffix.lower() == ".jsonl":
        return "jsonl"
    if path.suffix.lower() == ".gz" and path.name.endswith(".jsonl.gz"):
        return "jsonl"
    raise ValueError(f"could not infer input format for {path}")


class SQLiteHashStore:
    def __init__(self, path: Path, commit_every: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("CREATE TABLE IF NOT EXISTS exact_hashes (digest BLOB PRIMARY KEY)")
        self.conn.execute("CREATE TABLE IF NOT EXISTS prefix_hashes (digest BLOB PRIMARY KEY)")
        self.commit_every = commit_every
        self.pending = 0

    def insert_exact(self, digest: bytes) -> bool:
        cursor = self.conn.execute("INSERT OR IGNORE INTO exact_hashes (digest) VALUES (?)", (sqlite3.Binary(digest),))
        self.pending += 1
        self._maybe_commit()
        return cursor.rowcount == 1

    def insert_prefix(self, digest: bytes) -> bool:
        cursor = self.conn.execute("INSERT OR IGNORE INTO prefix_hashes (digest) VALUES (?)", (sqlite3.Binary(digest),))
        self.pending += 1
        self._maybe_commit()
        return cursor.rowcount == 1

    def _maybe_commit(self) -> None:
        if self.pending >= self.commit_every:
            self.conn.commit()
            self.pending = 0

    def close(self) -> None:
        if self.pending:
            self.conn.commit()
            self.pending = 0
        self.conn.close()


class LanguageDetector:
    def __init__(self, mode: str) -> None:
        self.mode = mode
        self._detect_langs = None
        if mode == "langdetect":
            try:
                from langdetect import detect_langs
            except ImportError as exc:
                raise RuntimeError(
                    "langdetect is required for --language-mode langdetect. Install it with `pip install langdetect`."
                ) from exc
            self._detect_langs = detect_langs

    def passes(self, row: dict, text: str, args: argparse.Namespace) -> bool:
        if args.language_mode == "none":
            return True

        if args.language_mode in {"auto", "metadata"}:
            lang = row.get(args.language_column)
            if isinstance(lang, str):
                if lang != args.language:
                    return False
                score = row.get(args.language_score_column)
                if score is None:
                    return True
                try:
                    return float(score) >= args.min_language_score
                except (TypeError, ValueError):
                    return False
            if args.language_mode == "metadata":
                return False

        if args.language_mode in {"auto", "langdetect"}:
            if self._detect_langs is None:
                if args.language_mode == "langdetect":
                    raise RuntimeError("langdetect backend requested but detector is unavailable")
                raise RuntimeError(
                    "language filtering needs metadata columns or langdetect. "
                    "For raw local files, install langdetect and use --language-mode langdetect."
                )
            sample = text[:2000]
            try:
                guesses = self._detect_langs(sample)
            except Exception:
                return False
            if not guesses:
                return False
            best = guesses[0]
            return best.lang == args.language and best.prob >= args.min_language_score

        return False


def hash_bytes(text: str) -> bytes:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).digest()


def iter_hf_rows(args: argparse.Namespace) -> Iterator[dict]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "datasets is required for --hf-dataset. Install it with `pip install datasets`."
        ) from exc

    dataset = load_dataset(
        args.hf_dataset,
        args.hf_config,
        split=args.hf_split,
        streaming=True,
        cache_dir=str(args.hf_cache_dir) if args.hf_cache_dir is not None else None,
    )
    yield from dataset


def iter_jsonl_rows(path: Path, text_column: str) -> Iterator[dict]:
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                yield row
            else:
                yield {text_column: str(row)}


def iter_txt_rows(path: Path, text_column: str) -> Iterator[dict]:
    with path.open("rt", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield {text_column: line}


def iter_parquet_rows(path: Path) -> Iterator[dict]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "pyarrow is required to read parquet input. Install it with `pip install pyarrow`."
        ) from exc

    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches():
        columns = batch.schema.names
        arrays = [batch.column(i).to_pylist() for i in range(len(columns))]
        for values in zip(*arrays):
            yield dict(zip(columns, values))


def iter_local_rows(args: argparse.Namespace) -> Iterator[dict]:
    files = collect_input_files(args.input_path, args.input_format)
    if not files:
        raise ValueError(f"no input files found under {args.input_path}")
    for path in files:
        fmt = detect_input_format(path, args.input_format)
        if fmt == "jsonl":
            yield from iter_jsonl_rows(path, args.text_column)
        elif fmt == "txt":
            yield from iter_txt_rows(path, args.text_column)
        elif fmt == "parquet":
            yield from iter_parquet_rows(path)
        else:
            raise ValueError(f"unsupported input format: {fmt}")


def count_urls(text: str) -> int:
    return len(URL_RE.findall(text))


def letter_ratio(text: str) -> float:
    if not text:
        return 0.0
    letters = sum(char.isalpha() for char in text)
    return letters / len(text)


def unique_word_ratio(text: str) -> float:
    words = WORD_RE.findall(text.lower())
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def repeated_line_score(text: str) -> float:
    words = WORD_RE.findall(text.lower())
    if len(words) < 30:
        return 0.0
    window = 3
    grams = [" ".join(words[i : i + window]) for i in range(len(words) - window + 1)]
    counts = Counter(grams)
    most_common = counts.most_common(1)[0][1]
    return most_common / len(grams)


def filter_document(
    row: dict,
    args: argparse.Namespace,
    language_detector: LanguageDetector,
    hash_store: SQLiteHashStore | None,
    junk_substrings: list[str],
) -> tuple[str | None, str | None]:
    raw_text = row.get(args.text_column)
    if not isinstance(raw_text, str):
        return None, "missing_text"

    text = normalize_text(raw_text, args.normalize)
    if not text:
        return None, "empty"
    if len(text) < args.min_chars:
        return None, "too_short"
    if len(text) > args.max_chars:
        return None, "too_long"
    if not language_detector.passes(row, text, args):
        return None, "language"

    lower = text.lower()
    if any(fragment in lower for fragment in junk_substrings):
        return None, "junk_substring"
    if text.count("=") > args.max_equals:
        return None, "too_many_equals"
    if text.count("{") + text.count("}") > args.max_braces:
        return None, "too_many_braces"
    if count_urls(text) > args.max_urls:
        return None, "too_many_urls"
    if letter_ratio(text) < args.min_letter_ratio:
        return None, "low_letter_ratio"

    words = WORD_RE.findall(lower)
    if len(words) >= args.min_words_for_unique_ratio and unique_word_ratio(text) < args.min_unique_word_ratio:
        return None, "low_unique_word_ratio"
    if repeated_line_score(text) > 0.2:
        return None, "repetition"

    if hash_store is not None:
        exact_digest = hash_bytes(text)
        if not hash_store.insert_exact(exact_digest):
            return None, "exact_duplicate"
        if len(text) >= args.prefix_dedupe_min_chars:
            prefix = text[: args.prefix_dedupe_chars]
            prefix_digest = hash_bytes(prefix)
            if not hash_store.insert_prefix(prefix_digest):
                return None, "prefix_duplicate"

    return text, None


def main() -> None:
    args = parse_args()
    max_output_bytes = parse_byte_size(args.max_output_bytes)
    junk_substrings = [fragment.lower() for fragment in DEFAULT_JUNK_SUBSTRINGS]
    if args.junk_substring:
        junk_substrings.extend(fragment.lower() for fragment in args.junk_substring)
    dedupe_db = args.dedupe_db
    if dedupe_db is None:
        dedupe_db = args.output.with_suffix(".dedupe.sqlite")

    language_detector = LanguageDetector(args.language_mode)
    hash_store = SQLiteHashStore(dedupe_db, commit_every=args.sqlite_commit_every)

    stats = {
        "source": {
            "input_path": str(args.input_path) if args.input_path is not None else None,
            "hf_dataset": args.hf_dataset,
            "hf_config": args.hf_config,
            "hf_split": args.hf_split,
            "text_column": args.text_column,
            "language": args.language,
            "language_mode": args.language_mode,
        },
        "processed_docs": 0,
        "kept_docs": 0,
        "written_bytes": 0,
        "skip_reasons": {},
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    iterator = iter_hf_rows(args) if args.hf_dataset is not None else iter_local_rows(args)

    try:
        with args.output.open("wt", encoding="utf-8") as handle:
            for row in iterator:
                stats["processed_docs"] += 1
                cleaned, reason = filter_document(row, args, language_detector, hash_store, junk_substrings)
                if cleaned is None:
                    stats["skip_reasons"][reason] = stats["skip_reasons"].get(reason, 0) + 1
                else:
                    encoded = (cleaned + "\n").encode("utf-8")
                    handle.write(cleaned)
                    handle.write("\n")
                    stats["kept_docs"] += 1
                    stats["written_bytes"] += len(encoded)

                if args.log_every and stats["processed_docs"] % args.log_every == 0:
                    print(json.dumps(stats, ensure_ascii=False), file=sys.stderr)

                if max_output_bytes is not None and stats["written_bytes"] >= max_output_bytes:
                    break
                if args.max_docs and stats["kept_docs"] >= args.max_docs:
                    break
    finally:
        hash_store.close()

    if args.stats_output is None:
        args.stats_output = args.output.with_suffix(args.output.suffix + ".stats.json")
    args.stats_output.parent.mkdir(parents=True, exist_ok=True)
    args.stats_output.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
