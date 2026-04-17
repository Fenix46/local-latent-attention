import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model-prefix", type=Path, required=True)
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--model-type", choices=["bpe", "unigram"], default="bpe")
    parser.add_argument("--character-coverage", type=float, default=1.0)
    parser.add_argument("--byte-fallback", action="store_true")
    parser.add_argument("--normalization-rule-name", type=str, default="nmt_nfkc")
    parser.add_argument("--pad-id", type=int, default=0)
    parser.add_argument("--unk-id", type=int, default=1)
    parser.add_argument("--bos-id", type=int, default=2)
    parser.add_argument("--eos-id", type=int, default=3)
    parser.add_argument("--shuffle-input-sentence", action="store_true")
    parser.add_argument("--input-sentence-size", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import sentencepiece as spm
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "sentencepiece is required for prototype.train_tokenizer. Install it in your venv with: pip install sentencepiece"
        ) from exc

    args.model_prefix.parent.mkdir(parents=True, exist_ok=True)
    spm.SentencePieceTrainer.train(
        input=str(args.input),
        model_prefix=str(args.model_prefix),
        vocab_size=args.vocab_size,
        model_type=args.model_type,
        character_coverage=args.character_coverage,
        byte_fallback=args.byte_fallback,
        normalization_rule_name=args.normalization_rule_name,
        pad_id=args.pad_id,
        unk_id=args.unk_id,
        bos_id=args.bos_id,
        eos_id=args.eos_id,
        shuffle_input_sentence=args.shuffle_input_sentence,
        input_sentence_size=args.input_sentence_size,
    )

    metadata = {
        "input": str(args.input),
        "model_prefix": str(args.model_prefix),
        "model_file": str(args.model_prefix.with_suffix(".model")),
        "vocab_file": str(args.model_prefix.with_suffix(".vocab")),
        "vocab_size": args.vocab_size,
        "model_type": args.model_type,
        "character_coverage": args.character_coverage,
        "byte_fallback": args.byte_fallback,
        "normalization_rule_name": args.normalization_rule_name,
        "pad_id": args.pad_id,
        "unk_id": args.unk_id,
        "bos_id": args.bos_id,
        "eos_id": args.eos_id,
    }
    metadata_path = args.model_prefix.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
