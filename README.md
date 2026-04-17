# Experimental Transformer Prototype

This prototype compares two small decoder-only language models on a synthetic
long-range retrieval task:

- `baseline`: standard causal Transformer attention
- `flash_dense`: dense causal attention backed by PyTorch scaled dot-product attention
- `local_latent`: causal local-window attention plus compressed latent memory

The goal is to test whether the `local_latent` attention can preserve useful
long-range signal while reducing the effective context cost.

The same training entrypoint can also run a more realistic byte-level language
modeling benchmark from any local text file.

## Files

- `prototype/models.py`: model definitions
- `prototype/tasks.py`: synthetic retrieval dataset
- `prototype/train.py`: training entrypoint
- `prototype/benchmark.py`: latency and parameter benchmark
- `prototype/inference.py`: autoregressive cache-growth estimate
- `prototype/generate.py`: text generation from a saved byte-level checkpoint
- `prototype/train_tokenizer.py`: train a SentencePiece tokenizer from a text corpus
- `prototype/check_sdpa.py`: inspect which PyTorch SDPA backends are usable for a given shape
- `prototype/plot_runs.py`: plot training curves from saved JSON run files
- `prototype/report.py`: merged comparison table for thesis-style reporting
- `prototype/incremental.py`: token-by-token cache benchmark with explicit state
- `prototype/incremental_block.py`: reusable incremental attention block with cache state
- `prototype/incremental_check.py`: sanity check for the reusable incremental block
- `prototype/incremental_model.py`: mini decoder LM using the incremental attention block
- `prototype/incremental_model_check.py`: forward vs step-by-step consistency check
- `prototype/incremental_train.py`: training harness for the incremental decoder LM
- `prototype/sweep.py`: multi-seed comparison sweep for incremental decoder experiments

## Quick start

Train the baseline:

```bash
python3 -m prototype.train --model baseline --device auto --steps 200
```

Train the local+latent variant:

```bash
python3 -m prototype.train --model local_latent --device auto --steps 200
```

Train the flash-backed dense baseline:

```bash
python3 -m prototype.train --model flash_dense --device auto --steps 200
```

Train on a local text corpus with byte-level next-token prediction:

```bash
python3 -m prototype.train --task text --text-path path/to/corpus.txt --model flash_dense --device auto --steps 200 --seq-len 512 --batch-size 8 --eval-batches 4
```

Train a SentencePiece tokenizer:

```bash
python3 -m prototype.train_tokenizer --input path/to/corpus.txt --model-prefix runs/tokenizers/tinystories_spm --vocab-size 4096 --model-type bpe --byte-fallback
```

Train on the same corpus with a learned tokenizer:

```bash
python3 -m prototype.train --task text --text-path path/to/corpus.txt --tokenizer-model runs/tokenizers/tinystories_spm.model --model local_latent --device auto --steps 200 --seq-len 512 --batch-size 8 --eval-batches 4
```

Train the strongest current long-context `local_latent` configuration on text:

```bash
python3 -m prototype.train --task text --text-path path/to/corpus.txt --model local_latent --device auto --steps 200 --seq-len 512 --batch-size 8 --latent-d-model 64 --latent-heads 2 --latent-query-block-size 128 --checkpoint-blocks --eval-batches 4
```

Save periodic checkpoints and a final model:

```bash
python3 -m prototype.train --task text --text-path path/to/corpus.txt --model local_latent --device auto --steps 1000 --seq-len 2048 --batch-size 8 --latent-d-model 64 --latent-heads 2 --latent-query-block-size 128 --checkpoint-blocks --save-dir runs/checkpoints/local_text --save-every 100 --save-final
```

Generate text from a saved checkpoint:

```bash
python3 -m prototype.generate --checkpoint runs/checkpoints/local_text/final.pt --prompt "Once upon a time" --max-new-tokens 200 --temperature 0.8 --top-k 50
```

Check which SDPA backend is usable for the dense baseline setup:

```bash
python3 -m prototype.check_sdpa --device cuda --batch-size 8 --seq-len 2048 --d-model 128 --n-heads 4 --dtype float16
```

Plot metrics from saved training runs:

```bash
python3 -m prototype.plot_runs runs/run_a.json runs/run_b.json --metrics eval_loss eval_bpb eval_acc --output-dir runs/plots/example
```

Run a quick benchmark:

```bash
python3 -m prototype.benchmark --context-lengths 128 256 512 1024
```

Run the inference-oriented cache estimate:

```bash
python3 -m prototype.inference --prompt-lengths 256 512 1024 --generate-tokens 128
```

Generate a compact comparison report:

```bash
python3 -m prototype.report --context-lengths 128 512 1024 --generate-tokens 128 --gate-mode improved
```

Run the explicit incremental cache benchmark:

```bash
python3 -m prototype.incremental --prompt-lengths 128 512 1024 --generate-tokens 128
```

Run the reusable incremental block sanity check:

```bash
python3 -m prototype.incremental_check --prompt-length 512 --generate-tokens 64
```

Run the incremental decoder consistency check:

```bash
python3 -m prototype.incremental_model_check --mode baseline --seq-len 64
```

Train the incremental decoder prototype:

```bash
python3 -m prototype.incremental_train --mode local_latent --device auto --gate-mode improved --steps 100 --seq-len 128
```

Run a small multi-seed sweep:

```bash
python3 -m prototype.sweep --seeds 0 1 --seq-lens 64 128 --steps 25
```

## Notes

- The scripts default to CPU because CUDA is not available in this environment.
- Training scripts support `--device auto|cpu|cuda|mps`; `auto` prefers CUDA,
  then MPS, then CPU.
- When running on CUDA, training summaries now also report
  `peak_cuda_allocated_mib` and `peak_cuda_reserved_mib` using PyTorch's
  internal peak-memory counters.
- `prototype.train` supports `--profile-attention` to add coarse timing logs for
  attention internals such as local, latent, and flash attention time.
- `prototype.train --task text` reads a local text file as raw bytes and trains
  a byte-level next-token model with `eval_loss`, `eval_acc`,
  `eval_perplexity`, and `eval_bpb`.
- Passing `--tokenizer-model path/to/model.model` switches the text task from
  raw bytes to SentencePiece token ids; in that mode the trainer reports
  `eval_bpt` instead of `eval_bpb`.
- Tokenized text runs now build and reuse an on-disk token cache under
  `runs/token_cache/`, so the first run over a large corpus is slower but
  subsequent train/eval splits reuse the same compact token stream.
- `prototype.train_tokenizer` currently trains SentencePiece tokenizers and
  writes `.model`, `.vocab`, and a small `.json` metadata file.
- `prototype.check_sdpa` probes whether the current PyTorch/CUDA stack can use
  Flash, memory-efficient, or math SDPA backends for a specific attention
  shape and dtype.
- `prototype.plot_runs` reads the full JSON payload written by `prototype.train`
  and saves one PNG per requested metric.
- `--text-path` is required for `--task text`; `--train-fraction` controls the
  train/eval split, and `--eval-batches` averages validation metrics across
  multiple random windows.
- `--save-dir` writes PyTorch checkpoints containing model state, optimizer
  state, run config, summary, and collected metrics.
- `--save-every N` saves `checkpoint_step_N.pt` style snapshots during training.
- `--save-final` writes `final.pt` at the end of the run.
- `prototype.train` also supports `--latent-d-model` and `--latent-heads` so
  the compressed remote branch can use a smaller hidden size than the local
  branch.
- `--latent-query-block-size` makes the remote latent branch operate on pooled
  query blocks instead of every token. Leaving it at `0` preserves the previous
  token-wise remote behavior.
- `--checkpoint-blocks` enables activation checkpointing across decoder blocks
  to trade extra compute for lower training memory without changing the model
  equations.
- The synthetic task plants a key token early in the sequence and asks the model
  to predict its paired value at the final position.
- The benchmark prints parameter counts, forward-pass timing, estimated attention
  score memory, and estimated KV-cache memory across different context lengths.
- The inference script estimates how KV-cache and attention memory grow during
  autoregressive decoding for the baseline and the compressed local+latent model.
- The decode simulation reports peak cache size and cumulative score workload
  during token-by-token generation for a full cache versus a hierarchical cache.
- The hierarchical cache simulation now compresses remote history in explicit
  chunks, so the reported remote memory reflects chunk-level condensation rather
  than a simple fixed-token cap.
- The report script merges benchmark and inference outputs into a compact table
  plus JSON payload, suitable for notes or thesis drafts.
- The incremental benchmark uses explicit token-by-token state to compare a full
  decode cache against a local window plus compressed remote summaries.
- The incremental training harness supports `simple` and `improved` gates so the
  routing policy can be studied without changing the remote-memory structure.
