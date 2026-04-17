# Local-Latent Transformer

A decoder-only language model that replaces dense self-attention with a two-path
mechanism: **exact local causal attention** over a recent window, plus a
**compressed latent memory** over the remote past, mixed through a learned
per-head gate. Built for long-context training on a single node (single or
multi-GPU) with a lean, production-oriented pipeline.

This repository contains the model, the training loop, a pre-tokenized data
pipeline, distributed-training helpers, and an experimental Triton kernel for
the fused local+latent attention block.

---

## Architecture at a glance

For each token `i`, the attention block computes two branches in parallel:

1. **Local branch** — exact causal attention restricted to a window of the
   nearest tokens. Uses PyTorch's flash scaled-dot-product attention on CUDA.
2. **Latent branch** — attention against a compressed memory of the remote
   past. The latent memory is obtained by pooling the sequence into
   `n_latents` chunks and projecting keys/values into a lower-dimensional
   `latent_dim`. The query side can operate token-wise or on pooled
   macro-blocks (`latent_query_block_size > 0`).
3. **Gate** — a per-head learned gate combines the two branches.

The design goal is bounded remote memory, exact local resolution, and a clean
fallback to a dense Transformer if the gate is saturated to one side.

Full model definition: [models.py](models.py) — `LocalLatentAttention` at
[models.py:61](models.py:61), the decoder stack at
[models.py:505](models.py:505).

---

## Files

**Core**
- [models.py](models.py) — `LocalLatentLM`, attention block, model presets,
  weight tying, `build_model` factory.
- [train.py](train.py) — single entry point. Supports single-GPU,
  DDP via `torchrun`, `torch.compile`, bf16/fp32/fp16, gradient accumulation,
  activation checkpointing, `torch.profiler`, checkpointing with
  atomic save + resume.
- [tasks.py](tasks.py) — three dataset classes:
  `RetrievalDataset` (synthetic), `ByteTextDataset`, `TokenizedTextDataset`,
  and `BinTextDataset` (memory-mapped pre-tokenized `.bin` files,
  map-style `torch.utils.data.Dataset`).
- [tokenizers.py](tokenizers.py) — SentencePiece loader with on-disk token
  cache under `runs/token_cache/`.
- [runtime.py](runtime.py) — device resolution, peak memory counters.
- [dist_utils.py](dist_utils.py) — thin wrapper around `torch.distributed`:
  `setup_distributed`, `all_reduce_mean`, `maybe_no_sync`, `unwrap`.
  Works transparently in single-process and `torchrun` modes.
- [launch_ddp.sh](launch_ddp.sh) — torchrun launcher, single-node, autodetects
  GPU count via `nvidia-smi`.

**Data preparation**
- [prepare_corpus.py](prepare_corpus.py) — download / deduplicate / tokenize a
  text corpus into a flat `.bin` file (uint16/uint32 depending on vocab size).
- [train_tokenizer.py](train_tokenizer.py) — train a SentencePiece tokenizer
  over a raw text file.

**Experimental Triton kernel (optional)**
- [llattn_triton.py](llattn_triton.py) — drop-in `LocalLatentAttentionTriton`
  replacement for the PyTorch attention block.
- [kernels/](kernels/) — forward/backward Triton kernels (`llattn_fwd.py`,
  `llattn_bwd.py`, `llattn_op.py`). Not on by default; enable with
  `--use-triton-kernel`. The PyTorch path remains the reference for
  correctness.

**Tests and generation**
- [test_triton_kernel.py](test_triton_kernel.py) — parity test: Triton
  forward/backward vs PyTorch reference.
- [test_generation.py](test_generation.py) — end-to-end generation sanity
  check.
- [generate.py](generate.py) — autoregressive generation from a saved
  checkpoint (temperature + top-k).

**Reports**
- [WEEKEND_REPORT.md](WEEKEND_REPORT.md) — hardware sweep, parity tests,
  bottleneck analysis from a single-H200 weekend.
- [PAPER_DRAFT.md](PAPER_DRAFT.md) — project paper draft.

---

## Requirements

- Python 3.10+ (tested on 3.12)
- PyTorch 2.4+ with CUDA for training on GPU (bf16 requires compute
  capability ≥ 8.0 — Ampere, Ada, Hopper)
- `sentencepiece` for tokenized text
- `triton` (optional) for the experimental fused kernel

---

## Data preparation

Train a SentencePiece tokenizer over a corpus:

```bash
python train_tokenizer.py --input path/to/corpus.txt \
  --model-prefix runs/tokenizers/my_spm --vocab-size 32000
```

Pre-tokenize the corpus into a flat `.bin` file (recommended for long runs —
memory-mapped, zero-copy reads):

```bash
python prepare_corpus.py --input path/to/corpus.txt \
  --tokenizer runs/tokenizers/my_spm.model \
  --output runs/token_cache/corpus.bin
```

---

## Training — single GPU

The strongest configuration tested so far on a single H200:

```bash
python train.py \
  --task bin --text-path runs/token_cache/corpus.bin \
  --vocab-size 32000 --model-preset 0.55b \
  --seq-len 2048 --batch-size 16 --accum-steps 1 \
  --steps 10000 --warmup-steps 200 --lr 3e-4 --lr-min 3e-5 \
  --torch-compile \
  --eval-every 500 --eval-batches 50 \
  --save-dir runs/my_pilot --save-every 1000 --save-final
```

Highlights:

- `--dtype auto` (default) selects bf16 on Ampere+ GPUs, fp32 otherwise.
- `--torch-compile` roughly doubles throughput; the first step pays graph
  capture and autotuning overhead.
- `--torch-profile --torch-profile-dir runs/profiles/X` writes a
  chrome trace + `top_ops.txt` for steps 11-20.
- `--checkpoint-blocks` trades compute for memory via activation
  checkpointing.

Other tasks:

```bash
# synthetic long-range retrieval (no external data)
python train.py --task retrieval --steps 200

# byte-level next-token on raw text
python train.py --task text --text-path path/to/corpus.txt \
  --seq-len 512 --batch-size 8 --steps 500

# SentencePiece-tokenized text (without pre-packing into a .bin)
python train.py --task text --text-path path/to/corpus.txt \
  --tokenizer-model runs/tokenizers/my_spm.model \
  --seq-len 2048 --batch-size 8 --steps 500
```

---

## Training — multi-GPU (single node)

```bash
./launch_ddp.sh \
  --task bin --text-path runs/token_cache/corpus.bin \
  --vocab-size 32000 --model-preset 0.55b \
  --seq-len 2048 --batch-size 8 --accum-steps 1 \
  --steps 10000 --warmup-steps 200 --lr 3e-4 --lr-min 3e-5 \
  --torch-compile \
  --save-dir runs/ddp_run --save-every 1000 --save-final
```

`launch_ddp.sh` autodetects the GPU count; override with
`NPROC=4 ./launch_ddp.sh ...`. Under DDP:

- `batch_size` is **per rank** — scale it down if you want to keep the
  effective global batch constant.
- `maybe_no_sync` gates the all-reduce to the final micro-step of gradient
  accumulation, avoiding (accum_steps − 1) wasted all-reduces.
- Checkpoint IO is gated on rank 0, tensors are all-reduced mean for eval.
- Resume works across different world sizes (state is layout-portable —
  `save_checkpoint` strips DDP / `torch.compile` prefixes).

A numerical parity test (same seed, `python train.py` vs
`NPROC=1 ./launch_ddp.sh ...`) produces bit-identical loss curves — see
[WEEKEND_REPORT.md](WEEKEND_REPORT.md).

---

## Resuming

```bash
python train.py ... --resume runs/my_pilot/checkpoint_latest.pt
```

The loader strips `_orig_mod.` / `module.` prefixes so checkpoints saved with
`torch.compile` or DDP are portable to any wrapper layout. LR schedule and
sampler epoch resume correctly.

---

## Generation

```bash
python generate.py --checkpoint runs/my_pilot/final.pt \
  --prompt "Once upon a time" --max-new-tokens 200 \
  --temperature 0.8 --top-k 50
```

---

## Model presets

Defined in [train.py](train.py), `MODEL_PRESETS`:

| preset | params | d_model | n_heads | n_layers | d_ff | latent_d_model | latent_heads |
|--------|-------:|--------:|--------:|---------:|-----:|---------------:|-------------:|
| `0.55b` | ~0.54B | 1408 | 16 | 22 | 4096 | 512 | 8 |
| `0.60b` | ~0.60B | 1536 | 16 | 22 | 4096 | 384 | 8 |

Override any field with `--d-model / --n-heads / --n-layers / --d-ff /
--latent-d-model / --latent-heads / --latent-tokens / --local-window`.

---

## Key CLI flags

| flag | purpose |
|------|---------|
| `--task {retrieval,text,bin}` | dataset family |
| `--model-preset {0.55b,0.60b}` | preset selector (or leave unset and specify each dim) |
| `--seq-len N` | training context length |
| `--batch-size N --accum-steps N` | per-rank batch, gradient accumulation |
| `--dtype {auto,float32,bfloat16,float16}` | autocast dtype |
| `--torch-compile` | enable `torch.compile` |
| `--checkpoint-blocks` | activation checkpointing per block |
| `--latent-query-block-size N` | pool queries into macro-blocks (0 = token-wise) |
| `--use-triton-kernel` | swap in the experimental fused Triton attention |
| `--torch-profile --torch-profile-dir DIR` | scheduled profiler trace |
| `--save-dir / --save-every / --save-final` | checkpointing |
| `--resume PATH` | resume from checkpoint (layout-portable) |
| `--num-workers N --prefetch-factor N` | DataLoader tuning |
| `--ddp-bucket-cap-mb N` | DDP all-reduce bucket size |

Full list: `python train.py --help`.

---

## Single-GPU H200 reference numbers

From a weekend sweep on one H200 (running thermally throttled at ~35% of
nominal clock — see [WEEKEND_REPORT.md](WEEKEND_REPORT.md) for details):

| config (540M params, seq 2048, bf16) | tok/s | peak VRAM |
|--------------------------------------|-------|-----------|
| baseline (bs=4, accum=4) | 57k | 23 GB |
| + `torch.compile`, bs=16 | **108k** | 60 GB |

Expected on a properly cooled H200: 2.5–3× these numbers.

---

## Status

- ✅ Single-GPU training pipeline stable, compile + bf16 + checkpointing
  validated.
- ✅ DDP pipeline bit-identical to single-process for `world_size=1`.
- ✅ Checkpoint resume validated across wrapper layouts.
- ⚠ Triton kernel parity tests exist but the fused path has not been
  benchmarked against the PyTorch baseline at scale — the cuDNN flash SDPA
  fallback is already competitive.
- ⚠ Multi-rank (`world_size ≥ 2`) not yet tested end-to-end on real workload.

---

## License

This project is licensed under the **GNU Affero General Public License v3.0
(AGPL-3.0)**. See [LICENSE](LICENSE) for the full text.

Key consequence: if you run a modified version of this software as a network
service, you must make the complete corresponding source code of your modified
version available to the users of that service under the same license.

Copyright (C) 2026 Emanuele Scarlata.
