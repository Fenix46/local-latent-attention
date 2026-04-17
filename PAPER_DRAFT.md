# Local-Latent Memory for Efficient Long-Context Transformer Training

## Abstract

Transformer self-attention provides strong sequence modeling performance but
incurs substantial memory and compute cost as context length grows. This work
investigates a constrained architectural modification, `local_latent`,
designed to preserve the overall Transformer training setup while replacing
dense global attention with a combination of exact local causal attention and a
compressed latent global memory, mixed through a learned per-head gate. The
goal is not to change the underlying modeling problem, but to study whether a
hierarchical attention structure can reduce memory pressure and remain
competitive in quality — and, crucially, whether those gains survive once the
prototype is ported to a production-oriented training stack at scales relevant
to modern hardware.

We report two complementary sets of experiments. The first is a suite of
controlled comparisons against a strong dense baseline (`flash_dense`,
implemented via PyTorch scaled dot-product attention) on a synthetic long-range
retrieval task, a mixed byte-level text corpus, and TinyStories. Across these
settings, the optimized `local_latent` variant shows a crossover behavior: it
becomes faster than `flash_dense` already at sequence length 2048, matches or
beats it in final loss at 4096 and 8192, and reduces peak CUDA allocation
substantially once blockwise latent queries and activation checkpointing are
enabled. On TinyStories at sequence length 2048 with 10k optimization steps
and parameter-matched settings, `local_latent` achieves mean eval loss 1.72
over three seeds versus 2.21 for `flash_dense`, with peak CUDA allocation
dropping from 565 MiB to 221 MiB.

The second set of experiments scales the system to a single NVIDIA H200
(141 GB) using a rewritten training pipeline with DistributedDataParallel
support, `torch.compile`, bf16 mixed precision, memory-mapped pre-tokenized
input, and gradient-accumulation-aware DDP synchronization. With a 540M
parameter `local_latent` model at sequence length 2048, `torch.compile`
approximately doubles throughput (from 57k to 108k tokens/second per rank)
while using 60 GB of VRAM, leaving ample headroom on the H200. A pilot run of
800 steps drops evaluation perplexity from 544 to 233 and raises next-token
accuracy from 12.7% to 18.3% with no observable overfitting gap. Bit-identical
numerical parity between single-process and DDP execution paths (`world_size=1`)
confirms that the distributed path introduces no drift. Checkpoint
save/load is validated across `torch.compile` and DDP wrapper layouts.

Together, these results support two claims. First, the local-latent design is
architecturally viable: properly implemented, it is Pareto-competitive with a
strong dense baseline in the long-context regime. Second, realizing that
benefit depends on implementation details at every level — local attention
kernel choice, latent-branch compression, blockwise queries, activation
checkpointing, compile, and distributed-training plumbing — in a way that
asymptotic complexity alone cannot predict.

## 1. Introduction

The standard Transformer uses dense self-attention:

\[
\mathrm{Att}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
\]

This design is highly effective but scales poorly with context length.
Theoretical improvements to long-context attention are abundant, yet many fail
to deliver practical speed or memory gains because the implementation itself
introduces unfavorable memory traffic, irregular kernel patterns, or excessive
framework overhead.

This project studies a specific hypothesis:

> A Transformer can remain competitive on long-context language modeling while
> replacing dense global attention with a structure that combines exact local
> attention and compressed global latent memory.

The emphasis is deliberately narrow. We do not attempt to redesign the
Transformer stack. We evolve only the attention mechanism in a structurally
conservative way, and we treat the choice of data pipeline, precision,
compiler, and distributed synchronization as first-class experimental
variables: a hierarchical attention design that is theoretically favorable but
impossible to train efficiently is of limited practical value.

## 2. Proposed Method

The proposed attention block decomposes context access into two paths:

1. Local causal attention over a fixed recent window.
2. Global latent attention over a compressed memory of the remote past.

For token position \(i\), the output is:

\[
Y_i = \lambda_i \cdot \mathrm{Att}_{local}(q_i, K_{\mathcal{N}(i)}, V_{\mathcal{N}(i)})
+ (1-\lambda_i) \cdot \mathrm{Att}_{latent}(q_i, \bar{K}, \bar{V})
\]

where:

- \(\mathcal{N}(i)\) is a causal local window of width \(w\);
- \(\bar{K}, \bar{V}\) are compressed latent summaries of the remote past,
  obtained by pooling the sequence into \(n_\ell\) chunks and projecting
  keys/values into a lower-dimensional latent space of width \(d_\ell\);
- \(\lambda_i \in [0,1]^H\) is a learned per-head gate.

The design goals are:

- exact high-resolution access to nearby tokens;
- low-resolution but non-zero access to remote context;
- bounded memory growth of the compressed remote branch with sequence length.

When the gate saturates to \(\lambda=1\), the block reduces to windowed local
attention; when it saturates to \(\lambda=0\), it reduces to attention against
a fixed-size latent memory. The mix is learned.

## 3. Implementation Details

### 3.1 Baselines

The prototype includes two dense decoder-only Transformer baselines:
token + positional embeddings, stacked attention/feed-forward blocks, causal
dense self-attention, and RMSNorm-style normalization. The first baseline
(`baseline`) uses an explicit masked attention implementation and served
mainly as a development reference. The stronger baseline used in all
comparisons reported here is `flash_dense`, which keeps the same dense causal
attention semantics but routes them through PyTorch scaled dot-product
attention (on CUDA this dispatches to cuDNN's flash SDPA on sm90).

### 3.2 Local-Latent Variant

The `local_latent` model keeps the decoder structure but replaces dense
self-attention with local attention + latent memory + learned gate. In the
latest version, the latent branch is low-rank: latent summaries are projected
into a smaller hidden space of width `latent_dim` before the remote attention
and projected back to model space only during the final mix. The strongest
long-context setting additionally uses blockwise latent queries
(`latent_query_block_size > 0`), so the remote branch is queried once per
macro-block rather than once per token.

For training-time memory reduction, we also evaluate activation checkpointing
across decoder blocks; this changes the systems-level memory/compute tradeoff
without altering the model's mathematical form.

### 3.3 Implementation Ablations

An important negative result emerged during development. A first
implementation of the local branch relied on `unfold`-based tensor
materialization. Although the architecture had better asymptotic memory
behavior, this implementation was substantially slower than the dense baseline
in practice and appeared to incur additional real-world memory overhead.

The turning point was to replace this local branch with a masked call to
`scaled_dot_product_attention`, allowing the local path to use a GPU-friendly
execution pattern. After this change, the expected practical benefits began
to appear. Later, a macro-block rewrite of the local attention further
reduced `local_attention_ms` and produced the throughput gains reported
below.

### 3.4 Production-Oriented Pipeline

To evaluate whether the architectural gains survive at scale, we rewrote the
training stack around the following components:

- **Memory-mapped pre-tokenized input.** `BinTextDataset` reads a flat
  `.bin` file produced by an external tokenizer pipeline (uint16 storage for
  vocab ≤ 65535), organized as a map-style `torch.utils.data.Dataset` with
  non-overlapping sequence windows. This is compatible with
  `DistributedSampler` and DataLoader workers, and scales to hundreds of GB
  corpora at negligible RAM cost.
- **Mixed precision.** bf16 autocast on Ampere+ (CC ≥ 8.0), auto-resolved at
  runtime. No loss scaler is required. No NaN/Inf observed in 800+ steps at
  540M parameters.
- **`torch.compile`.** Enabled with the default mode. The first step pays
  graph capture and autotune overhead (~80s for the 540M model); subsequent
  steps run against a warm cache. Checkpoints are saved with the
  `_orig_mod.` prefix stripped so they are portable across compile and DDP
  wrapper layouts.
- **DistributedDataParallel.** Wrapped with `gradient_as_bucket_view=True`
  and a tunable `bucket_cap_mb`. Gradient accumulation is made DDP-aware via
  a `maybe_no_sync` context manager that gates the all-reduce to the final
  micro-step, avoiding `(accum_steps − 1)` wasted collectives per optimizer
  step. Eval metrics are all-reduced as means. Checkpoint IO and logging are
  gated on rank 0.
- **`torchrun` launcher.** A thin wrapper around `torch.distributed.run`
  with single-node autodetection of GPU count, clean single-process
  fallback, and `expandable_segments` for long-run allocator stability.
- **Scheduled profiler.** `torch.profiler` with a fixed schedule
  (`wait=5, warmup=5, active=10`) exports chrome traces and a sorted
  `top_ops.txt`, making it straightforward to locate the real bottlenecks
  (rather than guessing from asymptotics).

## 4. Experimental Setup

### Tasks

Experiments were run on three families of benchmarks:

1. **Synthetic long-range retrieval.** Each sequence contains key-value
   marker pairs inserted earlier in the context and ends with a query token
   that asks the model to recover the corresponding value. Useful because
   it isolates long-context retrieval behavior without requiring a large
   external corpus.
2. **Byte-level next-token prediction** over a mixed local corpus and over
   TinyStories.
3. **Tokenized next-token prediction** over a pre-tokenized (.bin) corpus
   using a 32k SentencePiece vocabulary. This is the configuration used for
   the scaling experiments in §5.7.

### Model Family

- `flash_dense`: dense causal Transformer attention via PyTorch SDPA.
- `local_latent`: local attention + latent global memory + learned gate, as
  described in §2 and §3.

### Hardware and Framework

The quality comparisons at 2048/4096/8192 were performed on a Windows system
with an NVIDIA GeForce RTX 3060 Ti (8 GB VRAM) using CUDA-enabled PyTorch,
representative of memory-constrained research hardware. The scaling
experiments in §5.7 were performed on a single NVIDIA H200 (141 GB, Hopper,
sm90) running CUDA-enabled PyTorch. Due to cooling limitations on the
available H200 node, sustained SM clock ran at approximately 35–40% of
nominal during those experiments; observed throughput is therefore a
conservative lower bound.

### Metrics

We track training loss, evaluation loss, evaluation accuracy, evaluation
perplexity, bits-per-byte or bits-per-token where appropriate, steps per
second, tokens per second, and PyTorch peak CUDA allocation/reservation.

## 5. Results

### 5.1 Sequence Length 2048 (synthetic retrieval)

At sequence length 2048, batch size 16, 200 steps on RTX 3060 Ti:

| Model | Final Eval Loss | Final Eval Acc | Steps/s | Peak CUDA Allocated | Peak CUDA Reserved |
| --- | ---: | ---: | ---: | ---: | ---: |
| flash_dense | 0.002961 | 1.0 | 6.80 | 1050 MiB | 1126 MiB |
| local_latent | 0.004361 | 1.0 | 8.83 | 1387 MiB | 1506 MiB |

`local_latent` is ~1.30× faster; `flash_dense` retains a small final-loss
advantage; `local_latent` uses slightly more memory at this length.

### 5.2 Sequence Length 4096 (synthetic retrieval)

Batch size 8, 150 steps:

| Model | Final Eval Loss | Final Eval Acc | Steps/s | Peak CUDA Allocated | Peak CUDA Reserved |
| --- | ---: | ---: | ---: | ---: | ---: |
| flash_dense | 0.008842 | 1.0 | 4.05 | 1053 MiB | 1126 MiB |
| local_latent (`latent_d_model=64`, `latent_heads=2`, `latent_query_block_size=128`, `checkpoint_blocks`) | 0.005192 | 1.0 | 5.23 | 409 MiB | 450 MiB |

`local_latent` is ~1.29× faster, achieves better final loss, and uses ~2.6×
less peak CUDA allocation.

### 5.3 Sequence Length 8192 (synthetic retrieval, 2 seeds)

Batch size 16, 150 steps, mean over two seeds:

| Model | Final Eval Loss | Final Eval Acc | Steps/s | Peak CUDA Allocated | Peak CUDA Reserved |
| --- | ---: | ---: | ---: | ---: | ---: |
| flash_dense | 0.005743 | 1.0 | 0.58 | 4081 MiB | 4376 MiB |
| local_latent (`latent_d_model=64`, `latent_heads=2`, `latent_query_block_size=128`, `checkpoint_blocks`) | 0.002274 | 1.0 | 1.23 | 1529 MiB | 1548 MiB |

At 8192, `local_latent` is ~2.12× faster, better in loss, and uses ~2.67× less
peak CUDA allocation.

### 5.4 Byte-Level Text Benchmark at 8192

Mixed local corpus, batch size 4, 150 steps, mean over two seeds:

| Model | Final Eval Loss | Final Eval Acc | Eval BPB | Steps/s | Peak CUDA Allocated | Peak CUDA Reserved |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| flash_dense | 3.675423 | 0.1794 | 5.3025 | 2.27 | 1111 MiB | 1150 MiB |
| local_latent (…) | 3.754310 | 0.1863 | 5.4163 | 3.41 | 436 MiB | 476 MiB |

Near-parity in loss, ~1.50× faster, ~2.55× less peak CUDA.

### 5.5 TinyStories at 2048 (parameter-matched, 10k steps, 3 seeds)

Parameter-matched to within 64 parameters; batch size 8, 10k steps:

| Model | Mean Final Eval Loss | Mean Final Eval Acc | Mean Eval BPB | Peak CUDA Allocated | Peak CUDA Reserved | Params |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| flash_dense | 2.213773 | 0.3380 | 3.1938 | 565 MiB | 656 MiB | 823,936 |
| local_latent (`d_ff=223`, `latent_d_model=16`, `latent_heads=2`, `latent_query_block_size=128`, `checkpoint_blocks`) | 1.724723 | 0.4715 | 2.4882 | 221 MiB | 296 MiB | 824,000 |

The quality advantage holds per seed, not just in the mean. Throughput is
omitted because one `flash_dense` run was disturbed by host standby.

### 5.6 TinyStories at 8192 (parameter-matched scaling)

At sequence length 8192 and batch size 8, `flash_dense` was manually stopped
at 10k steps after flattening; the corresponding `local_latent` run was
continued to 50k steps. Single-seed results:

| Model | Horizon | Final Eval Loss | Final Eval Acc | Eval BPB | Peak CUDA Allocated | Peak CUDA Reserved | Params |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| flash_dense | 10k | 2.292234 | 0.3166 | 3.3070 | 2187 MiB | 2250 MiB | 1,610,368 |
| local_latent (…) | 10k | 1.907099 | 0.4246 | 2.7514 | 796 MiB | 1020 MiB | 1,610,432 |
| local_latent (…) | 50k | 0.814042 | 0.7437 | 1.1744 | 796 MiB | 1020 MiB | 1,610,432 |

At equal 10k steps, `local_latent` is already ahead. The 50k result is not a
symmetric benchmark but demonstrates that quality continues improving
substantially with the same peak memory.

### 5.7 Production Pipeline Scaling on H200

We evaluated the production pipeline on a single H200 with a 540M-parameter
`local_latent` model (`0.55b` preset: `d_model=1408`, `n_heads=16`,
`n_layers=22`, `d_ff=4096`, `latent_dim=512`, `latent_heads=8`) at sequence
length 2048, bf16 autocast, on a 32k SentencePiece-tokenized corpus stored as
a memory-mapped `.bin` file.

**Throughput.** `torch.compile` approximately doubles tokens-per-second per
rank. Increasing batch beyond bs=16 produces diminishing returns as the
workload leaves the GEMM-bound regime:

| Configuration | batch × accum | step_time | tokens/s | peak VRAM |
|---------------|---------------|-----------|---------:|----------:|
| baseline (no compile) | 4 × 4 | 0.58 s | 57k | 23 GB |
| + `torch.compile`, bs=16 | 16 × 1 | 0.30 s | 108k | 60 GB |
| + `torch.compile`, bs=32 | 32 × 1 | 0.59 s | 110k | 112 GB |

The thermal constraint on the available H200 node (sustained SM clock
~35–40% of nominal) means the 108k tok/s figure should be interpreted as a
floor; on a properly cooled H200 we expect approximately 2.5–3× this value.

**Numerical parity (single-process vs DDP).** We verified that invoking the
DDP path with `world_size=1` (`NPROC=1 ./launch_ddp.sh`) produces
bit-identical loss curves to single-process training under the same seed:

| step | `python train.py` | `torchrun nproc=1` |
|------|------------------:|-------------------:|
| 1    | 10.63702392578125 | 10.63702392578125 |
| 10   |  8.058294296264648 |  8.058294296264648 |
| 20   |  7.837799072265625 |  7.837799072265625 |
| eval |  7.71875 |  7.71875 |

The distributed path introduces no numerical drift. This rules out a broad
class of DDP integration bugs as a confound in downstream multi-rank
experiments.

**Learning curve.** An 800-step pilot on the H200 (bs=16, seq 2048,
`torch.compile`, bf16):

| step | train_loss | eval_loss | eval_acc | eval_ppl |
|------|-----------:|----------:|---------:|---------:|
|  100 |      6.82  |     —     |    —     |     —    |
|  250 |      6.23  |   6.30    |  0.127   |   544    |
|  500 |      5.94  |   5.90    |  0.149   |   365    |
|  750 |      5.33  |   5.45    |  0.183   |   233    |

Perplexity more than halves in 500 steps. Train and evaluation loss track
each other throughout, giving no evidence of overfitting at this training
horizon. No NaN or Inf were observed in bf16.

**Resume correctness.** After killing the pilot at step 500, resumption from
`checkpoint_latest.pt` produced step 525 training loss 5.86, continuous with
the pre-kill trajectory (step 500 loss 5.94). Resume works across the
`torch.compile` wrapper prefix after save/load were fixed to strip the
`_orig_mod.` / `module.` prefixes.

### 5.8 Bottleneck Analysis via `torch.profiler`

Profiler traces at three sequence lengths (1024, 2048, 4096; 10 active steps
each) summarized by CUDA self-time fraction:

| Category | s=1024 | s=2048 | s=4096 | Notes |
|----------|-------:|-------:|-------:|-------|
| GEMM (`mm` + `addmm`) | 29.9% | 29.5% | 25.4% | Dominant, scales sublinearly. |
| `aten::copy_` | 20.2% | 18.6% | 18.7% | Driven by `.contiguous()` after permutes in the attention block. |
| Attention fwd+bwd (cuDNN flash SDPA) | 11.0% | 12.2% | 14.1% | Scales O(S²); expected. |
| `aten::add_` | 8.3% | 10.4% | 14.2% | Residual + gradient scatter; grows with S. |

Two observations matter:

1. The attention path is already a fused flash kernel on sm90 — the
   `local_latent` block is not competing with a naïve attention
   implementation. Replacing it with a custom Triton kernel must therefore
   beat cuDNN's flash SDPA to be worth the complexity.
2. `aten::copy_` at ~20% is the largest non-GEMM bucket. It is dominated by
   contiguity conversions and pooling in the latent branch, both of which
   are amenable to a fused kernel or to eliminating redundant `.contiguous()`
   calls in the PyTorch path.

We implemented an experimental fused Triton kernel
(`LocalLatentAttentionTriton` in [llattn_triton.py](llattn_triton.py)) but
chose not to report scaling numbers from it until the cooling-limited H200
results are repeated on stable hardware. A parity test against the PyTorch
reference is included in the repository
([test_triton_kernel.py](test_triton_kernel.py)).

## 6. Discussion

### 6.1 What Worked

The optimized `local_latent` model achieves the intended qualitative result:
it preserves final task performance and, at sufficiently long context,
improves training throughput. With blockwise latent queries and activation
checkpointing, it also clearly dominates `flash_dense` in peak memory at
4096 and 8192 on the retrieval task, and achieves both a lower final loss
and a ~2.5× lower peak memory on the parameter-matched TinyStories 2048
comparison.

On the H200 scaling experiments, `torch.compile` nearly doubles throughput
on the 540M model, bf16 is stable, resume is robust across wrapper layouts,
and the DDP path is bit-identical to single-process. The pipeline is
therefore ready for multi-rank experiments; the remaining bottleneck is
hardware cooling on the available node.

### 6.2 What Failed Initially

The first `unfold`-based implementation of local attention produced a severe
regression. This negative result is important because it shows that
asymptotic complexity alone is not enough: a theoretically favorable
attention pattern can still lose badly if the implementation induces poor
memory access patterns or excessive tensor materialization.

A second, smaller failure mode emerged in the distributed path: saving a
checkpoint from a `torch.compile`d model without unwrapping leaves an
`_orig_mod.` prefix on every state-dict key, silently producing checkpoints
that do not load into the un-compiled model. The fix (unwrap on save, strip
prefixes on load) is trivial but illustrates how easily the wrapper stack
can produce subtle durability bugs.

### 6.3 Why the Optimized Version Matters

Once the local branch was reimplemented with masked scaled dot-product
attention, and once the latent branch was compressed into a lower-dimensional
space with blockwise queries, the practical behavior changed substantially.
This indicates that the viability of local-latent memory is not only an
architectural question but also a kernel- and implementation-level one.

### 6.4 Where the Advantage Appears

At 2048, `local_latent` is already faster than `flash_dense` but still
slightly behind in loss on the retrieval task. At 4096 it is better on
throughput, final loss, and peak memory. At 8192 the throughput advantage
widens further, the quality advantage is retained over two seeds, and
blockwise latent queries plus checkpointing produce a large peak-memory
reduction on the target 8 GB GPU. On TinyStories at 2048 with a
parameter-matched comparison, `local_latent` moves beyond near-parity and
achieves a large mean quality advantage over three seeds. On TinyStories at
8192, the same matched setting remains ahead at equal 10k steps and
continues improving in a longer single-seed 50k-step run, suggesting the
architecture may benefit from longer training horizons on natural-language
data.

### 6.5 Systems Lessons from the H200 Pilot

Three systems-level observations from the scaling run are worth highlighting:

- **`torch.compile` is a first-order speedup.** At 540M parameters it nearly
  doubles tokens/second per rank without changing the model. It is not a
  micro-optimization but a default.
- **Batch beyond GEMM saturation is wasted.** Raising bs from 16 to 32
  yielded +2% throughput for +87% VRAM on the H200. The useful regime for
  a given model/seq_len is narrower than intuition suggests; the profiler
  tells you when you have left it.
- **The "attention path is dense" assumption is wrong on modern CUDA
  stacks.** cuDNN's flash SDPA sm90 is already the baseline; a custom kernel
  must beat that, not a naïve implementation.

## 7. Limitations

- The synthetic retrieval task and the byte-level text corpus are not
  standardized public language-model benchmarks. TinyStories provides some
  corpus-based evidence but is itself a small simplified distribution.
- The 2048/4096/8192 quality comparisons use a small number of seeds
  (two to three), and the 8192 long-horizon scaling result is single-seed.
- The H200 scaling experiments were performed on a thermally limited node;
  throughput numbers should be read as a lower bound, and direct multi-rank
  (`world_size ≥ 2`) training was not exercised at scale in this work.
- The Triton fused kernel is implemented and passes a parity test but was
  not benchmarked against the PyTorch baseline at scale in this report.
- The strongest 4096 and 8192 results use activation checkpointing, so the
  final memory advantage reflects both architectural changes and a
  systems-level recomputation tradeoff.
- The latent branch still relies on simple pooled summaries; richer remote
  aggregation (e.g., learned pooling, cross-segment compression) is future
  work.

These limitations do not invalidate the core findings but should be
addressed before making broader claims.

## 8. Conclusion

This study evaluated a local-latent Transformer variant that replaces dense
global attention with a combination of local causal attention and compressed
latent memory. The central conclusion is threefold:

1. The architecture is functionally viable at long context lengths.
2. Practical benefit only emerges after implementation choices are aligned
   with efficient GPU execution — local-attention kernel, low-rank latent
   compression, blockwise latent queries, activation checkpointing.
3. The same emphasis on implementation carries over to the training stack:
   mixed-precision autocast, `torch.compile`, memory-mapped tokenized input,
   and DDP-aware gradient accumulation together convert the architectural
   gains into usable single-GPU throughput on modern hardware, without
   changing any of the model's mathematical properties.

After these optimizations, `local_latent` achieves a clear practical
advantage over `flash_dense` at long context lengths on the retrieval task
and a clear parameter-matched advantage on TinyStories. On a single H200 the
production pipeline reaches 108k tokens/second per rank (thermally limited)
with a 540M-parameter model, maintains bit-identical numerical parity between
single-process and DDP paths, and sustains stable bf16 training with robust
checkpoint save/resume across wrapper layouts.

The next steps are:

- multi-rank DDP training on a cooling-stable node to measure real scaling;
- longer training horizons on a standard tokenized corpus to characterize
  the scaling behavior of the architecture at fixed compute;
- a head-to-head benchmark of the fused Triton attention path against the
  PyTorch reference on stable hardware, with profiler-driven optimization
  of the `.contiguous()` and pooling overhead identified in §5.8.
