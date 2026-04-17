# Local-Latent Memory for Efficient Long-Context Transformer Training

## Abstract

Transformer self-attention provides strong sequence modeling performance but
 incurs substantial memory and compute cost as context length grows. This work
 investigates a constrained architectural modification, `local_latent`,
 designed to preserve the overall Transformer training setup while replacing
 dense global attention with a combination of local causal attention and a
 compressed latent global memory. The goal is not to change the underlying
 modeling problem, but to study whether a hierarchical attention structure can
 reduce memory pressure and remain competitive in quality.

We built a controlled PyTorch prototype and evaluated it first on a synthetic
 long-range retrieval task and then on byte-level next-token benchmarks over a
 mixed local text corpus and TinyStories. Early experiments showed that a naive
 implementation of local attention produced a severe practical regression
 despite better asymptotic behavior. Profiling then identified the local branch
 as the dominant bottleneck, leading first to a macro-block rewrite of local
 attention and then to a low-rank compressed latent branch. Against a strong
 dense baseline implemented with PyTorch scaled dot-product attention
 (`flash_dense`), the revised `local_latent` model became faster at long
 context lengths. On the synthetic retrieval task, at sequence length 2048,
 `local_latent` improved throughput from about 6.80 to about 8.83 steps/s,
 with a modest loss penalty (0.00436 vs. 0.00296) and peak CUDA allocation
 increasing from about 1050 MiB to about 1387 MiB. At sequence length 4096,
 the strongest practical configuration we observed (`latent_d_model=64`,
 `latent_heads=2`, `latent_query_block_size=128`, `checkpoint_blocks`)
 achieved 5.23 steps/s with final evaluation loss 0.00519, versus 4.05
 steps/s and 0.00884 for `flash_dense`, while using about 409 MiB instead of
 about 1053 MiB peak CUDA allocation. At sequence length 8192, the same
 setting achieved mean final evaluation loss about 0.00227 over two seeds,
 versus about 0.00574 for `flash_dense`, while also improving throughput from
 about 0.58 to about 1.23 steps/s and reducing peak CUDA allocation from about
 4081 MiB to about 1529 MiB. On the mixed byte-level text benchmark at
 sequence length 8192 and batch size 4, `local_latent` reached near-parity in
 loss with `flash_dense` (3.754 vs. 3.675 mean eval loss over two seeds) while
 remaining about 1.50x faster and reducing peak CUDA allocation from about
 1111 MiB to about 436 MiB. On TinyStories at sequence length 2048 and 10k
 optimization steps, a parameter-matched `local_latent` variant achieved mean
 eval loss about 1.72 over three seeds, versus about 2.21 for `flash_dense`,
 while also improving mean eval accuracy from about 0.338 to about 0.472 and
 reducing peak CUDA allocation from about 565 MiB to about 221 MiB. In a
 longer single-seed scaling run at sequence length 8192, the same
 parameter-matched `local_latent` setting already outperformed `flash_dense`
 at equal 10k optimization steps and then continued improving to final eval
 loss about 0.814 by 50k steps while staying below 0.8 GiB allocated CUDA
 memory.

These results suggest that the local-latent design is viable, but also show
 that implementation details are decisive: the architecture only becomes
 practically advantageous when the local attention path is implemented in a
 manner aligned with efficient GPU kernels.

## 1. Introduction

The standard Transformer uses dense self-attention:

\[
\mathrm{Att}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
\]

This design is highly effective, but it scales poorly with context length.
 Theoretical improvements to long-context attention are common, but many fail to
 deliver practical speed or memory gains because the implementation itself
 introduces unfavorable memory traffic, irregular kernel patterns, or excessive
 framework overhead.

This project studies a specific hypothesis:

> A Transformer can remain competitive on long-range retrieval while replacing
> dense global attention with a structure that combines exact local attention
> and compressed global latent memory.

The emphasis is deliberately narrow. We do not attempt to redesign the entire
 Transformer stack. We instead begin from a standard Transformer baseline and
 evolve only the attention mechanism in a structurally conservative way.

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

- \(\mathcal{N}(i)\) is a causal local window,
- \(\bar{K}, \bar{V}\) are compressed latent summaries of remote context,
- \(\lambda_i \in [0,1]\) is a learned gate.

The design goal is to preserve:

- exact high-resolution access to nearby tokens,
- low-resolution but non-zero access to remote context,
- bounded memory growth for the compressed remote branch.

## 3. Implementation Details

### 3.1 Baselines

The prototype includes two dense decoder-only Transformer baselines:

- token embeddings + positional embeddings,
- stacked attention/feed-forward blocks,
- causal dense self-attention,
- RMSNorm-style normalization.

The first baseline (`baseline`) uses an explicit masked attention
 implementation and served mainly as a development reference. The stronger
 baseline used in the main reported comparisons is `flash_dense`, which keeps
 the same dense causal attention semantics but routes them through PyTorch
 scaled dot-product attention.

### 3.2 Local-Latent Variant

The `local_latent` model keeps the same overall decoder structure but replaces
 dense self-attention with:

- local causal attention,
- latent memory attention,
- a learned gate per attention head.

In the latest version, the remote latent branch is also low-rank: the latent
 summaries are projected into a smaller hidden space before the remote
 attention, then projected back to model space only during the final mix. The
 strongest long-context setting additionally uses blockwise latent queries, so
 the remote branch is queried once per macro-block rather than once per token.
 For training-time memory reduction, we also evaluate activation checkpointing
 over decoder blocks; this changes the systems-level memory/compute tradeoff
 without altering the model's mathematical form.

Two gate variants were explored:

- `simple`: gate from the current token representation,
- `improved`: gate conditioned on query, local branch output, and remote branch
  output.

The improved gate produced better results than the simpler routing mechanism in
 short controlled experiments, without changing the memory architecture.

### 3.3 Implementation Ablation

An important negative result emerged during development. A first implementation
 of the local branch relied on `unfold`-based tensor materialization. Although
 the architecture had better asymptotic memory behavior, this implementation was
 substantially slower than the dense baseline in practice and appeared to incur
 additional real-world memory overhead.

The critical turning point was to replace this local branch with a masked call
 to `scaled_dot_product_attention`, allowing the local path to use a more
 GPU-friendly execution pattern. After this change, the expected practical
 benefits began to appear.

## 4. Experimental Setup

### Task

Experiments were run on three benchmarks.

The first is a synthetic long-range retrieval task. Each sequence contains
 key-value marker pairs inserted earlier in the context and ends with a query
 token that asks the model to recover the corresponding value. This task is
 useful because it isolates long-context retrieval behavior without requiring a
 large external corpus.

The second is a byte-level next-token prediction benchmark built from a mixed
 local corpus created by concatenating repository text files (`.py`, `.md`,
 `.txt`, `.json`, `.yaml`, `.yml`, `.toml`, `.csv`). This second benchmark is
 more realistic than retrieval, although it is still not a standardized public
 language-model corpus.

The third benchmark is TinyStories, trained in the same byte-level setup. This
 is a more realistic natural-language training setting than the mixed local
 corpus and serves as the first genuinely corpus-based training experiment in
 the project.

### Model Family

We evaluated:

- `flash_dense`: dense causal Transformer attention backed by PyTorch
  scaled dot-product attention
- `local_latent`: local attention + latent global memory

### Hardware and Framework

The key long-context training runs reported below were performed on a Windows
 system with an NVIDIA GeForce RTX 3060 Ti (8 GB VRAM) using CUDA-enabled
 PyTorch. Earlier development and debugging also took place on Apple Silicon,
 but PyTorch MPS was not available in that environment.

### Metrics

We tracked:

- training loss,
- evaluation loss,
- evaluation accuracy,
- steps per second,
- PyTorch peak CUDA memory allocation and reservation.

## 5. Results

### 5.1 Sequence Length 2048

Using framework-side CUDA memory logging, we compared `flash_dense` and the
 macro-block version of `local_latent` at sequence length 2048, batch size 16,
 and 200 optimization steps:

| Model | Final Eval Loss | Final Eval Acc | Steps/s | Peak CUDA Allocated | Peak CUDA Reserved |
| --- | ---: | ---: | ---: | ---: | ---: |
| flash_dense | 0.002961 | 1.0 | 6.80 | 1050 MiB | 1126 MiB |
| local_latent | 0.004361 | 1.0 | 8.83 | 1387 MiB | 1506 MiB |

Interpretation:

- `local_latent` is about 1.30x faster.
- The dense flash-backed baseline retains a quality advantage in final loss.
- `local_latent` currently uses more memory than `flash_dense` at this length.

### 5.2 Sequence Length 4096

Using framework-side CUDA memory logging, we compared `flash_dense` and the
 strongest practical `local_latent` configuration we observed at sequence
 length 4096, batch size 8, and 150 optimization steps:

| Model | Final Eval Loss | Final Eval Acc | Steps/s | Peak CUDA Allocated | Peak CUDA Reserved |
| --- | ---: | ---: | ---: | ---: | ---: |
| flash_dense | 0.008842 | 1.0 | 4.05 | 1053 MiB | 1126 MiB |
| local_latent (`latent_d_model=64`, `latent_heads=2`, `latent_query_block_size=128`, `checkpoint_blocks`) | 0.005192 | 1.0 | 5.23 | 409 MiB | 450 MiB |

Interpretation:

- At sequence length 4096, `local_latent` is about 1.29x faster.
- `local_latent` also achieves better final loss in this run.
- In this strongest practical setting, `local_latent` also reduces peak CUDA
  allocation substantially below `flash_dense`.

### 5.3 Sequence Length 8192

Using framework-side CUDA memory logging, we compared `flash_dense` and the
 strongest currently observed `local_latent` configuration at sequence length
 8192, batch size 16, and 150 optimization steps. The table reports the mean
 over two seeds for each model:

| Model | Final Eval Loss | Final Eval Acc | Steps/s | Peak CUDA Allocated | Peak CUDA Reserved |
| --- | ---: | ---: | ---: | ---: | ---: |
| flash_dense | 0.005743 | 1.0 | 0.58 | 4081 MiB | 4376 MiB |
| local_latent (`latent_d_model=64`, `latent_heads=2`, `latent_query_block_size=128`, `checkpoint_blocks`) | 0.002274 | 1.0 | 1.23 | 1529 MiB | 1548 MiB |

Interpretation:

- `local_latent` is about 2.12x faster.
- The low-rank `local_latent` variant also improves final loss over
  `flash_dense` at this length.
- With blockwise latent queries plus activation checkpointing, the strongest
  practical `local_latent` setting also reduces peak CUDA allocation
  substantially below `flash_dense` on the target 8 GB GPU.

### 5.4 Byte-Level Text Benchmark at 8192

To test whether the long-context behavior transferred beyond the synthetic
 retrieval setting, we also trained on the mixed local byte-level text corpus
 described above. We compared `flash_dense` and the strongest practical
 `local_latent` configuration at sequence length 8192, batch size 4, and 150
 optimization steps. The table reports the mean over two seeds for each model:

| Model | Final Eval Loss | Final Eval Acc | Eval BPB | Steps/s | Peak CUDA Allocated | Peak CUDA Reserved |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| flash_dense | 3.675423 | 0.1794 | 5.3025 | 2.27 | 1111 MiB | 1150 MiB |
| local_latent (`latent_d_model=64`, `latent_heads=2`, `latent_query_block_size=128`, `checkpoint_blocks`) | 3.754310 | 0.1863 | 5.4163 | 3.41 | 436 MiB | 476 MiB |

Interpretation:

- On this byte-level text benchmark, `local_latent` reaches near-parity with
  `flash_dense` in loss and bits-per-byte.
- `local_latent` is about 1.50x faster at the same batch size.
- `local_latent` also reduces peak CUDA allocation by about 2.55x.

### 5.5 TinyStories at 2048 (10k steps, parameter-matched)

To test whether the method remained competitive in a more realistic corpus
 setting, we trained on TinyStories in the same byte-level setup for 10k
 optimization steps at sequence length 2048 and batch size 8 over three seeds.
 One of the long `flash_dense` runs was affected by host standby, so we do not
 use wall-clock throughput from this experiment as evidence. The quality and
 memory results were stable and are summarized below as seed means. This
 comparison is now parameter-matched to within 64 parameters:

| Model | Mean Final Eval Loss | Mean Final Eval Acc | Mean Eval BPB | Peak CUDA Allocated | Peak CUDA Reserved | Params |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| flash_dense | 2.213773 | 0.3380 | 3.1938 | 565 MiB | 656 MiB | 823,936 |
| local_latent (`d_ff=223`, `latent_d_model=16`, `latent_heads=2`, `latent_query_block_size=128`, `checkpoint_blocks`) | 1.724723 | 0.4715 | 2.4882 | 221 MiB | 296 MiB | 824,000 |

Interpretation:

- On TinyStories, `local_latent` achieves a large quality advantage over
  `flash_dense` in mean eval loss, bits-per-byte, and eval accuracy.
- `local_latent` also reduces peak CUDA allocation by about 2.55x.
- In the three observed seeds, the quality advantage also holds per seed, not
  just in the mean.
- Qualitative sampling from the final checkpoints also suggested that the
  `local_latent` model produced more story-like completions than `flash_dense`
  at this training horizon, although this qualitative judgment is informal.

### 5.6 TinyStories at 8192 (parameter-matched scaling run)

We also tested whether the same parameter-matched setting would continue to
 improve at a much longer context length. At sequence length 8192 and batch
 size 8, `flash_dense` was manually stopped at 10k optimization steps after
 flattening substantially, while the corresponding `local_latent` run was
 continued to 50k steps. The table therefore reports both the equal-step
 comparison at 10k and the final single-seed `local_latent` result:

| Model | Training Horizon | Final Eval Loss | Final Eval Acc | Eval BPB | Peak CUDA Allocated | Peak CUDA Reserved | Params |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| flash_dense | 10k steps | 2.292234 | 0.3166 | 3.3070 | 2187 MiB | 2250 MiB | 1,610,368 |
| local_latent (`d_ff=223`, `latent_d_model=16`, `latent_heads=2`, `latent_query_block_size=128`, `checkpoint_blocks`) | 10k steps | 1.907099 | 0.4246 | 2.7514 | 796 MiB | 1020 MiB | 1,610,432 |
| local_latent (`d_ff=223`, `latent_d_model=16`, `latent_heads=2`, `latent_query_block_size=128`, `checkpoint_blocks`) | 50k steps | 0.814042 | 0.7437 | 1.1744 | 796 MiB | 1020 MiB | 1,610,432 |

Interpretation:

- At equal 10k optimization steps, the parameter-matched `local_latent` model
  is already better than `flash_dense` in eval loss, bits-per-byte, and eval
  accuracy.
- The parameter-matched `local_latent` model also reduces peak CUDA allocation
  by about 2.75x at this length.
- The long 8192 run does not show a hard plateau by 50k steps; quality
  continues improving substantially between 10k and 50k.
- Because the 8192 comparison beyond 10k is single-seed and the `flash_dense`
  run was stopped manually, this section should be read as strong scaling
  evidence rather than a fully symmetric final benchmark.

### 5.7 Development Ablations

The development process explored three qualitatively different local-attention
 implementations:

1. `unfold`-based local attention, which produced severe regressions.
2. masked scaled dot-product local attention, which was mathematically correct
   but still too slow against `flash_dense`.
3. macro-block local attention, which reduced `local_attention_ms`
   substantially and produced the throughput gains reported above.
4. a low-rank latent branch, which improved long-context quality while keeping
   most of the throughput advantage.
5. blockwise latent queries, which materially reduced the 8192-memory gap
   without giving back the quality advantage over `flash_dense`.
6. activation checkpointing, which converted the remaining 4096/8192 memory gap
   into a clear peak-memory advantage for the strongest practical training
   configuration.

The strongest pre-macro-block ablation setting combined `local_window=32` and
`latent_tokens=32`, improving quality somewhat but not changing the core
 conclusion that the local branch implementation itself was the decisive factor.

## 6. Discussion

### 6.1 What Worked

The optimized `local_latent` model achieves the intended qualitative result:

- it preserves final task performance,
- and, at sufficiently long context, improves training throughput.

However, after introducing a stronger dense baseline (`flash_dense`), the
 central tradeoff initially became more nuanced: the optimized
 `local_latent` model could outperform `flash_dense` in throughput and, after
 the low-rank latent update, also in final loss at 4096 and 8192, but it still
 trailed in peak memory. The final practical configuration resolves that gap:
 with blockwise latent queries and activation checkpointing, `local_latent`
 outperforms `flash_dense` in throughput, final loss, and peak memory at both
 4096 and 8192 on the synthetic retrieval task. On TinyStories at 2048,
 a parameter-matched `local_latent` model shows clearly better quality and much
 lower memory, with runtime measurements omitted there because one long
 baseline run was disturbed by system standby. At TinyStories 8192, the same
 parameter-matched setting remains ahead at equal 10k steps and then continues
 improving strongly in a longer single-seed run.

### 6.2 What Failed Initially

The initial implementation of the local branch produced a severe regression.
 This negative result is important because it shows that asymptotic complexity
 alone is not enough. A theoretically favorable attention pattern can still lose
 badly in practice if the implementation induces poor memory access patterns or
 excessive tensor materialization.

### 6.3 Why the Optimized Version Matters

Once the local branch was reimplemented with masked scaled dot-product
attention, the practical behavior changed substantially. This indicates that
the viability of local-latent memory is not only an architectural question but
also a kernel- and implementation-level one.

### 6.4 Where the Advantage Appears

The experimental picture now suggests a crossover behavior. At 2048,
 `local_latent` is already faster than `flash_dense`, but still behind in final
 loss in the measured run. At 4096, the strongest practical configuration is
 better on throughput, final loss, and peak memory. At 8192, the throughput
 advantage widens further, the quality advantage is retained over two seeds,
 and blockwise latent queries plus checkpointing produce a large peak-memory
 reduction on the target 8 GB GPU. On the byte-level text benchmark at 8192,
 `local_latent` reaches near-parity with `flash_dense` in loss while retaining
 a clear throughput and memory advantage. On TinyStories at 2048 and 10k
 steps, a parameter-matched `local_latent` model moves beyond near-parity and
 achieves a large mean quality advantage over three seeds. On TinyStories at
 8192, the same matched setting remains ahead at equal 10k steps and keeps
 improving in a longer 50k-step run, suggesting that the architecture may
 benefit from longer training horizons on natural-language data.

## 7. Limitations

This work remains a prototype study.

Main limitations:

- The main task remains synthetic, and the additional text benchmark uses a
  mixed repository-derived byte-level corpus rather than a standardized public
  language-model dataset.
- The strongest `flash_dense` versus `local_latent` results currently rely on a
  small number of seeds.
- Earlier development benchmarks against the weaker manual dense baseline are
  informative for engineering history, but are not the main evidence of the
  paper.
- The TinyStories 2048 comparison is now parameter-matched and uses three
  seeds, but the TinyStories 8192 scaling result is currently single-seed.
- Because one long TinyStories baseline run was affected by host standby, we
  do not treat the wall-clock timing from that experiment as paper-grade
  throughput evidence.
- The TinyStories 8192 baseline was manually stopped after flattening by 10k
  steps, so the longer-horizon comparison there is not fully symmetric.
- The incremental decoder path remains more prototype-oriented than
  production-optimized.
- The current latent branch still relies on simple pooled summaries, which may
  limit how much the model can reduce peak memory without giving back quality.
- All natural-language experiments remain byte-level. The generated samples are
  still visibly degraded, which motivates moving next to learned subword
  tokenization rather than staying in the byte-level regime.
- The strongest 4096 and 8192 results use activation checkpointing, so the
  final memory advantage reflects both architectural changes and a
  systems-level recomputation tradeoff.
- The strongest 8192 results use one blockwise latent-query setting (`128`) and
  only two seeds per model, so the exact Pareto frontier still needs broader
  confirmation.

These limitations do not invalidate the core findings, but they should be
 addressed in future work before making broader claims.

## 8. Conclusion

This study evaluated a local-latent Transformer variant that replaces dense
 global attention with a combination of local causal attention and compressed
 latent memory. The central conclusion is twofold:

1. The architecture is functionally viable at long context lengths.
2. Practical benefit only emerges after implementation choices are aligned with
   efficient GPU execution.

After optimizing the local attention path with macro-block processing,
compressing the remote latent branch into a lower-dimensional space, and then
introducing blockwise latent queries plus activation checkpointing, the
`local_latent` variant achieved a clear practical advantage over
`flash_dense` at long context lengths. At sequence length 2048 it became
faster, although still worse in final loss in the measured run. At sequence
length 4096 it was better in throughput, final loss, and peak memory. At
sequence length 8192 it remained substantially faster, also improved final
loss, and reduced peak CUDA allocation from about 4081 MiB to about 1529 MiB
in the best observed practical configuration. On the byte-level text benchmark
at sequence length 8192, it reached near-parity with `flash_dense` in final
loss while remaining about 1.50x faster and reducing peak CUDA allocation from
about 1111 MiB to about 436 MiB. On TinyStories at sequence length 2048 and
10k steps, a parameter-matched `local_latent` model achieved substantially
better mean eval loss and accuracy over three seeds than `flash_dense` while
also reducing peak CUDA allocation from about 565 MiB to about 221 MiB. At
sequence length 8192, the same matched setting was already ahead of
`flash_dense` at equal 10k optimization steps and then improved further to
0.814 final eval loss by 50k steps while staying below 0.8 GiB allocated CUDA
memory.

The main lesson is therefore not simply that hierarchical attention can work,
but that realizing its benefits requires both architectural design and careful
systems implementation. The next step is to move beyond byte-level language
modeling and test whether these gains survive once the prototype is trained
with a learned tokenizer on a more standard tokenized language-model setup.
