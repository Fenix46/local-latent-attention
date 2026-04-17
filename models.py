import math
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _is_compiling() -> bool:
    compiler = getattr(torch, "compiler", None)
    if compiler is not None and hasattr(compiler, "is_compiling"):
        return bool(compiler.is_compiling())
    dynamo = getattr(torch, "_dynamo", None)
    if dynamo is not None and hasattr(dynamo, "is_compiling"):
        return bool(dynamo.is_compiling())
    return False


@dataclass
class ModelConfig:
    vocab_size: int = 128
    max_seq_len: int = 1024
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 256
    dropout: float = 0.0
    local_window: int = 64
    local_block_size: int = 256
    latent_tokens: int = 16
    latent_d_model: int = 64
    latent_heads: int = 2
    latent_query_block_size: int = 0
    checkpoint_blocks: bool = False
    use_triton_kernel: bool = False


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (self.weight.shape[0],), self.weight, self.eps)


class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.w1 = nn.Linear(config.d_model, config.d_ff)
        self.w2 = nn.Linear(config.d_ff, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.gelu(self.w1(x))))


class LocalLatentAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        assert config.d_model % config.n_heads == 0
        assert config.latent_d_model % config.latent_heads == 0
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.window = config.local_window
        self.local_block_size = config.local_block_size
        self.latent_tokens = config.latent_tokens
        self.latent_heads = config.latent_heads
        self.latent_d_model = config.latent_d_model
        self.latent_head_dim = config.latent_d_model // config.latent_heads
        self.latent_query_block_size = max(0, config.latent_query_block_size)

        # These fused linears replace three separate GEMMs each. At the target
        # shapes the projection stack dominates far more than the attention
        # core, so reducing launch count here matters more than custom kernels.
        self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model)
        self.rq_proj = nn.Linear(config.d_model, config.latent_d_model)
        self.rkv_proj = nn.Linear(config.d_model, 2 * config.latent_d_model)
        self.r_out = nn.Linear(config.latent_d_model, config.d_model)
        self.out = nn.Linear(config.d_model, config.d_model)
        # Online-softmax gate: one scalar logit per head per branch,
        # conditioned on each branch's own output.  Numerically equivalent
        # to softmax([s_local, s_remote]) but computed without ever forming
        # a combined logit vector — same streaming trick as FlashAttention.
        self.gate_local  = nn.Linear(self.head_dim, 1, bias=False)
        self.gate_remote = nn.Linear(self.head_dim, 1, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self._n_layers = config.n_layers  # stored for weight init scaling
        self._mask_cache: dict[tuple[int, int, int, torch.device], torch.Tensor] = {}
        self.profile_enabled = False
        self.profile_totals: dict[str, float] = {}
        self.profile_steps = 0

    def _load_from_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
        local_metadata,
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        def _fuse_linear(new_name: str, old_names: tuple[str, ...]) -> None:
            new_weight = prefix + new_name + ".weight"
            new_bias = prefix + new_name + ".bias"
            old_weight_keys = [prefix + name + ".weight" for name in old_names]
            old_bias_keys = [prefix + name + ".bias" for name in old_names]

            if new_weight not in state_dict and all(key in state_dict for key in old_weight_keys):
                state_dict[new_weight] = torch.cat([state_dict.pop(key) for key in old_weight_keys], dim=0)
            if new_bias not in state_dict and all(key in state_dict for key in old_bias_keys):
                state_dict[new_bias] = torch.cat([state_dict.pop(key) for key in old_bias_keys], dim=0)

        _fuse_linear("qkv_proj", ("q_proj", "k_proj", "v_proj"))
        _fuse_linear("rkv_proj", ("rk_proj", "rv_proj"))
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def set_profile(self, enabled: bool) -> None:
        self.profile_enabled = enabled
        self.profile_totals = {}
        self.profile_steps = 0

    def get_profile_stats(self) -> dict[str, float]:
        if not self.profile_enabled or self.profile_steps == 0:
            return {}
        return {
            "profile_local_latent_total_ms": self.profile_totals["total_ms"] / self.profile_steps,
            "profile_qkv_ms": self.profile_totals["qkv_ms"] / self.profile_steps,
            "profile_pool_ms": self.profile_totals["pool_ms"] / self.profile_steps,
            "profile_local_attention_ms": self.profile_totals["local_attention_ms"] / self.profile_steps,
            "profile_latent_attention_ms": self.profile_totals["latent_attention_ms"] / self.profile_steps,
            "profile_gate_mix_ms": self.profile_totals["gate_mix_ms"] / self.profile_steps,
            "profile_gate_mean": self.profile_totals["gate_mean"] / self.profile_steps,
            "profile_seq_len": self.profile_totals["seq_len"] / self.profile_steps,
            "profile_local_block_size": self.profile_totals["local_block_size"] / self.profile_steps,
            "profile_latent_count": self.profile_totals["latent_count"] / self.profile_steps,
            "profile_latent_chunk_size": self.profile_totals["chunk_size"] / self.profile_steps,
            "profile_latent_query_block_size": self.profile_totals["latent_query_block_size"] / self.profile_steps,
        }

    @staticmethod
    def _sync(device: torch.device) -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def _pool_to_latents(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Pool x into at most latent_tokens summary vectors.

        `latent_tokens` is treated as an upper bound, not a guaranteed output count.
        The actual number of pooled latents depends on seq_len and the derived
        chunk_size.

        This avoids invalid reshapes for cases like:
          seq_len=130, latent_tokens=128 -> chunk_size=2 -> actual_chunks=65
        """
        batch, seq_len, dim = x.shape

        if seq_len <= 0:
            raise ValueError("seq_len must be positive")

        # Target maximum number of latent summaries.
        target_chunks = min(self.latent_tokens, seq_len)
        if target_chunks <= 0:
            raise ValueError("latent_tokens must be positive")

        # Chunk size chosen so that the number of chunks is <= target_chunks.
        chunk_size = math.ceil(seq_len / target_chunks)

        full_chunks = seq_len // chunk_size
        remainder = seq_len % chunk_size

        parts = []

        # Full chunks: [batch, full_chunks, chunk_size, dim] -> mean over chunk axis
        if full_chunks > 0:
            full = (
                x[:, : full_chunks * chunk_size, :]
                .contiguous()
                .view(batch, full_chunks, chunk_size, dim)
                .mean(dim=2)
            )
            parts.append(full)

        # Final partial chunk, averaged over real tokens only
        if remainder > 0:
            last = x[:, full_chunks * chunk_size :, :].mean(dim=1, keepdim=True)
            parts.append(last)

        if not parts:
            # This should never happen for seq_len > 0, but keep it explicit.
            raise RuntimeError("failed to create latent chunks")

        pooled = parts[0] if len(parts) == 1 else torch.cat(parts, dim=1)

        actual_chunks = pooled.size(1)

        # Map each token position to its latent chunk index.
        chunk_ids = torch.arange(seq_len, device=x.device) // chunk_size
        chunk_ids = chunk_ids.clamp(max=actual_chunks - 1)

        return pooled, chunk_ids

    def _pool_query_blocks(self, x: torch.Tensor, query_chunk_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, seq_len, dim = x.shape
        block_size = min(seq_len, self.latent_query_block_size)
        block_count = math.ceil(seq_len / block_size)
        pad_len = block_count * block_size - seq_len

        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))

        pooled = x.view(batch, block_count, block_size, dim).mean(dim=2)
        block_chunk_ids = []
        for start in range(0, seq_len, block_size):
            end = min(start + block_size, seq_len)
            block_chunk_ids.append(query_chunk_ids[start:end].min())
        return pooled, torch.stack(block_chunk_ids)

    def _local_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """Blockwise causal local attention.

        Each query at position i attends only to keys in [i-window+1, i-1],
        i.e. strictly past tokens (self excluded), matching the incremental
        generation semantics where the current token is appended *after*
        computing attention.
        """
        batch, _, seq_len, _ = q.shape
        if self.window <= 0:
            raise ValueError("local_window must be positive")

        block_size = min(seq_len, max(self.window, self.local_block_size))
        output = q.new_empty(batch, self.n_heads, seq_len, self.head_dim)
        for q_start in range(0, seq_len, block_size):
            q_end = min(q_start + block_size, seq_len)
            # Keys go up to q_end-1 (exclude position q_end, i.e. self of the
            # last query in the block).  This makes batch-mode causal
            # semantics consistent with token-by-token incremental generation.
            k_start = max(0, q_start - self.window + 1)
            k_end = q_end  # exclusive — keys are [k_start, k_end)
            q_block = q[:, :, q_start:q_end, :]
            k_block = k[:, :, k_start:k_end, :]
            v_block = v[:, :, k_start:k_end, :]
            mask = self._block_local_mask(
                query_len=q_end - q_start,
                key_len=k_end - k_start,
                left_context=q_start - k_start,
                device=q.device,
                dtype=q.dtype,
            )
            out_block = F.scaled_dot_product_attention(
                q_block,
                k_block,
                v_block,
                attn_mask=mask,
                dropout_p=self.dropout.p if self.training else 0.0,
            )
            output[:, :, q_start:q_end, :] = out_block
        return output

    def _block_local_mask(
        self,
        query_len: int,
        key_len: int,
        left_context: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        def _build_mask() -> torch.Tensor:
            query_pos = left_context + torch.arange(query_len, device=device)
            key_pos = torch.arange(key_len, device=device)
            distance = query_pos.unsqueeze(1) - key_pos.unsqueeze(0)
            allowed = (distance > 0) & (distance <= self.window)
            mask = torch.full((query_len, key_len), float("-inf"), device=device)
            return mask.masked_fill(allowed, 0.0)

        # torch.compile + cudagraphs does not like retaining graph-created CUDA
        # tensors inside Python caches across iterations. Build the mask fresh in
        # compiled mode and keep the cache only for eager execution.
        if _is_compiling():
            return _build_mask().to(dtype=dtype)

        cache_key = (query_len, key_len, left_context, device)
        mask = self._mask_cache.get(cache_key)
        if mask is None:
            mask = _build_mask()
            self._mask_cache[cache_key] = mask
        return mask.to(dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, dim = x.shape
        profile = self.profile_enabled
        if profile:
            self._sync(x.device)
            total_start = time.perf_counter()
            qkv_start = total_start

        qkv = self.qkv_proj(x).view(batch, seq_len, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if profile:
            self._sync(x.device)
            qkv_ms = (time.perf_counter() - qkv_start) * 1000.0
            pool_start = time.perf_counter()

        global_source, query_chunk_ids = self._pool_to_latents(x)
        chunk_count = global_source.size(1)
        chunk_size = math.ceil(seq_len / chunk_count)
        if profile:
            self._sync(x.device)
            pool_ms = (time.perf_counter() - pool_start) * 1000.0
            local_start = time.perf_counter()

        gk_gv = self.rkv_proj(global_source).view(
            batch,
            global_source.size(1),
            2,
            self.latent_heads,
            self.latent_head_dim,
        )
        gk, gv = gk_gv.unbind(dim=2)
        gk = gk.transpose(1, 2)
        gv = gv.transpose(1, 2)
        if self.latent_query_block_size > 1:
            query_source, latent_query_ids = self._pool_query_blocks(x, query_chunk_ids)
        else:
            query_source = x
            latent_query_ids = query_chunk_ids
        rq = self.rq_proj(query_source).view(
            batch,
            query_source.size(1),
            self.latent_heads,
            self.latent_head_dim,
        ).transpose(1, 2)

        local_out = self._local_attention(q, k, v)
        if profile:
            self._sync(x.device)
            local_ms = (time.perf_counter() - local_start) * 1000.0
            latent_start = time.perf_counter()

        latent_scores = torch.matmul(rq, gk.transpose(-2, -1)) / math.sqrt(self.latent_head_dim)
        latent_ids = torch.arange(global_source.size(1), device=x.device)
        latent_causal = latent_ids.unsqueeze(0) >= latent_query_ids.unsqueeze(1)
        latent_mask = latent_causal.unsqueeze(0).unsqueeze(0)
        latent_scores = latent_scores.masked_fill(latent_mask, float("-inf"))
        valid_latent = (~latent_causal).any(dim=-1)
        latent_scores = torch.where(
            valid_latent.view(1, 1, rq.size(2), 1),
            latent_scores,
            torch.zeros_like(latent_scores),
        )
        latent_attn = torch.softmax(latent_scores, dim=-1)
        latent_attn = latent_attn * valid_latent.view(1, 1, rq.size(2), 1)
        latent_attn = self.dropout(latent_attn)
        latent_out = torch.matmul(latent_attn, gv)
        if profile:
            self._sync(x.device)
            latent_ms = (time.perf_counter() - latent_start) * 1000.0
            gate_start = time.perf_counter()

        # ------------------------------------------------------------------ #
        # Online-softmax gate (per head, informed by each branch's output)   #
        #                                                                     #
        # local_out  : [batch, n_heads,       seq_len,   head_dim]           #
        # latent_out : [batch, latent_heads,  query_len, latent_head_dim]    #
        #                                                                     #
        # We need remote in [batch, n_heads, seq_len, head_dim] first.       #
        # Project latent_out back to d_model, then reshape per head.         #
        # ------------------------------------------------------------------ #

        # Remote: latent space → d_model, then split into n_heads
        remote_flat = latent_out.transpose(1, 2).contiguous().view(
            batch, query_source.size(1), self.latent_heads * self.latent_head_dim
        )
        remote_flat = self.r_out(remote_flat)  # [batch, query_len, d_model]

        # Expand remote to per-token resolution (match local_out shape)
        if self.latent_query_block_size > 1:
            remote_block_size = min(seq_len, self.latent_query_block_size)
            num_blocks = remote_flat.size(1)
            # Last block may cover fewer tokens — compute per-block sizes.
            full_blocks = seq_len // remote_block_size
            remainder   = seq_len % remote_block_size
            if remainder == 0:
                repeats = torch.full((num_blocks,), remote_block_size,
                                     dtype=torch.long, device=x.device)
            else:
                repeats = torch.full((num_blocks,), remote_block_size,
                                     dtype=torch.long, device=x.device)
                repeats[-1] = remainder
            # repeat_interleave is a single fused CUDA kernel — no Python loop.
            remote_per_token = torch.repeat_interleave(remote_flat, repeats, dim=1)
        else:
            remote_per_token = remote_flat  # [batch, seq_len, d_model]

        # Reshape both branches to per-head: [batch, n_heads, seq_len, head_dim]
        local_h  = local_out                                                  # already in this shape
        remote_h = remote_per_token.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Gate logits: one scalar per (token, head) from each branch's output.
        # gate_local/remote are Linear(head_dim, 1) applied over the last dim.
        # Input shape  : [batch, n_heads, seq_len, head_dim]
        # Output shape : [batch, n_heads, seq_len, 1]
        s_local  = self.gate_local(local_h)   # [B, H, S, 1]
        s_remote = self.gate_remote(remote_h)  # [B, H, S, 1]

        # Online softmax — numerically stable, no extra memory beyond two scalars
        m        = torch.maximum(s_local, s_remote)          # running max
        exp_l    = torch.exp(s_local  - m)
        exp_r    = torch.exp(s_remote - m)
        Z        = exp_l + exp_r                              # normaliser

        w_local  = exp_l / Z   # [B, H, S, 1]  sums with w_remote to 1
        w_remote = exp_r / Z   # [B, H, S, 1]

        # Weighted combination in head space, then flatten back to d_model
        mixed   = w_local * local_h + w_remote * remote_h    # [B, H, S, head_dim]
        output  = mixed.transpose(1, 2).contiguous().view(batch, seq_len, dim)

        if profile:
            self._sync(x.device)
            gate_ms = (time.perf_counter() - gate_start) * 1000.0
            total_ms = (time.perf_counter() - total_start) * 1000.0
            self.profile_totals["total_ms"] = self.profile_totals.get("total_ms", 0.0) + total_ms
            self.profile_totals["qkv_ms"] = self.profile_totals.get("qkv_ms", 0.0) + qkv_ms
            self.profile_totals["pool_ms"] = self.profile_totals.get("pool_ms", 0.0) + pool_ms
            self.profile_totals["local_attention_ms"] = self.profile_totals.get("local_attention_ms", 0.0) + local_ms
            self.profile_totals["latent_attention_ms"] = self.profile_totals.get("latent_attention_ms", 0.0) + latent_ms
            self.profile_totals["gate_mix_ms"] = self.profile_totals.get("gate_mix_ms", 0.0) + gate_ms
            self.profile_totals["gate_mean"] = self.profile_totals.get("gate_mean", 0.0) + w_local.mean().item()
            self.profile_totals["seq_len"] = self.profile_totals.get("seq_len", 0.0) + seq_len
            self.profile_totals["local_block_size"] = self.profile_totals.get("local_block_size", 0.0) + min(
                seq_len, max(self.window, self.local_block_size)
            )
            self.profile_totals["latent_count"] = self.profile_totals.get("latent_count", 0.0) + chunk_count
            self.profile_totals["chunk_size"] = self.profile_totals.get("chunk_size", 0.0) + chunk_size
            self.profile_totals["latent_query_block_size"] = self.profile_totals.get(
                "latent_query_block_size",
                0.0,
            ) + (min(seq_len, self.latent_query_block_size) if self.latent_query_block_size > 1 else 1)
            self.profile_steps += 1
        return self.out(output)


def _build_attention(config: "ModelConfig"):
    if config.use_triton_kernel:
        try:
            from .llattn_triton import LocalLatentAttentionTriton
        except ImportError:
            import importlib, pathlib, sys
            _dir = pathlib.Path(__file__).parent
            if str(_dir) not in sys.path:
                sys.path.insert(0, str(_dir))
            try:
                LocalLatentAttentionTriton = importlib.import_module("llattn_triton").LocalLatentAttentionTriton
            except ImportError as exc:
                raise RuntimeError(
                    "use_triton_kernel=True requires Triton and the custom kernel modules to be importable"
                ) from exc
        return LocalLatentAttentionTriton(
            d_model=config.d_model,
            n_heads=config.n_heads,
            latent_heads=config.latent_heads,
            latent_dim=config.latent_d_model,
            n_latents=config.latent_tokens,
            window=config.local_window,
        )
    return LocalLatentAttention(config)


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.norm1 = RMSNorm(config.d_model)
        self.norm2 = RMSNorm(config.d_model)
        self.attn = _build_attention(config)
        self.ff = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class LocalLatentLM(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.checkpoint_blocks = config.checkpoint_blocks
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position = nn.Embedding(config.max_seq_len, config.d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )
        self.norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight
        self._init_weights(config.n_layers)

    def _init_weights(self, n_layers: int) -> None:
        """GPT-2 style initialisation.

        All Linear layers start with std=0.02.  Output projections
        (those that feed directly into a residual stream) are additionally
        scaled by 1/sqrt(2*n_layers) so the residual stream variance stays
        O(1) regardless of depth — prevents the loss-630 explosion at init.
        """
        output_projs = {"out", "r_out", "w2"}   # names of output projections
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                # Scale output projections
                leaf = name.rsplit(".", 1)[-1]
                if leaf in output_projs:
                    module.weight.data.mul_(1.0 / math.sqrt(2 * n_layers))
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def set_profile(self, enabled: bool) -> None:
        for block in self.blocks:
            block.attn.set_profile(enabled)

    def get_profile_stats(self) -> dict[str, float]:
        totals: dict[str, float] = {}
        count = 0
        for block in self.blocks:
            block_stats = block.attn.get_profile_stats()
            if not block_stats:
                continue
            count += 1
            for key, value in block_stats.items():
                totals[key] = totals.get(key, 0.0) + value
        if count == 0:
            return {}
        return {key: value / count for key, value in totals.items()}

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch, seq_len = input_ids.shape
        pos = torch.arange(seq_len, device=input_ids.device)
        x = self.embedding(input_ids) + self.position(pos).unsqueeze(0)
        for block in self.blocks:
            if self.checkpoint_blocks and self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        x = self.norm(x)
        return self.lm_head(x)


def build_model(**overrides) -> LocalLatentLM:
    config = ModelConfig(**overrides)
    return LocalLatentLM(config)


def estimate_parameter_bytes(model: nn.Module, bytes_per_scalar: int = 4) -> int:
    return sum(param.numel() * bytes_per_scalar for param in model.parameters())
