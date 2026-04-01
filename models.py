import math
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


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


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * scale * self.weight


class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.w1 = nn.Linear(config.d_model, config.d_ff)
        self.w2 = nn.Linear(config.d_ff, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.gelu(self.w1(x))))


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        assert config.d_model % config.n_heads == 0
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.qkv = nn.Linear(config.d_model, config.d_model * 3)
        self.out = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, dim = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
            diagonal=1,
        )
        scores = scores.masked_fill(causal_mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        output = torch.matmul(attn, v)
        output = output.transpose(1, 2).contiguous().view(batch, seq_len, dim)
        return self.out(output)


class FlashCausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        assert config.d_model % config.n_heads == 0
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.qkv = nn.Linear(config.d_model, config.d_model * 3)
        self.out = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.profile_enabled = False
        self.profile_totals: dict[str, float] = {}
        self.profile_steps = 0

    def set_profile(self, enabled: bool) -> None:
        self.profile_enabled = enabled
        self.profile_totals = {}
        self.profile_steps = 0

    def get_profile_stats(self) -> dict[str, float]:
        if not self.profile_enabled or self.profile_steps == 0:
            return {}
        return {
            "profile_flash_attention_ms": self.profile_totals["attention_ms"] / self.profile_steps,
        }

    @staticmethod
    def _sync(device: torch.device) -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, dim = x.shape
        profile = self.profile_enabled
        if profile:
            self._sync(x.device)
            attention_start = time.perf_counter()
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            dropout_p=self.dropout.p if self.training else 0.0,
        )
        if profile:
            self._sync(x.device)
            self.profile_totals["attention_ms"] = self.profile_totals.get("attention_ms", 0.0) + (
                time.perf_counter() - attention_start
            ) * 1000.0
            self.profile_steps += 1
        output = output.transpose(1, 2).contiguous().view(batch, seq_len, dim)
        return self.out(output)


class LocalLatentAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        assert config.d_model % config.n_heads == 0
        assert config.latent_d_model % config.latent_heads == 0
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.window = config.local_window
        self.local_block_size = config.local_block_size
        self.latent_tokens = config.latent_tokens
        self.latent_heads = config.latent_heads
        self.latent_head_dim = config.latent_d_model // config.latent_heads
        self.latent_query_block_size = max(0, config.latent_query_block_size)

        self.q_proj = nn.Linear(config.d_model, config.d_model)
        self.k_proj = nn.Linear(config.d_model, config.d_model)
        self.v_proj = nn.Linear(config.d_model, config.d_model)
        self.rq_proj = nn.Linear(config.d_model, config.latent_d_model)
        self.rk_proj = nn.Linear(config.d_model, config.latent_d_model)
        self.rv_proj = nn.Linear(config.d_model, config.latent_d_model)
        self.r_out = nn.Linear(config.latent_d_model, config.d_model)
        self.out = nn.Linear(config.d_model, config.d_model)
        self.gate = nn.Linear(config.d_model * 2, 1)
        self.dropout = nn.Dropout(config.dropout)
        self._mask_cache: dict[tuple[int, int, int, torch.device], torch.Tensor] = {}
        self.profile_enabled = False
        self.profile_totals: dict[str, float] = {}
        self.profile_steps = 0

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
        batch, seq_len, dim = x.shape
        chunks = min(self.latent_tokens, seq_len)
        chunk_size = math.ceil(seq_len / chunks)
        pad_len = chunks * chunk_size - seq_len

        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))

        x = x.view(batch, chunks, chunk_size, dim)
        pooled = x.mean(dim=2)
        chunk_ids = torch.arange(seq_len, device=x.device) // chunk_size
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
        batch, _, seq_len, _ = q.shape
        if self.window <= 0:
            raise ValueError("local_window must be positive")

        block_size = min(seq_len, max(self.window, self.local_block_size))
        output = q.new_empty(batch, self.n_heads, seq_len, self.head_dim)
        for q_start in range(0, seq_len, block_size):
            q_end = min(q_start + block_size, seq_len)
            k_start = max(0, q_start - self.window + 1)
            q_block = q[:, :, q_start:q_end, :]
            k_block = k[:, :, k_start:q_end, :]
            v_block = v[:, :, k_start:q_end, :]
            mask = self._block_local_mask(
                query_len=q_end - q_start,
                key_len=q_end - k_start,
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
        cache_key = (query_len, key_len, left_context, device)
        mask = self._mask_cache.get(cache_key)
        if mask is None:
            query_pos = left_context + torch.arange(query_len, device=device)
            key_pos = torch.arange(key_len, device=device)
            distance = query_pos.unsqueeze(1) - key_pos.unsqueeze(0)
            allowed = (distance >= 0) & (distance < self.window)
            mask = torch.full((query_len, key_len), float("-inf"), device=device)
            mask = mask.masked_fill(allowed, 0.0)
            self._mask_cache[cache_key] = mask
        return mask.to(dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, dim = x.shape
        profile = self.profile_enabled
        if profile:
            self._sync(x.device)
            total_start = time.perf_counter()
            qkv_start = total_start

        q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
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

        gk = self.rk_proj(global_source).view(
            batch,
            global_source.size(1),
            self.latent_heads,
            self.latent_head_dim,
        ).transpose(1, 2)
        gv = self.rv_proj(global_source).view(
            batch,
            global_source.size(1),
            self.latent_heads,
            self.latent_head_dim,
        ).transpose(1, 2)
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

        local_out = local_out.transpose(1, 2).contiguous().view(batch, seq_len, dim)
        gate = torch.sigmoid(self.gate(torch.cat([x, local_out], dim=-1)))
        output = gate * local_out
        remote_blocks = latent_out.transpose(1, 2).contiguous().view(
            batch,
            query_source.size(1),
            self.latent_heads * self.latent_head_dim,
        )
        remote_blocks = self.r_out(remote_blocks)
        if self.latent_query_block_size > 1:
            remote_block_size = min(seq_len, self.latent_query_block_size)
            for block_idx, start in enumerate(range(0, seq_len, remote_block_size)):
                end = min(start + remote_block_size, seq_len)
                output[:, start:end, :].add_((1.0 - gate[:, start:end, :]) * remote_blocks[:, block_idx:block_idx + 1, :])
        else:
            output = output + (1.0 - gate) * remote_blocks
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
            self.profile_totals["gate_mean"] = self.profile_totals.get("gate_mean", 0.0) + gate.mean().item()
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


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig, attention_type: str) -> None:
        super().__init__()
        self.norm1 = RMSNorm(config.d_model)
        self.norm2 = RMSNorm(config.d_model)
        if attention_type == "baseline":
            self.attn = CausalSelfAttention(config)
        elif attention_type == "flash_dense":
            self.attn = FlashCausalSelfAttention(config)
        elif attention_type == "local_latent":
            self.attn = LocalLatentAttention(config)
        else:
            raise ValueError(f"Unsupported attention type: {attention_type}")
        self.ff = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class TinyDecoderLM(nn.Module):
    def __init__(self, config: ModelConfig, attention_type: str) -> None:
        super().__init__()
        self.config = config
        self.checkpoint_blocks = config.checkpoint_blocks
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position = nn.Embedding(config.max_seq_len, config.d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(config, attention_type) for _ in range(config.n_layers)]
        )
        self.norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight
        self._init_weights(config.n_layers)

    def _init_weights(self, n_layers: int) -> None:
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position.weight, mean=0.0, std=0.02)
        residual_std = 0.02 / math.sqrt(2 * n_layers)
        for block in self.blocks:
            # scale down the residual projections to keep residual stream variance stable
            nn.init.normal_(block.attn.out.weight, mean=0.0, std=residual_std)
            nn.init.normal_(block.ff.w2.weight, mean=0.0, std=residual_std)

    def set_profile(self, enabled: bool) -> None:
        for block in self.blocks:
            if hasattr(block.attn, "set_profile"):
                block.attn.set_profile(enabled)

    def get_profile_stats(self) -> dict[str, float]:
        totals: dict[str, float] = {}
        count = 0
        for block in self.blocks:
            if hasattr(block.attn, "get_profile_stats"):
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


def build_model(model_name: str, **overrides: int) -> TinyDecoderLM:
    config = ModelConfig(**overrides)
    return TinyDecoderLM(config, attention_type=model_name)


def attention_workload(model_name: str, seq_len: int, local_window: int, latent_tokens: int) -> int:
    if model_name in {"baseline", "flash_dense"}:
        return seq_len * seq_len
    if model_name == "local_latent":
        return seq_len * min(local_window, seq_len) + seq_len * min(latent_tokens, seq_len)
    raise ValueError(f"Unsupported model name: {model_name}")


def attention_score_elements(
    model_name: str,
    batch_size: int,
    seq_len: int,
    n_heads: int,
    local_window: int,
    latent_tokens: int,
) -> int:
    return batch_size * n_heads * attention_workload(
        model_name=model_name,
        seq_len=seq_len,
        local_window=local_window,
        latent_tokens=latent_tokens,
    )


def kv_cache_tokens(model_name: str, seq_len: int, local_window: int, latent_tokens: int) -> int:
    if model_name in {"baseline", "flash_dense"}:
        return seq_len
    if model_name == "local_latent":
        return min(seq_len, local_window) + min(seq_len, latent_tokens)
    raise ValueError(f"Unsupported model name: {model_name}")


def estimate_kv_cache_bytes(
    model_name: str,
    batch_size: int,
    seq_len: int,
    n_layers: int,
    n_heads: int,
    head_dim: int,
    local_window: int,
    latent_tokens: int,
    bytes_per_scalar: int = 4,
) -> int:
    tokens = kv_cache_tokens(
        model_name=model_name,
        seq_len=seq_len,
        local_window=local_window,
        latent_tokens=latent_tokens,
    )
    return batch_size * n_layers * n_heads * tokens * head_dim * 2 * bytes_per_scalar


def estimate_attention_bytes(
    model_name: str,
    batch_size: int,
    seq_len: int,
    n_heads: int,
    local_window: int,
    latent_tokens: int,
    bytes_per_scalar: int = 4,
) -> int:
    return attention_score_elements(
        model_name=model_name,
        batch_size=batch_size,
        seq_len=seq_len,
        n_heads=n_heads,
        local_window=local_window,
        latent_tokens=latent_tokens,
    ) * bytes_per_scalar


def estimate_parameter_bytes(model: nn.Module, bytes_per_scalar: int = 4) -> int:
    return sum(param.numel() * bytes_per_scalar for param in model.parameters())
