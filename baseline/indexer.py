import math
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class ModelArgs:
    max_batch_size: int = 4
    max_seq_len: int = 4096
    dtype: Literal["bf16", "fp8"] = "fp8"
    scale_fmt: Literal[None, "ue8m0"] = "ue8m0"
    expert_dtype: Literal[None, "fp4"] = None
    scale_dtype: Literal["fp32", "fp8"] = "fp8"
    vocab_size: int = 129280
    dim: int = 4096
    moe_inter_dim: int = 4096
    n_layers: int = 7
    n_hash_layers: int = 0
    n_mtp_layers: int = 1
    n_heads: int = 64
    n_routed_experts: int = 8
    n_shared_experts: int = 1
    n_activated_experts: int = 2
    score_func: Literal["softmax", "sigmoid", "sqrtsoftplus"] = "sqrtsoftplus"
    route_scale: float = 1.0
    swiglu_limit: float = 0.0
    q_lora_rank: int = 1024
    head_dim: int = 512
    rope_head_dim: int = 64
    norm_eps: float = 1e-6
    o_groups: int = 8
    o_lora_rank: int = 1024
    window_size: int = 128
    compress_ratios: Tuple[int, ...] = (0, 0, 4, 128, 4, 128, 4, 0)
    compress_rope_theta: float = 40000.0
    original_seq_len: int = 0
    rope_theta: float = 10000.0
    rope_factor: float = 40.0
    beta_fast: int = 32
    beta_slow: int = 1
    index_n_heads: int = 64
    index_head_dim: int = 128
    index_topk: int = 512
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6


def make_args() -> ModelArgs:
    return ModelArgs(
        max_batch_size=8,
        max_seq_len=2600,
        dim=1024,
        index_n_heads=16,
        index_head_dim=64,
        index_topk=128,
        q_lora_rank=256,
        rope_head_dim=32,
    )


def linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    assert bias is None
    return F.linear(x, weight)


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        dtype=None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        dtype = dtype or torch.bfloat16
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=dtype)
        )
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.register_parameter("scale", None)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
            bound = 1 / math.sqrt(in_features)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return linear(x, self.weight, self.bias)


class ColumnParallelLinear(Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        dtype=None,
    ):
        super().__init__(in_features, out_features, bias, dtype)


def apply_rotary_emb(
    x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False
) -> torch.Tensor:
    output = x
    complex_x = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if inverse:
        freqs_cis = freqs_cis.conj()
    if complex_x.ndim == 3:
        freqs_cis = freqs_cis.view(1, complex_x.size(1), complex_x.size(-1))
    else:
        freqs_cis = freqs_cis.view(
            1, complex_x.size(1), 1, complex_x.size(-1)
        )
    rotated = torch.view_as_real(complex_x * freqs_cis).flatten(-2)
    output.copy_(rotated)
    return output


class Model(nn.Module):
    """PyTorch reference model for the Indexer task."""

    def __init__(
        self,
        args: ModelArgs,
        freqs_cis: torch.Tensor,
        kv_cache: torch.Tensor,
        compress_ratio: int = 4,
    ):
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.index_n_heads
        self.n_local_heads = args.index_n_heads
        self.head_dim = args.index_head_dim
        self.rope_head_dim = args.rope_head_dim
        self.index_topk = args.index_topk
        self.q_lora_rank = args.q_lora_rank
        self.wq_b = ColumnParallelLinear(
            self.q_lora_rank, self.n_heads * self.head_dim
        )
        self.weights_proj = ColumnParallelLinear(
            self.dim, self.n_heads, dtype=torch.bfloat16
        )
        self.softmax_scale = self.head_dim ** -0.5
        self.compress_ratio = compress_ratio
        self.register_buffer("kv_cache", kv_cache)
        self.register_buffer("freqs_cis", freqs_cis)

    def forward(
        self, x: torch.Tensor, qr: torch.Tensor, start_pos: int, offset: int
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        freqs_cis = self.freqs_cis[start_pos : start_pos + seq_len]
        ratio = self.compress_ratio
        rope_dim = self.rope_head_dim
        end_pos = start_pos + seq_len

        q = self.wq_b(qr)
        q = q.unflatten(-1, (self.n_local_heads, self.head_dim))
        apply_rotary_emb(q[..., -rope_dim:], freqs_cis)
        weights = self.weights_proj(x) * (
            self.softmax_scale * self.n_heads ** -0.5
        )
        index_score = torch.einsum(
            "bshd,btd->bsht",
            q,
            self.kv_cache[:batch_size, : end_pos // ratio],
        )
        index_score = (
            index_score.relu_() * weights.unsqueeze(-1)
        ).sum(dim=2)

        if start_pos == 0:
            key_positions = torch.arange(
                seq_len // ratio, device=x.device
            ).repeat(seq_len, 1)
            query_limits = torch.arange(
                1, seq_len + 1, device=x.device
            ).unsqueeze(1) // ratio
            mask = key_positions >= query_limits
            index_score += torch.where(mask, float("-inf"), 0.0)

        topk_count = min(self.index_topk, end_pos // ratio)
        topk_idxs = index_score.topk(topk_count, dim=-1)[1]

        if start_pos == 0:
            query_limits = torch.arange(
                1, seq_len + 1, device=x.device
            ).unsqueeze(1) // ratio
            mask = topk_idxs >= query_limits
            topk_idxs = torch.where(mask, -1, topk_idxs + offset)
        else:
            topk_idxs += offset
        return topk_idxs


def get_inputs():
    args = make_args()
    batch_size = 8
    seq_len = 2600
    x = torch.randn(
        batch_size, seq_len, args.dim, dtype=torch.bfloat16
    )
    qr = torch.randn(
        batch_size, seq_len, args.q_lora_rank, dtype=torch.bfloat16
    )
    return [x, qr, 0, 0]


def get_init_inputs():
    args = make_args()
    compress_ratio = 4
    rope_theta = 10000.0
    freqs = 1.0 / (
        rope_theta
        ** (
            torch.arange(0, args.rope_head_dim, 2).float()
            / args.rope_head_dim
        )
    )
    positions = torch.arange(args.max_seq_len, dtype=torch.float32)
    angles = torch.outer(positions, freqs)
    freqs_cis = torch.polar(torch.ones_like(angles), angles)
    kv_cache = torch.randn(
        args.max_batch_size,
        args.max_seq_len // compress_ratio,
        args.index_head_dim,
        dtype=torch.bfloat16,
    )
    return [args, freqs_cis, kv_cache, compress_ratio]
