import math
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import torch
import triton
import triton.language as tl
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


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        dtype = dtype or torch.bfloat16
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.register_parameter("scale", None)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
            bound = 1 / math.sqrt(in_features)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _linear_triton(x, self.weight, self.bias)


class ColumnParallelLinear(Linear):
    pass


@triton.jit
def _linear_kernel(
    X, W, Y, M, K, N,
    sxm, sxk, wno, wni, sym, syn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    pn = tl.program_id(1)
    pm = pid // tl.cdiv(N, BLOCK_N)
    pn = pid % tl.cdiv(N, BLOCK_N)
    mo = pm * BLOCK_M + tl.arange(0, BLOCK_M)
    no = pn * BLOCK_N + tl.arange(0, BLOCK_N)
    ko = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, tl.cdiv(K, BLOCK_K)):
        kk = k0 * BLOCK_K + ko
        xm = tl.load(X + mo[:, None] * sxm + kk[None, :] * sxk,
                     mask=(mo[:, None] < M) & (kk[None, :] < K), other=0.0)
        ww = tl.load(W + no[:, None] * wno + kk[None, :] * wni,
                     mask=(no[:, None] < N) & (kk[None, :] < K), other=0.0)
        acc += tl.dot(xm, tl.trans(ww))
    tl.store(Y + mo[:, None] * sym + no[None, :] * syn, acc,
             mask=(mo[:, None] < M) & (no[None, :] < N))


def _linear_triton(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    if bias is not None:
        raise AssertionError("Indexer linear layers do not use bias")
    shape = x.shape
    m, k, n = x.numel() // x.shape[-1], x.shape[-1], weight.shape[0]
    x2 = x.reshape(m, k).contiguous()
    w = weight.contiguous()
    y = torch.empty((m, n), device=x.device, dtype=x.dtype)
    bm, bn, bk = 32, 64, 64
    grid = (triton.cdiv(m, bm) * triton.cdiv(n, bn), 1)
    _linear_kernel[grid](x2, w, y, m, k, n, x2.stride(0), x2.stride(1),
                         w.stride(0), w.stride(1), y.stride(0), y.stride(1),
                         BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk)
    return y.reshape(*shape[:-1], n)


@triton.jit
def _score_kernel(
    Q, KVC, WEIGHTS, OUT, B, S, T,
    qs_b, qs_s, qs_h, qs_d, ks_b, ks_t, ks_d, ws_b, ws_s, ws_h,
    N_HEADS: tl.constexpr, HEAD_DIM: tl.constexpr,
    BLOCK_S: tl.constexpr, BLOCK_T: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    tiles_t = tl.cdiv(T, BLOCK_T)
    tiles_s = tl.cdiv(S, BLOCK_S)
    b = pid // (tiles_s * tiles_t)
    rem = pid % (tiles_s * tiles_t)
    ps = rem // tiles_t
    pt = rem % tiles_t
    so = ps * BLOCK_S + tl.arange(0, BLOCK_S)
    to = pt * BLOCK_T + tl.arange(0, BLOCK_T)
    dm = tl.arange(0, BLOCK_D)
    valid_s = so < S
    valid_t = to < T
    out = tl.zeros((BLOCK_S, BLOCK_T), dtype=tl.float32)
    for h in range(N_HEADS):
        qptr = Q + b * qs_b + so[:, None] * qs_s + h * qs_h + dm[None, :] * qs_d
        kptr = KVC + b * ks_b + to[:, None] * ks_t + dm[None, :] * ks_d
        qv = tl.load(qptr, mask=valid_s[:, None] & (dm[None, :] < HEAD_DIM), other=0.0)
        kv = tl.load(kptr, mask=valid_t[:, None] & (dm[None, :] < HEAD_DIM), other=0.0)
        dot = tl.dot(qv, tl.trans(kv))
        dot = tl.maximum(dot, 0.0)
        w = tl.load(WEIGHTS + b * ws_b + so * ws_s + h * ws_h, mask=valid_s, other=0.0)
        out += dot * w[:, None]
    optr = OUT + b * S * T + so[:, None] * T + to[None, :]
    tl.store(optr, out, mask=valid_s[:, None] & valid_t[None, :])


def _score_triton(q: torch.Tensor, kv: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    b, s, h, d = q.shape
    t = kv.shape[1]
    out = torch.empty((b, s, t), device=q.device, dtype=torch.float32)
    bs, bt, bd = 8, 64, d
    grid = (b * triton.cdiv(s, bs) * triton.cdiv(t, bt),)
    _score_kernel[grid](q, kv, weights, out, b, s, t,
                        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                        kv.stride(0), kv.stride(1), kv.stride(2),
                        weights.stride(0), weights.stride(1), weights.stride(2),
                        N_HEADS=h, HEAD_DIM=d, BLOCK_S=bs, BLOCK_T=bt, BLOCK_D=bd)
    return out


@triton.jit
def _rotary_inplace_kernel(
    Q, FREQS, TOTAL, SEQ_LEN,
    qs_b, qs_s, qs_h, qs_d, fs_s, fs_p,
    N_HEADS: tl.constexpr, HEAD_DIM: tl.constexpr, ROPE_DIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    tok = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tm = tok < TOTAL
    pos = tok % SEQ_LEN
    h = tl.arange(0, N_HEADS)
    p = tl.arange(0, ROPE_DIM // 2)
    base = tok[:, None, None] * qs_s + h[None, :, None] * qs_h + (HEAD_DIM - ROPE_DIM) + 2 * p[None, None, :]
    m = tm[:, None, None]
    a = tl.load(Q + base, mask=m, other=0.0).to(tl.float32)
    b = tl.load(Q + base + 1, mask=m, other=0.0).to(tl.float32)
    fb = pos[:, None] * fs_s + p[None, :] * fs_p
    fm = tm[:, None]
    c = tl.load(FREQS + fb, mask=fm, other=0.0).to(tl.float32)[:, None, :]
    si = tl.load(FREQS + fb + 1, mask=fm, other=0.0).to(tl.float32)[:, None, :]
    tl.store(Q + base, a * c - b * si, mask=m)
    tl.store(Q + base + 1, a * si + b * c, mask=m)


def _rotary_triton(q: torch.Tensor, freqs: torch.Tensor, rope_dim: int) -> None:
    fr = torch.view_as_real(freqs)
    block = 8
    _rotary_inplace_kernel[(triton.cdiv(q.shape[0] * q.shape[1], block),)](
        q, fr, q.shape[0] * q.shape[1], q.shape[1],
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        fr.stride(0), fr.stride(1), N_HEADS=q.shape[2], HEAD_DIM=q.shape[3],
        ROPE_DIM=rope_dim, BLOCK=block)


@triton.jit
def _mask_kernel(X, TOTAL, S, T, RATIO: tl.constexpr, BLOCK: tl.constexpr):
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = o < TOTAL
    key = o % T
    query = (o // T) % S
    lim = (query + 1) // RATIO
    v = tl.load(X + o, mask=m, other=0.0)
    tl.store(X + o, tl.where(key >= lim, -float("inf"), v), mask=m)


@triton.jit
def _index_offset_kernel(X, TOTAL, S, K, OFFSET, RATIO: tl.constexpr, CAUSAL: tl.constexpr, BLOCK: tl.constexpr):
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = o < TOTAL
    v = tl.load(X + o, mask=m, other=0)
    if CAUSAL:
        q = (o // K) % S
        lim = (q + 1) // RATIO
        v = tl.where(v >= lim, -1, v + OFFSET)
    else:
        v += OFFSET
    tl.store(X + o, v, mask=m)


def _apply_mask(scores: torch.Tensor, ratio: int) -> None:
    block = 256
    _mask_kernel[(triton.cdiv(scores.numel(), block),)](scores, scores.numel(), scores.shape[1], scores.shape[2], RATIO=ratio, BLOCK=block)


def _apply_offset(indices: torch.Tensor, ratio: int, offset: int, causal: bool) -> None:
    block = 256
    _index_offset_kernel[(triton.cdiv(indices.numel(), block),)](indices, indices.numel(), indices.shape[1], indices.shape[2], offset, RATIO=ratio, CAUSAL=causal, BLOCK=block)


class ModelNew(nn.Module):
    def __init__(self, args: ModelArgs, freqs_cis: torch.Tensor, kv_cache: torch.Tensor, compress_ratio: int = 4):
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.index_n_heads
        self.n_local_heads = args.index_n_heads
        self.head_dim = args.index_head_dim
        self.rope_head_dim = args.rope_head_dim
        self.index_topk = args.index_topk
        self.q_lora_rank = args.q_lora_rank
        self.wq_b = ColumnParallelLinear(self.q_lora_rank, self.n_heads * self.head_dim)
        self.weights_proj = ColumnParallelLinear(self.dim, self.n_heads, dtype=torch.bfloat16)
        self.softmax_scale = self.head_dim ** -0.5
        self.compress_ratio = compress_ratio
        self.register_buffer("kv_cache", kv_cache)
        self.register_buffer("freqs_cis", freqs_cis)

    def forward(self, x: torch.Tensor, qr: torch.Tensor, start_pos: int, offset: int) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        ratio = self.compress_ratio
        end_pos = start_pos + seq_len
        key_len = end_pos // ratio
        q = _linear_triton(qr, self.wq_b.weight).unflatten(-1, (self.n_local_heads, self.head_dim))
        _rotary_triton(q, self.freqs_cis[start_pos:end_pos], self.rope_head_dim)
        weights = _linear_triton(x, self.weights_proj.weight) * (self.softmax_scale * self.n_heads ** -0.5)
        scores = _score_triton(q, self.kv_cache[:batch_size, :key_len], weights)
        if start_pos == 0 and key_len:
            _apply_mask(scores, ratio)
        topk_count = min(self.index_topk, key_len)
        topk_idxs = torch.topk(scores, topk_count, dim=-1)[1]
        if topk_count:
            _apply_offset(topk_idxs, ratio, offset, start_pos == 0)
        return topk_idxs


def get_inputs():
    args = make_args()
    x = torch.randn(8, 2600, args.dim, dtype=torch.bfloat16)
    qr = torch.randn(8, 2600, args.q_lora_rank, dtype=torch.bfloat16)
    return [x, qr, 0, 0]


def get_init_inputs():
    args = make_args()
    compress_ratio = 4
    freqs = 1.0 / (10000.0 ** (torch.arange(0, args.rope_head_dim, 2).float() / args.rope_head_dim))
    positions = torch.arange(args.max_seq_len, dtype=torch.float32)
    freqs_cis = torch.polar(torch.ones_like(torch.outer(positions, freqs)), torch.outer(positions, freqs))
    kv_cache = torch.randn(args.max_batch_size, args.max_seq_len // compress_ratio, args.index_head_dim, dtype=torch.bfloat16)
    return [args, freqs_cis, kv_cache, compress_ratio]
