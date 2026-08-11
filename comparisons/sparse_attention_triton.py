import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _sparse_attention_kernel(
    q_ptr,
    kv_ptr,
    sink_ptr,
    topk_ptr,
    out_ptr,
    n_batch,
    n_seq,
    n_heads,
    softmax_scale,
    stride_qb,
    stride_qm,
    stride_qh,
    stride_qd,
    stride_kvb,
    stride_kvn,
    stride_kvd,
    stride_ib,
    stride_im,
    stride_it,
    stride_ob,
    stride_om,
    stride_oh,
    stride_od,
    BLOCK_D: tl.constexpr,
    BLOCK_K: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    TOPK: tl.constexpr,
):
    """Each physical program processes strided logical attention rows."""
    program_id = tl.program_id(0)
    num_programs = tl.num_programs(0)
    logical_rows = n_seq * n_heads
    total_rows = logical_rows * n_batch

    for pid in range(program_id, total_rows, num_programs):
        head = pid % n_heads
        row = pid // n_heads
        seq = row % n_seq
        batch = row // n_seq

        d = tl.arange(0, BLOCK_D)
        d_mask = d < HEAD_DIM
        q_offsets = (
            batch * stride_qb
            + seq * stride_qm
            + head * stride_qh
            + d * stride_qd
        )
        q = tl.load(q_ptr + q_offsets, mask=d_mask, other=0.0).to(tl.float32)

        t = tl.arange(0, BLOCK_K)
        topk_offsets = batch * stride_ib + seq * stride_im + t * stride_it
        indices = tl.load(topk_ptr + topk_offsets, mask=t < TOPK, other=-1)
        valid = (t < TOPK) & (indices >= 0)
        safe_indices = tl.where(valid, indices, 0)

        kv_offsets = (
            batch * stride_kvb
            + safe_indices[:, None] * stride_kvn
            + d[None, :] * stride_kvd
        )
        kv_mask = valid[:, None] & d_mask[None, :]
        gathered = tl.load(
            kv_ptr + kv_offsets, mask=kv_mask, other=0.0
        ).to(tl.float32)

        scores = tl.sum(gathered * q[None, :], axis=1) * softmax_scale
        masked_scores = tl.where(valid, scores, float("-inf"))
        score_max = tl.max(masked_scores, axis=0)
        sink = tl.load(sink_ptr + head).to(tl.float32)
        row_max = tl.maximum(score_max, sink)

        exp_scores = tl.math.exp(masked_scores - row_max)
        exp_scores = tl.where(valid, exp_scores, 0.0)
        exp_sink = tl.math.exp(sink - row_max)
        denom = tl.sum(exp_scores, axis=0) + exp_sink
        weights = exp_scores / denom

        output = tl.sum(gathered * weights[:, None], axis=0)
        out_offsets = (
            batch * stride_ob
            + seq * stride_om
            + head * stride_oh
            + d * stride_od
        )
        tl.store(out_ptr + out_offsets, output, mask=d_mask)


def _next_power_of_two(value: int) -> int:
    result = 1
    while result < value:
        result *= 2
    return result


def sparse_attention_triton(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Sparse attention with gather, sink-aware softmax, and value reduction."""
    if q.ndim != 4 or kv.ndim != 3 or topk_idxs.ndim != 3:
        raise ValueError("expected q[B,M,H,D], kv[B,N,D], topk_idxs[B,M,K]")
    batch, seq_len, n_heads, head_dim = q.shape
    if kv.shape[0] != batch or kv.shape[2] != head_dim:
        raise ValueError("q and kv shapes are incompatible")
    if topk_idxs.shape[0] != batch or topk_idxs.shape[1] != seq_len:
        raise ValueError("q and topk_idxs shapes are incompatible")
    if attn_sink.numel() != n_heads:
        raise ValueError("attn_sink must have one value per query head")

    out = torch.empty_like(q)
    topk = topk_idxs.shape[-1]
    block_d = _next_power_of_two(head_dim)
    block_k = _next_power_of_two(topk)
    _sparse_attention_kernel[(min(48, batch * seq_len * n_heads),)](
        q,
        kv,
        attn_sink,
        topk_idxs,
        out,
        batch,
        seq_len,
        n_heads,
        softmax_scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        kv.stride(0),
        kv.stride(1),
        kv.stride(2),
        topk_idxs.stride(0),
        topk_idxs.stride(1),
        topk_idxs.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        BLOCK_D=block_d,
        BLOCK_K=block_k,
        HEAD_DIM=head_dim,
        TOPK=topk,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, n_heads: int, head_dim: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.softmax_scale = head_dim ** -0.5
        self.attn_sink = nn.Parameter(torch.zeros(n_heads, dtype=torch.float32))

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        topk_idxs: torch.Tensor,
    ) -> torch.Tensor:
        return sparse_attention_triton(
            q, kv, self.attn_sink, topk_idxs, self.softmax_scale
        )


batch_size = 8
seq_len = 2600
n_kv = 32
n_heads = 64
head_dim = 128
topk = 16


def get_inputs():
    q = torch.randn(
        batch_size, seq_len, n_heads, head_dim, dtype=torch.bfloat16
    )
    kv = torch.randn(batch_size, n_kv, head_dim, dtype=torch.bfloat16)
    topk_idxs = torch.randint(
        0, n_kv, (batch_size, seq_len, topk), dtype=torch.int32
    )
    return [q, kv, topk_idxs]


def get_init_inputs():
    return [n_heads, head_dim]
