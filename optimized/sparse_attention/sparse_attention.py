"""SparseAttention — Ascend C (Cube + Vector) competition submission.

Timed path: `torch.ops.npu.sparse_attn_megakernel_basic` (the compiled Ascend C
kernels loaded from `build/libsparse_attn_ops.so`).  No PyTorch
matmul/softmax/gather runs on the timed path.
"""

import os

import torch
import torch.nn as nn
import torch_npu  # noqa: F401  (registers the PrivateUse1/npu backend)

_LOADED = [False]


def _get_op():
    if not _LOADED[0]:
        _dir = os.path.dirname(os.path.abspath(__file__))
        _so = os.path.join(_dir, "build", "libsparse_attn_ops.so")
        torch.ops.load_library(_so)
        _LOADED[0] = True
    return torch.ops.npu.sparse_attn_megakernel_basic


def sparse_attention_ascendc(q, kv, attn_sink, topk_idxs, softmax_scale):
    return _get_op()(q, kv, attn_sink, topk_idxs, float(softmax_scale))


class ModelNew(nn.Module):
    def __init__(self, n_heads: int, head_dim: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.softmax_scale = head_dim ** -0.5
        self.attn_sink = nn.Parameter(torch.zeros(n_heads, dtype=torch.float32))

    def forward(self, q, kv, topk_idxs):
        return sparse_attention_ascendc(
            q, kv, self.attn_sink, topk_idxs, self.softmax_scale
        )


batch_size = 8
seq_len = 2600
n_kv = 32
n_heads = 64
head_dim = 128
topk = 16


def get_inputs():
    q = torch.randn(batch_size, seq_len, n_heads, head_dim, dtype=torch.bfloat16)
    kv = torch.randn(batch_size, n_kv, head_dim, dtype=torch.bfloat16)
    topk_idxs = torch.randint(0, n_kv, (batch_size, seq_len, topk), dtype=torch.int32)
    return [q, kv, topk_idxs]


def get_init_inputs():
    return [n_heads, head_dim]
