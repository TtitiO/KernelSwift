import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _sinkhorn_kernel(
    x_ptr,
    out_ptr,
    matrix_count,
    eps: tl.constexpr,
    repeat: tl.constexpr,
    rows: tl.constexpr,
    cols: tl.constexpr,
    block_rows: tl.constexpr,
    block_cols: tl.constexpr,
    num_programs: tl.constexpr,
):
    pid = tl.program_id(0)
    row_offsets = tl.arange(0, block_rows)
    col_offsets = tl.arange(0, block_cols)
    offsets = row_offsets[:, None] * cols + col_offsets[None, :]
    mask = (row_offsets[:, None] < rows) & (col_offsets[None, :] < cols)
    matrix_size: tl.constexpr = rows * cols

    for matrix_id in range(pid, matrix_count, num_programs):
        base = matrix_id * matrix_size
        values = tl.load(x_ptr + base + offsets, mask=mask, other=-float("inf"))
        values = values.to(tl.float32)

        row_max = tl.max(values, axis=1)
        values = tl.where(mask, tl.exp(values - row_max[:, None]), 0.0)
        row_sum = tl.sum(values, axis=1)
        values = tl.where(mask, values / row_sum[:, None] + eps, 0.0)

        col_sum = tl.sum(values, axis=0)
        values = tl.where(mask, values / (col_sum[None, :] + eps), 0.0)

        for _ in range(repeat - 1):
            row_sum = tl.sum(values, axis=1)
            values = tl.where(mask, values / (row_sum[:, None] + eps), 0.0)
            col_sum = tl.sum(values, axis=0)
            values = tl.where(mask, values / (col_sum[None, :] + eps), 0.0)

        tl.store(out_ptr + base + offsets, values, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, repeat: int = 10, eps: float = 1e-6):
        super().__init__()
        self.repeat = repeat
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim < 2:
            raise ValueError("Sinkhorn input must have at least two dimensions")
        if not x.is_contiguous():
            x = x.contiguous()

        rows = x.shape[-2]
        cols = x.shape[-1]
        matrix_count = x.numel() // (rows * cols)
        out = torch.empty_like(x, dtype=torch.float32)

        block_rows = triton.next_power_of_2(rows)
        block_cols = triton.next_power_of_2(cols)
        num_programs = min(matrix_count, 32)
        _sinkhorn_kernel[(num_programs,)](
            x,
            out,
            matrix_count,
            eps=self.eps,
            repeat=self.repeat,
            rows=rows,
            cols=cols,
            block_rows=block_rows,
            block_cols=block_cols,
            num_programs=num_programs,
        )
        return out


n0 = 1
n1 = 1024
mhc = 4


def get_inputs():
    return [torch.randn(n0, n1, mhc, mhc, dtype=torch.float32)]


def get_init_inputs():
    return []
