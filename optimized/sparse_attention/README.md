# SparseAttention — Ascend C (Cube + Vector) submission

C-like Ascend C implementation of the SparseAttention task for Huawei Atlas
A2 (Ascend 910B, `dav-2201`, CANN 8.5.2).  The timed path launches three
custom Ascend C kernels through a torch op (`npu::sparse_attn`); no PyTorch
matmul / softmax / gather runs on the timed path.

## Files

- `op_kernel/matmul_kernel.asc` — QK and PV GEMMs (`MatmulImpl`).
- `op_kernel/sparse_softmax_kernel.asc` — row-batched sink-aware softmax (Vector + `Gather`).
- `op_kernel/sparse_attn_tiling.h` — host/kernel shared tiling structs.
- `op_extension/sparse_attn_torch.cpp` — host tiling (`MatmulApiTiling`) + launch.
- `op_extension/register.cpp` — `npu::sparse_attn` op registration.
- `sparse_attention.py` — `ModelNew` wrapper (loads the `.so`, calls the op).

## Algorithm

```
scores = q @ kv^T                     (QK GEMM, BF16×BF16 → FP32, batched over b)
p      = softmax(gather_K(scores), sink)   (row-batched Vector kernel)
agg    = scatter_N(p)                      ([H,N] layout)
out    = agg @ kv                     (PV GEMM, BF16×BF16 → BF16, batched over b)
```

The softmax runs in the `[H,N]` layout (natural for both GEMMs): the `K`
selected columns are gathered with the offset-table `Gather` API, the
sink-aware softmax is batched over 4 rows so the math runs on full-width
vectors, probabilities are accumulated in an `[N,H]` scratch tile (contiguous
`Add`, correct for duplicate indices), and the result is transposed back to
`[H,N]` with a single precomputed-offset BF16 `Gather` (`Scatter` is
unsupported on `dav_2201`).

## Build

```bash
source /usr/local/Ascend/cann-8.5.2/set_env.sh
bash run.sh            # -> build/libsparse_attn_ops.so
```

## Measure

```bash
python benchmarks/ks/auto_bench.py \
  --v0_file baseline/sparse_attention.py \
  --v1_file optimized/sparse_attention/sparse_attention.py \
  --atol 1e-2 --rtol 1e-2 --warmup 200 --repeat 500
```

Measured on `liteserver-4db9` (8× 910B3): **~4.20x** speedup
(v0 ≈ 8.05 ms, v1 ≈ 1.91 ms), correctness PASS at `atol=rtol=1e-2`
(`max_abs_diff ≈ 0.016`).
