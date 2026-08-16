# SparseAttention — Ascend C (Cube + Vector) submission

C-like Ascend C implementation of the SparseAttention task for Huawei Atlas
A2 (Ascend 910B, `dav-2201`, CANN 8.5.2).  The timed path launches two custom
Ascend C kernels through a torch op (`npu::sparse_attn_megakernel_basic`):
a tiny kvT transpose kernel and the **fused QK+softmax+PV megakernel**
(`__mix__(1,2)`, raw `DataCopy/Nd2Nz + LoadData + Mmad + Fixpipe` basic API).
No PyTorch matmul / softmax / gather runs on the timed path.

## Files

- `op_kernel/fused_sparse_attn_basic_kernel.asc` — the fused megakernel:
  QK GEMM + sink-aware sparse softmax + PV GEMM in one launch, partitioned
  over 20 AI cores with double-buffered cube pipelines, AIV ping-pong
  prefetch and batched (4-tile) cross-core flags.
- `op_kernel/transpose_kv_kernel.asc` — kv [B,N,D] -> kvT [B,D,N] prep.
- `op_kernel/sparse_attn_tiling.h` — host/kernel shared tiling structs.
- `op_extension/sparse_attn_torch.cpp` — host tiling + launch.
- `op_extension/register.cpp` — `npu::sparse_attn_megakernel_basic` op.
- `sparse_attention.py` — `ModelNew` wrapper (loads the `.so`, calls the op).

## Algorithm

```
kvT      = transpose(kv)                     (tiny vector kernel)
scores   = q @ kvT                           (QK GEMM, BF16xBF16 -> FP32)
p        = softmax(gather_K(scores), sink)   (AIV, broadcast-batched math)
agg      = scatter_N(p)                      ([H,N] layout, duplicate-safe)
out      = agg @ kv                          (PV GEMM, BF16xBF16 -> BF16)
```

The softmax runs in the `[H,N]` layout (natural for both GEMMs): the `K`
selected columns are gathered with the offset-table `Gather` API, the
sink-aware softmax is batched over 4 rows on full-width vectors with Brcb
broadcasts for maxH/reciprocal, probabilities accumulate in `[N,H]` (contig-
uous `Add`, correct for duplicate indices), and the result is transposed back
to `[H,N]` with a precomputed-offset BF16 `Gather` (`Scatter` is unsupported
on `dav_2201`).  See `OPTIMIZATION_JOURNEY.md` for the full story.

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

Measured on `liteserver-4db9` (8x 910B3): **~7.1x** speedup
(v0 ~= 7.98 ms, v1 ~= 1.12 ms, raw fused-kernel time ~0.91 ms), correctness
PASS at `atol=rtol=1e-2` (max_abs_diff ~= 0.0156) across multiple seeds.
