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
  prefetch and batched (2-tile) cross-core flags.
- `op_kernel/transpose_kv_kernel.asc` — kv [B,N,D] -> kvT [B,D,N] prep.
- `op_kernel/sparse_attn_tiling.h` — host/kernel shared tiling structs.
- `op_extension/sparse_attn_torch.cpp` — host tiling + launch.
- `op_extension/register.cpp` — `npu::sparse_attn_megakernel_basic` op.
- `sparse_attention.py` — `ModelNew` wrapper (loads the `.so`, calls the op).

## Algorithm

```
kvT      = transpose(kv)                     (tiny vector kernel)
scores   = q @ kvT                           (QK GEMM, BF16xBF16 -> FP32)
p        = dense masked softmax(scores, sink, counts)  (AIV, dense-32 tile)
agg      = p (already [mh, n] PV A-layout)   (duplicate-safe via counts)
out      = agg @ kv                          (PV GEMM, BF16xBF16 -> BF16)
```

The softmax runs on the **dense [128, 32] scores tile in place** (no gather):
per-row duplicate counts `c[r][32]` are built on the vector pipe (a static
offset-table Gather replicates the idx entries across lanes, then
`onehot = 1 - min(|idxF - j|, 1)` and a `ReduceSum<RA>`), the row max/sum use
the AR-pattern hardware reductions over the contiguous N axis, maxH/recip are
fanned out with axis-1 (last-dim) Broadcasts, and `W = E * c * rcp` restores
the baseline duplicate-multiplicity semantics exactly (c=0 lanes contribute
an exact 0).  The result is already in the PV A-layout, so one Cast writes it
straight out (no zero-fill, no accumulate Adds, no transpose Gather).

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

Measured on an 8x Ascend 910B3 server: **~8.8x** speedup
(v0 ~= 8.06 ms, v1 ~= 0.92 ms), correctness PASS at `atol=rtol=1e-2`.
