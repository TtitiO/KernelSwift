# SparseAttention — Ascend C Cube + Vector design

> Current design (2026-08-19): a single basic-API megakernel (QK GEMM +
> **dense-32 masked softmax** + PV GEMM in one `__mix__(1,2)` launch), plus a
> tiny kvT transpose kernel.  Official-bench speedup **≈ 8.8x**.
> This document describes only the shipped design; the step-by-step
> optimization history (KFC two-stage → basic-API megakernel → AIV vector-pipe
> reductions → dense softmax) lives in the git log.

## Contract (frozen from `baseline/sparse_attention.py`)

Inputs: `q[B,M,H,D]` (BF16), `kv[B,N,D]` (BF16), `attn_sink[H]` (FP32),
`topk_idxs[B,M,K]` (INT32, negative = invalid), `softmax_scale = D**-0.5`.
Output: `out[B,M,H,D]` (BF16).  For each `(b,m)`:

```
s[h,t]   = dot(q[b,m,h,:], kv[b, topk[b,m,t], :]) * scale       (valid t only)
z[h]     = max( max_t s[h,t], sink[h] )
num[h,t] = exp(s[h,t] - z[h])  (0 for invalid t)
den[h]   = sum_t num[h,t] + exp(sink[h] - z[h])
p[h,t]   = num[h,t] / den[h]
out[b,m,h,:] = sum_t p[h,t] * kv[b, topk[b,m,t], :]
```

Duplicate indices count **with multiplicity** (each occurrence is a separate
softmax entry and a separate PV contribution); invalid indices contribute
neither score nor probability.  Official geometry `B=8, M=2600, N=32, H=64,
D=128, K=16`.

## Algorithm — the dense-softmax megakernel

Two custom kernels on the timed path (`npu::sparse_attn_megakernel_basic`);
no PyTorch matmul/softmax/gather runs on the timed path.

0. **kvT transpose** (tiny vector kernel): `kvT[B,D,N] = transpose(kv[B,N,D])`
   (QK needs B in `[D,N]`; PV uses `kv` directly).
1. **Fused megakernel** (`op_kernel/fused_sparse_attn_basic_kernel.asc`,
   `__mix__(1,2)`, raw basic API — `DataCopy`/`Nd2Nz` + `LoadData` + `Mmad` +
   `Fixpipe`, no KFC Matmul objects).  One tile = 4 m-rows × 64 heads = 256
   `(m,h)` rows × `N=32`, `K=128`; `blockDim = 20` AI cores; each of the 2 AIV
   sub-blocks per core owns half of each tile's rows (`[128, 32]` each).
   - **AIC**: depth-2 double-buffered `CopyIn → SplitA/SplitB → Mmad →
     CopyOut` pipeline for both GEMMs:
     `scores[256,32](FP32) = q[256,128] @ kvT[128,32]` then, after the AGG
     handshake, `out[256,128](BF16) = W[256,32](BF16) @ kv[32,128]`.
   - **AIV — dense-32 masked softmax** (the current design's core idea):
     the QK GEMM already computes **all N=32** scores densely, so instead of
     gathering the 16 selected columns, the AIV consumes the full `[128,32]`
     FP32 scores tile in place and keeps sparsity **via a count mask**:
     - **count vector** `c[r][n]` = number of occurrences of `n` in
       `topk_idxs[b,m,:]` (0 = unselected).  Built on the vector pipe with
       zero scalar work: a static-table `Gather` replicates the 16 idx values
       across lanes, an in-place `Cast<int32→float>`, the arithmetic one-hot
       `onehot = 1 − min(|idxF − j|, 1)` (`Sub`/`Abs`/`Mins`/`Muls`/`Adds`),
       and two `ReduceSum<Pattern::Reduce::RA>` — ~9 vector calls.
       (Scalar RMW and `Compare`+`Select` one-hot variants were tried and
       rejected — see journey §18.)
     - **softmax over all 32 lanes**: unmasked `ReduceMax` (shift-invariance
       makes the unselected lanes harmless — verified against the baseline),
       `max` with the sink, batched `Sub`/`Exp`, `ReduceSum` (AR pattern over
       the contiguous N axis), `den += exp(sink − z)`, one `Reciprocal`.
       maxH and the reciprocal fan out with `Broadcast<float,2,1>` (Brcb).
     - **weights = dense agg**: `W = E · c · rcp` restores the baseline's
       duplicate-multiplicity semantics exactly (unselected lanes are an
       exact 0), and one `Cast` writes W (BF16) **directly into the PV
       A-layout `[mh,n]` slot**.  There is no column gather, no per-k
       accumulate `Add`s, and no `[N,H]→[H,N]` transpose gather — the three
       AIV instruction classes that used to dominate the 85%-busy vector
       pipe.
   - **Synchronization**: `CrossCoreSetFlag` handshakes (QK: AIC→AIV; AGG:
     AIV→AIC), batched `FLAG_BATCH=2` tiles per handshake; both AIV
     sub-blocks must consume and return flags (an early-returning sub-block
     deadlocks the AIC wait).  Scores/idx/agg UB buffers are double-buffered
     so async MTE transfers never race vector math.
   - **Pipelining**: QK(t+1) is issued before softmax(t) so the cube and
     vector overlap; the PV follows the AGG handshake.  (Fusion only pays
     because of this overlap — a serialized fused kernel measured *slower*
     than the old three-kernel split.)

## Tiling

- Megakernel: `blockDim = cubeCores` (20), `totalTiles = B·M/4 = 5200` tiles
  partitioned over cores; each AIV sub-block processes a `[128,32]` half-tile.
  Host tiling in `computeFusedSparseAttnBasicTiling`
  (`op_extension/sparse_attn_torch.cpp`).
- UB buffers are sized from `tiling_.topk`/`tiling_.h` (not tile maxima) to
  stay inside the 192 KB AIV UB budget; `ReduceMax/Min/Sum` (RA/AR patterns)
  need a 1024-float sub-block-relative scratch.
- **Tiling tensors are cached** in a content-keyed static map
  (`makeTilingTensor`): they are read by the asynchronously launched kernel,
  so freeing them at op return was a use-after-free whenever allocator churn
  recycled the block (journey §19).  The cache also removes a per-call
  `at::empty` + synchronous H2D memcpy from the timed path (~85 µs).

## Precision

Scores are produced and consumed in FP32 (the fixpipe writes FP32; perf-
neutral vs FP16 but removes one rounding step).  Softmax math is FP32
throughout; only the weights `W` are rounded to BF16 before the PV GEMM, and
the output is rounded to BF16 at the end, matching the reference's final
`out.to(BF16)`.  Measured `max_abs_diff ≈ 0.0156` (one BF16 ULP), inside
`atol=rtol=1e-2`.  Duplicate-index cases (all-same c=16, pairs c=8) verified
explicitly against the baseline.  (BF16 logits were tried and rejected:
`max_abs_diff` ~0.023 > tolerance.)

## Build & portability (two toolchains)

- `run.sh` — **contest server**: sources the generic
  `/usr/local/Ascend/ascend-toolkit/set_env.sh`; kernel entry points are
  declared `extern "C"` (default in `sparse_attn_torch.cpp`).
- `run_huawei.sh` — **huawei dev server**: sources
  `/usr/local/Ascend/cann-8.5.2/set_env.sh` and passes
  `-DSPARSE_ATTN_KERNEL_CXX_LINKAGE=ON`, because this bisheng toolchain
  exports the kernel stubs **C++-mangled** (plain `extern "C"` declarations
  fail to load with `undefined symbol: <kernel>`).
- Host device checks use `.device().type() == c10::DeviceType::PrivateUse1`
  (portable across torch_npu versions; `is_privateuseone()` is deprecated).

## Result

`benchmarks/ks/auto_bench.py` (warmup 200, repeat 500, `atol=rtol=1e-2`) on
an 8x Ascend 910B3 server: **PASS accuracy, speedup ≈ 8.74–8.80x**
(v0 ≈ 8.0 ms, v1 ≈ 0.914–0.919 ms), stable across seeds {7, 42, 123}.

## Known open issue (CANN runtime, not our kernel)

Queuing a heavy torch op (e.g. a big FP32 `einsum`, or the full baseline
model) immediately before any of our raw-launched kernels **without an
intervening synchronize** triggers a runtime sqe fault (`507035`) — it hits
even the trivial pure-vector transpose kernel and the non-mixed three-kernel
path, on both the old gather and the new dense builds (journey §19).  Clean
under `torch.npu.synchronize()`, `ASCEND_LAUNCH_BLOCKING=1`, or
`TASK_QUEUE_ENABLE=0`; the official `auto_bench` calling pattern never
triggers it.  Best current model: an sqe/launch-ordering violation between
torch_npu's host-side async task queue and raw bisheng launches — a minimal
standalone repro should be filed upstream.
