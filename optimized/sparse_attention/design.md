# SparseAttention — Ascend C Cube + Vector design

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

Duplicates stay separate softmax entries; invalid indices contribute neither
score nor probability.  Official geometry `B=8, M=2600, N=32, H=64, D=128, K=16`.

## Algorithm

One **fused mixed kernel** (QK GEMM + softmax) plus one Cube GEMM (PV), in the
`[H,N]` layout so the GEMMs need **no transA** (the `MatmulImpl` transA path on
CANN 8.5.2 mis-addresses its internal base-blocks and is avoided).

1. **Fused QK + softmax** (`__mix__(1,2)` mixed kernel).  One tile = 4 m-rows ×
   64 heads = `singleCoreM=256` `(m,h)` rows × `N=32`, `K=128`.  The cube (AIC)
   runs a KFC server (`REGIST_MATMUL_OBJ`); the vector (AIV) drives everything:
   - **QK** (`Matmul` client, transB): `scores[256, 32] = q[256,128] @ kv^T`,
     issued **asynchronously** (`IterateAll<false>(gm, 0, false, true)`) to a
     GM tile so the cube computes tile `t+1` while the vector runs the softmax
     of tile `t`; `WaitIterateAll()` hands the written tile back.  The tile is
     read right back into UB (the L2 write-back is hit, avoiding the HBM
     round-trip of the old three-kernel split).  `C=scores` is FP16 (halving
     the traffic; the logits are re-cast to FP32 inside the softmax, costing
     one BF16 ULP).
   - **Softmax** (Vector, row-batched ×4): identical math to the earlier
     standalone kernel — gather the `K` selected columns into `sK[K,ROWS*H]`,
     batched-cast+scale to FP32, sink-aware softmax on full-width `ROWS*H`
     vectors, accumulate into `[ROWS,N,H]` and a single transposing BF16
     `Gather` to `[ROWS,H,N]`.  `Scatter` is not used (unsupported on
     `dav_2201`).  The scores/idx/agg UB buffers are **double-buffered** so the
     async MTE transfers never race the vector math.
2. **PV GEMM** (`MatmulImpl`, no transpose): `out[M*H, D] = agg @ kv`, batched
   over `b`; `A=agg` (BF16), `B=kv` (BF16), `C=out` (BF16).  Per batch
   `M=166400, N=128, K=32`.

### KFC handshake workaround

On CANN 8.5.2 `dav_2201`, the workspace clear + `WORKSPACE_SYNC_ID`(=15)
notify inside `ClearWorkspace` is dead code unless the super-kernel define is
set, so the KFC client's `WaitEvent(15)` would hang forever.  The kernel
replicates the handshake before `REGIST_MATMUL_OBJ`: the AIC runs
`ClearWorkspaceImpl(workspace)` + `NotifyEvent<PIPE_MTE3>(15)`; the AIV sets the
vector masks.

## Tiling

- Fused QK: `MatmulApiTiling` per tile `SetShape(256, 32, 128)` +
  `SetFixSplit(256, 32, 64)` (one base block = the full `[256,32]` tile, so a
  single `IterateAll` delivers the whole tile).  The kernel launches with
  `blockDim = cubeCores` (20); each of the `2*blockDim` AIV blocks drives
  `tilesPerAiv = 5200 / 40 = 130` tiles, and the AIC of each AI core serves its
  two AIV blocks via the shared KFC mailbox (a `~4 MB` GM workspace).
- PV: the original batched `MatmulImpl` tiling (unchanged from the 4.4x
  submission): `SetFixSplit(256, -1, 64)` + multi-core split.
- MDL config for the QK enables N-direction MTE2 preload (`doMTE2Preload=2`),
  which measurably hides the cube→L0C write-back gap for the skinny GEMM.

## Precision

Logits are stored FP16 from the QK GEMM (halving the scores round-trip), then
batched-cast back to FP32 and scaled before the FP32 softmax, so the softmax
math stays FP32 (no FP16/BF16 rounding of the exp/sum/div).  Only the weights
are rounded to BF16 before the PV GEMM and the result rounded to BF16 at the
end, matching the reference's final `out.to(BF16)`.  Measured `max_abs_diff`
≈ 0.0156 (one BF16 ULP), inside `atol=rtol=1e-2`.  (BF16 logits were tried and
rejected: `max_abs_diff` grew to ~0.023 > tolerance.)

## Result

`benchmarks/ks/auto_bench.py` (warmup 200, repeat 500) on `liteserver-4db9`
(8× 910B3): **PASS accuracy, speedup ≈ 5.4x** (v0 ≈ 8.05 ms, v1 ≈ 1.48 ms).
Seven consecutive fresh runs: 5.488x / 5.443x / 5.418x / 5.373x / 5.421x /
5.449x / 5.465x.  The fused QK+softmax removes the 170 MB scores round-trip and
overlaps the cube (QK) with the vector (softmax) via the async `IterateAll`
pipeline.

## PV fusion status (not completed)

Fusing the PV GEMM into the same kernel was attempted.  The two-object
`REGIST_MATMUL_OBJ(pipe, ws, qkMm, &qkTiling, pvMm, &pvTiling)` compiles and
runs, but the PV's `Matmul` client `IterateAll` (sync or async, with or without
`waitIterateAll`) **hangs** on CANN 8.5.2 `dav_2201` (the KFC multi-object
message routing / fixpipe-wait protocol for the second cube object never
replies), so the PV was kept as the separate batched `MatmulImpl` kernel.  The
remaining ~170 MB agg round-trip is worth ≈ 0.12 ms (→ ~5.9x if fused).

### Megakernel attempt matrix (all on CANN 8.5.2 `dav_2201`)

A full megakernel here means one `__mix__(1,2)` launch that runs QK GEMM,
sparse softmax, and PV GEMM on the same cube, so the `agg` intermediate never
leaves L2 and the PV kernel launch disappears.  Expected gain from the removed
launch + L2-resident `agg` is ≈ 0.12 ms (→ ≈ 5.9x).  Each variant below was
compiled with Ascend C and run on the huawei server; "hang" means the AIV never
returns from `IterateAll`/`WaitIterateAll` and the host call times out.

| # | Configuration | Result |
|---|---------------|--------|
| 1 | Two KFC `Matmul` objects (`qkMm`, `pvMm`); PV C=BF16, sync `IterateAll` | **hang** in PV `IterateAll` |
| 2 | Same as (1), but PV C=BF16, async `IterateAll<false>` + `WaitIterateAll` | **hang** in `WaitIterateAll` |
| 3 | Same as (1), but PV C=FP16 (write half, cast after launch) | **hang** in PV `IterateAll` |
| 4 | Second object forced to the **same QK tiling** (`256×32×128`, C=FP16) | **hang** on its first `IterateAll` |
| 5 | Single KFC object re-used for PV: after QK, `SetSingleShape(256, 128, 32)` | **hang** in the retargeted `IterateAll` |
| 6 | Single KFC object initialized with max dims `(256, 128, 128)`, then QK `SetSingleShape(256, 32, 128)` and PV `SetSingleShape(256, 128, 32)` | **hang** in the retargeted PV `IterateAll` |
| 7 | Control: single KFC object, QK-only loop with unified tiling, no PV loop | returns correctly |

Interpretation:
- The KFC high-level `Matmul` path on this CANN version can drive **one** cube
  object repeatedly at **one** tiled shape, but it cannot drive a second object
  and cannot be retargeted to a different `N`/`K` at runtime.
- The failure is not caused by dtype (BF16 vs FP16 C), by the PV tiling shape,
  or by the AIV→AIC `agg` data path; control (7) isolates the hang to the
  second-object / retarget call itself.
- Therefore a full single-launch QK+softmax+PV megakernel is blocked by this
  KFC limitation **if using the high-level `Matmul` objects**; the shipped
  kernel keeps the two-stage form for that path.

### Basic-API (`Mmad`) full megakernel — works

A raw-Ascend-C full megakernel was then implemented and validated on
`dav_2201` (CANN 8.5.2):

- `op_kernel/transpose_kv_kernel.asc` — vector kernel that writes `kvT[B,D,N]`
  from `kv[B,N,D]` (QK needs B as `[D,N]`; PV uses `kv` directly).
- `op_kernel/fused_sparse_attn_basic_kernel.asc` — one `__mix__(1,2)` launch:
  AIC runs `CopyIn (DataCopy/Nd2Nz) -> SplitA/SplitB (LoadData) -> Compute
  (Mmad) -> CopyOut (Fixpipe)` for both QK and PV; AIV0 runs the sink-aware
  sparse softmax; AIV1 participates in the cross-core flag handshake.
- Synchronization uses `CrossCoreSetFlag<2, PIPE_FIX>` (AIC→AIV) and
  `CrossCoreSetFlag<2, PIPE_MTE3>` (AIV→AIC) with matching default waits.
  Empirically, **both AIV sub-blocks must consume and return the flags** — an
  early-returning AIV1 hangs the AIC wait, so AIV1 runs the handshake while
  only AIV0 writes `agg`.

Optimization results (official shape `B=8, M=2600, H=64, D=128, N=32, K=16`):
- multi-core tile partition (20 AI cores, `blockDim=20`);
- each AIV processes half of each tile's rows, so both vector sub-blocks are
  used and each scores tile is only loaded once per half;
- AIC runs a depth-2 QK lookahead pipeline: up to two QK tiles are issued
  ahead, so `QK(t+2)` overlaps the AIVs' `softmax(t)`, then `PV(t)` follows
  the AGG handshake.

Measured with `auto_bench.py` (warmup 20, repeat 50): **PASS accuracy,
speedup ≈ 5.52x** (v0 ≈ 8.02 ms, v1 ≈ 1.45 ms), faster than the KFC-based
two-stage kernel's ≈ 5.45x.  `max_abs_diff` remains ≈ one BF16 ULP.

Other empirically established facts (CANN 8.5.2 `dav_2201`): a mixed kernel is
declared `__global__ __mix__(1, 2)` (`__attribute__((core_ratio(1,2)))`); with
`blockDim=N` it launches `N` AIC + `2N` AIV blocks; vector ops must be guarded
by `if ASCEND_IS_AIV`.  The `Matmul` alias resolves to `MatmulClient` (KFC) on
the vector side / `MatmulImpl` on the cube side; cross-core sync is
`ffts_cross_core_sync(PIPE_MTE3, GetffstMsg(0x02, flag))` / `wait_flag_dev(flag)`.
