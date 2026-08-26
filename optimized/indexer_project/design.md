# Indexer Ascend C mixed-kernel design

## 1. Frozen operator contract

The reference is `baseline/indexer.py`; its constructor and `forward(x, qr,
start_pos, offset)` signatures are frozen.

For the competition case:

- `B=8`, `S=2600`, `dim=1024`, `H=16`, `D=64`, `rope_dim=32`;
- `compress_ratio=4`, so the unpadded index-key length is `T=650`;
- `index_topk=128`, BF16 projections/scores, FP32 GEMM accumulation, and INT64
  output indices;
- score semantics are
  `sum_h(bf16(relu(bf16(q @ kv)) * weight_h))`, followed by the causal mask,
  TopK, invalid-index replacement with `-1`, and `offset` addition;
- integer outputs must match exactly. The project floating tolerance of `1e-2`
  does not relax TopK index equality.

## 2. Current timed path

The wrapper uses the framework projections and KV transpose/copy. The custom
path contains three compiled Ascend C kernels:

1. `indexer_rope_kernel`: in-place BF16 RoPE with FP32 arithmetic;
2. `fused_indexer_qk_reduce`: `__mix__(1,2)` QK GEMM, BF16 rounding, ReLU,
   weighted head reduction, causal score masking, and BF16 score output;
3. `indexer_topk_kernel`: exact top-128 selection over the padded score rows
   with the baseline postprocessing (invalid index -> `-1`, `+offset`) folded
   in, producing the final INT64 `[B,S,128]` indices.

The wrapper loads `build/libindexer_ops.so` and launches all kernels on the
current NPU stream. There is no exception or availability fallback around the
custom calls, and no PyTorch built-in TopK/masking remains on the timed path.

## 3. Mixed-kernel decomposition

### AIC

- One tile covers eight query tokens: `M=8*H=128`, `K=D=64`.
- The padded key dimension is `N=672`, split into seven `N_CHUNK=96` pieces
  (CO1 = 128x96 fp32 = 48 KB; two buffers fit L0C (<112 KB), so Mmad of
  chunk c+1 overlaps Fixpipe of chunk c — breaking the per-chunk pipeline
  serialization that dominated the kernel: 0.977 -> 0.854 ms).
- AIC performs ND-to-NZ movement, L1/L0 loads, FP32 `Mmad`, and `Fixpipe` into
  per-core GM/L2 ring slots. Each core's kvT batch operands (at most two,
  contiguous tiles) are preloaded into L1 in NZ layout once per kernel.
- There are `B*S/8=2600` tiles. On the measured 910B3, runtime discovery gives
  20 Cube cores and exactly 130 tiles per core.

### AIV

- Each Cube core has two AIV sub-blocks; each AIV reduces four of the eight
  tokens in a tile.
- Each token consumes `[H,N]=[16,672]` FP32 score elements.
- The baseline's BF16 rounding points are reproduced before ReLU and after
  multiplication by the per-head BF16 weight.
- The 16 head rows are accumulated in FP32, the causal suffix becomes `-inf`,
  and the final `[N]` row is rounded to BF16.

### Cross-core pipeline

- AIC and AIV synchronize with `CrossCoreSetFlag`/`CrossCoreWaitFlag`.
- `FLAG_BATCH=2` tiles share one signal; `NUM_SLOTS=2` creates a ring so AIC
  can run ahead without overwriting data still consumed by AIV.
- A/B L1 queues and AIV score/output queues are double-buffered.
- `TPipe` is created outside the kernel classes and passed by pointer.

## 4. Memory plan

For one AIV at `N=672`:

| Buffer | Shape/type | Bytes |
| --- | --- | ---: |
| score queue, two buffers | `2 * [16,672] bf16` | 43,008 |
| FP32 work | `[16,672] fp32` | 43,008 |
| BF16 rounding scratch | `[16,672] bf16` | 21,504 |
| four reduced rows | `[4,672] fp32` | 10,752 |
| output queue, two buffers | `2 * [4,672] bf16` | 10,752 |
| weights | `[16] bf16 + [16] fp32` | 96 |
| **Total** |  | **129,120** |

This fits the runtime-reported 196,352-byte UB on the Huawei 910B3.

The AIC-to-AIV workspace is a per-core double ring of **BF16** score tiles
(the fixpipe rounds fp32->bf16 with RNE, identical to the einsum-output
rounding the baseline applies, so numerics are unchanged; slot traffic is
halved vs the earlier fp32 ring: megakernel 1.24 -> 1.12 ms). The full
`[B,S,H,T]` intermediate is never materialized; only the reduced
`[B,S,672]` BF16 tensor reaches the framework TopK.

**Tiling is passed as scalar kernel arguments** (10 int32), not via a GM
tiling tensor: reading an H2D-copied GM tiling struct proved
context-dependent — in quiet process states the kernel read back all-zeros
and silently no-oped (see audit 2026-08-24 evening).

## 5. Exact TopK selector (`indexer_topk_kernel`)

Tie semantics: measured on this NPU (probe over all-`-inf` rows, heavy
duplicate values, causal-style rows), `torch.topk` behaves exactly like a
**stable descending sort** — equal values (including the `-inf` blocks from
padding/causal masking) keep ascending original index order. The advanced
`AscendC::TopK` (v220 impl: `Sort32` + stable `MrgSort` tree, largest-first)
reproduces that order bit-exactly, so INT64 outputs match `torch.topk`
exactly, ties included.

Kernel structure (40 AIV cores, 16 rows per `TopK` call):

- batches are strided across cores (batch `bi` -> core `bi % 40`) so every
  core sees a uniform mix of row limits; contiguous row ranges would bound
  the kernel time by the slowest core and defeat prefix pruning;
- valid-prefix pruning: for causal rows only the first `limit=(s+1)/ratio`
  columns are finite; `-inf` sorts last in stable order with ascending
  original index, so any prefix `>= max(limit, 128)` is bit-identical to
  the full 672. Each batch picks the smallest compile-time inner tier in
  `{128, 192, 256, 320, 384, 448, 512, 576, 672}` covering its max limit
  (average limit ~325) — template dispatch, because the advanced TopK is
  ~2.2x slower with a runtime inner, and sort cost scales with inner
  (0.545 -> 0.421 ms going from two tiers {224, 672} to nine).  The nine
  `TopkTiling` structs (112 B = 28 int32 each) are compile-time constants
  in `indexer_topk_tiling_consts.h` (generated by
  `tools/dump_topk_tiling.cpp`, verified against `TopKTilingFunc` once per
  process in the extension): passing nine structs as scalar kernel args
  does not fit (the launch stub marshalling fails somewhere between 65 and
  93 total args), and a cached H2D-copied GM tiling tensor proved
  unreadable for raw kernel launches in quiet process states (the
  confirmed origin of no-op/507035-class failures). Tail batches are
  zero-padded to 16 rows and their extra outputs discarded;
- rows are loaded as BF16 (prefix columns only, strided `DataCopyPad`),
  cast to FP32;
- for `causal=0` the padding columns `>= actualT` are forced to `-inf`
  (for `causal=1` the upstream megakernel already writes `-inf` there);
- `AscendC::TopK<float>` per batch (`k=128`, `TopKMode::TOPK_NORMAL`);
- postprocessing in INT32 vector arithmetic:
  `out = (idx >= valid) ? -1 : idx + offset` with `valid = (s+1)/ratio`,
  then INT32->INT64 cast and INT64 store.

Key platform findings (CANN 8.5.2 / DAV_2201), which explain the earlier
LiteTopK/TopK failures recorded in the audit:

- on dav-c220 `Sort32`/`Sort` take **raw values** (no `Concat` proposal
  packing; `Concat` is only for the v200 path);
- `MrgSort`/`vmrgsort4` on this chip always stops a call when the **first**
  input list drains (exhausted-suspension), so hand-rolled merge loops that
  ignore the returned `sortedNum` counts corrupt results and/or hit UB
  alignment exceptions (`507015`) on the unaligned continuation offsets;
- the advanced `AscendC::TopK` implements the correct continuation
  internally and, once buffer sizes/alignment are right, runs stably.

## 6. Measured status and remaining boundary

Optimization arc (official harness, median-ish fresh runs):

`2.489x -> 2.768x` (custom exact TopK fusion replacing framework
slice+topk+mask/where) `-> 2.97x` (no-op tiling bug fix: tiling passed as
scalar kernel args; bf16 score slots) `-> 3.16x` (TILE_M=128 larger-tile
redesign) `-> 3.26x` (TopK valid-prefix pruning + strided core load
balance) `-> 3.58x` (N_CHUNK=96 depth-2 CO1/B2 pipeline overlapping
Mmad/Fixpipe; CPU dispatch trims) `-> 3.83-3.92x` (TopK nine-tier inner
pruning with hardcoded tiling constants; see audit 2026-08-26 for the
rejected in-megakernel TopK fusion experiment).

- Exact score diagnostic (`test_repro.py`): max absolute difference `0.0`,
  TopK agreement `1.0`.
- Exact index diagnostic (`test_topk.py`, standalone op vs baseline-style
  reference): INT64 equality `1.0` on 8 shape/seed cases including heavy ties,
  `valid < 128` rows, garbage padding, and non-multiple-of-40 row counts.
- Official harness (`auto_bench.py --warmup 20 --repeat 100`, 2026-08-26,
  after the nine-tier TopK inner pruning):
  `PASS accuracy; v0=6.685191 ms, v1=1.744950 ms, speedup=3.831x`,
  `PASS accuracy; v0=6.699551 ms, v1=1.710194 ms, speedup=3.917x`,
  `PASS accuracy; v0=6.664751 ms, v1=1.740904 ms, speedup=3.828x`.
  Stress run (`--warmup 200 --repeat 500`):
  `PASS accuracy; v0=6.704736 ms, v1=1.740715 ms, speedup=3.852x`.
- Stage probe (`test_stages.py`, batched): `wq_b` 0.045, RoPE 0.169,
  weights 0.070, KV transpose 0.063, megakernel 0.850,
  **custom topk+mask 0.421** (was 0.545 before the nine-tier pruning).
  Full model: 1.573 ms batched, 1.697-1.728 ms sync/call — consistent
  with the official harness.
  CPU-side enqueue: 0.441 ms/call (down from 0.468 after removing the
  redundant `_kvT_pad` tail `zero_()` and caching the cube-core-count acl
  queries).
- Bare-loop stress (`test_topk_stress.py`, 500 consecutive topk_mask
  launches with no framework ops in between): no 507035, outputs exact.
- Megakernel bottleneck attribution (2026-08-25): skipping the entire AIV
  vector chain leaves the kernel at ~1.10 ms — the cost is on the AIC side.
  AIC skip-build breakdown: fixpipe slot writes 0.29 ms, Mmad 0.13 ms,
  Nd2Nz/SplitA/SplitB ≈ 0, residual ~0.7 ms = inter-pipe serialization
  latency between the ~12 queue ops per tile (invariant across all
  cross-core ring geometries — see below). Landed: per-core kvT L1 preload
  (1.110 -> 1.090 ms).
- Cross-core ring sweep (2026-08-25, runtime-parameterized flagBatch/
  numSlots, kept at 2/2 defaults): (4,2)=1.103, (2,4)=1.089, (4,4)=1.100,
  (8,2)=1.103, (8,4)=1.106, (16,2)=1.128, (2,8)=1.092, (8,8)=1.110 ms —
  handshake count is NOT the bottleneck, disproving the flag-round-trip
  latency theory. Deadlock root causes: the earlier FLAG_BATCH=4/NUM_SLOTS=4
  "deadlock" was a network-timeout artifact (the combo runs fine); the
  depth-2 CO1/B2 hang is real and consistent with L0C capacity < 112 KB
  (2x56 KB).
- Larger-tile redesign (2026-08-25, landed): TILE_M=128 (8 tokens/tile) with
  N_CHUNK=224 single-buffered CO1 (112 KB — the largest chunk fitting L0C),
  halving per-core tile count 260 -> 130 and thus the inter-pipe
  serialization that dominates the kernel: megakernel 1.087 -> 0.977 ms,
  bit-identical numerics (test_repro.py diff 0.0). AIV reworked to 4 tokens
  per sub-block (UB budget 129 KB). Alternatives evaluated on paper:
  N_CHUNK=112 (safe but ~2x more chunk ops), TILE_M=256 (CO1 exceeds L0C).
- Harness-vs-stages gap RESOLVED (2026-08-24 evening, see audit): the gap
  was a real megakernel correctness bug, not a scheduling slowdown. The
  kernel read its tiling from an H2D-copied GM tensor; in quiet process
  states (all v1-only stage tests) that read returned all-zeros, so the
  kernel exited immediately — the 0.065 ms "fast mode" was a no-op
  returning `at::empty` garbage. In the official harness (baseline ops run
  first) the read worked and the kernel did its real ~1.24 ms of work.
  Fix: tiling is now passed as scalar kernel arguments (like the rope/topk
  kernels), making execution context-independent; score slots were also
  switched fp32 -> bf16 (1.24 -> 1.12 ms). A new regression test
  (`test_baseline_first.py`, baseline forward before optimized model,
  exact INT64 equality) guards the harness-order scenario.

The timed path is now fully C-like: projections and KV layout conversion are
thin framework data movement; selection, masking, and offsetting run in the
custom kernels. Remaining headroom: megakernel 0.850 ms (residual
serialization now partially overlapped by the depth-2 CO1/B2 pipeline), the
TopK kernel (0.421 ms after nine-tier inner pruning — measured to be ~95%
Sort32/merge compute on the vector pipe, not launch/MTE2 overhead; see audit
2026-08-26), RoPE 0.170 ms (the 4-s block rewrite was tried and reverted:
0.170 -> 0.260 ms), and ~0.44 ms/call of CPU dispatch (overlapped with
device work; ~0.15 ms exposed at the sync boundary).
