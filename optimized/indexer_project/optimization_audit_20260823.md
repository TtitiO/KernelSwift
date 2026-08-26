# Indexer optimization audit — 2026-08-23

## Outcome

The retained candidate is the pre-existing exact mixed AIC/AIV implementation.
It compiles and launches on Huawei Ascend 910B3, passes the official accuracy
check, and provides a stable `2.489x` median speedup over the frozen Torch
reference. LiteTopK was attempted through three aligned Ascend C integration
routes; all compiled but raised reproducible AIV UB-alignment exceptions. The
known-good source and binary were restored.

## Environment and artifact

- Device: Huawei Ascend 910B3, 20 Cube cores / 40 Vector cores visible
- Target: `DAV_2201` (`--npu-arch=dav-2201`)
- CANN: `8.5.2`, inner version `V100R001C25SPC005B220`
- Bisheng: clang `15.0.5`, build `5c68a1cb1231`
- Python `3.10.12`, PyTorch `2.9.0+cpu`, torch_npu `2.9.0`
- Final remote binary SHA-256:
  `3b68832da4826a615cd955672607246221d21d342a8952cec7ac00b8a9c1cf5e`

## Correctness evidence

Direct score-path diagnostic on `[8,2600,16,64] x [8,650,64]`:

- output dtype: BF16;
- `inf` mismatch count: `0`;
- max absolute score difference: `0.0`;
- elements above `0.02` difference: `0`;
- TopK index agreement: `1.0`.

Final official smoke after restoring the retained candidate:

```text
PASS accuracy; v0=6.717716 ms, v1=2.712658 ms, speedup=2.476x
Summary: 1 passed, 0 failed, 1 total.
```

The final restoration check after the aligned LiteTopK retry also passed:

```text
PASS accuracy; v0=6.706536 ms, v1=2.691603 ms, speedup=2.492x
Summary: 1 passed, 0 failed, 1 total.
```

## Stable paired benchmark

Command, repeated in three fresh processes:

```bash
python benchmarks/ks/auto_bench.py \
  --v0_file baseline/indexer.py \
  --v1_file optimized/indexer_project/indexer_ver1.py \
  --atol 1e-2 --rtol 1e-2 --warmup 20 --repeat 100
```

| Run | Torch baseline (ms) | Candidate (ms) | Speedup |
| ---: | ---: | ---: | ---: |
| 1 | 6.722496 | 2.738518 | 2.455x |
| 2 | 6.720822 | 2.700247 | 2.489x |
| 3 | 6.701976 | 2.664093 | 2.516x |
| **Median** | **6.720822** | **2.700247** | **2.489x** |

The candidate was faster in all three processes and all three accuracy checks
passed.

## Stage probe

| Stage | Approximate latency (ms) |
| --- | ---: |
| `wq_b` projection | 0.052 |
| custom RoPE | 0.169 |
| weight projection + scale | 0.067 |
| padded KV transpose/copy | 0.062 |
| custom mixed QK/reduction kernel | 0.062 |
| score slice + framework TopK | 0.572 |
| final mask + `where` | 0.179 |

These component timings are diagnostic batched timings, not additive substitutes
for the synchronized official result. They identify framework TopK and its
postprocessing as the largest remaining stages.

## Paper-method applicability

| Method | Fits this fixed benchmark? | Decision |
| --- | --- | --- |
| IndexCache | No | Requires multiple layers, persistent cross-layer indices, and model-level calibration/training. The benchmark invokes one isolated Indexer without a layer identity or shared-index state. |
| LiteTopK | Conceptually yes | It preserves exact TopK and targets the measured bottleneck. Its sampling/histogram machinery is aimed at very long rows and large `k`; for `T=650`, `k=128`, a direct UB-resident exact selector is more appropriate. The CANN native-TopK integration attempted here was not launch-safe. |
| PIVOT | No for scoring correctness | Group proxy/reuse changes per-query indices; Refine is exact only within a proxy candidate set, not necessarily against the exhaustive Indexer. The competition compares every integer index exactly. |
| You Only Index Once / CLSA | No | Requires a KV-sharing multi-layer architecture and a routing index shared across cross-decoder layers, outside the operator interface. |
| HISA | No for scoring correctness | Coarse block filtering can discard a block containing a true top-k token. Reported model quality and >99% IoU are not exact index equality. At this benchmark's short `T=650`, the coarse pass also has little room to amortize. |
| MISA | No for scoring correctness | Head routing changes the score function and reports partial token-set recovery rather than exact equality. This task has only 16 heads, so an eight-head route is a 2x head reduction rather than the paper's 8x DeepSeek case. |

Primary sources:

- IndexCache: https://arxiv.org/abs/2603.12201
- LiteTopK: https://arxiv.org/abs/2607.11976
- PIVOT: https://arxiv.org/abs/2607.24593
- You Only Index Once: https://arxiv.org/abs/2606.06467
- HISA: https://arxiv.org/abs/2603.28458
- MISA: https://arxiv.org/abs/2605.07363

## Rejected experiments

### 1. UB-resident native TopK fusion

The score row was retained in UB and passed to CANN 8.5.2
`AscendC::TopK<float>`, with TopK tiling generated on the host for
`inner=672`, `k=128`. Compilation succeeded. The first scratch allocation bug
was corrected after a host probe showed `tmpLocalSize=3360` FP32 elements
(`13,440` bytes), with total planned UB still below the measured `196,352`
bytes. The corrected version and an int32-output isolation version both failed
at launch with runtime error `507015`: "The UB address accessed by the VEC
instruction is not aligned." The experiment was removed.

The subsequent aligned retry tested three additional layouts:

| Route | Alignment change | Result |
| --- | --- | --- |
| Mixed-kernel explicit scratch | Two reduced rows used a 704-float stride (`2816 = 11 * 256` bytes); every TopK operand and scratch buffer started at a 256-byte boundary | Compiled; launch failed with runtime `507015` and the same VEC UB-alignment exception |
| Mixed-kernel CANN stack scratch | Kept the 704-float row stride but used TopK's `PopStackBuffer` overload | Compiled; launch failed with the same exception |
| Isolated AIV-only selector | Restored the exact score kernel and moved TopK into a minimal 40-AIV kernel with one aligned row buffer and isolated scratch | Compiled; launch failed with the same exception |

Because the fault reproduces even in the isolated AIV-only kernel, it is not
caused by the original mixed-kernel row stride or TPipe allocation order. On
this CANN 8.5.2 / DAV_2201 combination, the advanced `AscendC::TopK<float>`
path is therefore classified as **compile-only, launch-failed** for this shape.

### 2. Custom post-TopK mask/offset kernel

An AIV postpass attempted to replace framework `where` and offset addition.
Both direct INT64 Cast and INT32 gather/scatter packing variants compiled but
failed at launch with the same UB-alignment exception. DAV_2201 has no SIMT
hardware, so the clean per-index SIMT alternative available on DAV_3510 is not
available. The experiment was removed.

## Next optimization recommendation

Build the selector as a separate, minimal AIV-only direct-invoke prototype
before integrating it with the mixed kernel. Use the A2-supported
`Sort<float,true>` plus `MrgSort<float,true>` path (never `MrgSort4`), retain
only 128 proposals per merge stage, and validate tie/`-inf` behavior against
the exact INT64 reference. This avoids relying on the failing advanced TopK
layout inside a large mixed-kernel UB allocation. Promote it only if it passes
exact integer equality and improves the official three-process benchmark.

Secondary opportunities, after selector correctness, are replacing the
16-step head-add chain with an A2 reduction pattern and fusing the padded KV
layout conversion. Their measured ceilings are smaller than the TopK target.

## 2026-08-23 (late): exact fused TopK landed via advanced `AscendC::TopK`

### What was done

- Replaced the framework `slice+topk` (0.573 ms) and `mask/where` (0.179 ms)
  stages with a new custom AIV kernel `indexer_topk_kernel`
  (`op_kernel/indexer_topk.asc`, op `indexer_ops.topk_mask` in
  `op_extension/indexer_torch.cpp`, launched from `indexer_ver1.py`).
- The kernel batches 16 rows per `AscendC::TopK<float>` call
  (`inner=672`, `k=128`, `TOPK_NORMAL`), folds the baseline postprocessing
  (`idx >= (s+1)/ratio -> -1`, else `idx + offset`) into INT32 vector
  arithmetic, and emits INT64 `[B,S,128]` directly. Host-side
  `TopKTilingFunc` tilings for batch sizes 1..16 are computed once and cached
  on device.
- Tie-breaking was probed empirically (`all -inf` rows, 50 duplicates of one
  value, causal-style rows, relu-zero rows): **torch.topk on this NPU is
  exactly a stable descending sort** (ties keep ascending original index).
  The advanced TopK's `Sort32` + stable `MrgSort` tree reproduces that order
  bit-exactly, including `-inf` blocks.

### Why the earlier native-TopK failures happened (root causes found)

1. On dav-c220 `Sort32`/`Sort` consume **raw values**; `Concat` proposal
   packing is a v200-only path. Sorting Concat-built proposals produced
   garbage order.
2. `MrgSort`/`vmrgsort4` here always suspends a call when the **first** input
   list drains (measured `sortedNum = 32,7,0,0` for a 4x32 merge). A
   hand-rolled merge loop that ignores the returned counts and keeps writing
   at unaligned continuation offsets corrupts data and trips the UB
   alignment exception (`507015`). The advanced `AscendC::TopK` handles this
   continuation internally and is stable once buffer sizes/alignment are
   correct.

### Correctness evidence

- `optimized/indexer_project/test_topk.py` (standalone op vs baseline-style
  torch reference): INT64 `torch.equal` on all 8 cases (competition shape
  with offsets 0/7, non-causal with garbage padding, `valid<128` rows,
  S=512 boundary, non-multiple-of-40 rows, heavy duplicate values):
  `ALL PASS`, agreement `1.000000`.
- `test_repro.py` score-path diagnostic unchanged: max abs diff `0.0`,
  TopK agreement `1.0`.
- Official harness: `PASS accuracy; v0=6.727836 ms, v1=2.436955 ms,
  speedup=2.761x` (`--warmup 20 --repeat 100`), and a stress run
  `--warmup 200 --repeat 500`: `v0=6.704340 ms, v1=2.410705 ms,
  speedup=2.781x`.

### Stage timings after fusion (test_stages.py, batched)

| Stage | Latency (ms) |
| --- | ---: |
| `wq_b` projection | 0.052 |
| custom RoPE | 0.169 |
| weight projection + scale | 0.071 |
| padded KV transpose/copy | 0.059 |
| custom mixed QK/reduction kernel | 0.065 |
| **custom topk+mask (new)** | **0.648** |
| full model batched | 0.974 |
| full model sync/call (median) | 1.172 |

### Known issues / notes

- Back-to-back unsynchronized launches of the standalone op (micro-benchmark
  loop with no intervening framework ops) can trip a vector-core exception
  (507035) after ~40-60 rapid launches; the official harness (per-call sync,
  warmup 200) and the full model path are unaffected. Per-call synced runs
  (200 iterations) are stable.
- Remaining headroom: the TopK kernel (0.648 ms) is dominated by per-row
  `Sort32`+merge work inside `AscendC::TopK`; batching 8->16 rows did not
  change it. RoPE (0.169 ms) is the second target.

## 2026-08-24 — Harness-vs-stages gap root-cause investigation

### Symptom

`test_stages.py` measures the full model at 1.172 ms (sync/call, megakernel
0.065 ms), while `auto_bench.py` reports v1 ≈ 2.41 ms. Gap ≈ 1.2 ms.

### Finding (bisected with probe_gap*.py, since deleted)

The gap is **not** measurement overhead (seeding, cloning, `no_grad`,
allocation churn, CPU dispatch, clock throttling were all excluded). It is a
real device-side slowdown of exactly one stage — the `fused_indexer_qk_reduce`
megakernel: 0.065 ms in a clean process, ~1.24 ms (≈20x, enqueue still
0.1 ms) inside the official harness flow. All other stages are identical in
both contexts.

Empirical rules established by ~20 controlled reproductions:

- The megakernel's execution mode is fixed at its **first launch in the
  process** and persists afterwards (cannot be flipped back by re-warmup,
  `empty_cache`, or buffer reallocation).
- Fast mode requires the first launch to happen **before the baseline
  model's first forward** (more precisely before the harness's framework-op
  activity). The official harness always runs the baseline correctness
  forward first, so v1's megakernel is locked into the slow mode there.
- In a clean process (v1-only tests) the first launch happens immediately,
  hence stage probes always see the fast mode.

### Fix attempts and why they were reverted

Warmup hooks were tried to force the first launch early (in `__init__`, in
an `_apply` override, in a `.to` override — all fire before the harness's
first baseline forward):

- Warmup including a topk call on all-zero data: megakernel stayed slow
  (correct, no gain).
- Megakernel-only warmup on the persistent `_kvT_pad`: megakernel became
  fast (0.065 ms in probes) but the official harness then FAILED
  correctness deterministically (2,442,457/2,662,400 INT64 mismatches).
  Mismatch structure: identical index *sets* per row, different *order* —
  masked (-inf) positions outrank valid scores, i.e. in this state the
  fast-mode megakernel produces causal-mask-broken scores. Weights/inputs
  integrity after warmup verified intact; in clean processes the same
  warmup does not corrupt anything (fast mode fully correct there,
  `test_repro.py` agreement 1.0).

Conclusion: in the harness context the fast mode cannot be entered without
breaking numerics, and the mechanism (runtime/driver-level state keyed to
first-launch context of the mix AIC/AIV kernel) is opaque. The warmup was
reverted; the submission keeps the verified slow-but-correct configuration.

### Verification after revert (fresh runs, official harness)

- `PASS accuracy; v0=6.679162 ms, v1=2.413256 ms, speedup=2.768x`
- `PASS accuracy; v0=6.669427 ms, v1=2.408445 ms, speedup=2.769x`
- `PASS accuracy; v0=6.680111 ms, v1=2.403829 ms, speedup=2.779x`
- `test_topk.py`: ALL PASS (8/8 exact). `test_repro.py`: agreement 1.0.
- `test_stages.py`: full 0.986 ms batched / 1.183 ms sync-per-call;
  megakernel 0.066 ms; custom topk+mask 0.655 ms.

### Follow-up ideas (not pursued)

- Rewrite the megakernel launch so its first invocation in the harness is
  indistinguishable from a clean-process launch (the corruption pattern
  suggests first-launch context contaminates the causal-mask path; possibly
  address/workspace related). Needs kernel-level instrumentation (printf /
  DumpTensor) on the server to pin down.
- If the fast mode ever becomes usable in-harness, expected official v1
  ≈ 1.2 ms → speedup ≈ 5.5x; this is the single largest known headroom,
  larger than the TopK kernel (0.648 ms) rewrite.

## 2026-08-24 (evening) — Gap ROOT-CAUSED and FIXED; megakernel 1.24 -> 1.12 ms

### Root cause (supersedes the earlier "slow mode" theory)

The morning session established that the megakernel runs at 0.065 ms in
quiet processes and 1.24 ms in the official harness, and that warmups make
it "fast but mask-broken". Kernel-level printf instrumentation
(`INDEXER_DEBUG_PRINT` debug build) settled the mechanism:

**In the "fast" state the kernel reads an ALL-ZERO tiling struct**
(`totalTiles=0 tilesPerCore=0 causal=0 n=0 mPerBatch=0` on every core) and
exits immediately. The 0.065 ms "fast mode" was a NO-OP returning
`at::empty` garbage (rows for tokens 2-3 of every tile never written;
`frac_diff=0.5` under changed inputs was the tell). The megakernel was
never fast; the official harness's 1.24 ms is its true cost, and all
previous fast-mode observations (stage probes, warmup experiments) were
measuring a no-op. The earlier "warmup corrupts the causal mask" finding
was the same bug seen from the output side.

The tiling was copied host->device once per process via raw `aclrtMemcpy`
into a cached `at::Tensor` and read back by the kernel through scalar GM
loads; that read is context-dependent (returns zeros in quiet process
states — the H2D copy's visibility to subsequently launched raw kernels is
not guaranteed there). Notably this was a latent CORRECTNESS risk for the
official run itself, not just a performance anomaly.

### Fix

- Tiling is now passed as **scalar kernel arguments** (10 int32), the same
  convention the rope/topk kernels already used successfully
  (`fused_indexer_qk_reduce.asc` Init signature, host
  `indexer_qk_reduce_torch`; `makeTilingTensor` removed). The kernel is now
  context-independent: in the former no-op state it runs correctly
  (`inf_mismatch=0 bad=0 agree=1.0000`).
- Score-slot ring switched fp32 -> bf16 (fixpipe `F322BF16` RNE quant,
  bit-identical to the einsum-output rounding; AIV widens bf16->fp32 with
  CAST_NONE as before). Halves slot traffic: megakernel 1.244 -> 1.113 ms
  in the gap-state probe, `test_repro.py` still max abs diff 0.0.

### Verification (all after the fix)

- Official harness, three fresh runs (`--warmup 20 --repeat 100`):
  `PASS accuracy; v0=6.701793 ms, v1=2.289803 ms, speedup=2.927x`
  `PASS accuracy; v0=6.662097 ms, v1=2.224927 ms, speedup=2.994x`
  `PASS accuracy; v0=6.685263 ms, v1=2.248023 ms, speedup=2.974x`
- `test_topk.py`: ALL PASS (8/8 exact). `test_repro.py`: max abs diff 0.0,
  topk agreement 1.0.
- New regression test `test_baseline_first.py` (baseline Model forward
  first — the official-harness launch order — then ModelNew; exact INT64
  equality, x5 repeats): PASS.
- Stage probe (now truthful): wq_b 0.045, rope 0.169, weights 0.067,
  kvT 0.060, megakernel 1.113, topk+mask 0.654; full 2.075 ms batched /
  2.201 ms sync-per-call — consistent with the official harness.

### Remaining headroom (real numbers now)

- Megakernel 1.113 ms: dominated by the AIV reduce's fp32 UB vector work
  (exact bf16-rounding emulation chain: Cast/Maxs/16xMuls/round/16xAdd per
  token). A restructure (e.g. cube-side head reduction) must preserve the
  exact bf16 product-rounding semantics or TopK tie order breaks.
- TopK kernel 0.654 ms (`AscendC::TopK` Sort32+merge per row).
- Latent risk (documented, not triggered in any harness run): the TopK
  kernel still reads its `TopkTiling` struct from an H2D-copied GM tensor —
  the same fragile pattern that bit the megakernel. If a quiet-context
  evaluation ever runs, this could read stale data (possibly the origin of
  the bare-loop 507035 crashes). Fix would be passing the struct fields as
  scalar args or a stream-ordered copy; left untouched because the harness
  path is verified and the struct is consumed whole by `AscendC::TopK`.

## 2026-08-25 — TopK scalar-tiling conversion; megakernel bottleneck attribution

### TopK kernel: GM tiling eliminated (stability fix)

The topk kernel read 16 `TopkTiling` structs from an H2D-copied cached GM
tensor — the same fragile pattern that caused the megakernel no-op bug and
the suspected origin of the bare-loop 507035 vector-core exception. Fixed
the same way: the single 16-row `TopkTiling` (112 B = 28 int32) is computed
once on the host and passed as 28 scalar kernel arguments; tail batches
(`rem < 16`) are zero-padded to a full 16-row `TopK` call and their extra
outputs are discarded, so one tiling struct covers every batch.

Verification: `test_topk.py` ALL PASS (8/8 exact); new
`test_topk_stress.py` — 500 consecutive bare `topk_mask` launches with no
framework ops in between — no 507035, outputs exact (previously the
exception appeared after ~40-60 such launches).

### Megakernel AIV reduce: profiling says NOT the bottleneck

Temporary `profMode` kernel arg (since reverted) measured the kernel with
parts of the AIV chain disabled: full 1.111 ms / no-mask 1.117 / no-round
1.116 / no-mul-round-sum 1.114 / handshake-only (no AIV data movement at
all) 1.099 ms. The entire AIV fp32 vector chain (Cast/Maxs/16xMuls/
bf16-round/15xAdd per token) contributes < ~2% — the cost is on the AIC
side (strided Nd2Nz kvT re-reads: every tile re-reads its batch's 86 KB
kvT operand, ~447 MB strided traffic, plus 447 MB fixpipe slot writes)
and/or the CrossCoreSetFlag/WaitFlag ring protocol. A quick protocol
retune (FLAG_BATCH=4, NUM_SLOTS=4) deadlocked/timed out and was reverted;
the last PASS state (FLAG_BATCH=2, NUM_SLOTS=2) is kept. Also ruled out
this round: bf16 vector math for the reduce chain (Maxs/Muls/Relu do not
support bf16 in the dav-c220 basic API).

### Verification (final state of the round)

- Official harness, three fresh runs (`--warmup 20 --repeat 100`):
  `PASS accuracy; v0=6.691055 ms, v1=2.253309 ms, speedup=2.969x`
  `PASS accuracy; v0=6.720316 ms, v1=2.273539 ms, speedup=2.956x`
  `PASS accuracy; v0=6.714605 ms, v1=2.298504 ms, speedup=2.921x`
- `test_topk.py` 8/8 ALL PASS; `test_topk_stress.py` PASS (no 507035);
  `test_repro.py` max abs diff 0.0; `test_baseline_first.py` PASS.
- Stage probe: wq_b 0.045, rope 0.170, weights 0.069, kvT 0.060,
  megakernel 1.110, topk+mask 0.637; full 2.068 ms batched /
  2.217 ms sync-per-call.

## 2026-08-25 (round 4) — Megakernel AIC-side profiling; kvT L1 preload

### Bottleneck attribution (skip-build profiling)

Building the kernel with individual AIC stages compiled out (timing-only
builds) against the 1.092 ms baseline: no-fixpipe 0.802 (fixpipe slot
writes = 0.29 ms), no-Mmad 0.964 (0.13 ms), no-SplitB 1.098 (~0),
no-A-loads 1.089 (~0). Combined with round-3's finding that the entire AIV
reduce chain costs < ~2%, the residual ~0.7 ms is per-chunk pipeline
serialization: the depth-1 CO1 (L0C) and B2 (L0B) queues force
Mmad -> Fixpipe -> Mmad alternation for each 224-wide chunk.

### Landed: per-core kvT L1 preload (1.110 -> 1.090 ms)

Every tile previously re-read its batch's full kvT operand (86 KB) from GM
through strided Nd2Nz — ~447 MB/call of redundant traffic. Each core's
contiguous tiles span at most two batches, so both operands are now
preloaded once into L1 in NZ layout (held in the B1 queue's two buffers
for the kernel's lifetime) and SplitB slices each 224-wide chunk from L1
(panel offset `c*N_CHUNK*d_`, bit-identical layout). Verified
`test_repro.py` max abs diff 0.0, agreement 1.0. The small wall-clock gain
confirms bandwidth was never the limiter — serialization is.

### Reverted attempts

- FLAG_BATCH=4 / NUM_SLOTS=4 cross-core ring retune: deadlocked.
- Depth-2 CO1/B2 queues (to overlap Mmad with Fixpipe): deadlocked, most
  likely L0C capacity (2x56 KB CO1). Both reverted to the PASS state.

### Verification (final state)

- Official harness, three fresh runs (`--warmup 20 --repeat 100`):
  `PASS accuracy; v0=6.701761 ms, v1=2.261249 ms, speedup=2.964x`
  `PASS accuracy; v0=6.678766 ms, v1=2.257609 ms, speedup=2.958x`
  `PASS accuracy; v0=6.684971 ms, v1=2.270229 ms, speedup=2.945x`
- `test_topk.py` 8/8 ALL PASS; `test_topk_stress.py` PASS (no 507035);
  `test_repro.py` max abs diff 0.0; `test_baseline_first.py` PASS.
- Stage probe: wq_b 0.045, rope 0.170, weights 0.069, kvT 0.060,
  megakernel 1.090, topk+mask 0.637; full 2.054 ms batched /
  2.200 ms sync-per-call.

## 2026-08-25 (round 5) — Cross-core handshake investigation: root causes and verdict

### Deadlock root causes

- FLAG_BATCH=4 / NUM_SLOTS=4 "deadlock" (round 3): **never happened** — that
  test ran in a background ssh session that died from network flakiness; no
  output was ever observed. With the ring geometry now runtime-parameterized
  (`INDEXER_FLAG_BATCH`/`INDEXER_NUM_SLOTS` env -> scalar kernel args,
  prologue generalized to `numSlots` batches in flight), the same (4,4)
  combo runs correctly at 1.100 ms.
- Depth-2 CO1/B2 queues: real hang, consistent with L0C capacity < 112 KB
  (2 x 56 KB CO1 buffers). Not retried.

### Verdict: handshake count is NOT the bottleneck

Sweep of ring geometries (flagBatch, numSlots), all correct:
(4,2)=1.103, (2,4)=1.089, (4,4)=1.100, (8,2)=1.103, (8,4)=1.106,
(16,2)=1.128, (2,8)=1.092, (8,8)=1.110 ms (baseline (2,2)=1.087). Cutting
cross-core round-trips ~8x (FLAG_BATCH=16) makes things slightly WORSE, so
the flag-round-trip latency theory (130 trips x ~5 us) is disproved. The
residual ~0.7 ms is inter-pipe serialization latency between the ~12 queue
ops per tile (queue event handoffs MTE2 -> LoadData -> cube -> FIX), which
no ring parameter changes. Direction 4 (flag polling) was therefore skipped
as it targets the same disproved mechanism.

The only remaining megakernel lever is fewer/larger ops per tile (e.g.
TILE_M=128 to halve chunk and sync counts), bounded by L0B (64 KB) and
L0C (<112 KB observed) capacities; deferred to a dedicated round with the
AIV token-mapping rework it implies.

### Landed state

Runtime-parameterized ring (defaults 2/2 = the verified protocol) plus the
kvT L1 preload; megakernel 1.087-1.090 ms, effectively unchanged.

### Verification

- Official harness, three fresh runs (`--warmup 20 --repeat 100`):
  `PASS accuracy; v0=6.687885 ms, v1=2.233254 ms, speedup=2.995x`
  `PASS accuracy; v0=6.690476 ms, v1=2.260098 ms, speedup=2.960x`
  `PASS accuracy; v0=6.682316 ms, v1=2.276909 ms, speedup=2.935x`
- Stress (`--warmup 200 --repeat 500`):
  `PASS accuracy; v0=6.657021 ms, v1=2.263639 ms, speedup=2.941x`.
- `test_topk.py` 8/8 ALL PASS; `test_topk_stress.py` PASS (no 507035);
  `test_repro.py` max abs diff 0.0; `test_baseline_first.py` PASS.

## 2026-08-25 (round 6) — Larger-tile megakernel redesign (TILE_M=128): 1.087 -> 0.977 ms

### Design (paper geometry)

Round 5 proved the kernel is bound by inter-pipe serialization across ~12
queue ops per tile, invariant to ring geometry. Halving the tile count is
the only lever. Buffer geometry for TILE_M=128 (8 tokens/tile, M=128,
K=64): A2 (L0A) 16 KB, B2 (L0B) 28 KB, CO1 (L0C) 128x224 fp32 = 112 KB
single-buffered — the largest N chunk still fitting L0C (the earlier
depth-2 2x56 KB CO1 hang pins L0C below 112 KB usable for two buffers;
a single 112 KB buffer works). Alternatives rejected on paper:
N_CHUNK=112 (safe CO1 but ~2x chunk ops), TILE_M=256 (CO1 224 KB > L0C).

### Implementation

- AIC: 8 tokens/tile (M=128), 3 chunks of 224; tiles/core 260 -> 130;
  kvT L1 preload (round 4) unchanged; ring protocol (runtime
  flagBatch/numSlots, 2/2 defaults) unchanged.
- AIV: 4 tokens per sub-block (tokens sub*4+tok, tok 0..3); outF32/outQ
  widened to 4 rows (UB total 129 KB < 192 KB).
- Numerics bit-identical by construction: same bf16 RNE slot rounding,
  same per-token fp32 reduce order, same causal mask.

### Verification

- `test_repro.py`: max abs diff 0.0, topk agreement 1.0.
- `test_topk.py` 8/8 ALL PASS; `test_baseline_first.py` PASS;
  `test_topk_stress.py` PASS (no 507035).
- Official harness, three fresh runs (`--warmup 20 --repeat 100`):
  `PASS accuracy; v0=6.688526 ms, v1=2.114828 ms, speedup=3.163x`
  `PASS accuracy; v0=6.693576 ms, v1=2.116968 ms, speedup=3.162x`
  `PASS accuracy; v0=6.696196 ms, v1=2.245224 ms, speedup=2.982x`
- Stress (`--warmup 200 --repeat 500`):
  `PASS accuracy; v0=6.659595 ms, v1=2.071107 ms, speedup=3.215x`.
- Stage probe: megakernel 1.087 -> 0.977 ms; full model 2.054 -> 1.936 ms
  batched, 2.066 ms sync-per-call.

## 2026-08-25 (round 7) — TopK valid-prefix pruning: 0.637 -> 0.545 ms

### Step 1 landed: valid-prefix pruning + strided batch assignment

- Observation: for causal rows only the first `limit=(s+1)/ratio` columns
  are finite; `-inf` sorts last in stable order with ascending original
  index, so any prefix `>= max(limit, 128)` is bit-identical to the full
  672 (finite entries first, then exactly indices `limit..127` for the
  `-inf` tail — no tail synthesis needed).
- TopK cost scales strongly with inner (measured: inner=224 -> 0.288 ms,
  inner=672 -> 0.637 ms for the whole tensor).
- Each 16-row batch now runs TopK at compile-time `inner=224` when its max
  limit fits (34.5% of batches) and `inner=672` otherwise. Two `TopkTiling`
  structs are passed as 56 scalar int32 args — 84 args (3 groups) exceeds
  the launch-stub marshalling limit (fails somewhere between 65 and 93
  total args; a 93-arg build crashed with 507035). Template dispatch is
  required: a runtime `inner` is as slow as 672 (the advanced TopK needs
  the compile-time constant).
- Critical fix #2: batch->core assignment changed from contiguous row
  ranges to strided (batch `bi` -> core `bi % 40`). With contiguous ranges
  ~75% of cores processed only high-limit rows, so the kernel time stayed
  bounded by the slowest core (0.639 ms) despite pruning working correctly
  (verified via on-device printf: grp selection was right all along).
- Prefix loading uses strided `DataCopyPad` (compact [16, inner] rows),
  also cutting MTE2 traffic for pruned batches.

Result: topk+mask 0.637 -> 0.545 ms; full model 1.936 -> 1.829 ms batched.

### Step 2 (partial bitonic top-128): not landed

Not implemented this round. The remaining 0.545 ms decomposes into a
~0.29 ms launch/MTE2 floor (inner=224 for everything measures 0.288 ms)
plus the `inner=672` batches' compute; a custom bitonic network would have
to beat 0.545 ms while preserving the exact stable-descending tie order
bit-for-bit. Given the exactness risk and the verified gains above, step 1
is kept as the round's deliverable; the bitonic route remains documented
as a candidate for a dedicated round.

### Verification

- `test_topk.py` 8/8 ALL PASS (covers valid<128 rows); `test_repro.py` max
  abs diff 0.0; `test_baseline_first.py` PASS; `test_topk_stress.py` 500
  bare launches, no 507035.
- Official harness, three fresh runs (`--warmup 20 --repeat 100`):
  `PASS accuracy; v0=6.699687 ms, v1=2.055252 ms, speedup=3.260x`
  `PASS accuracy; v0=6.673932 ms, v1=2.047227 ms, speedup=3.260x`
  `PASS accuracy; v0=6.682457 ms, v1=2.057742 ms, speedup=3.247x`
- Stress (`--warmup 200 --repeat 500`):
  `PASS accuracy; v0=6.685217 ms, v1=2.036182 ms, speedup=3.283x`.

## 2026-08-25 (round 8) — Megakernel N_CHUNK=96 + depth-2 queues; CPU trims; rope attempt reverted

### Item 1 landed: megakernel 0.977 -> 0.854 ms

Round-5/6 profiling showed per-chunk pipeline serialization (depth-1
CO1/B2 queues force Mmad and Fixpipe to alternate). TILE_M=128 with
N_CHUNK=96 gives CO1 = 128x96 fp32 = 48 KB per buffer, so TWO buffers fit
L0C (<112 KB usable, established earlier): depth-2 CO1 and B2 queues let
Mmad of chunk c+1 overlap Fixpipe of chunk c. nChunks = 672/96 = 7
exactly. Numerics unchanged (same GEMM/rounding paths, just smaller
chunks). Verified no deadlock; test_repro.py diff 0.0.
Rejected alternatives (paper): N_CHUNK=448/TILE_M=64 (2080 sync ops/core
vs 1430), N_CHUNK=112 depth-2 CO1 (112 KB, the previously-hung capacity),
TILE_M=256 (CO1 > L0C).

### Item 2 landed: CPU/launch overhead trims

CPU-side enqueue measured at 0.468 ms/call (probe: per-dispatch costs).
Two safe trims: (a) removed the per-call `_kvT_pad[...].zero_()` tail
clear — the persistent buffer is torch.zeros at allocation, nothing ever
writes the tail, and pad-region scores are masked to -inf downstream in
both causal paths anyway; (b) cached the cube-core-count query (two acl
calls per megakernel launch before). CPU enqueue now 0.441 ms; batched
full model 1.722 -> 1.687 ms. Rejected: folding the weights scale into
the projection (changes bf16 rounding of the baseline product),
eliminating the kvT transpose copy (needs an Nd2Nz layout rework for
marginal gain).

### Item 3 rejected: rope 4-s block rewrite (0.170 -> 0.260 ms)

Grouping 4 consecutive s values per unit (one DataCopyPad in/out, vector
chain on 4x elements, per-group cs/sn broadcast planes) made the rope
kernel SLOWER (0.260 ms, correct but slower), so it was reverted to the
verified 0.170 ms version; the file state is identical to the pre-attempt
kernel apart from the item-1 changes.

### Verification (final state)

- `test_repro.py` max abs diff 0.0, agreement 1.0; `test_topk.py` 8/8;
  `test_topk_stress.py` PASS (no 507035); `test_baseline_first.py` PASS.
- Official harness, three fresh runs (`--warmup 20 --repeat 100`):
  `PASS accuracy; v0=6.710800 ms, v1=1.873442 ms, speedup=3.582x`
  `PASS accuracy; v0=6.692379 ms, v1=1.877592 ms, speedup=3.564x`
  `PASS accuracy; v0=6.710320 ms, v1=1.880212 ms, speedup=3.569x`
- Stress (`--warmup 200 --repeat 500`):
  `PASS accuracy; v0=6.693604 ms, v1=1.866051 ms, speedup=3.587x`.
- Stage probe: megakernel 0.857, topk+mask 0.545, rope 0.170; full model
  1.690 ms batched, 1.836 ms sync-per-call.

## 2026-08-26: in-megakernel TopK fusion rejected; nine-tier TopK pruning landed

Goal: eliminate the suspected ~0.29 ms launch/MTE2 floor of the standalone
`indexer_topk` kernel (0.545 ms stage) by fusing exact top-128 selection
into the megakernel AIV epilogue (Plan A), with a radix-select standalone
kernel as Plan B.

### Finding 1: the "launch/MTE2 floor" does not exist — TopK is sort compute

Skip-build attribution on the fused variant (PROF_SKIP_TOPK, timing-only):
megakernel with the full epilogue minus the TopK/postprocess block = 0.871 ms
(baseline megakernel 0.854 ms, so all scaffolding — bf16 rounding, prefix
casts, INT64 writes — costs ~0.017 ms); the advanced-TopK block itself adds
~0.53 ms of vector-pipe compute spread over the 40 AIVs. The standalone
kernel is therefore ~95% Sort32/merge compute, not launch overhead; fusing
can only move that compute onto AIVs that are already saturated by the
reduce stage (AIV reduce time ≈ AIC time, 0.85 ms).

### Finding 2: Plan A (fused epilogue) measured break-even to worse

Implemented the fused epilogue with the exact proven selector (advanced
AscendC::TopK on the bf16-rounded rows in UB, valid-prefix pruning, INT32
mask/offset postprocessing, hardcoded per-tier tiling constants; correctness
was bit-exact on all gates). The planned RTop-K-style radix/binary-search
select was NOT used: a threshold select cannot emit torch.topk's fully
sorted (value desc, index asc) output order without an extra sort stage, and
a cost model of exact counting (16 adaptive compare+count passes over the
inner prefix plus the candidate-ordering stage) lands in the same
instruction-count class as Sort32-TopK while adding per-row scalar
synchronization — so reusing the proven stable-sort TopK was the
exactness-safe route for the experiment. Measured fused megakernel:
outter=4/tile 1.429 ms, outter=8/batch 1.398 ms, outter=16/2 batches
1.456 ms (1.430 with six inner tiers) vs 0.854 unfused; the AIVs are the
bottleneck once TopK is added. Official harness on the best fused variant:
3.495x < 3.58x baseline — Plan A reverted per the revert rule (megakernel,
wrapper, and tests restored to c855ae05).

### Item landed: nine-tier inner pruning in the standalone topk kernel

The standalone kernel previously dispatched each 16-row batch to
inner=224/672 (avg inner ~518). Sort cost scales with inner, so the dispatch
was refined to nine compile-time tiers {128, 192, 256, 320, 384, 448, 512,
576, 672} (avg inner ~360), with all nine TopkTiling structs hardcoded in
`op_kernel/indexer_topk_tiling_consts.h` (generated by
`tools/dump_topk_tiling.cpp` via TopKTilingFunc; the extension memcmp-verifies
them once per process — passing nine structs as scalar args would exceed the
65-93 launch-stub limit, and GM tiling tensors are unreadable in quiet
process states). Result: custom topk+mask 0.545 -> 0.421 ms; full model
1.690 -> 1.573 ms batched.

### Verification (final state)

- `test_repro.py` max abs diff 0.0, agreement 1.0; `test_topk.py` 8/8 exact;
  `test_topk_stress.py` PASS (500 bare launches, no 507035);
  `test_baseline_first.py` PASS.
- Official harness, three fresh runs (`--warmup 20 --repeat 100`):
  `PASS accuracy; v0=6.685191 ms, v1=1.744950 ms, speedup=3.831x`
  `PASS accuracy; v0=6.699551 ms, v1=1.710194 ms, speedup=3.917x`
  `PASS accuracy; v0=6.664751 ms, v1=1.740904 ms, speedup=3.828x`
- Stress (`--warmup 200 --repeat 500`):
  `PASS accuracy; v0=6.704736 ms, v1=1.740715 ms, speedup=3.852x`.
- Stage probe: megakernel 0.850, topk+mask 0.421, rope 0.169; full model
  1.573 ms batched, 1.697-1.728 ms sync-per-call.

### Build hygiene note

The server `build/` cache once retained a stale `-DPROF_SKIP_TOPK`/`-DPROF_SKIP_A`
flag from a failed probe configure (`CMAKE_ASC_FLAGS`/`CMAKE_CXX_FLAGS` cache
variables persist across reconfigures and silently no-op kernels). Always
wipe `build/` after probe-flag experiments; verification runs after this
incident used a fresh build dir.
