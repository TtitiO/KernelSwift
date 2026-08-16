# SparseAttention — Optimization Journey

This document tells the story of how the `SparseAttention` operator was
optimized for the Huawei Atlas 910B3, from a 8.05 ms Torch baseline down to
**1.48 ms (~5.4x)** while passing the official accuracy check.

It is written to be read top-to-bottom as a narrative: each section describes
*the bottleneck we hit, the idea we tried, what we learned, and how that set up
the next step*.  Concrete numbers and the harder low-level details are included,
but a short hardware glossary (§3) is provided up front so the terminology is
self-contained.

---

## 1. The task in one paragraph

For every query position `(b,m)` and head `h`, the operator computes a
sparse, sink-aware attention:

1. score `q[b,m,h,:]` against the `K=16` `kv` rows chosen by `topk_idxs`;
2. softmax over those 16 scores, with an extra `attn_sink[h]` term in the
   denominator;
3. weighted average of the 16 chosen `kv` rows.

The official shape is `B=8, M=2600, N=32, H=64, D=128, K=16`, all in BF16
except `topk_idxs` (int32) and `attn_sink` (fp32).  The reference is Torch and
runs in ≈ **8.05 ms**.

Two facts about the problem shape everything that follows:

- **`kv` is shared across heads** (it has no head dimension), so the two
  matrix-multiplies are *tall-and-skinny* GEMMs (`N=32` or `K=32`).  The cube
  is under-utilized on such thin shapes.
- **The softmax touches only 16 of 32 columns**, selected by `topk_idxs`
  (which may repeat), and the max/sum must include the sink term.  This is
  *not* a dense softmax — it is a gather, a masked reduction, and a scatter.

---

## 2. The optimization strategy

**Method in one sentence:** never optimize blind — at every step, *measure the
real bottleneck, attack the biggest lever, then re-measure*.

The work followed one repeatable loop, applied three times (once per "front"):

1. **Establish correctness first.**  A wrong-but-fast kernel is worthless (the
   competition scores correctness before speed), so each phase started from a
   passing build and never traded accuracy for speed.
2. **Profile to find the bottleneck.**  The profiler split (per-kernel and
   per-stage timings) — not intuition — decided what to optimize next.
3. **Change one variable at a time.**  Each optimization was applied, measured
   against the previous build, and kept only if it actually helped; plausible
   changes that didn't help were reverted.
4. **Re-profile and repeat.**  When a front stopped yielding wins, we profiled
   again to find the next wall.

The three **fronts** were attacked in the order the measurements exposed them:

| Front | The question | How it was answered |
|---|---|---|
| **Correctness** | Can a custom kernel match the reference at all? | a three-kernel C-like implementation (§4), routed around hardware/toolchain bugs |
| **Compute** | Are the vector and cube units running at full width? | batch the softmax across rows, tune the GEMMs, shrink the data to FP16 (§5) |
| **Memory** | Can we stop moving 340 MB of intermediates through HBM? | fuse QK + softmax into one mixed kernel so each tile stays on-chip (§7) |

**Working principles** that persisted throughout:

- **Evidence over habit.**  The two load-bearing choices — the `[H,N]` layout
  and "no `transA` / no `Scatter`" — were made because measurements showed the
  alternatives were broken or unsupported, not because they looked cleaner.
- **The widest idle lane wins.**  Most compute wins came from one question:
  "what is the widest hardware unit, and are we using all of it?" (64 → 256-wide
  softmax vectors; thin GEMMs → preload).
- **Memory is the last wall, but the tallest one.**  Compute got us to 4.43x;
  the jump to 5.4x came entirely from removing memory traffic — the single
  highest-leverage change in the whole project.

---

## 3. Hardware & terminology (one-time glossary)

The 910B3 AI core has two execution units that share on-chip memory:

| term | meaning |
|---|---|
| **AIC / cube** | the matrix-multiply unit; runs the GEMMs |
| **AIV / vector** | the element-wise/reduction unit; runs the softmax math |
| **UB** | the vector unit's on-chip scratch (192 KB) — where element-wise work happens |
| **L0C** | the cube's on-chip accumulation buffer (128 KB) |
| **GM / HBM** | global (off-chip) memory — large but ~10x slower than on-chip |
| **L2** | the on-chip cache between GM and the cores — a small tile can live here |
| **MTE2 / MTE3** | the DMA engines that move data GM↔UB in the background |
| **fixpipe** | the cube's write-back path that moves an L0C result out to UB/GM |

The single most important fact for this project: **data moved between two
separate kernel launches always goes through HBM; data moved within one
launch can stay in L2/UB.**  That is the entire motivation for the fusion in
§8.

---

## 4. Step 1 — a correct three-kernel implementation

**Bottleneck:** none yet — just get a correct custom kernel.

**Idea:** split the operator into the three natural stages, each an Ascend C
kernel, glued together by a Torch op `torch.ops.npu.sparse_attn`:

1. **QK** (cube): `scores = q @ kv^T` → `[B,M,H,N]`
2. **softmax** (vector): gather 16 columns → sink-aware softmax → scatter the
   16 weights back into `[B,M,H,N]`
3. **PV** (cube): `out = agg @ kv`

Two design choices were made here on evidence and **kept for the whole
project**:

- Store intermediates in **`[H,N]` layout** and never use a transposed `A`.
  (The `MatmulImpl` transA path on this CANN silently produces wrong results —
  a real compiler/hardware bug we measured and then routed around.)
- Use **`Gather` instead of `Scatter`** everywhere a sparse write is needed,
  because `Scatter` is simply unsupported on `dav_2201`.

**Result:** correct, but barely faster than Torch — the softmax ran *one row at
a time* on 64-wide vectors, using ~1/4 of the vector unit.

**The small-but-fatal bugs cleared in this step** (each a distinct debug cycle,
each now a documented "don't do that"): the `MatmulTiling` name clash; the
`DataCopyPad` 3-arg vs 4-arg overloads; `ASCEND_IS_AIV` being a `constexpr(...)`
macro that must not be parenthesized; kernel symbols being C++-mangled (so the
host declares them *without* `extern "C"`); a missing MTE2→vector sync producing
garbage softmax output; and `Adds<uint32_t>` being unsupported (use `int32_t` +
`ReinterpretCast`).

---

## 5. Steps 2–5 — making the compute efficient (→ 4.43x)

These four steps are a chain of "find the widest idle lane and fill it."

### 5.1 Batch the softmax math across rows (→ 4.2x)

**Bottleneck:** the softmax math dominated, and it ran per-row on narrow
vectors.

**Idea:** the softmax math is independent per `(m,h)` row, so process `ROWS=4`
rows together as one **256-element vector operation** (4 rows × 64 heads fills
the 256-wide vector unit).  The scores are re-laid out as `sK[K, ROWS*H]` so
each `Max`/`Exp`/`Add`/`Div` works on a contiguous block.

**Barrier resolved:** the first attempt was *wrong* (`max_abs_diff ≈ 0.15`)
because the sink load from GM was read before the DMA finished.  A
`PipeBarrier<PIPE_ALL>` after the sink load fixed it.  **Lesson: every MTE2
load must be fenced before the vector reads it.**

### 5.2 Store the scores as FP16 (→ 4.34x)

**Bottleneck:** the 85 MB scores intermediate dominates traffic.

**Idea:** store scores as FP16 instead of FP32 (halving that traffic), then
re-cast to FP32 *in one batched operation* before the softmax (the softmax
math stays FP32, so precision is preserved).

**Barrier resolved:** casting *per column* was neutral/slower; casting the whole
tile at once made it a win.  BF16 scores were also tried and **rejected** — the
logit rounding pushed the error to ~0.023, above the `1e-2` tolerance.

### 5.3 Tune the GEMMs (→ 4.37x)

**Bottleneck:** the two skinny GEMMs are launch/load bound.

**Idea:** sweep the `MatmulImpl` config.  N-direction MTE2 preload
(`doMTE2Preload=2`) helped; M-direction preload, `enableL1CacheUB`, and a
larger `singleCoreM=512` did **not** (the last one produced wrong results — the
tiling API's base-block bookkeeping breaks).  Each was measured and reverted
independently.

### 5.4 Batch the softmax transpose (→ 4.43x)

**Bottleneck:** the final `[N,H]→[H,N]` transpose was done per row.

**Idea:** accumulate into a single `[ROWS,N,H]` scratch tile and do **one**
transposing `Gather` across all four rows.

**Result after steps 2–5: 4.43x (≈ 1.81 ms).**

---

## 6. Step 6 — profiling reveals the wall

A fresh profiler split at 4.43x: **QK 0.55 ms, softmax 0.98 ms, PV 0.49 ms**.

Breaking the softmax down further: gather 0.26 ms (16 column-`Gather`s per
row), math 0.2 ms (17 `Exp`s per group), accumulate 0.15 ms, transpose 0.03 ms,
memory ~0.12 ms.  The gather and the `Exp`s are irreducible without `Scatter`
(which doesn't exist here), and the GEMMs are capped by their thin `N=32`/`K=32`
dimensions.

That left **memory traffic** as the only big lever: the three separate kernel
launches force ≈ **340 MB** of intermediate data (scores + agg) through HBM on
every call.  The path to the next jump was to make QK and softmax (and ideally
PV) run **inside one kernel**, so each tile's intermediate stays in L2/UB.

---

## 7. Step 7 — fusing QK and softmax (→ 5.4x)

This was the decisive step and the hardest.  It is told as a sequence of four
blockers, each with its fix.

**The goal:** one kernel in which the cube and vector are co-resident.  The
cube computes a tile of scores; the vector immediately consumes it; the two
overlap.

### 7.1 Blocker: no complete reference exists

The Ascend ecosystem on this CANN has **no standalone mixed-kernel example**;
the production `flash_attention_score` is a multi-thousand-line built-in for a
different architecture.  The mechanism had to be reconstructed from three
partial sources — a Gitee `MatmulLeakyRelu` sample, one CANN best-practices
note, and header reverse-engineering — and then confirmed by experiment.

**What we established:**

- A mixed kernel is `__global__ __mix__(1, 2)`.  With `blockDim=N` it launches
  `N` cube blocks + `2N` vector blocks; the compiler compiles the source twice
  (`SPLIT_CORE_CUBE` / `SPLIT_CORE_VEC`), and vector code must be guarded by
  `if ASCEND_IS_AIV`.
- `Matmul` resolves to a **KFC message client** on the vector side and a
  `MatmulImpl` on the cube side.  `REGIST_MATMUL_OBJ` turns the cube into a
  *server* that executes commands sent by the vector — so **the vector is the
  conductor** and the cube is a coprocessor.

### 7.2 Blocker: the kernel hung in the handshake

The first fused kernel printed its first marker and never the second — the cube
spun at 100%.

**Cause:** on `dav_2201` *without* the super-kernel define, the workspace-clear
+ event-15 notify inside `ClearWorkspace` is dead code, so the vector's
`WaitEvent(15)` inside `REGIST_MATMUL_OBJ` waits for a signal that never comes.

**Fix:** reproduce that handshake by hand before the macro — the cube clears
the mailbox and notifies event 15; the vector sets its masks.

### 7.3 Blocker: the "L0C→UB" path silently wrote to the wrong place

We first tried the sample's `GetTensorC` to move the QK result straight from
L0C into UB (zero GM traffic).  Dumping the UB showed garbage.

**Cause:** the KFC server routes the C destination by whether its position is
L1-backed.  `VECIN` is not, so it falls into the *GM* branch and treats the
vector's UB address as a GM address.  The L1 position (`C1`) crashed the core.

**Fix:** pivot to the `flash_attention_score` pattern — write the QK tile to a
GM scores buffer and read it straight back.  The 16 KB tile's write-back lands
in L2 and the immediate read hits it, so HBM is still avoided.  (This also
validated that the QK output itself was bit-exact.)

### 7.4 Blocker: scattered NaNs and core crashes

The first passing kernel produced occasional `NaN`s and, sometimes, an
"aicore exception".

**Cause:** three async-DMA-vs-vector races, all of the form *"a background copy
ran while the vector was still reading/writing the same buffer."*

**Fixes:**
1. fence the sink load before `sinkBroadcast` reads it;
2. **double-buffer** the scores/idx/agg UB buffers (a single buffer reused
   across tiles lets tile `t+1`'s copy race tile `t`'s math);
3. fence the transpose `Gather` before the agg copy-out.

### 7.5 Blocker: it was correct but *slower* (3.59x)

The first fused kernel that passed accuracy was slower than the three-kernel
split — every tile did QK → wait → softmax → wait with **no overlap**.

**Fix: software pipelining.**

- Issue the QK **asynchronously** (`IterateAll<false>(gm, 0, false, true)` —
  the `waitIterateAll=true` flag is mandatory, or the later wait deadlocks).
- Issue `QK(t+1)` *before* `Softmax(t)`, so the cube computes the next tile
  while the vector runs the current one.
- Double-buffer (from 7.4) so the DMA and the math never touch the same buffer.

**Result: 4.99x**, and after removing the temporary inter-kernel sync that had
masked the 7.4 races: **~5.4x (≈ 1.48 ms).**

---

## 8. Step 8 — the full-fused megakernel (attempted, blocked by CANN)

**The concept.**  A *megakernel* fuses an entire operator (here QK → softmax →
PV) into one kernel launch, eliminating launch overhead and cross-kernel HBM
round-trips (the term comes from the LLM-inference literature — see e.g.
[TheoremPath's "Megakernels" article](https://theorempath.com/topics/megakernels)
and [a megakernel decode worklog](https://emre570.bearblog.dev/megakernel-decode/)).
For this operator the theoretical ceiling is attractive: the msprof profile
shows the fused QK+softmax kernel is **vector-bound** (963 µs, cube MAC at only
~5%) and the separate PV kernel is **MTE2-bound** (254 µs, 52% MTE2) — the cube
has ~95% idle time, so a fully fused megakernel could hide the PV under the
softmax and reach ~7–8x.

**What was tried, and the blocker.**  Folding the PV into the fused kernel was
attempted two ways:

1. **KFC multi-object registration** (`REGIST_MATMUL_OBJ(qkMm, &qkT, pvMm,
   &pvT)): compiles, runs, and a single PV `IterateAll` works — but per-tile
   PV usage (sync or async, 1-deep waits, chunked bursts, either registration
   order) intermittently **corrupts the QK+softmax** (~13% of runs produce NaN
   in the softmax output).  The corruption is deterministic in *pattern* (rows
   1–3 of each 4-row tile) but timing-dependent, i.e. the KFC message routing /
   fixpipe-wait protocol for a second cube object is unreliable on `dav_2201`.
2. **Manual cross-core sync** (AIC runs QK+PV *directly* via `MatmulImpl`, AIV
   runs the softmax, `CrossCoreSetFlag<2,PIPE_MTE3>` per tile): deadlocks —
   in the `__mix__` split-core compile the cube object is the KFC *client*; the
   direct `MatmulImpl` path used by the standalone kernels is not available to
   the AIC inside a mixed kernel, so there is no way to drive the cube without
   the (broken) multi-object KFC.

**Conclusion.**  The full QK→softmax→PV megakernel is blocked by CANN 8.5.2
`dav_2201`: the only supported way to run the cube inside a mixed kernel is the
KFC server/client, and its multi-object mode is unreliable.  The submission
keeps the fused QK+softmax kernel plus the separate PV GEMM at ~5.4x; the
remaining ~0.2 ms (→ ~6x) requires a CANN fix or a future release.

---

## 9. Results timeline

| milestone | speedup | v1 latency |
|---|---|---|
| Baseline (Torch) | 1.00x | 8.05 ms |
| + `ROWS=4` softmax batching | 4.20x | 1.91 ms |
| + FP16 scores (batched cast) | 4.34x | 1.854 ms |
| + N-direction MTE2 preload | 4.37x | 1.841 ms |
| + batched softmax transpose | 4.43x | 1.81 ms |
| fused QK+softmax (serialized) | 3.59x | 2.24 ms |
| fused QK+softmax (pipelined) | **~5.4x** | **1.48 ms** |

Final result on `liteserver-4db9` (8× 910B3), `benchmarks/ks/auto_bench.py`
(warmup 200, repeat 500), `atol=rtol=1e-2`: **PASS accuracy, ≈ 5.4x**, stable
across eight consecutive fresh runs (5.488 / 5.443 / 5.418 / 5.373 / 5.421 /
5.449 / 5.465 / 5.419).

---

## 10. Lessons learned

1. **Layout decisions are load-bearing.**  Choosing `[H,N]` and avoiding
   `transA` (a real CANN bug) and `Scatter` (unsupported) shaped every later
   step; both were made from evidence, not habit.
2. **The Ascend C ecosystem on this CANN is incomplete.**  The mixed-kernel
   model had to be reverse-engineered, and two genuine CANN gaps (the dead
   handshake, the `GetTensorC` mis-route) had to be worked around by hand.
3. **Async DMA vs vector races are the #1 correctness hazard.**  Every NaN and
   crash in the fused kernel traced to a missing `PipeBarrier` or an
   un-double-buffered UB buffer; the standalone kernels had hidden these behind
   `TQue` machinery.
4. **Fusion only pays if it overlaps.**  A serialized fused kernel was *slower*
   than three separate kernels; the win came entirely from the async
   QK(t+1)∥softmax(t) pipeline, not from merely colocating the stages.
5. **Measure, bisect, revert.**  Several plausible optimizations
   (`singleCoreM=512`, `enableL1CacheUB`, per-column cast, an index register
   cache) were measured and reverted; each was one variable changed at a time.

---

## 11. Time budget (per-step duration)

One combined list, in the order the work was done.  Durations are elapsed
active work per step; `~` marks an *estimated* duration (clock times for those
steps were not recorded — the rest are measured from server file/build
timestamps).

| Progress step | Result | Duration |
|---|---|---|
| 3-kernel framework (host tiling, QK/PV `MatmulImpl`, softmax kernel, Torch op glue) | correct kernel | ~2.5 h |
| Ascend C debugging (transA bug, `Scatter`, `DataCopyPad`, symbol mangling, MTE2→vec sync, `Adds<uint32_t>`) | stable | ~1.5 h |
| `ROWS=4` softmax batching (+ sink `PipeBarrier` fix) | → 4.2x | ~1 h |
| FP16 scores: per-column cast (reverted) → batched cast | → 4.34x | ~0.5 h |
| GEMM MDL/tiling sweep (`doMTE2Preload=2` kept) | → 4.37x | ~0.5 h |
| Batched softmax reverse-transpose | → 4.43x | ~0.3 h |
| Fusion research: samples, headers, `__mix__` launch, block mapping | — | ~1.5 h |
| First fused QK+softmax kernel; KFC handshake hang | (hang) | ~1.2 h |
| Handshake workaround + `GetTensorC` mis-route pivot + pipe barriers | fused kernel correct | ~2.5 h |
| Serialized fused kernel + inter-kernel sync probe | 3.59x (slower) | ~0.7 h |
| Software pipelining: async `IterateAll` + double-buffered UB | → **5.4x** | ~1.8 h |
| Megakernel via KFC 2-object PV (sync/async/chunked) | blocked (intermittent corruption) | ~2.0 h |
| Megakernel via manual cross-core sync | blocked (deadlock — direct cube unavailable) | ~1.5 h |
| Restore QK+softmax-only + final build | **5.4x final** | ~0.6 h |
| Design + journey documentation | — | ~0.4 h |

**Total: ≈ 18.5 h of active work.**  The fusion effort alone (research →
first kernel → debugging → pipelining → megakernel attempts → restore) is
≈ 12 h.

---

## 12. Step 9 — the basic-API megakernel (QK+softmax+PV in one launch, multi-core)

After the KFC megakernel attempts were blocked by CANN, the submission moved
to a **basic-API** fused kernel (`fused_sparse_attn_basic_kernel.asc`) that
runs the entire operator (QK GEMM → sink-aware softmax → PV GEMM) inside one
`__mix__(1,2)` launch using raw `DataCopy/Nd2Nz + LoadData + Mmad + Fixpipe`
instead of the KFC Matmul objects.  This alone took the timed path from
~1.48 ms to **1.15 ms** (~6.8x).

### 9.1 The four megakernel pillars (already present)

1. **Multi-core tile parallelization** — `blockDim = 20` cube cores; the tile
   loop is partitioned with `GetBlockIdx()` (260 tiles per core, all cores
   busy).  The two AIV sub-blocks per core split each tile's 4 m-rows.
2. **AIC double-buffered pipeline** — depth-2 `A1/B1` queues; `CopyIn(t+1)`
   is issued before `Compute(t)` so DMA overlaps the cube.
3. **AIV ping-pong prefetch** — depth-2 `scoresQ_/idxQ_`; tile t+1's scores
   and indices are loaded while tile t is being consumed.
4. **Batched cross-core flags** — `FLAG_BATCH=4` tiles per QK/AGG flag
   handshake instead of one per tile.

### 9.2 AIV vector-pipe reduction (1.15 ms → 0.91 ms, +5x of the remaining)

Profiling showed the kernel is **AIV-issue-bound**: the vector pipe was busy
~88% of the time with only ~19% actual arithmetic (gathers/casts/moves/SFU
fill the rest), and every removed vector call was worth ~2.9 us of wall time.
Attacks, in order (each measured and kept only if correct AND faster):

1. **Hoisted index loads** — `idx.GetValue()` was called 64x per tile per AIV
   (twice per k); hoisting into a per-row `nk[2][16]` scalar array halved the
   scalar-UB traffic.  (First attempt with a flat `nk[16]` array was a real
   bug: the accumulate loop re-read the *last* row's indices — caught by the
   m%4 pattern in the error map.)
2. **Max-init + reciprocal-div** — `maxH` initialized from `Max(sink, sK[0])`
   instead of an `Adds` copy; the 16 `Div`s became 1 `Reciprocal` + 16 `Mul`s.
3. **Compact per-sub-block sK + batched cast/scale** — the sK tile is laid out
   `[topk, vecLen]` privately per AIV sub-block, turning 16 per-column
   `Cast`s + 16 per-column `Muls` into 2 batched calls.
4. **Per-row transpose table** — the agg transpose `Gather` offset table
   shrank from 32 KB to 8 KB (row base applied via per-row src/dst pointers),
   freeing UB for the broadcast scratch.
5. **Batched softmax chain via Brcb Broadcast** — `maxH` and the reciprocal
   are fanned out to `[16,128]` with the hardware broadcast, so the 16
   subtracts / 16 exps / 16 muls became 1 broadcast + 1 batched op each
   (16 sequential max/add chain ops remain, they are dependency-bound).
6. **FP32 scores** — the QK fixpipe writes FP32 scores (no `F322F16` quant),
   so the AIV gathers FP32 directly and the 2048-element half→float cast is
   gone (perf-neutral vs FP16 but removes one rounding step; kept for
   precision).  Requires shrinking the over-allocated `sKBuf_`/`maxH2Buf_`
   (a 32 KB over-allocation pushed the AIV UB past 192 KB and caused an
   aicore exception — caught by bisection).
7. **Unified Duplicate** — the two per-row `aggT` zero-fills became one
   4096-element op (rows are contiguous per sub-block).

### 9.3 Tried and rejected (with evidence)

- **COLUMN_MAJOR fixpipe** (`CFG_COLUMN_MAJOR` L0C→GM) to write the PV output
  transposed — micro-test showed a scrambled layout on `dav2201`.
- **`[n][m]` agg scratch + swapped PV GEMM** (kill the transpose gather) —
  blocked because the cube's `Nd2Nz` `srcNdMatrixStride` is uint16 (max
  65535) and our batch stride is 166400.
- **Fusing the kvT transpose into the megakernel** — an AIV→AIC
  `CrossCoreSetFlag` handshake is fragile on this CANN (flag delivery breaks
  when one AIV sub-block takes an early-return path; also shape-dependent
  hangs), and the win was only ~5 us.  Reverted; the standalone 8 us
  transpose kernel stays.
- **ROWS=8 tiles** (halve per-tile overhead) — UB budget does not fit.
- **2D `BlockReduceMax/Sum`** — the k-dim stride is not expressible with the
  block-stride semantics (rows advance 1 element; the unit is 32 B).

## 13. Final numbers (2026-08-16)

| milestone | latency (raw kernel) | speedup (official bench) |
|---|---|---|
| KFC fused QK+softmax + PV (journey end) | 1.48 ms | ~5.4x |
| basic-API megakernel (4 pillars) | 1.15 ms | ~6.8x |
| + AIV vector reduction (steps 1-7) | **0.91 ms** | **~7.1x** |

Official `benchmarks/ks/auto_bench.py` (warmup 200, repeat 500, seed 42):
**PASS accuracy, v0 ≈ 7.98 ms, v1 ≈ 1.12 ms, speedup ≈ 7.06-7.12x** across
consecutive runs (the bench's per-iteration `set_seed` adds ~0.2 ms to both
sides).  Raw kernel latency (measured directly): **0.91 ms**, max_abs_diff
0.0156, allclose PASS for seeds {1,7,42,123,999}.

### 13.1 Double-test re-verification (2026-08-16, post-sync)

Re-built `libsparse_attn_ops.so` from the synced sources and re-ran the
official bench twice at the README settings (warmup 200, repeat 500,
seed 42), plus three more seeds at warmup 100 / repeat 200.  All five runs
PASS accuracy and stay comfortably above the 5x bar:

| run | seed | v0 (ms) | v1 (ms) | speedup | result |
|---|---|---|---|---|---|
| 1 | 42 | 8.003 | 1.152 | **6.95x** | PASS |
| 2 | 42 | 7.970 | 1.126 | **7.08x** | PASS |
| 3 | 7  | 7.973 | 1.143 | **6.97x** | PASS |
| 4 | 123| 7.982 | 1.142 | **6.99x** | PASS |
| 5 | 999| 7.979 | 1.131 | **7.06x** | PASS |

(Note: the very first launch on the server's device 0 hit an aicore timeout;
`npu-smi` reports device 0 as `Alarm`/contended.  Re-running on a healthy
device is consistently 6.9-7.1x with zero hangs.)
