---
name: kernelswift-a2-kernel-dev
description: "Route, implement, review, and validate KernelSwift operators for Huawei Atlas A2/Ascend 910B. Use for Ascend C, CATLASS, PTOAS/PTODSL, PyPTO, TileLang-Ascend, or Triton-Ascend work on SparseAttention, Indexer, Sinkhorn, and related custom kernels; enforce source-faithful semantics, a real custom-kernel timed path, and evidence-backed performance claims."
---

# KernelSwift A2 kernel development — v0.3

This is the project-specific front door for the 2026 KernelSwift A2/910B
track. It is a routing and admission policy, not a promise that a generated
kernel is fast. Read [a2-validation-contract.md](references/a2-validation-contract.md)
before making a correctness or speed claim. Read [dsl-landscape.md](references/dsl-landscape.md)
only when a route decision needs it; do not load the entire skill catalog by
default.

## Why v0.3 exists

The Huawei SparseAttention experiment exposed a workflow failure, not a
successful optimization:

- the known-correct Ascend C scalar-vector candidate measured about
  `99.99 ms` against a Torch baseline of about `8.05 ms` (`0.081x`);
- the attempted vectorized candidate compiled but failed correctness
  (`max_abs_diff` about `3.9`, `mean_abs_diff` about `0.29`);
- v0.2 was then restored to a copy of the correctness-only implementation;
- no checked-in v0.2 benchmark report or raw profiler artifact was found.

Therefore v0.1/v0.2 are **correctness-only prototypes**, not optimized
submissions. A folder name, successful compilation, or a README number never
promotes a candidate. v0.3 makes that promotion gate explicit.

## Non-negotiable project rules

1. `baseline/` is frozen. Never modify it to make a candidate pass.
2. `optimized/` is reserved for a candidate that has passed the submission
   contract and the measured performance gate. Put exploratory Torch, Triton,
   TileLang, or failed Ascend C variants in `comparisons/` or an explicitly
   labelled prototype directory.
3. The timed path must load and launch the compiled custom artifact. No
   `torch.einsum`, `torch.matmul`, `torch.softmax`, `torch.gather`, ATen/NPU
   composite, exception fallback, or availability branch may replace it.
4. Preserve the reference constructor and `forward` signatures exactly.
5. Do not claim an A2 result from CPU execution, syntax checks, compilation,
   or a single noisy timing.
6. Keep the last known-good artifact until the replacement passes both
   correctness and performance. Revert a failed or slower candidate.

## Route selection

Choose the narrowest route that preserves the real operator boundary. Load
only the listed follow-up skills for the selected route.

| Need | Route | Follow-up skills |
|---|---|---|
| C/C++-like GM/L1/L0/UB movement, vector/reduction/sort/index/fusion | **Ascend C** | `npu-arch`, `ascendc-env-check`, `ascendc-operator-design`, `ascendc-tiling-design`, `ascendc-api-best-practices`, `ascendc-operator-code-gen`, `ascendc-operator-compile-debug`, then precision/profiling |
| GEMM, grouped GEMM, matrix epilogue, Cube-heavy attention | **CATLASS** or Ascend C Matmul/Mmad | `catlass-op-design`, `catlass-op-develop`, `catlass-op-perf-tune`, plus the A2 validation contract |
| Explicit tile research route | **PTOAS/PTODSL** or **PTO-ISA C++** | Verify the pinned toolchain and retain lowered artifacts before investing in a candidate |
| Higher-level Python tile/runtime route | **PyPTO** or **TileLang-Ascend** | Use only for comparison/prototyping unless deliberately promoted after the C-like submission review |
| Existing Python Triton migration | **Triton-Ascend** | Treat as a comparison route; it is not C-like source |

Do not choose a route because a skill has a similar name. Confirm the source,
compiler invocation, target architecture, binary, registration, and adapter
all use that route. Filter CANNBot material to A2/`DAV_2201`; ignore A5,
`DAV_3510`, and Kirin assumptions.

### Operator-specific default routes

- **SparseAttention:** default to a mixed AIC/AIV or Cube-plus-Vector design.
  After one `(batch, sequence)` row gathers the K selected KV rows, the dense
  subproblems are `QK: [H,D] × [D,K] → [H,K]` and
  `PV: [H,K] × [K,D] → [H,D]`. A vector-only scalar loop is a diagnostic
  baseline, not an optimization plan for the official `H=64,D=128,K=16` case.
- **Indexer:** preserve the full score, mask, top-k, offset, and cache/state
  contract. Use Ascend C sort/index capabilities or a source-backed comparison;
  do not replace the layer with a score helper.
- **Sinkhorn:** use vector reductions and a stable accumulation dtype; measure
  the repeated normalization passes and avoid silently changing iteration count.

## Workflow

### 1. Freeze the contract before editing

Write a short design record containing the formula, real model/layer boundary,
all shapes and strides, layouts, dtypes and accumulation dtypes, masks,
duplicate/invalid-index behavior, state/cache lifetime, tail rules, core
partitioning, and tolerance. For the current SparseAttention benchmark the
contract is `q[B,M,H,D]`, `kv[B,N,D]`, `sink[H]`, and
`topk_idxs[B,M,K]`, with BF16 q/kv, FP32 sink/accumulation, INT32 indices,
sink as an extra softmax item, and BF16 output. Duplicates remain separate
softmax entries; invalid indices contribute neither value nor probability.

Inspect the matching `baseline/`, `comparisons/`, `dlblas/kernels/ascend/`,
benchmark, and test files. Do not change the boundary or layout merely to make
an implementation easier.

### 2. Design for the actual A2

Record all of the following in `design.md` or an equivalent artifact:

1. AIC/AIV responsibility and block/core decomposition;
2. GM → L1/L0/UB movement and layout conversions;
3. tile dimensions, alignment, padding, masks, and buffer lifetimes;
4. CopyIn → Compute → CopyOut stages, double buffering, and synchronization;
5. accumulator precision, overflow/NaN behavior, tie handling, and sparse
   index semantics;
6. host tiling, registration, adapter, exact `dav-2201` target, and artifact
   path/hash.

For SparseAttention, do not spend the whole iteration tuning a scalar
per-head/per-index loop. Attempt a source-backed Cube/Mmad/Matmul or
CATLASS design first. If that route cannot compile and launch after one bounded
debugging window (about 10–15 minutes), stop and report the blocker; retain a
correct prototype, but do not label it v0.3 optimized.

### 3. Implement and prove the launch path

The Python wrapper must load the shared library and call the registered custom
operator. The C++ adapter must launch the generated kernel on the current NPU
stream. Inspect the symbol, registration, and binary rather than trusting a
function name. Keep framework code out of the timed kernel path.

Use the direct-invoke stream contract deliberately: `stream(true)` may be
needed to drain queued framework work before a raw ACL launch. Do not add
unmeasured `torch.npu.synchronize()` calls around every candidate invocation.
If an explicit synchronization is required for correctness, document the data
dependency, measure its cost separately, and keep the same policy for the
baseline and candidate timing protocol.

### 4. Validate in layers, then promote

Run the layers in order:

1. **Static:** source, registration, adapter, UTF-8, no fallback, no obvious
   UB overlap, and exact interface.
2. **Compile:** CANN/compiler version, `DAV_2201` target, command, exit status,
   and generated binary hash.
3. **Load/launch:** named 910B device and runtime evidence that the custom
   symbol was loaded and launched.
4. **Correctness:** official shapes plus invalid indices, duplicates, tails,
   extreme sinks, masks, and state transitions; report max absolute/relative
   error and mismatch counts.
5. **Performance:** same device, inputs, dtype, layout, stream policy, and
   process environment; warmup-separated paired measurements, dispersion, and
   raw profiler output.

For the competition harness, use these exact stages:

```bash
# smoke: compile/load/accuracy and a cheap timing
python benchmarks/ks/auto_bench.py --v0_file baseline/<op>.py \
  --v1_file optimized/<candidate>/<candidate>.py \
  --atol 1e-2 --rtol 1e-2 --warmup 5 --repeat 20

# stable candidate measurement
python benchmarks/ks/auto_bench.py --v0_file baseline/<op>.py \
  --v1_file optimized/<candidate>/<candidate>.py \
  --atol 1e-2 --rtol 1e-2 --warmup 20 --repeat 100

# final only after the candidate is already correct and faster
python benchmarks/ks/auto_bench.py --v0_file baseline/<op>.py \
  --v1_file optimized/<candidate>/<candidate>.py \
  --atol 1e-2 --rtol 1e-2 --warmup 200 --repeat 500
```

Repeat the stable command in at least three fresh processes. Promote only if
correctness passes and the candidate is faster than the frozen baseline in at
least two of three runs, with no run showing a material correctness or launch
failure. Record the median and the raw outputs. `auto_bench.py` times v0 then
v1 serially and synchronizes after each call; this is the competition metric,
but it is not a substitute for a profiler that separates launch, synchronization,
memory, Vector, and Cube time.

### 5. Profiler and handoff gate

Before calling a candidate optimized, retain:

- source, host/tiling source, design record, and build log;
- shared library/kernel object and SHA-256;
- correctness case list and machine-readable results;
- exact benchmark command, all raw outputs, and profiler CSV/JSON;
- device/CANN/driver/Python/Torch/torch_npu versions and dirty-state summary;
- a concise report with limitations.

If the gate fails, label the result `static-only`, `compile-only`,
`correctness-only`, or `performance-regressed`. Do not create a new promoted
version just to preserve a failed experiment. The v0.2 Huawei result remains
`correctness-only` under this policy.

## Remote synchronization policy

The Huawei checkout is a shared working tree. Synchronize the project skill
and its references by explicit file list. Make a recoverable backup before
overwriting a remote file. Never use a deletion-enabled sync for this project:
inspect a dry run first and preserve remote-only prototypes, build outputs, and
other users' work. A remote copy is not a Git release; report transport,
checkout status, build, correctness, performance, and publication separately.

## Review handoff

Every completion report must show the selected route and source pin, changed
files/artifacts, contract and tolerance, correctness evidence, paired NPU
timings, profiler evidence, and unresolved blockers. Separate RTX and Huawei
results. If CANN, `bisheng`, `torch_npu`, or a physical 910B is unavailable,
stop at the highest honest evidence layer.

## Source caveats

- Ascend C, CATLASS, PTOAS, PTO-ISA, PyPTO, TileLang-Ascend, and Triton-Ascend
  are different abstraction layers; lowering to Ascend C does not make a
  Python DSL C-like source.
- Standalone PTO-DSL is maintenance-only; current work is in PTOAS, which
  requires the LLVM 21 VPTO toolchain.
- High-level Matmul objects can consume internal synchronization flags. If a
  custom CrossCore protocol is also used, verify flag allocation or use a raw
  Mmad path with an explicit protocol.
- Vendored upstream skills describe their own checkout layouts. Adapt paths to
  KernelSwift and let this contract take precedence.

## v0.3 change record

- Added an explicit correctness/performance promotion gate and result labels.
- Made Cube/Mmad the default SparseAttention direction and bounded failed
  vector-only experiments.
- Added stream/synchronization measurement rules and three-process stability
  checks.
- Added a minimal skill-loading policy and non-destructive Huawei sync policy.
- Recorded the Huawei v0.2 incident in
  [v0.3-performance-audit.md](references/v0.3-performance-audit.md).
