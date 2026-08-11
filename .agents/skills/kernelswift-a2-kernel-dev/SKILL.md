---
name: kernelswift-a2-kernel-dev
description: "Route, implement, review, and validate optimized KernelSwift operators for Huawei Atlas A2/Ascend 910B. Use when choosing or using Ascend C, CATLASS, PTOAS/PTODSL, PyPTO, TileLang-Ascend, or Triton-Ascend for SparseAttention, Indexer, Sinkhorn, or another custom kernel; enforce source-faithful semantics, no PyTorch-built-in fallback, and artifact-backed 910B performance claims."
---

# KernelSwift A2 kernel development

Use this skill as the project-specific front door for custom kernels. It selects an authoring route, keeps the operator contract stable, and separates local/static evidence from real Ascend hardware evidence. Read [dsl-landscape.md](references/dsl-landscape.md) when selecting a DSL and [a2-validation-contract.md](references/a2-validation-contract.md) before making a correctness or speed claim.

## Route selection

Choose the narrowest route that preserves the requested kernel boundary.

| Need | Route | What to load next |
|---|---|---|
| C/C++-like source, explicit GM/L1/L0/UB movement, vector/reduction/sort/index/fusion | **Ascend C** (default for the A2 competition track) | `npu-arch` → `ascendc-env-check` → `ascendc-operator-design` + `ascendc-tiling-design` → `ascendc-api-best-practices` → `ascendc-operator-code-gen` → `ascendc-operator-compile-debug` → precision/`ascendc-perf-optimize`/profiling skills |
| GEMM, grouped GEMM, matrix epilogue, attention or quantization assembled from expert templates | **CATLASS** | `catlass-op-design` → `catlass-op-develop` → `catlass-op-perf-tune`; the `catlass-operator-*` skills remain compatible alternatives |
| CuTeDSL-like explicit tile programming and research iteration | **PTOAS/PTODSL (SPMD)** | Use the pinned sources in `dsl-landscape.md`; treat the toolchain as experimental and verify the LLVM/VPTO dependency before investing in an operator |
| Higher-level Python tile/runtime programming over PTO | **PyPTO (MPMD)** | Use the vendored `pypto-*` suite for API exploration, design, implementation, verification, and tuning; it is not a C-like source path |
| Productive Python tile authoring with Ascend C/PTO or AscendNPU IR lowering | **TileLang-Ascend** | `tilelang-op-design` → `tilelang-op-develop` → `tilelang-api-best-practices`/`tilelang-programming-model-guide` → `tilelang-perf-optimization` |
| Existing Python Triton workflow or GPU→NPU migration | **Triton-Ascend** | Use the repository's existing Triton integration and the upstream Triton-Ascend documentation; it is not a C-like language |

Do not select a route solely because a skill name says “Ascend.” Confirm that the generated source, compiler invocation, binary, and adapter actually use the selected backend.

## Workflow

### 1. Freeze the operator contract

Before changing code, record:

- the full mathematical formula and the real upstream/model layer boundary;
- every input/output shape, stride/layout, dtype, accumulation dtype, mask, and state/cache lifetime;
- dynamic-shape and tail behavior, core partitioning, and allowed error (`atol`/`rtol`, normally no looser than the project’s `1e-2` ceiling);
- the current KernelSwift entry point and a reference implementation used only for checking, never as the optimized implementation.

For SparseAttention, Indexer, and Sinkhorn, inspect the corresponding `baseline/`, `comparisons/`, `dlblas/kernels/ascend/`, benchmark, and test files before designing a new boundary. Do not replace a full layer with a helper kernel or silently change layouts. For irregular Indexer work, consult the Ascend C sort/index references and the PyPTO API/tiling feasibility reports before selecting a backend.

### 2. Design for A2/910B

Use the selected design skill and make the following explicit in `design.md` or its equivalent:

1. block/core decomposition and AIC/AIV responsibility;
2. GM → L1/L0 (Cube) or GM → UB (Vector) movement;
3. tile dimensions, alignment/padding, tail masks, and buffer lifetimes;
4. CopyIn → Compute → CopyOut stages, double buffering/software pipeline, and every event/cross-core synchronization;
5. accumulator precision, overflow/NaN behavior, and deterministic handling of ties or sparse indices;
6. launch/registration/adapter contract and the exact `dav-2201`/910B build target.

Prefer a small, fused kernel when launch and intermediate-memory overhead dominate. For matrix-heavy stages, compare a direct Ascend C implementation with CATLASS rather than assuming a template is faster. For PTOAS, PyPTO, or TileLang, keep the generated Ascend C/PTO/IR artifact so the route is auditable.

### 3. Implement without semantic shortcuts

Load the relevant vendored upstream skill files before editing. Keep the implementation independent of PyTorch built-ins and framework composites in the timed path. A Torch/NPU function may be retained as a golden/reference path, but an adapter must load and launch the compiled kernel artifact. Preserve the existing public API unless the contract explicitly changes.

### 4. Validate in layers

Follow [a2-validation-contract.md](references/a2-validation-contract.md):

- run static checks and source inspection locally;
- on a real 910B host, compile and load the artifact, then run shape/dtype/tail/corner cases against the frozen reference;
- collect synchronized, warmup-separated, repeated timings and profiler data for custom and baseline paths in the same environment; use `ops-profiling`/`ascendc-operator-performance-eval` only on a named NPU and retain the raw profiler output;
- report each result with the device, CANN/driver/compiler versions, command, artifact path/hash, per-case correctness, and per-case timing. Never turn a CPU import or syntax check into an NPU pass.

Accept an optimization only when correctness remains within the contract and paired measurements show a repeatable improvement. Preserve the last known-good artifact and roll back a noisy or regressed candidate.

### 5. Review handoff

Before declaring completion, show:

1. the selected route and pinned source;
2. changed source files and generated artifacts;
3. correctness evidence and tolerance;
4. benchmark/profiler evidence, with RTX and Ascend results separated if both exist;
5. unresolved environment or hardware blockers.

If CANN, `bisheng`, `torch_npu`, or a physical 910B is unavailable, stop at the highest honest validation layer and state the missing evidence.

## Important source caveats

- Ascend C, CATLASS, PTOAS, TileLang-Ascend, and Triton-Ascend are different abstraction layers. Do not call a Python DSL “C-like” merely because it lowers to Ascend C.
- PyPTO is a higher-level MPMD runtime/framework that uses PTO-ISA primitives; PTO-DSL/PTODSL is the lower-level SPMD CuTile/CuTeDSL-like route. Keep their skills and generated artifacts separate.
- Standalone PTO-DSL is maintenance-only; current development is in PTOAS. PTOAS requires the LLVM 21 VPTO branch and is not a drop-in replacement for CANN/Ascend C.
- CANNBot's `ascendc-performance-best-practices` contains mixed-generation material, including DAV_3510-only sections. On A2, use only DAV_2201/A2-tagged guidance and confirm every capacity or instruction claim with `npu-arch` and live CANN platform metadata.
- CANNBot's direct-invoke template includes A5/Kirin branches. For this project, use its A2-compatible Vector/direct-launch material only; do not inherit A5 `dav-3510` or Kirin assumptions.
- The vendored Ascend Agent Skills are upstream playbooks. Their `ascend-kernel/csrc/ops` assumptions do not automatically match this repository; this skill’s contract and the current checkout take precedence.
