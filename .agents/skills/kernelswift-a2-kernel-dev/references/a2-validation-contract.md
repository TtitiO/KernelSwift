# KernelSwift A2/910B validation contract

Use this checklist for every custom kernel, regardless of authoring DSL.

## Evidence layers

Report the highest completed layer and do not collapse the layers into a single “pass.”

| Layer | Minimum evidence | What it does not prove |
|---|---|---|
| Static/source | Selected DSL source exists; registration and adapter paths are inspectable; no timed-path Torch fallback | Compilation, device execution, correctness, or performance |
| Compile | Exact command, CANN/compiler versions, A2 target, successful exit, and produced binary/WHL/kernel object | That the adapter loads it or that results are correct |
| Load/launch | Runtime log or trace shows the adapter loading and launching the compiled artifact on a named 910B device | Numerical correctness across the contract or speedup |
| Correctness | Frozen reference, all required shape/dtype/tail/corner cases, per-case error metrics, and tolerance | Performance |
| Performance | Same-environment paired measurements, synchronization, warmup, repetitions, dispersion, and profiler evidence | Generalization beyond tested cases |

## Environment record

Capture at least:

- Atlas product/device (`npu-smi info`), 910B variant, driver and firmware;
- CANN, `bisheng`, Python, PyTorch, `torch_npu`, and DSL/library versions;
- repository commit and dirty-state summary;
- exact build and run commands;
- source and binary hashes or stable artifact paths.

For the vendored CANNBot helpers, `ascendc-env-check`/`npu-arch` are discovery aids, not proof of a successful kernel run. Record the resolved `SocVersion`/`NpuArch` (A2 should resolve to `ASCEND910B`/`DAV_2201`) and query runtime buffer capacities instead of copying a generic A3/A5 value. `ops-profiling` is usable only after a real NPU profiler collection; preserve its raw CSV/JSON output alongside the summary.

For direct Ascend C/CATLASS on A2/A3, verify that the build targets architecture `2201`/the matching 910B SoC rather than relying on a default.

## Correctness gate

1. Run all competition shapes and dtypes.
2. Add non-aligned tails, minimum/maximum extents, empty/sparse rows where legal, extreme values, ties, masks, and state/cache transitions.
3. Compare in an appropriate accumulation dtype and report max absolute error, max relative error, and mismatch location/count.
4. Use the project ceiling of `1e-2` only when the operator contract permits it. Prefer a tighter threshold when stable.
5. Reject NaN/Inf changes unless the frozen semantics explicitly permit them.
6. Re-run the full correctness matrix after each performance edit.

The Torch implementation is a golden path, not an implementation shortcut. Inspect the optimized adapter and compiled artifact path to rule out `torch`, `torch_npu`, ATen, `torch.ops`, or saved-framework-call fallbacks in the timed kernel.

For PyPTO/PTOAS/TileLang routes, retain both the source-level program and the lowered PTO/IR/binary artifact. CPU simulation or an API feasibility report is static/functional evidence only; it does not establish 910B launch or performance.

## Performance gate

1. Synchronize before and after timed regions.
2. Separate compilation/JIT, allocation, data transfer, and initialization from steady-state kernel latency unless the competition includes them.
3. Warm up both candidate and baseline, alternate/ pair their trials, and use enough repetitions to expose noise.
4. Report median plus dispersion and raw per-case values; do not infer speedup from unrelated runs.
5. Compare identical shapes, dtypes, layouts, streams, and state.
6. Use profiler data to classify Vector/Cube/MTE/synchronization/launch bottlenecks before changing tiling.
7. Accept only repeatable improvement with correctness unchanged. Keep the previous artifact until the candidate passes.

## Artifact gate

For a completed A2 result, retain:

- kernel source and host/tiling source;
- design/tiling record;
- compiler/build log and compiled object or package;
- adapter/registration source proving the artifact is loaded;
- correctness case list and machine-readable results;
- benchmark/profiler command and raw output;
- concise Markdown summary with failures and limitations.

If the current machine lacks CANN or a physical 910B, label the result `static-only` or `compile-only` as appropriate and list the exact remote validation still required.
