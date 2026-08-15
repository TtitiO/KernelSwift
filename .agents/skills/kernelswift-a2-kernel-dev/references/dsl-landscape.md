# Ascend A2 kernel DSL and skill landscape

Research snapshot: **2026-08-11**. Re-check upstream compatibility before changing the toolchain.

## Bottom line

- **Ascend C** is the direct C/C++-like programming language for custom Ascend kernels and the primary route for the KernelSwift A2/910B competition goal.
- **CATLASS** is the closest analogue to **CUTLASS**: a layered C++ template library built on Ascend C. It is especially relevant to matrix-heavy kernels, not a Python DSL.
- **PTOAS/PTODSL** is the closest current Ascend analogue to **CuTeDSL/cuTile**: explicit, low-level, SPMD tile-oriented Python authoring over PTO. It is promising but still a research/experimental toolchain.
- **PTO-ISA C++** is a separate C++ header-only tile library and virtual ISA. It exposes `pto/pto-inst.hpp`, tile types/layouts/valid regions, tile instructions, and Auto/Manual placement and synchronization modes. It is C-like source, but its abstraction and generated artifacts must still be distinguished from direct Ascend C.
- **PyPTO** is a separate, higher-level MPMD Python tile/runtime stack built on PTO-ISA. It is useful for framework-integrated kernels and has a richer CANNBot skill suite, but it is not a C-like source language and should not be presented as a direct Ascend C replacement.
- **TileLang-Ascend** is a higher-level Python tile DSL with Ascend C/PTO and AscendNPU IR backends. It is a productivity route, not C-like source.
- **Triton-Ascend** is a Python DSL and already matches much of KernelSwift's current implementation style. It is useful for migration and iteration, but does not satisfy a strict C-like-source requirement.

## Comparison

| Route | Authoring model | A2/910B evidence in source | Control level | KernelSwift role |
|---|---|---|---|---|
| Ascend C | C/C++ syntax and AscendC APIs | Official docs list Atlas A2/A3 and use the A2/A3 architecture target | Highest supported direct control of AIC/AIV, GM/L1/L0/UB, pipelines and events | Primary implementation path |
| CATLASS | C++ templates over Ascend C | Current CATLASS supports Atlas A2/A3; current mainline requires CANN 8.5.0 and uses architecture `2201` | High, but organized around reusable matrix/epilogue components | GEMM, grouped GEMM, attention, quantization |
| PTO-ISA C++ | C++ header-only tile library / virtual ISA | Current PTO-ISA advertises Ascend A2/A3/A5 and CPU profiles; A2/A3 profile targets 910B/910C | Explicit tile types, layouts, valid regions, tile placement, synchronization, and pipeline contracts | C-like PTO route; compare against direct Ascend C |
| PTOAS/PTODSL | Python tracing/JIT to PTO/MLIR/LLVM | PTOAS validation includes `Ascend910B1`; PTO-ISA lists A2 (910B), A3, A5 and CPU simulation | Explicit tile/data-movement control with compiler-managed pipeline opportunities | CuTeDSL-like research route; prototype before adoption |
| PyPTO | Higher-level Python tensor/tile runtime (MPMD) | CANNBot materials explicitly cover A2/A3 constraints and 910B/910C execution | More framework/runtime abstraction; PTO-ISA remains the low-level primitive | Framework-integrated alternative; use when its runtime contract fits, not for strict C-like source |
| TileLang-Ascend | Python/TVM tile DSL | Upstream says tested on A2/A3; minimum CANN 8.3.RC1 | Developer and expert modes; explicit L1/L0/UB, Cube/Vector scopes, pipelining and synchronization | Fast experimentation; strong SparseAttention/Indexer examples |
| Triton-Ascend | Python Triton DSL | Current docs support Atlas A2/A3/950; release 3.2.2 recommends CANN 9.1.0 | Lower-level than Torch, less explicit than direct Ascend C/PTO for native hierarchy | Preserve existing KernelSwift route and GPU→NPU migration |

## GPU analogy

| NVIDIA GPU ecosystem | Closest Ascend-side concept | Important difference |
|---|---|---|
| CUDA C++ | Ascend C | Different execution units, scratchpad hierarchy, compiler and synchronization model |
| CUTLASS | CATLASS | CATLASS is Ascend-specific and built over Ascend C; APIs/templates are not source-compatible |
| CuTe C++ layouts/templates | CATLASS tile/layout components and PTO-ISA C++ tiles | No one-to-one layout algebra compatibility |
| CuTeDSL / cuTile | PTOAS/PTODSL | PTOAS uses PTO/MLIR/LLVM and requires the VPTO toolchain; this is the closest low-level analogy |
| PyTorch/JAX-style NPU runtime | PyPTO | Higher-level MPMD execution and tensor APIs; it still lowers through PTO-ISA but is not SPMD CuTeDSL |
| TileLang CUDA backend | TileLang-Ascend | Ascend backend maps GPU-like shared/register concepts to L1/UB and L0, with AIC/AIV concerns |
| Triton | Triton-Ascend | Ascend extensions and memory/core behavior differ from GPU Triton |

## Recommended order for the three target operators

1. **Sinkhorn**: start with direct Ascend C Vector/reduction design. Use FP32 accumulation where required, fuse normalization stages when legal, and measure whether extra passes or exponentials dominate.
2. **Indexer**: start with direct Ascend C for score/reduction/sort/top-k and irregular index handling. Compare TileLang-Ascend's LightningIndexer examples only after confirming the exact formula, layout, and output contract.
3. **SparseAttention**: start with a source-faithful decomposition of sparse index/mask handling, Cube matmuls, Vector softmax, and output accumulation. Compare direct Ascend C, CATLASS components, and TileLang/PTO prototypes because the best abstraction can differ by stage.

Do not admit a route based only on a similarly named example. Compare formula, kernel boundary, input/output layout, state lifecycle, dtype, shapes, masking, and parallel schedule.

## Installed project skills

From [`ascend/agent-skills`](https://github.com/ascend/agent-skills) at commit [`155ac37bd169ddb89479af528297cfb2237400aa`](https://github.com/ascend/agent-skills/commit/155ac37bd169ddb89479af528297cfb2237400aa):

- Ascend C: `ascendc-operator-project-init`, `ascendc-operator-design`, `ascendc-operator-testcase-gen`, `ascendc-operator-code-gen`, `ascendc-operator-compile-debug`, `ascendc-operator-precision-eval`, `ascendc-operator-precision-debug`, `ascendc-operator-performance-eval`, `ascendc-operator-performance-optim`, and `ascendc-operator-code-review`.
- CATLASS: `catlass-operator-design`, `catlass-operator-code-gen`, and `catlass-operator-performance-optim`.

The two upstream YAML descriptions containing unquoted colons (`ascendc-operator-code-gen` and `ascendc-operator-compile-debug`) were quoted locally so Codex can parse them. Three additional portability repairs were made: the generated build template now uses its project root and quotes deletion/build paths, the test-case skill no longer claims a missing UT skill is installed, and the code-review skill treats its missing external style file as optional.

`ascendc-operator-performance-eval` retains upstream's valid `argument-hint` frontmatter extension. The lightweight system `quick_validate.py` does not whitelist that extension, although the Skills CLI parser accepts it; validate its YAML separately rather than deleting upstream metadata.

From [`tile-ai/tilelang-ascend`](https://github.com/tile-ai/tilelang-ascend) at commit [`272c0ab3928df30f84d7ac644456856366ff60c4`](https://github.com/tile-ai/tilelang-ascend/commit/272c0ab3928df30f84d7ac644456856366ff60c4):

- `tilelang-op-design`, `tilelang-op-develop`, and `tilelang-perf-optimization`.

The upstream TileLang skills assume a TileLang-Ascend checkout. In KernelSwift, use them as design/implementation guidance and adapt paths deliberately.

From Huawei's [`cann/cannbot-skills`](https://gitcode.com/cann/cannbot-skills) at commit [`c46fa548c71183a62e16b96d598fdabe5199d8f7`](https://gitcode.com/cann/cannbot-skills/commit/c46fa548c71183a62e16b96d598fdabe5199d8f7):

- Ascend C support: `ascendc-api-best-practices`, `ascendc-tiling-design`, `ascendc-env-check`, `npu-arch`, `ascendc-direct-invoke-template`, `ascendc-performance-best-practices`, `ascendc-perf-optimize`, `ops-profiling`, and `ops-precision-standard`.
- CATLASS support: `catlass-op-design`, `catlass-op-develop`, and `catlass-op-perf-tune` (the latter two complement the existing CATLASS skills and are especially A2-oriented).
- PyPTO support: the complete `pypto-*` suite, including intent/planning, API exploration, design, construction, implementation, verification, precision debugging, knowledge lookup, and performance tuning.
- TileLang support: `tilelang-api-best-practices`, `tilelang-programming-model-guide`, `tilelang-env-check`, `tilelang-op-test-design`, and `tilelang-review`; the project keeps the already-pinned Tile-AI design/develop/performance skills where names overlap.

The CANNBot material is retained under its CANN Open Software License v2 text in `.agents/THIRD_PARTY_LICENSES/cannbot-skills-CANN-OSL-v2.txt`. Its documentation mixes generations: filter `DAV_2201`/A2 guidance for 910B and verify all capacities through the live platform API. The direct-invoke template also contains A5/Kirin branches that are out of scope for this project.

Some deep CANNBot references intentionally name files generated during an operator workflow or paths inside a full CATLASS/PyPTO/CANN checkout. They are not bundled artifacts in KernelSwift; resolve them against the selected toolchain checkout and pin that checkout before implementation.

## Agent-skill availability review

The official [`ascend/agent-skills`](https://github.com/ascend/agent-skills) repository is the relevant public Ascend skill collection. The skills.sh snapshot observed on 2026-08-07 showed roughly 155–166 installs for the core Ascend C skills and passing registry security checks. This is a small ecosystem, so official ownership and source review matter more than install count.

NVIDIA's [`NVIDIA/skills`](https://github.com/NVIDIA/skills) repository contains GPU/CuTe-oriented entries such as `kernel-cute-writing`, but the skills.sh entry did not expose a directly installable `SKILL.md` at the time of review. Treat it as optional reference material, not a KernelSwift dependency.

No maintained, standalone PTOAS-specific public agent skill was found. The project-local router therefore records PTOAS sources and validation gates instead of pretending an upstream skill exists.

## Authoritative sources and pins

- [Ascend C introduction](https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/opdevg/Ascendcopdevg/atlas_ascendc_10_0001.html): describes Ascend C as natively supporting C/C++ standards and compiling for Ascend AI processors.
- [Ascend A2/A3 architecture and synchronization](https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/opdevg/Ascendcopdevg/atlas_ascendc_10_0011.html): AIC/AIV, memory hierarchy, pipelines, and synchronization.
- [CATLASS](https://gitcode.com/cann/catlass), inspected at `0dfcf9df304f297361edab86448ac5ea6aed1647`.
- [PTO-DSL](https://github.com/huawei-csl/pto-dsl), commit [`b10afbea191dcce6f718d1f1240d5fdc4fca990a`](https://github.com/huawei-csl/pto-dsl/commit/b10afbea191dcce6f718d1f1240d5fdc4fca990a). Its README says future development moved into PTOAS and the standalone repository is maintenance-only.
- [PTOAS](https://github.com/hw-native-sys/PTOAS), commit [`988d50e245217669a27448c96641bb7eaf26baed`](https://github.com/hw-native-sys/PTOAS/commit/988d50e245217669a27448c96641bb7eaf26baed). It requires the LLVM 21 VPTO branch and includes PTODSL/Python bindings and A2/A3 validation generation for `Ascend910B1`.
- [PTO-ISA](https://github.com/hw-native-sys/pto-isa), inspected at commit [`40e741bf1cfce99da3b1caa514e08c2f72894922`](https://github.com/hw-native-sys/pto-isa/commit/40e741bf1cfce99da3b1caa514e08c2f72894922). The current source documents 124 tile interfaces, C++ header-only expansion through `pto/pto-inst.hpp`, CPU simulation, and A2/A3/A5 support; its upstream A2/A3 performance tables remain unverified KernelSwift claims.
- [PyPTO](https://gitcode.com/cann/pypto), inspected at commit [`fad83293ecb575dbe626050467e17cfabdb87278`](https://gitcode.com/cann/pypto/commit/fad83293ecb575dbe626050467e17cfabdb87278): higher-level Python MPMD framework in the PTO ecosystem; the current source declares package version `9.1.0` and documents CANN-version matching. Use the vendored CANNBot PyPTO skills with the installed PyPTO devkit and live A2 validation.
- [TileLang-Ascend](https://github.com/tile-ai/tilelang-ascend), commit [`272c0ab3928df30f84d7ac644456856366ff60c4`](https://github.com/tile-ai/tilelang-ascend/commit/272c0ab3928df30f84d7ac644456856366ff60c4).
- [Triton-Ascend](https://github.com/triton-lang/triton-ascend), commit [`77023a376129f7adb0d912de9a46697e48e5c290`](https://github.com/triton-lang/triton-ascend/commit/77023a376129f7adb0d912de9a46697e48e5c290).

Repository commits were live-checked on 2026-08-07. Hardware performance claims remain upstream claims until reproduced on the project's A2/910B environment.
