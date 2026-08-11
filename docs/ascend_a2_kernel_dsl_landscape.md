# Ascend A2/910B kernel DSL and skill landscape

Research snapshot: 2026-08-07. The reusable project skill is [`kernelswift-a2-kernel-dev`](../.agents/skills/kernelswift-a2-kernel-dev/SKILL.md); its references contain the full route comparison and the A2 validation contract.

## Recommendation

For the KernelSwift competition track, make **Ascend C** the default implementation route. It is the direct C/C++-like language for Ascend AI processors and exposes the memory movement, AIC/AIV split, tiling, pipeline, and synchronization decisions needed for custom SparseAttention, Indexer, and Sinkhorn kernels.

Use **CATLASS** when the hot part is GEMM/grouped GEMM, a matrix epilogue, attention, or quantization that matches its reusable C++ templates. CATLASS is the Ascend analogue of CUTLASS, not a Python DSL.

Use **PTOAS/PTODSL** for CuTeDSL-like, explicit SPMD tile experiments. Standalone PTO-DSL is maintenance-only; current development is in PTOAS, which requires the LLVM 21 VPTO toolchain. Treat it as experimental until it compiles, launches, and beats the direct baseline on the real 910B. **PyPTO** is a separate, higher-level MPMD Python runtime over PTO-ISA; it is useful for framework-integrated operators but is not a C-like authoring language.

Use **TileLang-Ascend** for faster Python tile iteration (tested upstream on A2/A3, with Ascend C/PTO and AscendNPU IR backends). Use **Triton-Ascend** for the repository’s existing Triton workflow or GPU-to-NPU migration; neither is a C-like authoring language.

## Installed skills

The project now vendors the following source-pinned playbooks under `.agents/skills/`:

- Ascend C: project initialization, design, test-case generation, code generation, compile/debug, precision evaluation/debug, performance evaluation/optimization, and code review.
- CATLASS: design, code generation, and performance optimization.
- TileLang-Ascend: operator design, development, and performance optimization.
- CANNBot extensions (pin `c46fa548c71183a62e16b96d598fdabe5199d8f7`): A2 architecture and environment checks, Ascend C API/tiling/pipeline references, direct-invoke templates, NPU profiling and precision standards, complementary CATLASS implementation/tuning skills, the complete PyPTO workflow suite, and TileLang API/programming-model/test/review skills.
- A project router: `kernelswift-a2-kernel-dev`.

The two Ascend upstream skill frontmatters that were not valid YAML were repaired locally by quoting their descriptions; the bodies remain upstream. See the router reference for exact commits, links, and caveats.

Vendored frontmatter was normalized to the portable `name`/`description` schema where upstream used loader-specific metadata.

The corresponding Mulan PSL v2, MIT, and CANN Open Software License v2 texts are retained in `.agents/THIRD_PARTY_LICENSES/`.

## Quick selection table

| Kernel requirement | First choice | Alternate to measure |
|---|---|---|
| Vector/reduction/sort/index/fused C-like kernel | Ascend C | PTOAS or TileLang-Ascend prototype |
| GEMM/grouped GEMM/matrix epilogue | CATLASS | Direct Ascend C |
| Explicit tile research, CuTeDSL-style control | PTOAS/PTODSL | TileLang-Ascend |
| Higher-level Python tensor/tile runtime | PyPTO | PTOAS/PTODSL or TileLang-Ascend |
| Existing Triton code or migration | Triton-Ascend | Ascend C rewrite only if the C-like contract requires it |

No route is accepted on syntax alone: retain the source, compilation artifact, adapter load evidence, correctness matrix, and synchronized paired timing before claiming an A2 speedup.

## Source index

- [Ascend C introduction](https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/opdevg/Ascendcopdevg/atlas_ascendc_10_0001.html)
- [Ascend C A2/A3 architecture](https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/opdevg/Ascendcopdevg/atlas_ascendc_10_0011.html)
- [Ascend Agent Skills](https://github.com/ascend/agent-skills)
- [CATLASS](https://gitcode.com/cann/catlass)
- [PTOAS](https://github.com/hw-native-sys/PTOAS) · [PTO-ISA](https://github.com/hw-native-sys/pto-isa) · [PTO-DSL (maintenance)](https://github.com/huawei-csl/pto-dsl)
- [PyPTO](https://gitcode.com/cann/pypto) · [CANNBot skills](https://gitcode.com/cann/cannbot-skills)
z- [TileLang-Ascend](https://github.com/tile-ai/tilelang-ascend)
- [Triton-Ascend](https://github.com/triton-lang/triton-ascend)
- [NVIDIA skills](https://github.com/NVIDIA/skills) (CuTe/CUTLASS references)
