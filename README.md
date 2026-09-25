# KernelSwift — Ascend C operator optimizations for Atlas A2 (910B3)

Custom C-like **Ascend C** kernels for the 2026 KernelSwift Operator
Innovation Competition (Huawei Ascend A2 / 910B3 track, `dav-2201`,
CANN 8.5.2).  Three operators are implemented as fused Cube+Vector
megakernels behind thin torch extensions — no PyTorch matmul/softmax/gather
runs on the timed path.

| Operator | Speedup vs Torch baseline | Timed path |
|---|---|---|
| [Sinkhorn](optimized/sinkhorn_project/) | **~9.4x** | one pure-AIV fused kernel (all row/column normalization iterations in UB) |
| [SparseAttention](optimized/sparse_attention/) | **~8.8x** | kvT transpose + `__mix__(1,2)` QK+softmax+PV megakernel (raw `DataCopy/Nd2Nz + LoadData + Mmad + Fixpipe`) |
| [Indexer](optimized/indexer_project/) | **~3.9x** | mixed AIC/AIV pipeline: RoPE + score GEMM + 9-tier exact TopK |

All numbers: official harness (`benchmarks/ks/auto_bench.py`), warmup 200 /
repeat 500, median, 8x Ascend 910B3, correctness PASS at `atol=rtol=1e-2`.

## Repository layout

- [`baseline/`](baseline/) — competition-provided Torch reference
  implementations (frozen correctness oracle and speedup baseline).
- [`optimized/`](optimized/) — the submitted Ascend C kernels, torch
  extensions, and `ModelNew` Python wrappers.  Each project has a
  `design.md` describing the kernel architecture.
- [`benchmarks/ks/auto_bench.py`](benchmarks/ks/auto_bench.py) — correctness
  check + baseline-vs-optimized latency measurement.

## Requirements

- Ascend 910B3 with CANN 8.5.2 (`dav-2201` toolchain, `find_package(ASC)`)
- Python 3.10 with PyTorch 2.9.0 and torch_npu 2.9.0
- CMake >= 3.16

## Build

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh   # or your CANN install
cd optimized/<project>
bash run.sh        # -> build/lib*_ops.so
```

`run.sh` uses `python3` by default; set `KERNELSWIFT_PYTHON` if your default
interpreter does not have torch/torch_npu:

```bash
KERNELSWIFT_PYTHON=/path/to/venv/bin/python bash run.sh
```

On a bisheng toolchain host whose kernel stubs are C++-mangled, use
`run_huawei.sh` (passes the `*_KERNEL_CXX_LINKAGE` flag) where available.

## Measure

```bash
python benchmarks/ks/auto_bench.py \
  --v0_file baseline/sparse_attention.py \
  --v1_file optimized/sparse_attention/sparse_attention.py \
  --atol 1e-2 --rtol 1e-2 --warmup 200 --repeat 500
```

## License

[MIT](LICENSE)
