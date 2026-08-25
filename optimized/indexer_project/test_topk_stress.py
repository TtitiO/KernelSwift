"""Stress: hundreds of bare consecutive topk_mask launches (no framework ops
between them) — regression for the vector-core 507035 exception seen with the
old GM-tiling topk kernel. Also verifies output equality across all launches.
"""
import os, sys, torch, torch_npu

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
torch.ops.load_library(os.path.join(_here, "build", "libindexer_ops.so"))
topk_op = torch.ops.indexer_ops.topk_mask

torch.npu.set_device(0)
torch.manual_seed(0)
B, S, N, T = 8, 2600, 672, 650
rs = torch.randn(B, S, N, dtype=torch.bfloat16, device="npu")
# causal masking like the megakernel produces
s = torch.arange(S, device="npu")
lim = (s + 1) // 4
col = torch.arange(N, device="npu")
mask = col.unsqueeze(0) >= lim.unsqueeze(1)
rs = torch.where(mask.unsqueeze(0), float("-inf"), rs)

ref = topk_op(rs, True, 4, T, 0)
torch.npu.synchronize()

N_ITER = 500
ok = True
for i in range(N_ITER):
    out = topk_op(rs, True, 4, T, 0)
    if i % 100 == 99:
        torch.npu.synchronize()
        ok &= torch.equal(out, ref)
torch.npu.synchronize()
ok &= torch.equal(out, ref)
print(f"bare-loop x{N_ITER}: no exception, outputs equal={ok}")
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
