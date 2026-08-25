import os, sys, torch, torch_npu

_dir = os.path.dirname(os.path.abspath(__file__))
torch.ops.load_library(os.path.join(_dir, "build", "libindexer_ops.so"))
op = torch.ops.indexer_ops.fused_qk_reduce

import argparse
p = argparse.ArgumentParser()
p.add_argument("--ones_w", action="store_true")
p.add_argument("--no_causal", action="store_true")
cli = p.parse_args()

torch.npu.set_device(0)
B, S, H, D, N = 8, 2600, 16, 64, 672
T = 650
torch.manual_seed(0)
q = torch.randn(B, S, H, D, dtype=torch.bfloat16).npu()
kv = torch.randn(B, T, D, dtype=torch.bfloat16).npu()
kvT = torch.zeros(B, D, N, dtype=torch.bfloat16).npu()
kvT[:, :, :T] = kv.transpose(1, 2)
if cli.ones_w:
    w = torch.ones(B, S, H, dtype=torch.bfloat16).npu()
else:
    w = torch.randn(B, S, H, dtype=torch.bfloat16).npu()

print("launching op...", flush=True)
out = op(q, kvT, w, not cli.no_causal, 4)
print("launched, syncing...", flush=True)
torch.npu.synchronize()
print("done, out:", out.shape, out.dtype, flush=True)

# reference (mirrors baseline forward math, start_pos == 0)
score = torch.einsum("bshd,btd->bsht", q, kv)
ref = (score.relu_() * w.unsqueeze(-1)).sum(dim=2)
s = torch.arange(S).npu()
limit = (s + 1) // 4
col = torch.arange(T).npu()
mask = col.unsqueeze(0) >= limit.unsqueeze(1)
if not cli.no_causal:
    ref = torch.where(mask.unsqueeze(0), float("-inf"), ref)

got = out[:, :, :T].float()
reff = ref.float()
both_inf = torch.isinf(reff) & torch.isinf(got)
diff = torch.where(both_inf, torch.zeros_like(got), (got - reff).abs())
inf_mismatch = (torch.isinf(reff) != torch.isinf(got)).sum().item()
print("inf mismatch count:", inf_mismatch, flush=True)
print("max abs diff:", diff.max().item(), flush=True)
bad = (diff > 0.02) & ~both_inf
print("bad count:", bad.sum().item(), flush=True)
if bad.any():
    idx = bad.nonzero()[0]
    b_, s_, t_ = idx.tolist()
    print("sample bad at", idx.tolist(), "got", got[b_, s_, t_].item(), "ref", reff[b_, s_, t_].item(), flush=True)

# topk index agreement (the actual harness metric: exact int equality)
K = 128
ref_idx = ref.topk(K, dim=-1)[1]
got_idx = out[:, :, :T].topk(K, dim=-1)[1]
agree = (ref_idx == got_idx).float().mean().item()
print("topk index agreement:", agree, flush=True)
