# Standalone correctness test for indexer_ops.topk_mask (exact INT64 equality).
import os
import sys
import torch
import torch_npu

_dir = os.path.dirname(os.path.abspath(__file__))
torch.ops.load_library(os.path.join(_dir, "build", "libindexer_ops.so"))
topk_op = torch.ops.indexer_ops.topk_mask

torch.npu.set_device(0)
N = 672
K = 128


def ref_topk(scores, causal, ratio, actualT, offset, S):
    # mirrors baseline/indexer.py postprocessing on the sliced score tensor
    k = min(K, actualT)
    idx = scores[:, :, :actualT].topk(k, dim=-1)[1]
    if causal:
        qlims = torch.arange(1, S + 1, device=scores.device).unsqueeze(1) // ratio
        mask = idx >= qlims
        idx = torch.where(mask, -1, idx + offset)
    else:
        idx = idx + offset
    return idx


def run_case(name, B, S, ratio, causal, offset, seed, garbage_pad=False):
    torch.manual_seed(seed)
    actualT = S // ratio
    scores = torch.randn(B, S, N, dtype=torch.bfloat16).npu()
    if causal:
        s = torch.arange(S, device=scores.device)
        limit = (s + 1) // ratio
        col = torch.arange(N, device=scores.device)
        m = col.unsqueeze(0) >= limit.unsqueeze(1)
        scores = torch.where(m.unsqueeze(0), float("-inf"), scores)
    if garbage_pad:
        # simulate megakernel output when causal mask does not cover padding
        scores[:, :, actualT:] = torch.randn(
            B, S, N - actualT, dtype=torch.bfloat16).npu().abs() + 3.0
    else:
        scores[:, :, actualT:] = float("-inf")
    scores = scores.contiguous()

    torch.npu.synchronize()  # isolate from async input construction
    got = topk_op(scores, causal, ratio, actualT, offset)
    ref = ref_topk(scores, causal, ratio, actualT, offset, S)
    same = torch.equal(got, ref)
    agree = (got == ref).float().mean().item()
    print(f"{name}: exact={same} agree={agree:.6f}", flush=True)
    if not same:
        bad = (got != ref).nonzero()
        b, s, i = bad[0].tolist()
        print(f"  first mismatch at {(b, s, i)}: got {got[b, s, :16].tolist()}")
        print(f"                          ref {ref[b, s, :16].tolist()}")
        v = (s + 1) // ratio
        print(f"  valid={v} actualT={actualT}")
        print("  row scores around:", scores[b, s, :min(v + 4, N)].float().tolist()[:20])
    return same


ok = True
ok &= run_case("competition causal", 8, 2600, 4, True, 0, 0)
ok &= run_case("competition causal off=7", 8, 2600, 4, True, 7, 1)
ok &= run_case("non-causal garbage pad", 4, 2600, 4, False, 3, 2, garbage_pad=True)
ok &= run_case("tiny S (valid<128)", 8, 64, 4, True, 0, 3)
ok &= run_case("S=512 boundary", 2, 512, 4, True, 1000000, 4)
ok &= run_case("odd rows not mult of 40", 1, 252, 4, True, 0, 5)
ok &= run_case("non-causal exact", 8, 2600, 4, False, 0, 6)

# heavy-tie stress: many duplicate values
torch.manual_seed(7)
B, S, ratio = 4, 2600, 4
scores = torch.randint(0, 8, (B, S, N)).to(torch.bfloat16).npu()  # only 8 distinct values
s = torch.arange(S, device=scores.device)
limit = (s + 1) // ratio
col = torch.arange(N, device=scores.device)
m = col.unsqueeze(0) >= limit.unsqueeze(1)
scores = torch.where(m.unsqueeze(0), float("-inf"), scores).contiguous()
got = topk_op(scores, True, ratio, S // ratio, 5)
ref = ref_topk(scores, True, ratio, S // ratio, 5, S)
same = torch.equal(got, ref)
print(f"heavy ties: exact={same}", flush=True)
ok &= same

print("ALL PASS" if ok else "FAILURES PRESENT", flush=True)
sys.exit(0 if ok else 1)
