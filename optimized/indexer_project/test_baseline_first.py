"""Regression test: official-harness launch order.

Runs the baseline Model's forward FIRST (like auto_bench's v0 correctness
pass), then the optimized ModelNew, and requires exact INT64 index equality.
Guards the fixed megakernel-tiling bug: previously, when the megakernel's
first launch followed the baseline forward it silently read a zeroed GM
tiling tensor and no-oped (garbage output), and warmup tricks to dodge that
produced mask-broken scores.
"""
import os, sys, torch, torch_npu

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
sys.path.insert(0, os.path.join(_here, "..", "..", "baseline"))
import indexer_ver1 as opt
import indexer as base

torch.npu.set_device(0)


def seed(s):
    torch.manual_seed(s)
    torch.npu.manual_seed_all(s)


seed(42)
v0_init = base.get_init_inputs()
seed(42)
v1_init = opt.get_init_inputs()
model = base.Model(*v0_init).eval().npu()
model_new = opt.ModelNew(*v1_init).eval().npu()
try:
    model_new.load_state_dict(model.state_dict())
except Exception:
    pass
seed(42)
inputs = [t.npu() if isinstance(t, torch.Tensor) else t for t in base.get_inputs()]

with torch.no_grad():
    # baseline first (harness order), then optimized
    out0 = model(*inputs)
    out1 = model_new(*inputs)
    torch.npu.synchronize()

exact = torch.equal(out0, out1)
mismatch = (out0 != out1).sum().item()
print(f"harness-order (baseline first): exact={exact} mismatched={mismatch}")

with torch.no_grad():
    # repeat a few times to catch order/state drift
    ok = True
    for i in range(5):
        ok &= torch.equal(model(*inputs), model_new(*inputs))
    torch.npu.synchronize()
print(f"repeat x5: exact={ok}")

if exact and ok:
    print("PASS")
else:
    print("FAIL")
    sys.exit(1)
