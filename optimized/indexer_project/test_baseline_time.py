import sys, time, torch
sys.path.insert(0, "/home/tinglin/KernelSwift/baseline")
sys.path.insert(0, "/home/tinglin/KernelSwift/optimized/indexer_project")
import indexer as base

torch.npu.set_device(0)
init = base.get_init_inputs()
model = base.Model(*init).eval()
inputs = base.get_inputs()
model = model.npu()
inputs = [i.npu() if isinstance(i, torch.Tensor) else i for i in inputs]
with torch.no_grad():
    for _ in range(5):
        out = model(*inputs)
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        out = model(*inputs)
    torch.npu.synchronize()
    t1 = time.perf_counter()
print(f"baseline: {(t1-t0)/20*1000:.3f} ms")
torch.save(out.cpu(), "/tmp/indexer_ref_out.pt")
