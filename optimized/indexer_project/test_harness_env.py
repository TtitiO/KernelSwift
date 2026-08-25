import os, sys, time, torch, torch_npu
sys.path.insert(0, "/home/tinglin/KernelSwift/baseline")
sys.path.insert(0, "/home/tinglin/KernelSwift/optimized/indexer_project")
import indexer as base
import indexer_ver1 as opt

torch.npu.set_device(0)
def sync(): torch.npu.synchronize()

# harness order: v0 model built and run first
b_init = base.get_init_inputs()
b_model = base.Model(*b_init).eval()
b_inputs = base.get_inputs()
b_model = b_model.npu()
b_inputs = [i.npu() if isinstance(i, torch.Tensor) else i for i in b_inputs]
with torch.no_grad():
    for _ in range(5):
        b_model(*b_inputs)
sync()

o_init = opt.get_init_inputs()
o_model = opt.ModelNew(*o_init).eval().npu()
o_inputs = opt.get_inputs()
with torch.no_grad():
    for _ in range(5):
        o_model(*o_inputs)
    sync()
    ts = []
    for _ in range(30):
        a = time.perf_counter(); o_model(*o_inputs); sync()
        ts.append((time.perf_counter()-a)*1000)
    ts.sort()
    print(f"v1 after v0 (median): {ts[15]:.3f} ms  min: {ts[0]:.3f}")
    # free baseline memory and re-measure
    del b_model, b_inputs
    torch.npu.empty_cache()
    for _ in range(3): o_model(*o_inputs)
    sync()
    ts = []
    for _ in range(30):
        a = time.perf_counter(); o_model(*o_inputs); sync()
        ts.append((time.perf_counter()-a)*1000)
    ts.sort()
    print(f"v1 after freeing v0 (median): {ts[15]:.3f} ms  min: {ts[0]:.3f}")
