import os, sys, time, torch, torch_npu
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
import indexer_ver1 as opt

torch.npu.set_device(0)
init = opt.get_init_inputs()
model = opt.ModelNew(*init).eval()
model = model.npu()
inputs = opt.get_inputs()

def sync(): torch.npu.synchronize()

with torch.no_grad():
    for _ in range(5):
        out = model(*inputs)
    sync()
    t0 = time.perf_counter()
    for _ in range(20): out = model(*inputs)
    sync(); t1 = time.perf_counter()
    print(f"full (batched): {(t1-t0)/20*1000:.3f} ms")
    # per-call synced (harness style)
    ts = []
    for _ in range(20):
        a = time.perf_counter(); out = model(*inputs); sync()
        ts.append((time.perf_counter()-a)*1000)
    ts.sort()
    print(f"full (sync/call, median): {ts[10]:.3f} ms")

    x, qr, start_pos, offset = inputs
    fc = model.freqs_cis[0:2600]
    for _ in range(3):
        q = model.wq_b(qr).unflatten(-1, (16, 64))
        opt._get_rope_op()(q, fc)
        weights = model.weights_proj(x) * (model.softmax_scale * 16 ** -0.5)
    sync()
    def timeit(name, fn, n=50):
        for _ in range(3): fn()
        sync()
        a = time.perf_counter()
        for _ in range(n): fn()
        sync(); b = time.perf_counter()
        print(f"{name}: {(b-a)/n*1000:.3f} ms")
    timeit("wq_b", lambda: model.wq_b(qr))
    timeit("rope", lambda: opt._get_rope_op()(q, fc))
    timeit("weights+scale", lambda: model.weights_proj(x) * 0.0442)
    timeit("kvT copy", lambda: (
        model._kvT_pad[:8, :, 650:].zero_(),
        model._kvT_pad[:8, :, :650].copy_(model.kv_cache[:8, :650].transpose(1,2))))
    op = torch.ops.indexer_ops.fused_qk_reduce
    timeit("megakernel", lambda: op(q, model._kvT_pad[:8], weights, True, 4))
    rs = op(q, model._kvT_pad[:8], weights, True, 4)
    timeit("custom topk+mask", lambda: opt._get_topk_op()(rs, True, 4, 650, 0))
