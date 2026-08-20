"""
Sinkhorn — Ascend C (Pure AIV) Optimization
Fully fused temporal loop implementation. 0 HBM overhead during iterations.
"""
import os
import torch
import torch.nn as nn
import torch_npu  # 必须导入，注册 NPU 后端

# ---------------------------------------------------------------------------
# 动态加载编译好的 .so 动态链接库
# ---------------------------------------------------------------------------
_OP = None

def _get_op():
    global _OP
    if _OP is None:
        _dir = os.path.dirname(os.path.abspath(__file__))
        _so = os.path.join(_dir, "build", "libsinkhorn_ops.so")
        # pybind11 直调入口（与 torch.ops 注册的是同一个自定义 kernel，开销更小）
        import importlib.util
        spec = importlib.util.spec_from_file_location("sinkhorn_ext", _so)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _OP = mod.sinkhorn
    return _OP

# ---------------------------------------------------------------------------
# 赛题要求的 Model 定义 (API 不能变)
# ---------------------------------------------------------------------------
class Model(nn.Module):
    """
    Highly optimized Ascend C implementation of sinkhorn_normalize.
    """
    def __init__(self, repeat: int = 10, eps: float = 1e-6):
        super().__init__()
        self.repeat = repeat
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [..., mhc, mhc] float32
        Returns:
            [..., mhc, mhc] float32  (doubly stochastic)
        """
        # 拦截调用，导向我们无敌的 NPU 底层算子
        return _get_op()(x, self.repeat, self.eps)

# auto_bench.py 要求 v1 文件导出 ModelNew（AST 过滤会丢弃别名赋值，必须用真实类定义）
class ModelNew(Model):
    pass

# ---------------------------------------------------------------------------
# 测试样例与初始化 (必须保留原样)
# ---------------------------------------------------------------------------
n0 = 1
n1 = 1024
mhc = 4

def get_inputs():
    x = torch.randn(n0, n1, mhc, mhc, dtype=torch.float32)
    return [x]

def get_init_inputs():
    return []

if __name__ == "__main__":
    # 本地跑个简单的测试
    model = Model().npu()
    inputs = [t.npu() for t in get_inputs()]
    out = model(*inputs)
    print("Forward passed! Output shape:", out.shape)
    