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
_LOADED = [False]

def _get_op():
    if not _LOADED[0]:
        # 指向通过 CMake/bash 脚本编译出来的库文件
        _dir = os.path.dirname(os.path.abspath(__file__))
        _so = os.path.join(_dir, "build", "libsinkhorn_ops.so") 
        torch.ops.load_library(_so)
        _LOADED[0] = True
    # 返回刚才在 C++ 里注册的算子
    return torch.ops.sinkhorn_ops.sinkhorn_kernel_basic

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
    