#include <torch/extension.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"

// 声明底层 Ascend C 算子
extern "C" void sinkhorn_kernel(uint32_t blockDim, void *l2Ctrl, aclrtStream stream,
                                uint8_t *x, uint32_t total_matrices, uint32_t repeat, float eps);

at::Tensor sinkhorn_torch(const at::Tensor &x, int64_t repeat, double eps) {
    // 1. 基本校验
    TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");
    
    // 换成底层枚举判断 NPU Device
    TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1, "x must be on NPU");
    
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(x.dim() == 4, "x must be [n0, n1, mhc, mhc]");

    // 2. 获取维度信息
    uint32_t n0 = x.size(0);
    uint32_t n1 = x.size(1);
    uint32_t mhc = x.size(2);
    TORCH_CHECK(mhc == 4 && x.size(3) == 4, "Optimized kernel only supports 4x4 inner matrices");
    
    uint32_t total_matrices = n0 * n1;

    // 3. 核心策略：因为底层是 In-place 原地计算，深拷贝一份 x 作为最终输出
    at::Tensor out = x.clone();

    // 4. 计算调度的核数 (BlockDim)
    uint32_t blockDim = 40; 
    if (total_matrices < blockDim) {
        blockDim = total_matrices;
    }

    // 5. 获取当前 NPU 硬件流并 Launch 算子
    auto aclStream = c10_npu::getCurrentNPUStream().stream(true);
    sinkhorn_kernel(blockDim, nullptr, aclStream, 
                    reinterpret_cast<uint8_t*>(out.mutable_data_ptr()), 
                    total_matrices, (uint32_t)repeat, (float)eps);

    return out;
}

// 6. 将 C++ 函数注册绑定到 PyTorch
TORCH_LIBRARY(sinkhorn_ops, m) {
    m.def("sinkhorn_kernel_basic", &sinkhorn_torch);
}
