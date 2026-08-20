#include <torch/extension.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"

// 声明底层 Ascend C 算子（huawei bisheng 工具链导出 C++ mangled stub，不要用 extern "C"）
void sinkhorn_kernel(uint32_t blockDim, void *l2Ctrl, aclrtStream stream,
                     uint8_t *x, uint8_t *out, uint32_t total_matrices, uint32_t repeat, float eps);

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

    // 2. 输出显存（out-of-place，避免 clone 的 D2D 拷贝开销）
    at::Tensor out = at::empty_like(x);

    // 4. 计算调度的核数 (BlockDim)
    uint32_t blockDim = 40; 
    if (total_matrices < blockDim) {
        blockDim = total_matrices;
    }
    // 向量化 kernel 单核 UB 按 32 个矩阵（512 floats）分配；AIV 子块数为 2
    TORCH_CHECK((total_matrices + blockDim * 2 - 1) / (blockDim * 2) <= 32,
                "too many matrices per core for optimized sinkhorn kernel");

    // 5. 获取当前 NPU 硬件流并 Launch 算子
    auto aclStream = c10_npu::getCurrentNPUStream().stream(true);
    sinkhorn_kernel(blockDim, nullptr, aclStream, 
                    reinterpret_cast<uint8_t*>(const_cast<void*>(x.const_data_ptr())),
                    reinterpret_cast<uint8_t*>(out.mutable_data_ptr()), 
                    total_matrices, (uint32_t)repeat, (float)eps);

    return out;
}

// 6. 将 C++ 函数注册绑定到 PyTorch
TORCH_LIBRARY(sinkhorn_ops, m) {
    m.def("sinkhorn_kernel_basic", &sinkhorn_torch);
}

// 7. pybind11 直调入口：绕过 torch.ops 调度器，降低每次调用的固定开销
PYBIND11_MODULE(sinkhorn_ext, m) {
    m.def("sinkhorn", &sinkhorn_torch, "sinkhorn npu (direct)");
}
