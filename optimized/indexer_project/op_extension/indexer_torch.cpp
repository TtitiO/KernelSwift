#include <torch/extension.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"

extern "C" {
    // 声明刚才写的 __mix__(1,2) 算子
    void fused_indexer_qk_reduce(uint32_t blockDim, void *l2Ctrl, aclrtStream stream,
                                 uint8_t *q, uint8_t *kvT, uint8_t *weights,
                                 uint8_t *out, uint8_t *workspace, uint32_t total_tiles);
}

at::Tensor indexer_qk_reduce_torch(const at::Tensor &q, const at::Tensor &kvT, const at::Tensor &weights) {
    // 强制检查类型和连续性
    TORCH_CHECK(q.scalar_type() == at::kBFloat16, "q must be bf16");
    TORCH_CHECK(kvT.scalar_type() == at::kBFloat16, "kvT must be bf16");
    TORCH_CHECK(weights.scalar_type() == at::kBFloat16, "weights must be bf16");
    TORCH_CHECK(q.device().type() == c10::DeviceType::PrivateUse1, "must be on NPU");

    int64_t B = q.size(0);
    int64_t S = q.size(1);
    int64_t H = q.size(2);
    int64_t T_pad = kvT.size(2); // 已经被 Python pad 到了 704
    
    // 总计需要计算的 [64, 64] 矩阵块数量: (8 * 2600 * 16) / 64 = 5200
    uint32_t total_tiles = (B * S * H) / 64; 

    // 输出的归约结果是 float32
    at::Tensor out = at::empty({B, S, T_pad}, q.options().dtype(at::kFloat));
    
    // 【核心分配】: Workspace 内存 (用于 L2 Cache 驻留)
    // 20 个核 * 2(DoubleBuffer) * 64 行 * 704 列 * 4 Bytes(FP32) = 7,208,960 Bytes (~7.2MB)
    int64_t ws_size = 20 * 2 * 64 * 704 * 4;
    at::Tensor workspace = at::empty({ws_size}, q.options().dtype(at::kByte));
    
    // 启动 20 个 Cube 核心 (硬件会自动带起 40 个 Vector 核心)
    auto aclStream = c10_npu::getCurrentNPUStream().stream(true);
    fused_indexer_qk_reduce(20, nullptr, aclStream, 
                            reinterpret_cast<uint8_t*>(q.contiguous().mutable_data_ptr()),
                            reinterpret_cast<uint8_t*>(kvT.contiguous().mutable_data_ptr()),
                            reinterpret_cast<uint8_t*>(weights.contiguous().mutable_data_ptr()),
                            reinterpret_cast<uint8_t*>(out.mutable_data_ptr()),
                            reinterpret_cast<uint8_t*>(workspace.mutable_data_ptr()),
                            total_tiles);
    return out;
}

TORCH_LIBRARY(indexer_ops, m) {
    m.def("fused_qk_reduce", &indexer_qk_reduce_torch);
}
