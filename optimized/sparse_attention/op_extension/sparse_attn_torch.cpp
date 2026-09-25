#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include "acl/acl.h"
#include <torch/extension.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"

#include "sparse_attn_tiling.h"

// Kernel entry points.  Linkage is toolchain-dependent: the huawei bisheng
// build exports C++-mangled stubs (SPARSE_ATTN_KERNEL_CXX_LINKAGE, set by
// run_huawei.sh); the contest server toolchain exports C linkage (default).
#ifdef SPARSE_ATTN_KERNEL_CXX_LINKAGE
void transpose_kv_kernel(uint32_t blockDim, void *l2Ctrl, aclrtStream stream,
                         uint8_t *kv, uint8_t *kvT, uint8_t *tiling);
void fused_sparse_attn_basic_kernel(uint32_t blockDim, void *l2Ctrl, aclrtStream stream,
                                    uint8_t *q, uint8_t *kv, uint8_t *kvT,
                                    uint8_t *sink, uint8_t *topk, uint8_t *scores,
                                    uint8_t *agg, uint8_t *out, uint8_t *tiling);
#else
extern "C" {
    void transpose_kv_kernel(uint32_t blockDim, void *l2Ctrl, aclrtStream stream,
                             uint8_t *kv, uint8_t *kvT, uint8_t *tiling);
    void fused_sparse_attn_basic_kernel(uint32_t blockDim, void *l2Ctrl, aclrtStream stream,
                                        uint8_t *q, uint8_t *kv, uint8_t *kvT,
                                        uint8_t *sink, uint8_t *topk, uint8_t *scores,
                                        uint8_t *agg, uint8_t *out, uint8_t *tiling);
}
#endif

namespace ascend_kernel {


static int32_t getCubeCoreNum()
{
    int32_t deviceId = -1;
    aclrtGetDevice(&deviceId);
    int64_t num = 0;
    aclrtGetDeviceInfo(deviceId, ACL_DEV_ATTR_CUBE_CORE_NUM, &num);
    return num > 0 ? (int32_t)num : 20;
}

static void computeTransposeKvTiling(TransposeKvTiling &t, int64_t B, int64_t N,
                                     int64_t D, double softmaxScale)
{
    t.batchNum = static_cast<int32_t>(B);
    t.n = static_cast<int32_t>(N);
    t.d = static_cast<int32_t>(D);
    t.scale = static_cast<float>(softmaxScale);
}

static void computeFusedSparseAttnBasicTiling(FusedSparseAttnBasicTiling &t,
                                              int64_t B, int64_t M, int64_t H,
                                              int64_t N, int64_t D, int64_t K,
                                              double softmaxScale)
{
    t.totalTiles = static_cast<int32_t>(B * M / 4);
    t.batchNum = static_cast<int32_t>(B);
    t.mPerBatch = static_cast<int32_t>(M);
    t.h = static_cast<int32_t>(H);
    t.n = static_cast<int32_t>(N);
    t.d = static_cast<int32_t>(D);
    t.topk = static_cast<int32_t>(K);
    t.scale = static_cast<float>(softmaxScale);
    const int32_t cubeCores = getCubeCoreNum();
    t.blockNum = (cubeCores < t.totalTiles) ? cubeCores : t.totalTiles;
    if (t.blockNum < 1) t.blockNum = 1;

    t.reserved0 = 0;
}

// The tiling tensor is read by the asynchronously-launched kernel, so a
// per-call local would be freed while the kernel is still queued/running and
// its recycled block can be overwritten by unrelated torch ops (observed as
// deterministic tile corruption / aivec MPU faults under torch-op churn).
// The tiling content depends only on (shape, scale), so cache it by content:
// allocated once, alive for the process, and the per-call H2D memcpy also
// disappears from the timed path.
static at::Tensor makeTilingTensor(const void *data, size_t bytes, const at::Tensor &ref)
{
    static std::mutex tilingMu;
    static std::unordered_map<std::string, at::Tensor> cache;
    std::string key(static_cast<const char *>(data), bytes);
    std::lock_guard<std::mutex> lk(tilingMu);
    auto it = cache.find(key);
    if (it != cache.end()) {
        return it->second;
    }
    at::Tensor t = at::empty({(int64_t)bytes}, ref.options().dtype(at::kByte));
    aclrtMemcpy(t.mutable_data_ptr(), bytes, data, bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    cache.emplace(std::move(key), t);
    return t;
}

at::Tensor sparse_attn_megakernel_basic_torch(const at::Tensor &q,
                                              const at::Tensor &kv,
                                              const at::Tensor &attn_sink,
                                              const at::Tensor &topk_idxs,
                                              double softmax_scale)
{
    TORCH_CHECK(q.scalar_type() == at::kBFloat16, "q must be bfloat16");
    TORCH_CHECK(kv.scalar_type() == at::kBFloat16, "kv must be bfloat16");
    TORCH_CHECK(attn_sink.scalar_type() == at::kFloat, "attn_sink must be float32");
    TORCH_CHECK(topk_idxs.scalar_type() == at::kInt, "topk_idxs must be int32");
    TORCH_CHECK(q.device().type() == c10::DeviceType::PrivateUse1, "q must be on NPU");
    TORCH_CHECK(kv.device().type() == c10::DeviceType::PrivateUse1, "kv must be on NPU");
    TORCH_CHECK(attn_sink.device().type() == c10::DeviceType::PrivateUse1, "attn_sink must be on NPU");
    TORCH_CHECK(topk_idxs.device().type() == c10::DeviceType::PrivateUse1, "topk_idxs must be on NPU");
    TORCH_CHECK(q.is_contiguous() && kv.is_contiguous() &&
                attn_sink.is_contiguous() && topk_idxs.is_contiguous(),
                "inputs must be contiguous");

    TORCH_CHECK(q.dim() == 4, "q must be [b, m, h, d]");
    TORCH_CHECK(kv.dim() == 3, "kv must be [b, n, d]");
    TORCH_CHECK(attn_sink.dim() == 1, "attn_sink must be [h]");
    TORCH_CHECK(topk_idxs.dim() == 3, "topk_idxs must be [b, m, topk]");

    const int64_t B = q.size(0);
    const int64_t M = q.size(1);
    const int64_t H = q.size(2);
    const int64_t D = q.size(3);
    const int64_t N = kv.size(1);
    const int64_t K = topk_idxs.size(2);

    TORCH_CHECK(kv.size(0) == B && kv.size(2) == D, "kv shape mismatch");
    TORCH_CHECK(attn_sink.size(0) == H, "attn_sink shape mismatch");
    TORCH_CHECK(topk_idxs.size(0) == B && topk_idxs.size(1) == M, "topk_idxs shape mismatch");

    at::Tensor kvT = at::empty({B, D, N}, kv.options());
    at::Tensor scores = at::empty({B, M, H, N}, q.options().dtype(at::kFloat));
    at::Tensor agg = at::empty({B, M, H, N}, q.options().dtype(at::kBFloat16));
    at::Tensor out = at::empty({B, M, H, D}, q.options());

    auto aclStream = c10_npu::getCurrentNPUStream().stream(true);

    TransposeKvTiling kvTiling{};
    computeTransposeKvTiling(kvTiling, B, N, D, softmax_scale);
    at::Tensor kvTilingT = makeTilingTensor(&kvTiling, sizeof(kvTiling), kv);
    transpose_kv_kernel((uint32_t)kvTiling.batchNum, nullptr, aclStream,
        reinterpret_cast<uint8_t *>(kv.mutable_data_ptr()),
        reinterpret_cast<uint8_t *>(kvT.mutable_data_ptr()),
        reinterpret_cast<uint8_t *>(kvTilingT.mutable_data_ptr()));

    FusedSparseAttnBasicTiling tiling{};
    computeFusedSparseAttnBasicTiling(tiling, B, M, H, N, D, K, softmax_scale);
    at::Tensor tilingT = makeTilingTensor(&tiling, sizeof(tiling), q);
    fused_sparse_attn_basic_kernel((uint32_t)tiling.blockNum, nullptr, aclStream,
        reinterpret_cast<uint8_t *>(q.mutable_data_ptr()),
        reinterpret_cast<uint8_t *>(kv.mutable_data_ptr()),
        reinterpret_cast<uint8_t *>(kvT.mutable_data_ptr()),
        reinterpret_cast<uint8_t *>(attn_sink.mutable_data_ptr()),
        reinterpret_cast<uint8_t *>(topk_idxs.mutable_data_ptr()),
        reinterpret_cast<uint8_t *>(scores.mutable_data_ptr()),
        reinterpret_cast<uint8_t *>(agg.mutable_data_ptr()),
        reinterpret_cast<uint8_t *>(out.mutable_data_ptr()),
        reinterpret_cast<uint8_t *>(tilingT.mutable_data_ptr()));

    return out;
}

} // namespace ascend_kernel
