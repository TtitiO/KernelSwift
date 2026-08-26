#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include "acl/acl.h"
#include <torch/extension.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"

#include "indexer_tiling.h"
#include "indexer_topk_tiling_consts.h"
#include "adv_api/tiling_api.h"
#include "adv_api/kernel_tiling.h"
#include "utils/tiling/platform/platform_ascendc.h"

// Kernel entry point.  Linkage is toolchain-dependent: the huawei bisheng
// build exports a C++-mangled stub (INDEXER_KERNEL_CXX_LINKAGE, set by
// run.sh); the contest server toolchain exports C linkage (default).
#ifdef INDEXER_KERNEL_CXX_LINKAGE
// NOTE: this branch must NOT be extern "C" — the bisheng-built .asc stubs
// are C++-mangled (verified: `nm -D` shows _Z23fused_indexer_qk_reduce...);
// extern "C" declarations here leave the references undefined and the .so
// fails to dlopen (upstream ecb19c2a regressed this; reverted locally).
extern "C" {
void fused_indexer_qk_reduce(uint32_t blockDim, void *l2Ctrl, aclrtStream stream,
                             uint8_t *q, uint8_t *kvT, uint8_t *weights,
                             uint8_t *out, uint8_t *scores, int32_t totalTiles,
                             int32_t tilesPerCore, int32_t mPerBatch, int32_t h,
                             int32_t n, int32_t d, int32_t causal, int32_t ratio,
                             int32_t blockNum, int32_t nChunks,
                             int32_t flagBatch, int32_t numSlots);
void indexer_rope_kernel(uint32_t blockDim, void *l2Ctrl, aclrtStream stream,
                         uint8_t *q, uint8_t *freqs, int32_t b, int32_t s,
                         int32_t h, int32_t d, int32_t pairs);
void indexer_topk_kernel(uint32_t blockDim, void *l2Ctrl, aclrtStream stream,
                         uint8_t *scores, uint8_t *out, int32_t totalRows,
                         int32_t rowsPerCore, int32_t seqlen, int32_t causal,
                         int32_t ratio, int32_t actualT, int64_t offset,
                         int32_t tt0, int32_t tt1, int32_t tt2, int32_t tt3, int32_t tt4, int32_t tt5, int32_t tt6, int32_t tt7, int32_t tt8, int32_t tt9, int32_t tt10, int32_t tt11, int32_t tt12, int32_t tt13, int32_t tt14, int32_t tt15, int32_t tt16, int32_t tt17, int32_t tt18, int32_t tt19, int32_t tt20, int32_t tt21, int32_t tt22, int32_t tt23, int32_t tt24, int32_t tt25, int32_t tt26, int32_t tt27, int32_t tt28, int32_t tt29, int32_t tt30, int32_t tt31, int32_t tt32, int32_t tt33, int32_t tt34, int32_t tt35, int32_t tt36, int32_t tt37, int32_t tt38, int32_t tt39, int32_t tt40, int32_t tt41, int32_t tt42, int32_t tt43, int32_t tt44, int32_t tt45, int32_t tt46, int32_t tt47, int32_t tt48, int32_t tt49, int32_t tt50, int32_t tt51, int32_t tt52, int32_t tt53, int32_t tt54, int32_t tt55);
}
 #else
extern "C" {
    void fused_indexer_qk_reduce(uint32_t blockDim, void *l2Ctrl, aclrtStream stream,
                                 uint8_t *q, uint8_t *kvT, uint8_t *weights,
                                 uint8_t *out, uint8_t *scores,
                                 int32_t totalTiles, int32_t tilesPerCore,
                                 int32_t mPerBatch, int32_t h, int32_t n,
                                 int32_t d, int32_t causal, int32_t ratio,
                                 int32_t blockNum, int32_t nChunks,
                                 int32_t flagBatch, int32_t numSlots);
    void indexer_rope_kernel(uint32_t blockDim, void *l2Ctrl, aclrtStream stream,
                             uint8_t *q, uint8_t *freqs, int32_t b, int32_t s,
                             int32_t h, int32_t d, int32_t pairs);
    void indexer_topk_kernel(uint32_t blockDim, void *l2Ctrl, aclrtStream stream,
                             uint8_t *scores, uint8_t *out, int32_t totalRows,
                             int32_t rowsPerCore, int32_t seqlen,
                             int32_t causal, int32_t ratio, int32_t actualT,
                             int64_t offset, int32_t tt0, int32_t tt1, int32_t tt2, int32_t tt3, int32_t tt4, int32_t tt5, int32_t tt6, int32_t tt7, int32_t tt8, int32_t tt9, int32_t tt10, int32_t tt11, int32_t tt12, int32_t tt13, int32_t tt14, int32_t tt15, int32_t tt16, int32_t tt17, int32_t tt18, int32_t tt19, int32_t tt20, int32_t tt21, int32_t tt22, int32_t tt23, int32_t tt24, int32_t tt25, int32_t tt26, int32_t tt27, int32_t tt28, int32_t tt29, int32_t tt30, int32_t tt31, int32_t tt32, int32_t tt33, int32_t tt34, int32_t tt35, int32_t tt36, int32_t tt37, int32_t tt38, int32_t tt39, int32_t tt40, int32_t tt41, int32_t tt42, int32_t tt43, int32_t tt44, int32_t tt45, int32_t tt46, int32_t tt47, int32_t tt48, int32_t tt49, int32_t tt50, int32_t tt51, int32_t tt52, int32_t tt53, int32_t tt54, int32_t tt55);
}
#endif

namespace ascend_kernel {

static constexpr int32_t TOKENS_PER_TILE = 8;
static constexpr int32_t HEADS = 16;
static constexpr int32_t TILE_M = TOKENS_PER_TILE * HEADS;  // 64
static constexpr int32_t N_CHUNK = 96;

static int32_t getCubeCoreNum()
{
    // aclrtGetDevice + aclrtGetDeviceInfo cost ~10-20us per call; the core
    // count is fixed for the process, so query it once.
    static const int32_t cached = [] {
        int32_t deviceId = -1;
        aclrtGetDevice(&deviceId);
        int64_t num = 0;
        aclrtGetDeviceInfo(deviceId, ACL_DEV_ATTR_CUBE_CORE_NUM, &num);
        return num > 0 ? (int32_t)num : 20;
    }();
    return cached;
}

// Per-core ring of score-slot buffers (bf16 [4*H, N] tiles): the raw-Mmad
// kernel needs no zeroed KFC mailbox, so at::empty is enough and the buffer
// is cached across calls (allocated once, alive for the process).
static at::Tensor makeScoresBuffer(int64_t bytes, const at::Tensor &ref)
{
    static std::mutex wsMu;
    static std::unordered_map<int64_t, at::Tensor> cache;
    std::lock_guard<std::mutex> lk(wsMu);
    auto it = cache.find(bytes);
    if (it != cache.end()) {
        return it->second;
    }
    at::Tensor t = at::empty({bytes}, ref.options().dtype(at::kByte));
    cache.emplace(bytes, t);
    return t;
}

at::Tensor indexer_qk_reduce_torch(const at::Tensor &q, const at::Tensor &kvT,
                                   const at::Tensor &weights, bool causal,
                                   int64_t ratio)
{
    TORCH_CHECK(q.scalar_type() == at::kBFloat16, "q must be bfloat16");
    TORCH_CHECK(kvT.scalar_type() == at::kBFloat16, "kvT must be bfloat16");
    TORCH_CHECK(weights.scalar_type() == at::kBFloat16, "weights must be bfloat16");
    TORCH_CHECK(q.device().type() == c10::DeviceType::PrivateUse1, "q must be on NPU");
    TORCH_CHECK(q.is_contiguous() && kvT.is_contiguous() && weights.is_contiguous(),
                "inputs must be contiguous");

    TORCH_CHECK(q.dim() == 4, "q must be [b, s, h, d]");
    TORCH_CHECK(kvT.dim() == 3, "kvT must be [b, d, n] (n padded to a 16 multiple)");
    TORCH_CHECK(weights.dim() == 3, "weights must be [b, s, h]");

    const int64_t B = q.size(0);
    const int64_t S = q.size(1);
    const int64_t H = q.size(2);
    const int64_t D = q.size(3);
    const int64_t N = kvT.size(2);

    TORCH_CHECK(kvT.size(0) == B && kvT.size(1) == D, "kvT shape mismatch");
    TORCH_CHECK(weights.size(0) == B && weights.size(1) == S && weights.size(2) == H,
                "weights shape mismatch");
    TORCH_CHECK(N % 16 == 0, "kvT n dimension must be padded to a 16 multiple");
    TORCH_CHECK(D % 16 == 0, "head dim must be a 16 multiple");
    TORCH_CHECK(S % TOKENS_PER_TILE == 0, "S must be divisible by 8 tokens per tile");
    TORCH_CHECK(H == HEADS, "kernel is specialized for H=16 index heads");

    at::Tensor out = at::empty({B, S, N}, q.options());

    IndexerTiling tiling{};
    tiling.totalTiles = static_cast<int32_t>(B * S / TOKENS_PER_TILE);
    tiling.blockNum = getCubeCoreNum();
    tiling.tilesPerCore = static_cast<int32_t>(
        (tiling.totalTiles + tiling.blockNum - 1) / tiling.blockNum);
    tiling.mPerBatch = static_cast<int32_t>(S);
    tiling.h = static_cast<int32_t>(H);
    tiling.n = static_cast<int32_t>(N);
    tiling.d = static_cast<int32_t>(D);
    tiling.causal = causal ? 1 : 0;
    tiling.ratio = static_cast<int32_t>(ratio);
    tiling.nChunks = static_cast<int32_t>((N + N_CHUNK - 1) / N_CHUNK);

    // Ring geometry: env-overridable for handshake sweeps; defaults are the
    // verified 2 tiles/flag x 2 slots protocol.
    static const int32_t flagBatch = [] {
        const char *e = getenv("INDEXER_FLAG_BATCH");
        return e ? atoi(e) : 2;
    }();
    static const int32_t numSlots = [] {
        const char *e = getenv("INDEXER_NUM_SLOTS");
        return e ? atoi(e) : 2;
    }();
    const int64_t wsBytes = (int64_t)tiling.blockNum * numSlots * flagBatch *
                            TILE_M * N * (int64_t)sizeof(uint16_t);  // bf16 slots
    at::Tensor scores = makeScoresBuffer(wsBytes, q);

    auto aclStream = c10_npu::getCurrentNPUStream().stream(true);
    fused_indexer_qk_reduce((uint32_t)tiling.blockNum, nullptr, aclStream,
                            reinterpret_cast<uint8_t *>(q.mutable_data_ptr()),
                            reinterpret_cast<uint8_t *>(kvT.mutable_data_ptr()),
                            reinterpret_cast<uint8_t *>(weights.mutable_data_ptr()),
                            reinterpret_cast<uint8_t *>(out.mutable_data_ptr()),
                            reinterpret_cast<uint8_t *>(scores.mutable_data_ptr()),
                            tiling.totalTiles, tiling.tilesPerCore,
                            tiling.mPerBatch, tiling.h, tiling.n, tiling.d,
                            tiling.causal, tiling.ratio, tiling.blockNum,
                            tiling.nChunks, flagBatch, numSlots);
    return out;
}

// In-place RoPE on q [B, S, H, D] bf16; freqs [S, P, 2] fp32 (real view of
// freqs_cis).  Rotates the trailing 2*P segment of every head row.
void indexer_rope_torch(const at::Tensor &q, const at::Tensor &freqs)
{
    TORCH_CHECK(q.scalar_type() == at::kBFloat16, "q must be bfloat16");
    TORCH_CHECK(freqs.scalar_type() == at::kFloat, "freqs must be float32");
    TORCH_CHECK(q.is_contiguous() && freqs.is_contiguous(),
                "inputs must be contiguous");
    TORCH_CHECK(q.dim() == 4, "q must be [b, s, h, d]");
    TORCH_CHECK(freqs.dim() == 3 && freqs.size(2) == 2, "freqs must be [s, p, 2]");

    const int64_t B = q.size(0);
    const int64_t S = q.size(1);
    const int64_t H = q.size(2);
    const int64_t D = q.size(3);
    const int64_t P = freqs.size(1);
    TORCH_CHECK(freqs.size(0) >= S, "freqs shorter than seqlen");
    TORCH_CHECK(2 * P <= D, "rope segment larger than head dim");
    TORCH_CHECK((2 * P) % 16 == 0, "rope segment must be 16-aligned");

    auto aclStream = c10_npu::getCurrentNPUStream().stream(true);
    indexer_rope_kernel(40, nullptr, aclStream,
                        reinterpret_cast<uint8_t *>(q.mutable_data_ptr()),
                        reinterpret_cast<uint8_t *>(freqs.mutable_data_ptr()),
                        (int32_t)B, (int32_t)S, (int32_t)H, (int32_t)D,
                        (int32_t)P);
}

// Exact top-128 selection over the padded score rows [B,S,672] with the
// baseline postprocessing (invalid -> -1, +offset) folded into the kernel.
// Returns int64 [B, S, min(128, actualT)].
// The kernel's per-tier TopkTiling structs (outter=16, k=128, nine inner
// tiers) are compile-time constants (see indexer_topk_tiling_consts.h):
// passing nine structs as scalar kernel args does not fit the launch stub,
// and an H2D-copied GM tiling tensor proved unreadable for raw kernel
// launches in quiet process states (see megakernel audit, 2026-08-24
// evening).  Recompute them here once per process and fail loudly on any
// platform/toolchain drift.
static void checkTopkTilings()
{
    static std::once_flag once;
    std::call_once(once, [] {
        auto *platform =
            platform_ascendc::PlatformAscendCManager::GetInstance("Ascend910B3");
        TORCH_CHECK(platform != nullptr, "platform init failed");
        for (int i = 0; i < INDEXER_TOPK_TIERS; ++i) {
            AscendC::tiling::TopkTiling tt;
            bool ok = AscendC::TopKTilingFunc(
                *platform, INDEXER_TOPK_INNERS[i], 16 /*outter*/, 128 /*k*/,
                4 /*fp32*/, false /*isInitIndex*/,
                AscendC::TopKMode::TOPK_NORMAL, true /*isLargest*/, tt);
            TORCH_CHECK(ok, "TopKTilingFunc failed");
            TORCH_CHECK(tt.tmpLocalSize <= 3456, "topk tmp too large");
            TORCH_CHECK(std::memcmp(&tt, INDEXER_TOPK_TTS[i], sizeof(tt)) == 0,
                        "hardcoded topk tiling mismatch for inner=",
                        INDEXER_TOPK_INNERS[i]);
        }
    });
}

at::Tensor indexer_topk_torch(const at::Tensor &scores, bool causal,
                              int64_t ratio, int64_t actualT, int64_t offset)
{
    TORCH_CHECK(scores.scalar_type() == at::kBFloat16, "scores must be bfloat16");
    TORCH_CHECK(scores.device().type() == c10::DeviceType::PrivateUse1,
                "scores must be on NPU");
    TORCH_CHECK(scores.is_contiguous(), "scores must be contiguous");
    TORCH_CHECK(scores.dim() == 3 && scores.size(2) == 672,
                "scores must be [b, s, 672]");

    const int64_t B = scores.size(0);
    const int64_t S = scores.size(1);
    const int64_t k = actualT < 128 ? actualT : 128;
    int64_t totalRows = B * S;

    at::Tensor out = at::empty({B, S, 128}, scores.options().dtype(at::kLong));
    checkTopkTilings();
    // The kernel ignores the 56 trailing tiling args (kept for ABI
    // compatibility); pass the tier-0 constants for documentation value.
    const int32_t *a = INDEXER_TOPK_TTS[0];

    constexpr int32_t kAivCores = 40;
    const int32_t rowsPerCore =
        (int32_t)((totalRows + kAivCores - 1) / kAivCores);

    auto aclStream = c10_npu::getCurrentNPUStream().stream(true);
    indexer_topk_kernel(kAivCores, nullptr, aclStream,
                        reinterpret_cast<uint8_t *>(scores.mutable_data_ptr()),
                        reinterpret_cast<uint8_t *>(out.mutable_data_ptr()),
                        (int32_t)totalRows, rowsPerCore, (int32_t)S,
                        causal ? 1 : 0, (int32_t)ratio, (int32_t)actualT,
                        offset, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17], a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26], a[27], a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17], a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26], a[27]);
    if (k < 128) {
        return out.slice(2, 0, k).contiguous();
    }
    return out;
}

} // namespace ascend_kernel

TORCH_LIBRARY(indexer_ops, m) {
    m.def("fused_qk_reduce", &ascend_kernel::indexer_qk_reduce_torch);
    m.def("rope_inplace", &ascend_kernel::indexer_rope_torch);
    m.def("topk_mask", &ascend_kernel::indexer_topk_torch);
}
