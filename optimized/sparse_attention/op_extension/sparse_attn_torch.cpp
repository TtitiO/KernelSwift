#include <cmath>
#include <cstdint>
#include "acl/acl.h"
#include <torch/extension.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"

#include "sparse_attn_tiling.h"
#include "adv_api/matmul/matmul_tiling.h"
#include "tiling/platform/platform_ascendc.h"

// Kernel entry points (bisheng exports these host wrappers with C++ linkage).
void matmul_qk_kernel(uint32_t blockDim, void *l2Ctrl, aclrtStream stream,
                      uint8_t *a, uint8_t *b, uint8_t *c, uint8_t *tiling);
void matmul_pv_kernel(uint32_t blockDim, void *l2Ctrl, aclrtStream stream,
                      uint8_t *a, uint8_t *b, uint8_t *c, uint8_t *tiling);
void sparse_softmax_kernel(uint32_t blockDim, void *l2Ctrl, aclrtStream stream,
                           uint8_t *scores, uint8_t *agg, uint8_t *topk,
                           uint8_t *sink, uint8_t *tiling);
void fused_qk_softmax_kernel(uint32_t blockDim, void *l2Ctrl, aclrtStream stream,
                             uint8_t *q, uint8_t *kv, uint8_t *sink,
                             uint8_t *topk, uint8_t *scores, uint8_t *agg,
                             uint8_t *workspace, uint8_t *tiling);
void transpose_kv_kernel(uint32_t blockDim, void *l2Ctrl, aclrtStream stream,
                         uint8_t *kv, uint8_t *kvT, uint8_t *tiling);
void transpose_kv_kernel(uint32_t blockDim, void *l2Ctrl, aclrtStream stream,
                         uint8_t *kv, uint8_t *kvT, uint8_t *tiling);
void fused_sparse_attn_basic_kernel(uint32_t blockDim, void *l2Ctrl, aclrtStream stream,
                                    uint8_t *q, uint8_t *kv, uint8_t *kvT,
                                    uint8_t *sink, uint8_t *topk,
                                    uint8_t *scores, uint8_t *agg, uint8_t *out,
                                    uint8_t *tiling);

namespace ascend_kernel {

static int32_t getCubeCoreNum()
{
    int32_t deviceId = -1;
    aclrtGetDevice(&deviceId);
    int64_t num = 0;
    aclrtGetDeviceInfo(deviceId, ACL_DEV_ATTR_CUBE_CORE_NUM, &num);
    return num > 0 ? (int32_t)num : 20;
}

static int32_t getVectorCoreNum()
{
    int32_t deviceId = -1;
    aclrtGetDevice(&deviceId);
    int64_t num = 0;
    aclrtGetDeviceInfo(deviceId, ACL_DEV_ATTR_VECTOR_CORE_NUM, &num);
    return num > 0 ? (int32_t)num : 40;
}

// Compute the cube tiling for a per-batch GEMM C[M,N] = A[M,K] @ B[K,N].
static void computeSparseMatmulTiling(SparseMatmulTiling &t,
                                int64_t M, int64_t N, int64_t K,
                                bool transA, bool transB, bool fp32C,
                                int64_t batchNum,
                                int64_t aBatchStride, int64_t bBatchStride,
                                int64_t cBatchStride)
{
    auto *platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    TORCH_CHECK(platform != nullptr, "PlatformAscendC unavailable");

    matmul_tiling::MatmulApiTiling mmTiling(*platform);
    mmTiling.SetShape(static_cast<int32_t>(M), static_cast<int32_t>(N),
                      static_cast<int32_t>(K));
    mmTiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                      matmul_tiling::DataType::DT_BF16, transA);
    mmTiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                      matmul_tiling::DataType::DT_BF16, transB);
    mmTiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                      fp32C ? matmul_tiling::DataType::DT_FLOAT
                            : matmul_tiling::DataType::DT_BF16);
    mmTiling.SetBias(false);
    mmTiling.SetMadType(matmul_tiling::MatrixMadType::NORMAL);
    // Prefer a smaller M / larger K base-block for tall/skinny GEMMs.
    mmTiling.SetFixSplit(256, -1, 64);

    AscendC::tiling::TCubeTiling cube = {};
    int64_t rc = mmTiling.GetTiling(cube);
    TORCH_CHECK(rc == 0, "MatmulApiTiling::GetTiling failed");

    // Force the logical dims (different CANN versions may leave them stale).
    cube.M = static_cast<int32_t>(M);
    cube.N = static_cast<int32_t>(N);
    cube.Ka = static_cast<int32_t>(K);
    cube.Kb = static_cast<int32_t>(K);

    int32_t singleM = cube.singleCoreM > 0 ? cube.singleCoreM : 1;
    int32_t singleN = cube.singleCoreN > 0 ? cube.singleCoreN : 1;
    const int32_t baseM = cube.baseM > 0 ? cube.baseM : 16;
    const int32_t baseN = cube.baseN > 0 ? cube.baseN : 16;
    constexpr int32_t ALIGN = 16;

    int32_t mTotalCnt = (int32_t)((M + singleM - 1) / singleM);
    int32_t nTotalCnt = (int32_t)((N + singleN - 1) / singleN);
    int32_t totalBlock = mTotalCnt * nTotalCnt;

    // Multi-core split: MatmulApiTiling may assign the whole M/N to one core
    // (singleM == M), leaving only `batchNum` blocks.  Shrink to fill the cube
    // cores (keeps results identical, raises parallelism).
    const int32_t cubeCores = getCubeCoreNum();
    while (totalBlock < cubeCores && singleM > baseM) {
        singleM = std::max(baseM, (singleM / 2) / ALIGN * ALIGN);
        mTotalCnt = (int32_t)((M + singleM - 1) / singleM);
        totalBlock = mTotalCnt * nTotalCnt;
    }
    while (totalBlock < cubeCores && singleN > baseN) {
        singleN = std::max(baseN, (singleN / 2) / ALIGN * ALIGN);
        nTotalCnt = (int32_t)((N + singleN - 1) / singleN);
        totalBlock = mTotalCnt * nTotalCnt;
    }
    cube.singleCoreM = singleM;
    cube.singleCoreN = singleN;
    cube.usedCoreNum = totalBlock;

    const int32_t mBaseTail = (int32_t)(M - (int64_t)(mTotalCnt - 1) * singleM);
    const int32_t nBaseTail = (int32_t)(N - (int64_t)(nTotalCnt - 1) * singleN);

    t.cubeTiling = cube;
    t.batchNum = (int32_t)batchNum;
    t.mPerBatch = (int32_t)M;
    t.mTotalCnt = mTotalCnt;
    t.nTotalCnt = nTotalCnt;
    t.totalBlock = totalBlock;
    t.blockNum = getCubeCoreNum();
    const int64_t grid = batchNum * totalBlock;
    if (t.blockNum > grid) t.blockNum = (int32_t)grid;
    if (t.blockNum < 1) t.blockNum = 1;
    t.workPerBlock = (int32_t)((grid + t.blockNum - 1) / t.blockNum);
    t.mBaseTail = mBaseTail;
    t.nBaseTail = nBaseTail;
    t.reserved0 = 0;
    t.aBatchStride = aBatchStride;
    t.bBatchStride = bBatchStride;
    t.cBatchStride = cBatchStride;
}

// Tiling for the fused QK+softmax mixed kernel.  One tile = 4 m-rows x 64
// heads (singleCoreM=256) x N=32 columns; K=128.  Launched with blockDim =
// cubeCores (20 AI cores); each of the 2*blockDim AIV blocks drives
// tilesPerAiv tiles.
static void computeFusedQKSoftmaxTiling(FusedQKSoftmaxTiling &t,
                                        int64_t B, int64_t M, int64_t H,
                                        int64_t N, int64_t D, int64_t K,
                                        double softmaxScale)
{
    auto *platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    TORCH_CHECK(platform != nullptr, "PlatformAscendC unavailable");

    const int64_t TM = 4 * H;  // 256 (m,h) rows per tile

    matmul_tiling::MatmulApiTiling mmTiling(*platform);
    mmTiling.SetShape(static_cast<int32_t>(TM), static_cast<int32_t>(N),
                      static_cast<int32_t>(D));
    mmTiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                      matmul_tiling::DataType::DT_BF16, false);
    mmTiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                      matmul_tiling::DataType::DT_BF16, true);
    mmTiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                      matmul_tiling::DataType::DT_FLOAT16);
    mmTiling.SetBias(false);
    mmTiling.SetMadType(matmul_tiling::MatrixMadType::NORMAL);
    // One base block = the full [256, 32] tile, so a single Iterate +
    // GetTensorC delivers the whole tile to the vector side in row-major.
    mmTiling.SetFixSplit(static_cast<int32_t>(TM), static_cast<int32_t>(N), 64);

    AscendC::tiling::TCubeTiling cube = {};
    int64_t rc = mmTiling.GetTiling(cube);
    TORCH_CHECK(rc == 0, "MatmulApiTiling::GetTiling failed (fused QK)");

    cube.M = static_cast<int32_t>(TM);
    cube.N = static_cast<int32_t>(N);
    cube.Ka = static_cast<int32_t>(D);
    cube.Kb = static_cast<int32_t>(D);
    cube.singleCoreM = static_cast<int32_t>(TM);
    cube.singleCoreN = static_cast<int32_t>(N);
    cube.baseM = static_cast<int32_t>(TM);
    cube.baseN = static_cast<int32_t>(N);
    cube.singleCoreK = 64;
    cube.baseK = 64;
    cube.usedCoreNum = 1;

    const int32_t blockNum = getCubeCoreNum();
    const int64_t totalTiles = B * M / 4;
    const int64_t aivNum = 2LL * blockNum;
    fprintf(stderr, "[fused-tiling] M=%d N=%d K=%d singleCoreM=%d singleCoreN=%d singleCoreK=%d baseM=%d baseN=%d baseK=%d\n",
            (int)cube.M, (int)cube.N, (int)cube.Ka,
            (int)cube.singleCoreM, (int)cube.singleCoreN, (int)cube.singleCoreK,
            (int)cube.baseM, (int)cube.baseN, (int)cube.baseK);
    t.cubeTiling = cube;
    t.totalTiles = static_cast<int32_t>(totalTiles);
    t.tilesPerAiv = static_cast<int32_t>((totalTiles + aivNum - 1) / aivNum);
    t.mPerBatch = static_cast<int32_t>(M);
    t.h = static_cast<int32_t>(H);
    t.n = static_cast<int32_t>(N);
    t.topk = static_cast<int32_t>(K);
    t.scale = static_cast<float>(softmaxScale);
    t.blockNum = blockNum;
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

static at::Tensor makeTilingTensor(const void *data, size_t bytes, const at::Tensor &ref)
{
    at::Tensor t = at::empty({(int64_t)bytes}, ref.options().dtype(at::kByte));
    aclrtMemcpy(t.mutable_data_ptr(), bytes, data, bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    return t;
}

at::Tensor sparse_attn_torch(const at::Tensor &q,
                             const at::Tensor &kv,
                             const at::Tensor &attn_sink,
                             const at::Tensor &topk_idxs,
                             double softmax_scale)
{
    TORCH_CHECK(q.scalar_type() == at::kBFloat16, "q must be bfloat16");
    TORCH_CHECK(kv.scalar_type() == at::kBFloat16, "kv must be bfloat16");
    TORCH_CHECK(attn_sink.scalar_type() == at::kFloat, "attn_sink must be float32");
    TORCH_CHECK(topk_idxs.scalar_type() == at::kInt, "topk_idxs must be int32");
    TORCH_CHECK(q.is_privateuseone(), "q must be on NPU");
    TORCH_CHECK(kv.is_privateuseone(), "kv must be on NPU");
    TORCH_CHECK(attn_sink.is_privateuseone(), "attn_sink must be on NPU");
    TORCH_CHECK(topk_idxs.is_privateuseone(), "topk_idxs must be on NPU");
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
    const int64_t MH = M * H;

    TORCH_CHECK(kv.size(0) == B && kv.size(2) == D, "kv shape mismatch");
    TORCH_CHECK(attn_sink.size(0) == H, "attn_sink shape mismatch");
    TORCH_CHECK(topk_idxs.size(0) == B && topk_idxs.size(1) == M, "topk_idxs shape mismatch");

    // Intermediate: agg in [B, M, H, N] (h-major).  scores never touch GM in
    // the fused path (QK result goes L0C -> vector UB via the fixpipe).
    at::Tensor agg = at::empty({B, M, H, N}, q.options().dtype(at::kBFloat16));
    at::Tensor out = at::empty({B, M, H, D}, q.options());

    auto aclStream = c10_npu::getCurrentNPUStream().stream(true);

    // --- Fused QK GEMM + sparse softmax (mixed AIC+AIV kernel) ---
    FusedQKSoftmaxTiling fusedTiling{};
    computeFusedQKSoftmaxTiling(fusedTiling, B, M, H, N, D, K, softmax_scale);
    at::Tensor fusedTilingT = makeTilingTensor(&fusedTiling, sizeof(fusedTiling), q);
    auto *platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    const int64_t wsSize = 4LL * 1024 * 1024;  // KFC mailbox (oversized for safety)
    at::Tensor ws = at::zeros({wsSize}, q.options().dtype(at::kByte));
    at::Tensor scores = at::empty({B, M, H, N}, q.options().dtype(at::kFloat));
    fused_qk_softmax_kernel((uint32_t)fusedTiling.blockNum, nullptr, aclStream,
        reinterpret_cast<uint8_t *>(q.mutable_data_ptr()),
        reinterpret_cast<uint8_t *>(kv.mutable_data_ptr()),
        reinterpret_cast<uint8_t *>(attn_sink.mutable_data_ptr()),
        reinterpret_cast<uint8_t *>(topk_idxs.mutable_data_ptr()),
        reinterpret_cast<uint8_t *>(scores.mutable_data_ptr()),
        reinterpret_cast<uint8_t *>(agg.mutable_data_ptr()),
        reinterpret_cast<uint8_t *>(ws.mutable_data_ptr()),
        reinterpret_cast<uint8_t *>(fusedTilingT.mutable_data_ptr()));

    // --- PV GEMM: out[M*H, D] = agg[M*H, N] @ kv[N, D]  (no transpose) ---
    SparseMatmulTiling pvTiling{};
    computeSparseMatmulTiling(pvTiling, MH, D, N, false, false, false, B,
                        MH * N, N * D, MH * D);
    at::Tensor pvTilingT = makeTilingTensor(&pvTiling, sizeof(pvTiling), q);
    matmul_pv_kernel((uint32_t)pvTiling.blockNum, nullptr, aclStream,
        reinterpret_cast<uint8_t *>(agg.mutable_data_ptr()),
        reinterpret_cast<uint8_t *>(kv.mutable_data_ptr()),
        reinterpret_cast<uint8_t *>(out.mutable_data_ptr()),
        reinterpret_cast<uint8_t *>(pvTilingT.mutable_data_ptr()));

    return out;
}

// ---------------------------------------------------------------------------
// Debug ops: expose each stage for isolated correctness checks.
// ---------------------------------------------------------------------------
at::Tensor sparse_attn_qk(const at::Tensor &q, const at::Tensor &kv)
{
    const int64_t B = q.size(0), M = q.size(1), H = q.size(2), D = q.size(3);
    const int64_t N = kv.size(1), MH = M * H;
    at::Tensor scores = at::empty({B, M, H, N}, q.options().dtype(at::kFloat));
    auto aclStream = c10_npu::getCurrentNPUStream().stream(true);
    SparseMatmulTiling t{};
    computeSparseMatmulTiling(t, MH, N, D, false, true, false, B, MH * D, N * D, MH * N);
    at::Tensor tt = makeTilingTensor(&t, sizeof(t), q);
    matmul_qk_kernel((uint32_t)t.blockNum, nullptr, aclStream,
        reinterpret_cast<uint8_t *>(q.mutable_data_ptr()),
        reinterpret_cast<uint8_t *>(kv.mutable_data_ptr()),
        reinterpret_cast<uint8_t *>(scores.mutable_data_ptr()),
        reinterpret_cast<uint8_t *>(tt.mutable_data_ptr()));
    return scores;
}

at::Tensor sparse_attn_softmax(const at::Tensor &scores, const at::Tensor &topk_idxs,
                               const at::Tensor &attn_sink, double softmax_scale)
{
    const int64_t B = scores.size(0), M = scores.size(1), H = scores.size(2), N = scores.size(3);
    const int64_t K = topk_idxs.size(2);
    at::Tensor agg = at::empty({B, M, H, N}, scores.options().dtype(at::kBFloat16));
    auto aclStream = c10_npu::getCurrentNPUStream().stream(true);
    SparseSoftmaxTiling t{};
    t.totalRows = (int32_t)(B * M);
    t.m = (int32_t)M; t.h = (int32_t)H; t.n = (int32_t)N; t.topk = (int32_t)K;
    t.blockNum = getVectorCoreNum();
    if (t.blockNum > t.totalRows) t.blockNum = t.totalRows;
    if (t.blockNum < 1) t.blockNum = 1;
    t.workPerBlock = (t.totalRows + t.blockNum - 1) / t.blockNum;
    t.scale = (float)softmax_scale;
    at::Tensor tt = makeTilingTensor(&t, sizeof(t), scores);
    sparse_softmax_kernel((uint32_t)t.blockNum, nullptr, aclStream,
        reinterpret_cast<uint8_t *>(scores.mutable_data_ptr()),
        reinterpret_cast<uint8_t *>(agg.mutable_data_ptr()),
        reinterpret_cast<uint8_t *>(topk_idxs.mutable_data_ptr()),
        reinterpret_cast<uint8_t *>(attn_sink.mutable_data_ptr()),
        reinterpret_cast<uint8_t *>(tt.mutable_data_ptr()));
    return agg;
}

at::Tensor sparse_attn_pv(const at::Tensor &agg, const at::Tensor &kv)
{
    const int64_t B = agg.size(0), M = agg.size(1), H = agg.size(2), N = agg.size(3);
    const int64_t D = kv.size(2), MH = M * H;
    at::Tensor out = at::empty({B, M, H, D}, agg.options());
    auto aclStream = c10_npu::getCurrentNPUStream().stream(true);
    SparseMatmulTiling t{};
    computeSparseMatmulTiling(t, MH, D, N, false, false, false, B, MH * N, N * D, MH * D);
    at::Tensor tt = makeTilingTensor(&t, sizeof(t), agg);
    matmul_pv_kernel((uint32_t)t.blockNum, nullptr, aclStream,
        reinterpret_cast<uint8_t *>(agg.mutable_data_ptr()),
        reinterpret_cast<uint8_t *>(kv.mutable_data_ptr()),
        reinterpret_cast<uint8_t *>(out.mutable_data_ptr()),
        reinterpret_cast<uint8_t *>(tt.mutable_data_ptr()));
    return out;
}

at::Tensor sparse_attn_fused_qk_softmax(const at::Tensor &q, const at::Tensor &kv,
                                        const at::Tensor &attn_sink,
                                        const at::Tensor &topk_idxs,
                                        double softmax_scale)
{
    const int64_t B = q.size(0), M = q.size(1), H = q.size(2), D = q.size(3);
    const int64_t N = kv.size(1), K = topk_idxs.size(2);
    at::Tensor agg = at::empty({B, M, H, N}, q.options().dtype(at::kBFloat16));
    auto aclStream = c10_npu::getCurrentNPUStream().stream(true);
    FusedQKSoftmaxTiling t{};
    computeFusedQKSoftmaxTiling(t, B, M, H, N, D, K, softmax_scale);
    at::Tensor tt = makeTilingTensor(&t, sizeof(t), q);
    auto *platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    const int64_t wsSize = 4LL * 1024 * 1024;  // KFC mailbox (oversized for safety)
    at::Tensor ws = at::zeros({wsSize}, q.options().dtype(at::kByte));
    at::Tensor scores = at::empty({B, M, H, N}, q.options().dtype(at::kFloat));
    fused_qk_softmax_kernel((uint32_t)t.blockNum, nullptr, aclStream,
        reinterpret_cast<uint8_t *>(q.mutable_data_ptr()),
        reinterpret_cast<uint8_t *>(kv.mutable_data_ptr()),
        reinterpret_cast<uint8_t *>(attn_sink.mutable_data_ptr()),
        reinterpret_cast<uint8_t *>(topk_idxs.mutable_data_ptr()),
        reinterpret_cast<uint8_t *>(scores.mutable_data_ptr()),
        reinterpret_cast<uint8_t *>(agg.mutable_data_ptr()),
        reinterpret_cast<uint8_t *>(ws.mutable_data_ptr()),
        reinterpret_cast<uint8_t *>(tt.mutable_data_ptr()));
    return agg;
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
    TORCH_CHECK(q.is_privateuseone(), "q must be on NPU");
    TORCH_CHECK(kv.is_privateuseone(), "kv must be on NPU");
    TORCH_CHECK(attn_sink.is_privateuseone(), "attn_sink must be on NPU");
    TORCH_CHECK(topk_idxs.is_privateuseone(), "topk_idxs must be on NPU");
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

// Debug: same fused megakernel but also return the internal intermediates.
std::vector<at::Tensor> sparse_attn_megakernel_basic_debug_torch(const at::Tensor &q,
                                                                  const at::Tensor &kv,
                                                                  const at::Tensor &attn_sink,
                                                                  const at::Tensor &topk_idxs,
                                                                  double softmax_scale)
{
    const int64_t B = q.size(0), M = q.size(1), H = q.size(2), D = q.size(3);
    const int64_t N = kv.size(1), K = topk_idxs.size(2);
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
    return {out, agg, scores, kvT};
}

at::Tensor sparse_attn_transpose_kv_torch(const at::Tensor &kv)
{
    TORCH_CHECK(kv.scalar_type() == at::kBFloat16, "kv must be bfloat16");
    TORCH_CHECK(kv.is_privateuseone(), "kv must be on NPU");
    TORCH_CHECK(kv.dim() == 3, "kv must be [b, n, d]");
    const int64_t B = kv.size(0);
    const int64_t N = kv.size(1);
    const int64_t D = kv.size(2);
    at::Tensor kvT = at::empty({B, D, N}, kv.options());
    auto aclStream = c10_npu::getCurrentNPUStream().stream(true);
    TransposeKvTiling t{};
    computeTransposeKvTiling(t, B, N, D, 1.0);
    at::Tensor tt = makeTilingTensor(&t, sizeof(t), kv);
    transpose_kv_kernel((uint32_t)t.batchNum, nullptr, aclStream,
        reinterpret_cast<uint8_t *>(kv.mutable_data_ptr()),
        reinterpret_cast<uint8_t *>(kvT.mutable_data_ptr()),
        reinterpret_cast<uint8_t *>(tt.mutable_data_ptr()));
    return kvT;
}

} // namespace ascend_kernel
