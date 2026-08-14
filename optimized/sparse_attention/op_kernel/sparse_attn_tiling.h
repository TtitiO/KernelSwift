#ifndef SPARSE_ATTN_TILING_H
#define SPARSE_ATTN_TILING_H

#include <cstdint>

// AscendC::tiling::TCubeTiling is a plain POD (int32_t fields) defined in
// adv_api/kernel_tiling.h; it is available to both the host (g++) and the
// device (bisheng) compilers.
#include "adv_api/kernel_tiling.h"

// ---------------------------------------------------------------------------
// Vector softmax kernel tiling.  One work item is one (b, m) row.
// ---------------------------------------------------------------------------
struct SparseSoftmaxTiling {
    int32_t totalRows;     // B * M
    int32_t m;             // M (sequence length per batch)
    int32_t h;             // H (heads)
    int32_t n;             // N (kv entries)
    int32_t topk;          // K
    int32_t blockNum;      // number of vector blocks launched
    int32_t workPerBlock;  // rows per block = ceil(totalRows / blockNum)
    float   scale;         // softmax scale
};

// ---------------------------------------------------------------------------
// Fused QK+softmax mixed-kernel tiling.  blockDim = AI cores (20); the AIC
// runs a KFC server for its MatmulImpl object, and each of the 2*blockDim AIV
// blocks drives tilesPerAiv tiles (one tile = singleCoreM=256 (m,h) rows =
// 4 m-rows x 64 heads, N=32 columns, K=128).
// ---------------------------------------------------------------------------
struct FusedQKSoftmaxTiling {
    AscendC::tiling::TCubeTiling cubeTiling;  // QK cube tiling (per tile)

    int32_t totalTiles;    // B*M/4 = 5200
    int32_t tilesPerAiv;   // ceil(totalTiles / (2*blockNum)) = 130
    int32_t mPerBatch;     // M = 2600
    int32_t h;             // H = 64
    int32_t n;             // N = 32
    int32_t topk;          // K = 16
    float   scale;         // softmax scale
    int32_t blockNum;      // AI cores launched (20)
};

// ---------------------------------------------------------------------------
// Cube matmul tiling (QK and PV).  One work item is one (b, mTile) tile.
// grid = batchNum * totalBlock; blockNum = min(grid, cubeCores); each block
// loops over workPerBlock consecutive tiles.
// ---------------------------------------------------------------------------
struct SparseMatmulTiling {
    AscendC::tiling::TCubeTiling cubeTiling;  // filled by MatmulApiTiling

    int32_t batchNum;    // B
    int32_t mPerBatch;   // M*H (rows of A / C per batch)
    int32_t mTotalCnt;   // ceil(mPerBatch / singleCoreM)
    int32_t nTotalCnt;   // ceil(N / singleCoreN)
    int32_t totalBlock;  // mTotalCnt * nTotalCnt (per batch)
    int32_t blockNum;    // number of blocks launched
    int32_t workPerBlock;// tiles per block = ceil(batchNum*totalBlock / blockNum)
    int32_t mBaseTail;   // tail rows of the last m tile
    int32_t nBaseTail;   // tail cols of the last n tile
    int32_t reserved0;

    int64_t aBatchStride;  // A per-batch stride in elements
    int64_t bBatchStride;  // B per-batch stride in elements
    int64_t cBatchStride;  // C per-batch stride in elements
};

#endif  // SPARSE_ATTN_TILING_H
