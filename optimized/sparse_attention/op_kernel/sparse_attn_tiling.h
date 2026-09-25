#ifndef SPARSE_ATTN_TILING_H
#define SPARSE_ATTN_TILING_H

#include <cstdint>

// The tiling structs below are shared between the host launcher (g++) and the
// device kernels (bisheng).  adv_api/kernel_tiling.h is pulled in so both
// compilers see the same AscendC tiling PODs; keep it included even where the
// remaining structs no longer embed TCubeTiling directly, as the device build
// relies on its transitive includes.
#include "adv_api/kernel_tiling.h"

// ---------------------------------------------------------------------------
// KV transpose kernel tiling.  One work item is one batch.  The kernel also
// emits an fp16 copy of kv (for the fp16 PV GEMM) and folds softmax_scale
// into the transposed kvT (bf16), so the QK GEMM outputs pre-scaled scores
// and the AIV softmax needs no per-tile scale Mul.
// ---------------------------------------------------------------------------
struct TransposeKvTiling {
    int32_t batchNum;  // B
    int32_t n;         // N (kv entries)
    int32_t d;         // D (head dim)
    float   scale;     // softmax scale folded into kvT
};

// ---------------------------------------------------------------------------
// Basic-API fused sparse-attention megakernel tiling.
// First version: blockDim=1, AIC and AIV0 process all tiles sequentially.
// ---------------------------------------------------------------------------
struct FusedSparseAttnBasicTiling {
    int32_t totalTiles;  // B*M/4
    int32_t batchNum;    // B
    int32_t mPerBatch;   // M
    int32_t h;           // H
    int32_t n;           // N
    int32_t d;           // D
    int32_t topk;        // K
    float   scale;       // softmax scale
    int32_t blockNum;    // AI cores launched (<= cubeCores)
    int32_t reserved0;
};

#endif  // SPARSE_ATTN_TILING_H
