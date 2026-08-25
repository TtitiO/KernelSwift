#ifndef INDEXER_TILING_H
#define INDEXER_TILING_H

#include <cstdint>

// ---------------------------------------------------------------------------
// Fused indexer QK-reduce mixed-kernel tiling (raw Mmad + CrossCore flags).
// blockDim = AI cores (20).  One tile = 4 tokens x H head rows = 4*H (s,h)
// rows x N (padded kv length) columns; K = head dim.  The AIC runs the QK
// GEMM per tile in N_CHUNK-wide column chunks through L0C and fixpipes the
// bf16 score tile into a per-core ring of GM/L2 slots; the two AIV
// sub-blocks reduce the H head rows (ReLU -> per-row weight mul -> fp32 head
// sum -> bf16) with the causal mask folded in.
// ---------------------------------------------------------------------------
struct IndexerTiling {
    int32_t totalTiles;    // B*S/4 = 5200
    int32_t tilesPerCore;  // ceil(totalTiles / blockNum) = 260
    int32_t mPerBatch;     // S = 2600
    int32_t h;             // H = 16 index heads
    int32_t n;             // N = padded kv length (multiple of 16, e.g. 672)
    int32_t d;             // D = head dim (64)
    int32_t causal;        // 1 -> fold the start_pos==0 causal mask
    int32_t ratio;         // compress ratio (4)
    int32_t blockNum;      // AI cores launched (20)
    int32_t nChunks;       // ceil(n / 224)
};

#endif  // INDEXER_TILING_H
