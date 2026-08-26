#ifndef INDEXER_TOPK_TILING_CONSTS_H
#define INDEXER_TOPK_TILING_CONSTS_H

#include <cstdint>

// ---------------------------------------------------------------------------
// Hardcoded advanced-TopK tiling structs for the standalone indexer_topk
// kernel (outter = 16 rows per TopK call, k = 128, fp32 source, TOPK_NORMAL,
// largest-first), generated on Ascend910B3 / CANN 8.5.2 by
// tools/dump_topk_tiling.cpp via AscendC::TopKTilingFunc.
//
// Valid-prefix pruning dispatches each batch to the smallest compile-time
// inner tier in INDEXER_TOPK_INNERS covering its largest row limit (sort cost
// scales with inner; the advanced TopK is ~2.2x slower with a runtime inner).
//
// They are compile-time constants because the kernel-launch stub marshalling
// fails somewhere between 65 and 93 scalar args (nine structs = 252 int32),
// and an H2D-copied GM tiling tensor proved unreadable in quiet process
// states.  The extension recomputes all structs with TopKTilingFunc once per
// process and TORCH_CHECKs them against these arrays, so any platform drift
// fails loudly instead of corrupting results.
// ---------------------------------------------------------------------------

constexpr int32_t INDEXER_TOPK_TT_INTS = 28;  // sizeof(TopkTiling)/sizeof(int32)
constexpr int32_t INDEXER_TOPK_TIERS = 9;
constexpr int32_t INDEXER_TOPK_INNERS[INDEXER_TOPK_TIERS] = {
    128, 192, 256, 320, 384, 448, 512, 576, 672};

// TopKTilingFunc(platform, outter=16, k=128, fp32, TOPK_NORMAL, inner=...)
constexpr int32_t INDEXER_TOPK_TTS[INDEXER_TOPK_TIERS][INDEXER_TOPK_TT_INTS] = {
    {640, 2048, 256, 4, 32, 128, 128, 2048, 256, 512, 2, 4, 6, 2,
     256, 1, 512, 32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {960, 3072, 384, 6, 48, 128, 128, 2048, 256, 512, 2, 4, 6, 2,
     384, 1, 768, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {1280, 4096, 512, 8, 64, 128, 128, 2048, 256, 512, 2, 4, 6, 2,
     512, 1, 1024, 64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {1600, 5120, 640, 10, 80, 128, 128, 2048, 256, 512, 2, 4, 6, 2,
     640, 1, 1280, 80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {1920, 6144, 768, 12, 96, 128, 128, 2048, 256, 512, 2, 4, 6, 2,
     768, 1, 1536, 96, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {2240, 7168, 896, 14, 112, 128, 128, 2048, 256, 512, 2, 4, 6, 2,
     896, 1, 1792, 112, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {2560, 8192, 1024, 16, 128, 128, 128, 2048, 256, 512, 2, 4, 6, 2,
     1024, 1, 2048, 128, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {2880, 9216, 1152, 18, 144, 128, 128, 2048, 256, 512, 2, 4, 6, 2,
     1152, 1, 2304, 144, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {3360, 10752, 1344, 21, 168, 128, 128, 2048, 256, 512, 2, 4, 6, 2,
     1344, 1, 2688, 168, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
};

// UB scratch for the TopK call: max(tmpLocalSize) = 3360 floats.
constexpr int32_t INDEXER_TOPK_TMP_FLOATS = 3360;

#endif  // INDEXER_TOPK_TILING_CONSTS_H
