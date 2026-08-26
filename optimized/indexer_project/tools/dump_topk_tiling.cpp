// One-shot host utility: dump the advanced-TopK tiling structs used by the
// fused megakernel epilogue (outter=4 tokens per AIV tile, k=128, fp32 src,
// inner=224/672).  The printed int32 arrays are pasted into
// op_kernel/indexer_topk_tiling_consts.h and guarded at runtime by an
// extension-side memcmp against TopKTilingFunc.
#include <cstdio>
#include "adv_api/tiling_api.h"
#include "adv_api/kernel_tiling.h"
#include "utils/tiling/platform/platform_ascendc.h"

int main()
{
    auto *platform =
        platform_ascendc::PlatformAscendCManager::GetInstance("Ascend910B3");
    if (platform == nullptr) {
        printf("platform init failed\n");
        return 1;
    }
    const int32_t outters[1] = {16};
    const int32_t inners[6] = {160, 192, 256, 384, 512, 640};
    for (int o = 0; o < 1; ++o) {
        for (int i = 0; i < 6; ++i) {
            AscendC::tiling::TopkTiling tt;
            bool ok = AscendC::TopKTilingFunc(
                *platform, inners[i], outters[o], 128 /*k*/, 4 /*fp32*/,
                false /*isInitIndex*/, AscendC::TopKMode::TOPK_NORMAL,
                true /*isLargest*/, tt);
            if (!ok) {
                printf("TopKTilingFunc failed for outter=%d inner=%d\n",
                       outters[o], inners[i]);
                return 1;
            }
            printf("// outter=%d inner=%d  sizeof=%zu  tmpLocalSize=%u\n",
                   outters[o], inners[i], sizeof(tt), tt.tmpLocalSize);
            const int32_t *a = reinterpret_cast<const int32_t *>(&tt);
            printf("{");
            for (size_t j = 0; j < sizeof(tt) / sizeof(int32_t); ++j) {
                printf("%s%d", j ? ", " : "", a[j]);
            }
            printf("}\n");
        }
    }
    return 0;
}
