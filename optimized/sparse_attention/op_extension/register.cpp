#include <torch/extension.h>
#include <torch/library.h>
#include "ops.h"

namespace {

TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def("sparse_attn_megakernel_basic(Tensor q, Tensor kv, Tensor attn_sink, "
          "Tensor topk_idxs, float softmax_scale) -> Tensor");
}

TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    m.impl("sparse_attn_megakernel_basic", TORCH_FN(ascend_kernel::sparse_attn_megakernel_basic_torch));
}

} // namespace
