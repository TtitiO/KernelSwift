#include <torch/extension.h>
#include <torch/library.h>
#include "ops.h"

namespace {

TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def("sparse_attn(Tensor q, Tensor kv, Tensor attn_sink, Tensor topk_idxs, "
          "float softmax_scale) -> Tensor");
    m.def("sparse_attn_qk(Tensor q, Tensor kv) -> Tensor");
    m.def("sparse_attn_softmax(Tensor scores, Tensor topk_idxs, Tensor attn_sink, "
          "float softmax_scale) -> Tensor");
    m.def("sparse_attn_pv(Tensor agg, Tensor kv) -> Tensor");
    m.def("sparse_attn_fused_qk_softmax(Tensor q, Tensor kv, Tensor attn_sink, "
          "Tensor topk_idxs, float softmax_scale) -> Tensor");
}

TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    m.impl("sparse_attn", TORCH_FN(ascend_kernel::sparse_attn_torch));
    m.impl("sparse_attn_qk", TORCH_FN(ascend_kernel::sparse_attn_qk));
    m.impl("sparse_attn_softmax", TORCH_FN(ascend_kernel::sparse_attn_softmax));
    m.impl("sparse_attn_pv", TORCH_FN(ascend_kernel::sparse_attn_pv));
    m.impl("sparse_attn_fused_qk_softmax", TORCH_FN(ascend_kernel::sparse_attn_fused_qk_softmax));
}

at::Tensor sparse_attn_meta(const at::Tensor &q, const at::Tensor &kv,
                            const at::Tensor &attn_sink, const at::Tensor &topk_idxs,
                            double /*softmax_scale*/)
{
    return at::empty_like(q);
}

TORCH_LIBRARY_IMPL(npu, Meta, m)
{
    m.impl("sparse_attn", &sparse_attn_meta);
}

} // namespace
