#ifndef OPS_H
#define OPS_H

#include <torch/extension.h>

namespace ascend_kernel {

at::Tensor sparse_attn_torch(const at::Tensor &q,
                             const at::Tensor &kv,
                             const at::Tensor &attn_sink,
                             const at::Tensor &topk_idxs,
                             double softmax_scale);

at::Tensor sparse_attn_qk(const at::Tensor &q, const at::Tensor &kv);
at::Tensor sparse_attn_softmax(const at::Tensor &scores, const at::Tensor &topk_idxs,
                               const at::Tensor &attn_sink, double softmax_scale);
at::Tensor sparse_attn_pv(const at::Tensor &agg, const at::Tensor &kv);

at::Tensor sparse_attn_fused_qk_softmax(const at::Tensor &q, const at::Tensor &kv,
                                        const at::Tensor &attn_sink,
                                        const at::Tensor &topk_idxs,
                                        double softmax_scale);

at::Tensor sparse_attn_transpose_kv_torch(const at::Tensor &kv);

at::Tensor sparse_attn_megakernel_basic_torch(const at::Tensor &q,
                                             const at::Tensor &kv,
                                             const at::Tensor &attn_sink,
                                             const at::Tensor &topk_idxs,
                                             double softmax_scale);

std::vector<at::Tensor> sparse_attn_megakernel_basic_debug_torch(const at::Tensor &q,
                                                                  const at::Tensor &kv,
                                                                  const at::Tensor &attn_sink,
                                                                  const at::Tensor &topk_idxs,
                                                                  double softmax_scale);

} // namespace ascend_kernel

#endif // OPS_H
