// SPDX-License-Identifier: Apache-2.0
// Minimal registrations around the exact installed donor CANN adapter.
#include <torch/extension.h>
#include <torch/library.h>
#include <torch/torch.h>
#include <climits>
#include <cstdlib>
#include <string>
#include <tuple>
#include <algorithm>
#include <limits>
#include <unistd.h>
#include "include/op_api_common_213.h"
#include "include/actual_qli_candidate.h"
#include "include/actual_sparse_mla.h"

thread_local char g_hashBuf[kHashBufSize];
thread_local int g_hashOffset=0;

TORCH_LIBRARY(sglang_v41_a3_g4,m) {
 m.def("qli_metadata(int num_heads_q, int num_heads_k, int head_dim, int topk, int quant_mode, *, Tensor? cu_seqlens_q=None, Tensor? cu_seqlens_k=None, Tensor? seqused_q=None, Tensor? seqused_k=None, Tensor? cmp_residual_k=None, int batch_size=0, int max_seqlen_q=-1, int max_seqlen_k=-1, str layout_q='TND', str layout_k='PA_BBND', int mask_mode=3, int cmp_ratio=1) -> Tensor");
 m.impl("qli_metadata",torch::kPrivateUse1,&vllm_ascend::qli_v2::QuantLightningIndexerMetadata);
 m.def("qli(Tensor query, Tensor key, Tensor weights, Tensor query_dequant_scale, Tensor key_dequant_scale, int topk, int quant_mode, *, Tensor? candidate_topk_index=None, Tensor? cu_seqlens_q=None, Tensor? cu_seqlens_k=None, Tensor? seqused_q=None, Tensor? seqused_k=None, Tensor? cmp_residual_k=None, Tensor? block_table=None, Tensor? output_idx_offset=None, Tensor? metadata=None, int max_seqlen_q=-1, str layout_q='TND', str layout_k='PA_BBND', int mask_mode=3, int cmp_ratio=1, int candidate_mode=3, int candidate_topk_blocks=2048, int candidate_block_size=8) -> (Tensor,Tensor,Tensor)");
 m.impl("qli",torch::kPrivateUse1,&vllm_ascend::qli_v2::QuantLightningIndexerCandidate);
 m.impl("qli",torch::kMeta,&vllm_ascend::qli_v2::QuantLightningIndexerCandidate);
 m.def("sparse_metadata(int num_heads_q, int num_heads_kv, int head_dim, *, Tensor? cu_seqlens_q=None, Tensor? cu_seqlens_ori_kv=None, Tensor? cu_seqlens_cmp_kv=None, Tensor? seqused_q=None, Tensor? seqused_ori_kv=None, Tensor? seqused_cmp_kv=None, Tensor? cmp_residual_kv=None, Tensor? ori_topk_length=None, Tensor? cmp_topk_length=None, int batch_size=0, int max_seqlen_q=-1, int max_seqlen_ori_kv=-1, int max_seqlen_cmp_kv=-1, int ori_topk=0, int cmp_topk=0, int cmp_ratio=0, int ori_mask_mode=0, int cmp_mask_mode=0, int ori_win_left=-1, int ori_win_right=-1, str layout_q='TND', str layout_kv='PA_BBND', bool has_ori_kv=True, bool has_cmp_kv=True) -> Tensor");
 m.impl("sparse_metadata",torch::kPrivateUse1,&vllm_ascend::npu_sparse_flash_mla_metadata);
 m.def("sparse(Tensor q, *, Tensor? ori_kv=None, Tensor? cmp_kv=None, Tensor? ori_sparse_indices=None, Tensor? cmp_sparse_indices=None, Tensor? ori_block_table=None, Tensor? cmp_block_table=None, Tensor? cu_seqlens_q=None, Tensor? cu_seqlens_ori_kv=None, Tensor? cu_seqlens_cmp_kv=None, Tensor? seqused_q=None, Tensor? seqused_ori_kv=None, Tensor? seqused_cmp_kv=None, Tensor? cmp_residual_kv=None, Tensor? ori_topk_length=None, Tensor? cmp_topk_length=None, Tensor? sinks=None, Tensor? metadata=None, float softmax_scale=1., int cmp_ratio=0, int ori_mask_mode=0, int cmp_mask_mode=0, int ori_win_left=-1, int ori_win_right=-1, str layout_q='TND', str layout_kv='PA_BBND', int topk_value_mode=1, bool return_softmax_lse=False) -> (Tensor,Tensor)");
 m.impl("sparse",torch::kPrivateUse1,&vllm_ascend::npu_sparse_flash_mla);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {}

// Diagnostic only: dladdr on this pointer proves actual symbol ownership.
extern "C" __attribute__((visibility("default")))
void *sglang_v41_a3_resolve_symbol(const char *name) { return GetOpApiFuncAddr(name); }
