#ifndef VLLM_ASCEND_QUANT_LIGHTNING_INDEXER_V2_PROTO_H_
#define VLLM_ASCEND_QUANT_LIGHTNING_INDEXER_V2_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(VllmAscendQuantLightningIndexerV2)
    .INPUT(q, ge::TensorType::ALL())
    .INPUT(k, ge::TensorType::ALL())
    .INPUT(w, ge::TensorType::ALL())
    .INPUT(q_descale, ge::TensorType::ALL())
    .INPUT(k_descale, ge::TensorType::ALL())
    .OPTIONAL_INPUT(cu_seqlens_q, ge::TensorType::ALL())
    .OPTIONAL_INPUT(cu_seqlens_k, ge::TensorType::ALL())
    .OPTIONAL_INPUT(seqused_q, ge::TensorType::ALL())
    .OPTIONAL_INPUT(seqused_k, ge::TensorType::ALL())
    .OPTIONAL_INPUT(cmp_residual_k, ge::TensorType::ALL())
    .OPTIONAL_INPUT(block_table, ge::TensorType::ALL())
    .OPTIONAL_INPUT(output_idx_offset, ge::TensorType::ALL())
    .OPTIONAL_INPUT(metadata, ge::TensorType::ALL())
    .OPTIONAL_INPUT(candidate_topk_index, ge::TensorType::ALL())
    .OUTPUT(sparse_indices, ge::TensorType::ALL())
    .OUTPUT(sparse_values, ge::TensorType::ALL())
    .OUTPUT(candidate_topk_index_out, ge::TensorType::ALL())
    .REQUIRED_ATTR(topk, Int)
    .REQUIRED_ATTR(quant_mode, Int)
    .ATTR(max_seqlen_q, Int, -1)
    .ATTR(layout_q, String, "BSND")
    .ATTR(layout_k, String, "BSND")
    .ATTR(mask_mode, Int, 0)
    .ATTR(cmp_ratio, Int, 1)
    .ATTR(return_value, Int, 0)
    .ATTR(candidate_mode, Int, 3)
    .ATTR(candidate_topk_blocks, Int, 2048)
    .ATTR(candidate_block_size, Int, 8)
    .ATTR(key_stride0, Int, 0)
    .ATTR(key_dequant_scale_stride0, Int, 0)
    .OP_END_FACTORY_REG(VllmAscendQuantLightningIndexerV2);

}

#endif
