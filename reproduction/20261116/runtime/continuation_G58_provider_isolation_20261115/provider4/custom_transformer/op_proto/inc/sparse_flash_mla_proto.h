#ifndef SPARSE_FLASH_MLA_PROTO_H_
#define SPARSE_FLASH_MLA_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(SparseFlashMla)
    .INPUT(q, ge::TensorType::ALL())
    .OPTIONAL_INPUT(ori_kv, ge::TensorType::ALL())
    .OPTIONAL_INPUT(cmp_kv, ge::TensorType::ALL())
    .OPTIONAL_INPUT(ori_sparse_indices, ge::TensorType::ALL())
    .OPTIONAL_INPUT(cmp_sparse_indices, ge::TensorType::ALL())
    .OPTIONAL_INPUT(ori_block_table, ge::TensorType::ALL())
    .OPTIONAL_INPUT(cmp_block_table, ge::TensorType::ALL())
    .OPTIONAL_INPUT(cu_seqlens_q, ge::TensorType::ALL())
    .OPTIONAL_INPUT(cu_seqlens_ori_kv, ge::TensorType::ALL())
    .OPTIONAL_INPUT(cu_seqlens_cmp_kv, ge::TensorType::ALL())
    .OPTIONAL_INPUT(seqused_q, ge::TensorType::ALL())
    .OPTIONAL_INPUT(seqused_ori_kv, ge::TensorType::ALL())
    .OPTIONAL_INPUT(seqused_cmp_kv, ge::TensorType::ALL())
    .OPTIONAL_INPUT(cmp_residual_kv, ge::TensorType::ALL())
    .OPTIONAL_INPUT(ori_topk_length, ge::TensorType::ALL())
    .OPTIONAL_INPUT(cmp_topk_length, ge::TensorType::ALL())
    .OPTIONAL_INPUT(sinks, ge::TensorType::ALL())
    .OPTIONAL_INPUT(metadata, ge::TensorType::ALL())
    .OUTPUT(attn_out, ge::TensorType::ALL())
    .OUTPUT(softmax_lse, ge::TensorType::ALL())
    .ATTR(softmax_scale, Float, 1)
    .ATTR(cmp_ratio, Int, 0)
    .ATTR(ori_mask_mode, Int, 0)
    .ATTR(cmp_mask_mode, Int, 0)
    .ATTR(ori_win_left, Int, -1)
    .ATTR(ori_win_right, Int, -1)
    .ATTR(layout_q, String, "BSND")
    .ATTR(layout_kv, String, "BSND")
    .ATTR(topk_value_mode, Int, 1)
    .ATTR(return_softmax_lse, Bool, false)
    .OP_END_FACTORY_REG(SparseFlashMla);

}

#endif
