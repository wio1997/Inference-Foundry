#ifndef KV_QUANT_SPARSE_FLASH_ATTENTION_VLLM_PROTO_H_
#define KV_QUANT_SPARSE_FLASH_ATTENTION_VLLM_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(KvQuantSparseFlashAttentionVllm)
    .INPUT(query, ge::TensorType::ALL())
    .INPUT(key, "key_type")
    .INPUT(value, "key_type")
    .INPUT(sparse_indices, ge::TensorType::ALL())
    .OPTIONAL_INPUT(key_dequant_scale, ge::TensorType::ALL())
    .OPTIONAL_INPUT(value_dequant_scale, ge::TensorType::ALL())
    .OPTIONAL_INPUT(block_table, ge::TensorType::ALL())
    .OPTIONAL_INPUT(actual_seq_lengths_query, ge::TensorType::ALL())
    .OPTIONAL_INPUT(actual_seq_lengths_kv, ge::TensorType::ALL())
    .OUTPUT(attention_out, ge::TensorType::ALL())
    .OUTPUT(softmax_max, ge::TensorType::ALL())
    .OUTPUT(softmax_sum, ge::TensorType::ALL())
    .DATATYPE(key_type, ge::TensorType({ge::DT_INT8}))
    .REQUIRED_ATTR(scale_value, Float)
    .REQUIRED_ATTR(key_quant_mode, Int)
    .REQUIRED_ATTR(value_quant_mode, Int)
    .ATTR(sparse_block_size, Int, 1)
    .ATTR(layout_query, String, "BSND")
    .ATTR(layout_kv, String, "BSND")
    .ATTR(sparse_mode, Int, 3)
    .ATTR(pre_tokens, Int, 9223372036854775807)
    .ATTR(next_tokens, Int, 9223372036854775807)
    .ATTR(attention_mode, Int, 0)
    .ATTR(quant_scale_repo_mode, Int, 1)
    .ATTR(tile_size, Int, 128)
    .ATTR(rope_head_dim, Int, 64)
    .ATTR(return_softmax_lse, Bool, false)
    .OP_END_FACTORY_REG(KvQuantSparseFlashAttentionVllm);

}

#endif
