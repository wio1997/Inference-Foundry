#ifndef FUSED_SCATTER_COPY_SPARSE_FLASH_ATTENTION_PROTO_H_
#define FUSED_SCATTER_COPY_SPARSE_FLASH_ATTENTION_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(FusedScatterCopySparseFlashAttention)
    .INPUT(query, ge::TensorType::ALL())
    .INPUT(key, ge::TensorType::ALL())
    .INPUT(value, ge::TensorType::ALL())
    .INPUT(sparse_indices, ge::TensorType::ALL())
    .INPUT(cache_tokens, ge::TensorType::ALL())
    .INPUT(hbm_block_table, ge::TensorType::ALL())
    .INPUT(actual_seq_lengths_query, ge::TensorType::ALL())
    .INPUT(actual_seq_lengths_kv, ge::TensorType::ALL())
    .INPUT(query_rope, ge::TensorType::ALL())
    .INPUT(hbm_key_rope, ge::TensorType::ALL())
    .INPUT(dram_key_rope, ge::TensorType::ALL())
    .INPUT(dram_kv_cache, ge::TensorType::ALL())
    .INPUT(dram_block_table, ge::TensorType::ALL())
    .INPUT(topk_source_ids, ge::TensorType::ALL())
    .INPUT(topk_miss_counts, ge::TensorType::ALL())
    .INPUT(miss_src_ids, ge::TensorType::ALL())
    .INPUT(miss_dst_slots, ge::TensorType::ALL())
    .INPUT(miss_counts, ge::TensorType::ALL())
    .OUTPUT(attention_out, ge::TensorType::ALL())
    .REQUIRED_ATTR(scale_value, Float)
    .REQUIRED_ATTR(sparse_block_size, Int)
    .ATTR(layout_query, String, "TND")
    .ATTR(layout_kv, String, "PA_BSND")
    .ATTR(sparse_mode, Int, 3)
    .OP_END_FACTORY_REG(FusedScatterCopySparseFlashAttention);

}

#endif
