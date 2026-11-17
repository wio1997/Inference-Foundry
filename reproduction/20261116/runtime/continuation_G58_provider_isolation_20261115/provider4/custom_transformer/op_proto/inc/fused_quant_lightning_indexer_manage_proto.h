#ifndef FUSED_QUANT_LIGHTNING_INDEXER_MANAGE_PROTO_H_
#define FUSED_QUANT_LIGHTNING_INDEXER_MANAGE_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(FusedQuantLightningIndexerManage)
    .INPUT(index_weights, ge::TensorType::ALL())
    .INPUT(query_dequant_scale, ge::TensorType::ALL())
    .INPUT(query, ge::TensorType::ALL())
    .INPUT(index_key_dequant_scale, ge::TensorType::ALL())
    .INPUT(index_key_cache, ge::TensorType::ALL())
    .INPUT(index_block_table, ge::TensorType::ALL())
    .INPUT(actual_seq_lengths_query, ge::TensorType::ALL())
    .INPUT(actual_seq_lengths_key, ge::TensorType::ALL())
    .INPUT(offload_seq_lengths_key, ge::TensorType::ALL())
    .INPUT(num_cache_tokens, ge::TensorType::ALL())
    .INPUT(request_state, ge::TensorType::ALL())
    .INPUT(req_pool_entries, ge::TensorType::ALL())
    .INPUT(cache_slots_pool, ge::TensorType::ALL())
    .OUTPUT(topk_src_ids, ge::TensorType::ALL())
    .OUTPUT(topk_dst_slots, ge::TensorType::ALL())
    .OUTPUT(topk_miss_counts, ge::TensorType::ALL())
    .OUTPUT(miss_src_ids, ge::TensorType::ALL())
    .OUTPUT(miss_dst_slots, ge::TensorType::ALL())
    .OUTPUT(miss_counts, ge::TensorType::ALL())
    .OUTPUT(cache_slots_pool_out, ge::TensorType::ALL())
    .OP_END_FACTORY_REG(FusedQuantLightningIndexerManage);

}

#endif
