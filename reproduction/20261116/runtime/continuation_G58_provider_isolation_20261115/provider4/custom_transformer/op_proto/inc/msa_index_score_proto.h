#ifndef MSA_INDEX_SCORE_PROTO_H_
#define MSA_INDEX_SCORE_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(MsaIndexScore)
    .INPUT(query, ge::TensorType::ALL())
    .INPUT(key, ge::TensorType::ALL())
    .OPTIONAL_INPUT(block_table, ge::TensorType::ALL())
    .OPTIONAL_INPUT(scale, ge::TensorType::ALL())
    .OPTIONAL_INPUT(atten_mask, ge::TensorType::ALL())
    .OPTIONAL_INPUT(actual_seq_qlen, ge::TensorType::ALL())
    .OPTIONAL_INPUT(actual_seq_klen, ge::TensorType::ALL())
    .INPUT(start_loc, ge::TensorType::ALL())
    .OUTPUT(score, ge::TensorType::ALL())
    .ATTR(layout_key, String, "BBND")
    .ATTR(sparse_mode, Int, 3)
    .ATTR(init_blocks, Int, 0)
    .ATTR(local_blocks, Int, 1)
    .OP_END_FACTORY_REG(MsaIndexScore);

}

#endif
