#ifndef ATTN_RES_FWD_PROTO_H_
#define ATTN_RES_FWD_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(AttnResFwd)
    .INPUT(prefix_sum, ge::TensorType::ALL())
    .INPUT(block_residual, ge::TensorType::ALL())
    .INPUT(proj_weight, ge::TensorType::ALL())
    .INPUT(norm_weight, ge::TensorType::ALL())
    .INPUT(addend, ge::TensorType::ALL())
    .INPUT(output_norm, ge::TensorType::ALL())
    .OUTPUT(hidden_states, ge::TensorType::ALL())
    .OUTPUT(prefix_out, ge::TensorType::ALL())
    .OUTPUT(materialized, ge::TensorType::ALL())
    .ATTR(norm_eps, Float, 9.99999975e-06)
    .ATTR(need_backward, Bool, false)
    .REQUIRED_ATTR(valid_blocks, Int)
    .REQUIRED_ATTR(block_token_stride, Int)
    .ATTR(block_write_idx, Int, -1)
    .ATTR(output_norm_eps, Float, 0)
    .ATTR(save_materialized, Bool, false)
    .ATTR(mix, Bool, true)
    .ATTR(fuse_add, Bool, false)
    .ATTR(optimize_prefill, Bool, false)
    .OP_END_FACTORY_REG(AttnResFwd);

}

#endif
