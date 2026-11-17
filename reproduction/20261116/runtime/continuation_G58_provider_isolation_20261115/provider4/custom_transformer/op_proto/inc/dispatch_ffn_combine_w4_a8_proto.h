#ifndef DISPATCH_FFN_COMBINE_W4_A8_PROTO_H_
#define DISPATCH_FFN_COMBINE_W4_A8_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(DispatchFFNCombineW4A8)
    .INPUT(a, ge::TensorType::ALL())
    .DYNAMIC_INPUT(w1, ge::TensorType::ALL())
    .DYNAMIC_INPUT(w2, ge::TensorType::ALL())
    .INPUT(expertIdx, ge::TensorType::ALL())
    .DYNAMIC_INPUT(scale1, ge::TensorType::ALL())
    .DYNAMIC_INPUT(scale2, ge::TensorType::ALL())
    .DYNAMIC_INPUT(bias1, ge::TensorType::ALL())
    .DYNAMIC_INPUT(bias2, ge::TensorType::ALL())
    .INPUT(probs, ge::TensorType::ALL())
    .OPTIONAL_INPUT(xActiveMaskOptional, ge::TensorType::ALL())
    .OUTPUT(out, ge::TensorType::ALL())
    .OUTPUT(expert_token_nums, ge::TensorType::ALL())
    .REQUIRED_ATTR(group, String)
    .ATTR(M, Int, 0)
    .ATTR(transB, Bool, false)
    .ATTR(weightNz, Bool, false)
    .ATTR(swigluLimit, Float, 0)
    .REQUIRED_ATTR(worldSize, Int)
    .OP_END_FACTORY_REG(DispatchFFNCombineW4A8);

}

#endif
