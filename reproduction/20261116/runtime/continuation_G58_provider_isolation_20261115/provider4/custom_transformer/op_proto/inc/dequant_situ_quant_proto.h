#ifndef DEQUANT_SITU_QUANT_PROTO_H_
#define DEQUANT_SITU_QUANT_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(DequantSituQuant)
    .INPUT(x, ge::TensorType::ALL())
    .OPTIONAL_INPUT(weight_scale, ge::TensorType::ALL())
    .OPTIONAL_INPUT(activation_scale, ge::TensorType::ALL())
    .OPTIONAL_INPUT(bias, ge::TensorType::ALL())
    .OPTIONAL_INPUT(quant_scale, ge::TensorType::ALL())
    .OPTIONAL_INPUT(quant_offset, ge::TensorType::ALL())
    .OPTIONAL_INPUT(group_index, ge::TensorType::ALL())
    .OUTPUT(y, ge::TensorType::ALL())
    .OUTPUT(scale, ge::TensorType::ALL())
    .ATTR(beta, Float, 4)
    .ATTR(linear_beta, Float, 25)
    .ATTR(activate_left, Bool, true)
    .ATTR(quant_mode, String, "dynamic")
    .OP_END_FACTORY_REG(DequantSituQuant);

}

#endif
