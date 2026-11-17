#ifndef GMM_DEQUANT_SITU_QUANT_PROTO_H_
#define GMM_DEQUANT_SITU_QUANT_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(GmmDequantSituQuant)
    .INPUT(x, ge::TensorType::ALL())
    .DYNAMIC_INPUT(weight, ge::TensorType::ALL())
    .DYNAMIC_INPUT(weight_scale, ge::TensorType::ALL())
    .INPUT(x_scale, ge::TensorType::ALL())
    .INPUT(group_list, ge::TensorType::ALL())
    .OUTPUT(y, ge::TensorType::ALL())
    .OUTPUT(y_scale, ge::TensorType::ALL())
    .ATTR(beta, Float, 1)
    .ATTR(linear_beta, Float, 1)
    .ATTR(has_linear, Bool, false)
    .ATTR(group_list_type, Int, 1)
    .ATTR(weight_nz, Bool, false)
    .OP_END_FACTORY_REG(GmmDequantSituQuant);

}

#endif
