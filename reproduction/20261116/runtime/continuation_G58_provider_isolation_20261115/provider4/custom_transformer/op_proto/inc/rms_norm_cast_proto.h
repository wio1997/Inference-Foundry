#ifndef RMS_NORM_CAST_PROTO_H_
#define RMS_NORM_CAST_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(RmsNormCast)
    .INPUT(x, ge::TensorType::ALL())
    .INPUT(gamma, ge::TensorType::ALL())
    .OUTPUT(y, ge::TensorType::ALL())
    .OUTPUT(y_fp32, ge::TensorType::ALL())
    .ATTR(epsilon, Float, 9.99999997e-07)
    .OP_END_FACTORY_REG(RmsNormCast);

}

#endif
