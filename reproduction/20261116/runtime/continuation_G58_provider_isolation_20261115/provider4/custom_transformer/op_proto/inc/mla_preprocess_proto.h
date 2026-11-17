#ifndef MLA_PREPROCESS_PROTO_H_
#define MLA_PREPROCESS_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(MlaPreprocess)
    .INPUT(hiddenState, ge::TensorType::ALL())
    .OPTIONAL_INPUT(quantScale1, ge::TensorType::ALL())
    .OPTIONAL_INPUT(quantOffset1, ge::TensorType::ALL())
    .INPUT(wdqkv, ge::TensorType::ALL())
    .OPTIONAL_INPUT(bias1, ge::TensorType::ALL())
    .INPUT(gamma2, ge::TensorType::ALL())
    .OPTIONAL_INPUT(beta2, ge::TensorType::ALL())
    .OPTIONAL_INPUT(quantScale2, ge::TensorType::ALL())
    .OPTIONAL_INPUT(quantOffset2, ge::TensorType::ALL())
    .INPUT(gamma3, ge::TensorType::ALL())
    .OPTIONAL_INPUT(sin1, ge::TensorType::ALL())
    .OPTIONAL_INPUT(cos1, ge::TensorType::ALL())
    .OPTIONAL_INPUT(sin2, ge::TensorType::ALL())
    .OPTIONAL_INPUT(cos2, ge::TensorType::ALL())
    .INPUT(keycache, ge::TensorType::ALL())
    .INPUT(slotMapping, ge::TensorType::ALL())
    .INPUT(wuq, ge::TensorType::ALL())
    .OPTIONAL_INPUT(bias2, ge::TensorType::ALL())
    .INPUT(wuk, ge::TensorType::ALL())
    .OPTIONAL_INPUT(descale1, ge::TensorType::ALL())
    .OPTIONAL_INPUT(descale2, ge::TensorType::ALL())
    .OPTIONAL_INPUT(ctkvScale, ge::TensorType::ALL())
    .OPTIONAL_INPUT(qnopeScale, ge::TensorType::ALL())
    .OUTPUT(q, ge::TensorType::ALL())
    .OUTPUT(keycacheOut, ge::TensorType::ALL())
    .OUTPUT(q2, ge::TensorType::ALL())
    .OUTPUT(keycacheOut2, ge::TensorType::ALL())
    .OUTPUT(innerOut, ge::TensorType::ALL())
    .ATTR(cacheMode, Int, 1)
    .ATTR(quantMode, Int, 1)
    .ATTR(enableInnerOut, Bool, false)
    .ATTR(enableRope, Bool, true)
    .REQUIRED_ATTR(kvCacheStride0, Int)
    .REQUIRED_ATTR(kvCacheRopeStride0, Int)
    .OP_END_FACTORY_REG(MlaPreprocess);

}

#endif
