#ifndef K2Q_CSR_HIST_PROTO_H_
#define K2Q_CSR_HIST_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(K2qCsrHist)
    .INPUT(q2k, ge::TensorType::ALL())
    .INPUT(scratch, ge::TensorType::ALL())
    .OUTPUT(scratch, ge::TensorType::ALL())
    .REQUIRED_ATTR(total_rows, Int)
    .REQUIRED_ATTR(max_kv, Int)
    .REQUIRED_ATTR(use_simt, Int)
    .REQUIRED_ATTR(batch, Int)
    .OP_END_FACTORY_REG(K2qCsrHist);

}

#endif
