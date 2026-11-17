#ifndef K2Q_CSR_SCATTER_PROTO_H_
#define K2Q_CSR_SCATTER_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(K2qCsrScatter)
    .INPUT(q2k, ge::TensorType::ALL())
    .INPUT(cu_seqlens, ge::TensorType::ALL())
    .INPUT(scratch, ge::TensorType::ALL())
    .OUTPUT(q_ind, ge::TensorType::ALL())
    .OUTPUT(slot, ge::TensorType::ALL())
    .REQUIRED_ATTR(total_rows, Int)
    .REQUIRED_ATTR(max_kv, Int)
    .REQUIRED_ATTR(use_simt, Int)
    .REQUIRED_ATTR(q_global_offset, Int)
    .OP_END_FACTORY_REG(K2qCsrScatter);

}

#endif
