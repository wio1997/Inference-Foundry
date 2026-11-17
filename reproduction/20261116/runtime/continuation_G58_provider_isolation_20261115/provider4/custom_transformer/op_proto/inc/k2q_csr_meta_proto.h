#ifndef K2Q_CSR_META_PROTO_H_
#define K2Q_CSR_META_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(K2qCsrMeta)
    .INPUT(cu_seqlens, ge::TensorType::ALL())
    .INPUT(cu_block_lens, ge::TensorType::ALL())
    .INPUT(scratch, ge::TensorType::ALL())
    .OUTPUT(scratch, ge::TensorType::ALL())
    .REQUIRED_ATTR(order_method, Int)
    .REQUIRED_ATTR(total_rows, Int)
    .REQUIRED_ATTR(max_kv, Int)
    .REQUIRED_ATTR(num_heads, Int)
    .REQUIRED_ATTR(num_tokens, Int)
    .REQUIRED_ATTR(topk, Int)
    .OP_END_FACTORY_REG(K2qCsrMeta);

}

#endif
