#ifndef SPARSE_ATTENTION_SCORE_PROTO_H_
#define SPARSE_ATTENTION_SCORE_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(SparseAttentionScore)
    .INPUT(query, ge::TensorType::ALL())
    .INPUT(key, ge::TensorType::ALL())
    .INPUT(value, ge::TensorType::ALL())
    .INPUT(selectIdx, ge::TensorType::ALL())
    .INPUT(blockTable, ge::TensorType::ALL())
    .OPTIONAL_INPUT(selectNumIdx, ge::TensorType::ALL())
    .OPTIONAL_INPUT(actualSeqLengths, ge::TensorType::ALL())
    .OPTIONAL_INPUT(actualSeqLengthsKv, ge::TensorType::ALL())
    .OPTIONAL_INPUT(qDequantScale, ge::TensorType::ALL())
    .OPTIONAL_INPUT(kDequantScale, ge::TensorType::ALL())
    .OPTIONAL_INPUT(vDequantScale, ge::TensorType::ALL())
    .OUTPUT(attentionOut, ge::TensorType::ALL())
    .OUTPUT(softmaxLse, ge::TensorType::ALL())
    .ATTR(numKeyValueHeads, Int, 1)
    .ATTR(scaleValue, Float, 0)
    .ATTR(blockSize, Int, 128)
    .ATTR(topK, Int, 16)
    .ATTR(innerPrecise, Int, 4)
    .OP_END_FACTORY_REG(SparseAttentionScore);

}

#endif
