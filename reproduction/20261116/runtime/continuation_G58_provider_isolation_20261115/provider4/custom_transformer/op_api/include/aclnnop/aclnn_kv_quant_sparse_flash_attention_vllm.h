
/*
 * calution: this file was generated automatically do not change it.
*/

#ifndef ACLNN_KV_QUANT_SPARSE_FLASH_ATTENTION_VLLM_H_
#define ACLNN_KV_QUANT_SPARSE_FLASH_ATTENTION_VLLM_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* function: aclnnKvQuantSparseFlashAttentionVllmGetWorkspaceSize
 * parameters :
 * query : required
 * key : required
 * value : required
 * sparseIndices : required
 * keyDequantScaleOptional : optional
 * valueDequantScaleOptional : optional
 * blockTableOptional : optional
 * actualSeqLengthsQueryOptional : optional
 * actualSeqLengthsKvOptional : optional
 * scaleValue : required
 * keyQuantMode : required
 * valueQuantMode : required
 * sparseBlockSize : optional
 * layoutQueryOptional : optional
 * layoutKvOptional : optional
 * sparseMode : optional
 * preTokens : optional
 * nextTokens : optional
 * attentionMode : optional
 * quantScaleRepoMode : optional
 * tileSize : optional
 * ropeHeadDim : optional
 * returnSoftmaxLse : optional
 * attentionOutOut : required
 * softmaxMaxOut : required
 * softmaxSumOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnKvQuantSparseFlashAttentionVllmGetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *value,
    const aclTensor *sparseIndices,
    const aclTensor *keyDequantScaleOptional,
    const aclTensor *valueDequantScaleOptional,
    const aclTensor *blockTableOptional,
    const aclTensor *actualSeqLengthsQueryOptional,
    const aclTensor *actualSeqLengthsKvOptional,
    double scaleValue,
    int64_t keyQuantMode,
    int64_t valueQuantMode,
    int64_t sparseBlockSize,
    char *layoutQueryOptional,
    char *layoutKvOptional,
    int64_t sparseMode,
    int64_t preTokens,
    int64_t nextTokens,
    int64_t attentionMode,
    int64_t quantScaleRepoMode,
    int64_t tileSize,
    int64_t ropeHeadDim,
    bool returnSoftmaxLse,
    const aclTensor *attentionOutOut,
    const aclTensor *softmaxMaxOut,
    const aclTensor *softmaxSumOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* function: aclnnKvQuantSparseFlashAttentionVllm
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnKvQuantSparseFlashAttentionVllm(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
