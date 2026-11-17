
/*
 * calution: this file was generated automatically do not change it.
*/

#ifndef ACLNN_INNER_FUSED_SCATTER_COPY_SPARSE_FLASH_ATTENTION_H_
#define ACLNN_INNER_FUSED_SCATTER_COPY_SPARSE_FLASH_ATTENTION_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* function: aclnnInnerFusedScatterCopySparseFlashAttentionGetWorkspaceSize
 * parameters :
 * query : required
 * key : required
 * value : required
 * sparseIndices : required
 * cacheTokens : required
 * hbmBlockTable : required
 * actualSeqLengthsQuery : required
 * actualSeqLengthsKv : required
 * queryRope : required
 * hbmKeyRope : required
 * dramKeyRope : required
 * dramKvCache : required
 * dramBlockTable : required
 * topkSourceIds : required
 * topkMissCounts : required
 * missSrcIds : required
 * missDstSlots : required
 * missCounts : required
 * scaleValue : required
 * sparseBlockSize : required
 * layoutQueryOptional : optional
 * layoutKvOptional : optional
 * sparseMode : optional
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnInnerFusedScatterCopySparseFlashAttentionGetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *value,
    const aclTensor *sparseIndices,
    const aclTensor *cacheTokens,
    const aclTensor *hbmBlockTable,
    const aclTensor *actualSeqLengthsQuery,
    const aclTensor *actualSeqLengthsKv,
    const aclTensor *queryRope,
    const aclTensor *hbmKeyRope,
    const aclTensor *dramKeyRope,
    const aclTensor *dramKvCache,
    const aclTensor *dramBlockTable,
    const aclTensor *topkSourceIds,
    const aclTensor *topkMissCounts,
    const aclTensor *missSrcIds,
    const aclTensor *missDstSlots,
    const aclTensor *missCounts,
    double scaleValue,
    int64_t sparseBlockSize,
    char *layoutQueryOptional,
    char *layoutKvOptional,
    int64_t sparseMode,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* function: aclnnInnerFusedScatterCopySparseFlashAttention
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnInnerFusedScatterCopySparseFlashAttention(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
