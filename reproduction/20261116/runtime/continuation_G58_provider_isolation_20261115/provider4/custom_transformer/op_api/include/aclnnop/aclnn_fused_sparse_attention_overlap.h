
/*
 * calution: this file was generated automatically do not change it.
*/

#ifndef ACLNN_FUSED_SPARSE_ATTENTION_OVERLAP_H_
#define ACLNN_FUSED_SPARSE_ATTENTION_OVERLAP_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* function: aclnnFusedSparseAttentionOverlapGetWorkspaceSize
 * parameters :
 * query : required
 * key : required
 * value : required
 * sparseIndices : required
 * blockTableOptional : optional
 * actualSeqLengthsQueryOptional : optional
 * actualSeqLengthsKvOptional : optional
 * queryRopeOptional : optional
 * keyRopeOptional : optional
 * selectionKRope : required
 * selectionKvCache : required
 * selectionKvBlockTable : required
 * selectionKvBlockStatus : required
 * selectionMembershipMap : required
 * scaleValue : required
 * sparseBlockSize : required
 * layoutQueryOptional : optional
 * layoutKvOptional : optional
 * sparseMode : optional
 * attentionOutOut : required
 * selectionKvActualSeqOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnFusedSparseAttentionOverlapGetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *value,
    const aclTensor *sparseIndices,
    const aclTensor *blockTableOptional,
    const aclTensor *actualSeqLengthsQueryOptional,
    const aclTensor *actualSeqLengthsKvOptional,
    const aclTensor *queryRopeOptional,
    const aclTensor *keyRopeOptional,
    const aclTensor *selectionKRope,
    const aclTensor *selectionKvCache,
    const aclTensor *selectionKvBlockTable,
    const aclTensor *selectionKvBlockStatus,
    const aclTensor *selectionMembershipMap,
    double scaleValue,
    int64_t sparseBlockSize,
    char *layoutQueryOptional,
    char *layoutKvOptional,
    int64_t sparseMode,
    const aclTensor *attentionOutOut,
    const aclTensor *selectionKvActualSeqOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* function: aclnnFusedSparseAttentionOverlap
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnFusedSparseAttentionOverlap(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
