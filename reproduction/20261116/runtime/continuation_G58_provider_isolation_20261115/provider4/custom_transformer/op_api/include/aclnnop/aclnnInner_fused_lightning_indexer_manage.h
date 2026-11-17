
/*
 * calution: this file was generated automatically do not change it.
*/

#ifndef ACLNN_INNER_FUSED_LIGHTNING_INDEXER_MANAGE_H_
#define ACLNN_INNER_FUSED_LIGHTNING_INDEXER_MANAGE_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* function: aclnnInnerFusedLightningIndexerManageGetWorkspaceSize
 * parameters :
 * indexWeights : required
 * queryDequantScale : required
 * query : required
 * indexKeyDequantScale : required
 * indexKeyCache : required
 * indexBlockTable : required
 * actualSeqLengthsQuery : required
 * actualSeqLengthsKey : required
 * offloadSeqLengthsKey : required
 * numCacheTokens : required
 * requestState : required
 * reqPoolEntries : required
 * cacheSlotsPool : required
 * topkSrcIdsOut : required
 * topkDstSlotsOut : required
 * topkMissCountsOut : required
 * missSrcIdsOut : required
 * missDstSlotsOut : required
 * missCountsOut : required
 * cacheSlotsPoolOutOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnInnerFusedLightningIndexerManageGetWorkspaceSize(
    const aclTensor *indexWeights,
    const aclTensor *queryDequantScale,
    const aclTensor *query,
    const aclTensor *indexKeyDequantScale,
    const aclTensor *indexKeyCache,
    const aclTensor *indexBlockTable,
    const aclTensor *actualSeqLengthsQuery,
    const aclTensor *actualSeqLengthsKey,
    const aclTensor *offloadSeqLengthsKey,
    const aclTensor *numCacheTokens,
    const aclTensor *requestState,
    const aclTensor *reqPoolEntries,
    const aclTensor *cacheSlotsPool,
    const aclTensor *topkSrcIdsOut,
    const aclTensor *topkDstSlotsOut,
    const aclTensor *topkMissCountsOut,
    const aclTensor *missSrcIdsOut,
    const aclTensor *missDstSlotsOut,
    const aclTensor *missCountsOut,
    const aclTensor *cacheSlotsPoolOutOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* function: aclnnInnerFusedLightningIndexerManage
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnInnerFusedLightningIndexerManage(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
