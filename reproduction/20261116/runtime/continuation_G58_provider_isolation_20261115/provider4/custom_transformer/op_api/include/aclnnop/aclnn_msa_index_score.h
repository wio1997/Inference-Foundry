
/*
 * calution: this file was generated automatically do not change it.
*/

#ifndef ACLNN_MSA_INDEX_SCORE_H_
#define ACLNN_MSA_INDEX_SCORE_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* function: aclnnMsaIndexScoreGetWorkspaceSize
 * parameters :
 * query : required
 * key : required
 * blockTableOptional : optional
 * scaleOptional : optional
 * attenMaskOptional : optional
 * actualSeqQlenOptional : optional
 * actualSeqKlenOptional : optional
 * startLoc : required
 * layoutKeyOptional : optional
 * sparseMode : optional
 * initBlocks : optional
 * localBlocks : optional
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnMsaIndexScoreGetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *blockTableOptional,
    const aclTensor *scaleOptional,
    const aclTensor *attenMaskOptional,
    const aclTensor *actualSeqQlenOptional,
    const aclTensor *actualSeqKlenOptional,
    const aclTensor *startLoc,
    char *layoutKeyOptional,
    int64_t sparseMode,
    int64_t initBlocks,
    int64_t localBlocks,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* function: aclnnMsaIndexScore
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnMsaIndexScore(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
