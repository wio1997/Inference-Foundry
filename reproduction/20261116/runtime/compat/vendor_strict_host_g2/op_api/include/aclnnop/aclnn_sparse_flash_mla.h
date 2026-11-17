
/*
 * calution: this file was generated automatically do not change it.
*/

#ifndef ACLNN_SPARSE_FLASH_MLA_H_
#define ACLNN_SPARSE_FLASH_MLA_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* function: aclnnSparseFlashMlaGetWorkspaceSize
 * parameters :
 * q : required
 * oriKvOptional : optional
 * cmpKvOptional : optional
 * oriSparseIndicesOptional : optional
 * cmpSparseIndicesOptional : optional
 * oriBlockTableOptional : optional
 * cmpBlockTableOptional : optional
 * cuSeqlensQOptional : optional
 * cuSeqlensOriKvOptional : optional
 * cuSeqlensCmpKvOptional : optional
 * sequsedQOptional : optional
 * sequsedOriKvOptional : optional
 * sequsedCmpKvOptional : optional
 * cmpResidualKvOptional : optional
 * oriTopkLengthOptional : optional
 * cmpTopkLengthOptional : optional
 * sinksOptional : optional
 * metadataOptional : optional
 * softmaxScale : optional
 * cmpRatio : optional
 * oriMaskMode : optional
 * cmpMaskMode : optional
 * oriWinLeft : optional
 * oriWinRight : optional
 * layoutQOptional : optional
 * layoutKvOptional : optional
 * topkValueMode : optional
 * returnSoftmaxLse : optional
 * attnOutOut : required
 * softmaxLseOutOptional : optional
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnSparseFlashMlaGetWorkspaceSize(
    const aclTensor *q,
    const aclTensor *oriKvOptional,
    const aclTensor *cmpKvOptional,
    const aclTensor *oriSparseIndicesOptional,
    const aclTensor *cmpSparseIndicesOptional,
    const aclTensor *oriBlockTableOptional,
    const aclTensor *cmpBlockTableOptional,
    const aclTensor *cuSeqlensQOptional,
    const aclTensor *cuSeqlensOriKvOptional,
    const aclTensor *cuSeqlensCmpKvOptional,
    const aclTensor *sequsedQOptional,
    const aclTensor *sequsedOriKvOptional,
    const aclTensor *sequsedCmpKvOptional,
    const aclTensor *cmpResidualKvOptional,
    const aclTensor *oriTopkLengthOptional,
    const aclTensor *cmpTopkLengthOptional,
    const aclTensor *sinksOptional,
    const aclTensor *metadataOptional,
    double softmaxScale,
    int64_t cmpRatio,
    int64_t oriMaskMode,
    int64_t cmpMaskMode,
    int64_t oriWinLeft,
    int64_t oriWinRight,
    char *layoutQOptional,
    char *layoutKvOptional,
    int64_t topkValueMode,
    bool returnSoftmaxLse,
    const aclTensor *attnOutOut,
    const aclTensor *softmaxLseOutOptional,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* function: aclnnSparseFlashMla
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnSparseFlashMla(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
