
/*
 * calution: this file was generated automatically do not change it.
*/

#ifndef ACLNN_GMM_DEQUANT_SITU_QUANT_H_
#define ACLNN_GMM_DEQUANT_SITU_QUANT_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* function: aclnnGmmDequantSituQuantGetWorkspaceSize
 * parameters :
 * x : required
 * weight : dynamic
 * weightScale : dynamic
 * xScale : required
 * groupList : required
 * beta : optional
 * linearBeta : optional
 * hasLinear : optional
 * groupListType : optional
 * weightNz : optional
 * yOut : required
 * yScaleOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnGmmDequantSituQuantGetWorkspaceSize(
    const aclTensor *x,
    const aclTensorList *weight,
    const aclTensorList *weightScale,
    const aclTensor *xScale,
    const aclTensor *groupList,
    double beta,
    double linearBeta,
    bool hasLinear,
    int64_t groupListType,
    bool weightNz,
    const aclTensor *yOut,
    const aclTensor *yScaleOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* function: aclnnGmmDequantSituQuant
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnGmmDequantSituQuant(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
