
/*
 * calution: this file was generated automatically do not change it.
*/

#ifndef ACLNN_RMS_NORM_CAST_H_
#define ACLNN_RMS_NORM_CAST_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* function: aclnnRmsNormCastGetWorkspaceSize
 * parameters :
 * x : required
 * gamma : required
 * epsilon : optional
 * yOut : required
 * yFp32Out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnRmsNormCastGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *gamma,
    double epsilon,
    const aclTensor *yOut,
    const aclTensor *yFp32Out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* function: aclnnRmsNormCast
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnRmsNormCast(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
