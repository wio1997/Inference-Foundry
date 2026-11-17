
/*
 * calution: this file was generated automatically do not change it.
*/

#ifndef ACLNN_HC_PRE_H_
#define ACLNN_HC_PRE_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* function: aclnnHcPreGetWorkspaceSize
 * parameters :
 * x : required
 * hcFn : required
 * hcScale : required
 * hcBase : required
 * preMixOptional : optional
 * hcMult : optional
 * hcSinkhornIters : optional
 * hcEps : optional
 * normEps : optional
 * yOut : required
 * postOut : required
 * combFragOut : required
 * preOutOptional : optional
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnHcPreGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *hcFn,
    const aclTensor *hcScale,
    const aclTensor *hcBase,
    const aclTensor *preMixOptional,
    int64_t hcMult,
    int64_t hcSinkhornIters,
    double hcEps,
    double normEps,
    const aclTensor *yOut,
    const aclTensor *postOut,
    const aclTensor *combFragOut,
    const aclTensor *preOutOptional,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* function: aclnnHcPre
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnHcPre(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
