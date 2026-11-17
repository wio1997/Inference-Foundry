
/*
 * calution: this file was generated automatically do not change it.
*/

#ifndef ACLNN_INNER_DISPATCH_FFNCOMBINE_W4A8_H_
#define ACLNN_INNER_DISPATCH_FFNCOMBINE_W4A8_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* function: aclnnInnerDispatchFFNCombineW4A8GetWorkspaceSize
 * parameters :
 * a : required
 * w1 : dynamic
 * w2 : dynamic
 * expertIdx : required
 * scale1 : dynamic
 * scale2 : dynamic
 * bias1 : dynamic
 * bias2 : dynamic
 * probs : required
 * xActiveMaskOptionalOptional : optional
 * group : required
 * m : optional
 * transB : optional
 * weightNz : optional
 * swigluLimit : optional
 * worldSize : required
 * outOut : required
 * expertTokenNumsOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnInnerDispatchFFNCombineW4A8GetWorkspaceSize(
    const aclTensor *a,
    const aclTensorList *w1,
    const aclTensorList *w2,
    const aclTensor *expertIdx,
    const aclTensorList *scale1,
    const aclTensorList *scale2,
    const aclTensorList *bias1,
    const aclTensorList *bias2,
    const aclTensor *probs,
    const aclTensor *xActiveMaskOptionalOptional,
    char *group,
    int64_t m,
    bool transB,
    bool weightNz,
    double swigluLimit,
    int64_t worldSize,
    const aclTensor *outOut,
    const aclTensor *expertTokenNumsOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* function: aclnnInnerDispatchFFNCombineW4A8
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnInnerDispatchFFNCombineW4A8(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
