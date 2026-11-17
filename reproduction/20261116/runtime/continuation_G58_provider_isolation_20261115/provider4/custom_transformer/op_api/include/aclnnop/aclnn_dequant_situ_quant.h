
/*
 * calution: this file was generated automatically do not change it.
*/

#ifndef ACLNN_DEQUANT_SITU_QUANT_H_
#define ACLNN_DEQUANT_SITU_QUANT_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* function: aclnnDequantSituQuantGetWorkspaceSize
 * parameters :
 * x : required
 * weightScaleOptional : optional
 * activationScaleOptional : optional
 * biasOptional : optional
 * quantScaleOptional : optional
 * quantOffsetOptional : optional
 * groupIndexOptional : optional
 * beta : optional
 * linearBeta : optional
 * activateLeft : optional
 * quantModeOptional : optional
 * yOut : required
 * scaleOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnDequantSituQuantGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *weightScaleOptional,
    const aclTensor *activationScaleOptional,
    const aclTensor *biasOptional,
    const aclTensor *quantScaleOptional,
    const aclTensor *quantOffsetOptional,
    const aclTensor *groupIndexOptional,
    double beta,
    double linearBeta,
    bool activateLeft,
    char *quantModeOptional,
    const aclTensor *yOut,
    const aclTensor *scaleOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* function: aclnnDequantSituQuant
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnDequantSituQuant(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
