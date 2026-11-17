
/*
 * calution: this file was generated automatically do not change it.
*/

#ifndef ACLNN_MLA_PREPROCESS_H_
#define ACLNN_MLA_PREPROCESS_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* function: aclnnMlaPreprocessGetWorkspaceSize
 * parameters :
 * hiddenState : required
 * quantScale1Optional : optional
 * quantOffset1Optional : optional
 * wdqkv : required
 * bias1Optional : optional
 * gamma2 : required
 * beta2Optional : optional
 * quantScale2Optional : optional
 * quantOffset2Optional : optional
 * gamma3 : required
 * sin1Optional : optional
 * cos1Optional : optional
 * sin2Optional : optional
 * cos2Optional : optional
 * keycache : required
 * slotMapping : required
 * wuq : required
 * bias2Optional : optional
 * wuk : required
 * descale1Optional : optional
 * descale2Optional : optional
 * ctkvScaleOptional : optional
 * qnopeScaleOptional : optional
 * cacheMode : optional
 * quantMode : optional
 * enableInnerOut : optional
 * enableRope : optional
 * kvCacheStride0 : required
 * kvCacheRopeStride0 : required
 * qOut : required
 * keycacheOutOut : required
 * q2Out : required
 * keycacheOut2Out : required
 * innerOutOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnMlaPreprocessGetWorkspaceSize(
    const aclTensor *hiddenState,
    const aclTensor *quantScale1Optional,
    const aclTensor *quantOffset1Optional,
    const aclTensor *wdqkv,
    const aclTensor *bias1Optional,
    const aclTensor *gamma2,
    const aclTensor *beta2Optional,
    const aclTensor *quantScale2Optional,
    const aclTensor *quantOffset2Optional,
    const aclTensor *gamma3,
    const aclTensor *sin1Optional,
    const aclTensor *cos1Optional,
    const aclTensor *sin2Optional,
    const aclTensor *cos2Optional,
    const aclTensor *keycache,
    const aclTensor *slotMapping,
    const aclTensor *wuq,
    const aclTensor *bias2Optional,
    const aclTensor *wuk,
    const aclTensor *descale1Optional,
    const aclTensor *descale2Optional,
    const aclTensor *ctkvScaleOptional,
    const aclTensor *qnopeScaleOptional,
    int64_t cacheMode,
    int64_t quantMode,
    bool enableInnerOut,
    bool enableRope,
    int64_t kvCacheStride0,
    int64_t kvCacheRopeStride0,
    const aclTensor *qOut,
    const aclTensor *keycacheOutOut,
    const aclTensor *q2Out,
    const aclTensor *keycacheOut2Out,
    const aclTensor *innerOutOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* function: aclnnMlaPreprocess
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnMlaPreprocess(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
