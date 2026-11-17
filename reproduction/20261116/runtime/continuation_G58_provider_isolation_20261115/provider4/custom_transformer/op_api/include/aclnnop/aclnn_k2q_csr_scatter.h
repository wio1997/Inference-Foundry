
/*
 * calution: this file was generated automatically do not change it.
*/

#ifndef ACLNN_K2Q_CSR_SCATTER_H_
#define ACLNN_K2Q_CSR_SCATTER_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* function: aclnnK2qCsrScatterGetWorkspaceSize
 * parameters :
 * q2k : required
 * cuSeqlens : required
 * scratch : required
 * totalRows : required
 * maxKv : required
 * useSimt : required
 * qGlobalOffset : required
 * qIndOut : required
 * slotOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnK2qCsrScatterGetWorkspaceSize(
    const aclTensor *q2k,
    const aclTensor *cuSeqlens,
    const aclTensor *scratch,
    int64_t totalRows,
    int64_t maxKv,
    int64_t useSimt,
    int64_t qGlobalOffset,
    const aclTensor *qIndOut,
    const aclTensor *slotOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* function: aclnnK2qCsrScatter
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnK2qCsrScatter(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
