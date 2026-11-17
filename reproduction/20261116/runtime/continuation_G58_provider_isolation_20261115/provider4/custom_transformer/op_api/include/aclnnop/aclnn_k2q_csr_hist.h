
/*
 * calution: this file was generated automatically do not change it.
*/

#ifndef ACLNN_K2Q_CSR_HIST_H_
#define ACLNN_K2Q_CSR_HIST_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* function: aclnnK2qCsrHistGetWorkspaceSize
 * parameters :
 * q2k : required
 * scratchRef : required
 * totalRows : required
 * maxKv : required
 * useSimt : required
 * batch : required
 * scratchRef : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnK2qCsrHistGetWorkspaceSize(
    const aclTensor *q2k,
    aclTensor *scratchRef,
    int64_t totalRows,
    int64_t maxKv,
    int64_t useSimt,
    int64_t batch,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* function: aclnnK2qCsrHist
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnK2qCsrHist(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
