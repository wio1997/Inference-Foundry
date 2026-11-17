
/*
 * calution: this file was generated automatically do not change it.
*/

#ifndef ACLNN_K2Q_CSR_ROW_PREFIX_H_
#define ACLNN_K2Q_CSR_ROW_PREFIX_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* function: aclnnK2qCsrRowPrefixGetWorkspaceSize
 * parameters :
 * scratch : required
 * totalRows : required
 * maxKv : required
 * useSimt : required
 * numHeads : required
 * numTokens : required
 * topk : required
 * batch : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnK2qCsrRowPrefixGetWorkspaceSize(
    const aclTensor *scratch,
    int64_t totalRows,
    int64_t maxKv,
    int64_t useSimt,
    int64_t numHeads,
    int64_t numTokens,
    int64_t topk,
    int64_t batch,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* function: aclnnK2qCsrRowPrefix
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnK2qCsrRowPrefix(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
