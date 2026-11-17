
/*
 * calution: this file was generated automatically do not change it.
*/

#ifndef ACLNN_K2Q_CSR_META_H_
#define ACLNN_K2Q_CSR_META_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* function: aclnnK2qCsrMetaGetWorkspaceSize
 * parameters :
 * cuSeqlens : required
 * cuBlockLens : required
 * scratchRef : required
 * orderMethod : required
 * totalRows : required
 * maxKv : required
 * numHeads : required
 * numTokens : required
 * topk : required
 * scratchRef : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnK2qCsrMetaGetWorkspaceSize(
    const aclTensor *cuSeqlens,
    const aclTensor *cuBlockLens,
    aclTensor *scratchRef,
    int64_t orderMethod,
    int64_t totalRows,
    int64_t maxKv,
    int64_t numHeads,
    int64_t numTokens,
    int64_t topk,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* function: aclnnK2qCsrMeta
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnK2qCsrMeta(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
