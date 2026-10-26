# 执行日志

- `2026-10-19T02:05:08Z` Loop 已冻结。下一步：Review already dispatched L027-R001 frozen source collection; preserve post-dispatch metadata registration caveat and never rerun command.

- `2026-10-19T02:05:08Z` 为用例 `source-contract` 创建 Run `L027-R001`（review）。

- `2026-10-19T02:05:54Z` Run `L027-R001` 记录为 `fail`；正确性为 `not-applicable`。Frozen source collector executed once; inner exit255 SSH banner timeout, empty source payload. CLI exit0 is not success. Run metadata registered after dispatch due stale active-loop index; provenance caveat preserved. No native/inference execution and no new source credit.

- 2026-10-19T02:08:42.403998+00:00: R001metadata registered after dispatch: see notes/L027-registration-recovery.json. Frozen command executed exactly once, failed SSH banner before source collection. Direct BatchMode connectivity also timed out; do not rerun. Cached canonical4sources and historical3receipt hashes located; no runtime proof.

- `2026-10-19T02:10:39Z` 为用例 `source-contract` 创建 Run `L027-R002`（simulation）。

- `2026-10-19T02:11:33Z` Run `L027-R002` 记录为 `invalid`；正确性为 `invalid`。INVALID harness: py_compile system-cache permission failure prevented frozen protocol/spec creation; later shell lines incorrectly registered/dispatched missing spec. No inner receipt/result; zero source checks credited. R002 retained, not rerun; R003 generated with in-memory compile and separately verified frozen inputs.

- `2026-10-19T02:11:33Z` 为用例 `source-contract` 创建 Run `L027-R003`（simulation）。

- `2026-10-19T02:12:41Z` Run `L027-R003` 记录为 `pass`；正确性为 `pass`。138 checks across 25 exact cached canonical source-policy cases PASS; 11 complete methods. Explicit tensor/collective/network/free fixtures; 4 device-deadline cases release without all acks, host pages remain held; exception ledger policy tested. No native lifetime, current runtime flags, or performance proof.

- `2026-10-19T02:14:18Z` 为用例 `source-contract` 创建 Run `L027-R004`（review）。

- `2026-10-19T02:15:47Z` Run `L027-R004` 记录为 `pass`；正确性为 `not-applicable`。12 fixed public commit files verified by Git blob SHA1; 8 preexisting snapshots exactly match and 4 missing local files recovered. Remote installed runtime not revalidated; no native execution.

- `2026-10-19T02:15:54Z` 为用例 `source-contract` 创建 Run `L027-R005`（simulation）。

- `2026-10-19T02:17:09Z` Run `L027-R005` 记录为 `pass`；正确性为 `pass`。13 checks/6 cases of complete canonical _await_transfer_futures with real stdlib Future/thread PASS. Deferred nonzero status drains; RuntimeError escapes with another Future running even when deferred enabled. Explicit transfer boundary; no actual transport/UAF proof. Ascend direct custom-pool override still needs separate audit.

- `2026-10-19T02:18:42Z` 为用例 `source-contract` 创建 Run `L027-R006`（simulation）。

- `2026-10-19T02:19:53Z` Run `L027-R006` 记录为 `pass`；正确性为 `pass`。30 checks/6 cases PASS: complete Ascend.send_kvcache -> Mooncake._transfer_data -> engine.batch_transfer_sync with real stdlib executors and explicit SDK/layout fixtures. Custom-pool override returns with a running per-layer Future for both deferred flag values; SDK exceptions become -1. Actual custom-pool eligibility/runtime flags and DMA lifetime unproven.

- `2026-10-19T02:20:00Z` 为用例 `source-contract` 创建 Run `L027-R007`（review）。

- `2026-10-19T02:22:05Z` Run `L027-R007` 记录为 `pass`；正确性为 `not-applicable`。Fixed commit custom-pool selector verified by public Git blob. EnvStr defaultNone disables branch; selector recognizes NVLINK/BAREX/INTRA_NODE_NVLINK without hardware check, separate initializer may fall back. Actual Ascend custom-pool applicability remains runtime-unconfirmed.

- `2026-10-19T02:23:31Z` 暂存知识变化 `pd-drain-timeout-ascend-override-contract`：Fixed commit13publicGitblobverifiedfiles; separate CPU source experiments138/13/30checks. Device deferred deadline can release without full ACK; host path holds. Generic helper drains nonzero status when enabled but uncaught Exception escapes with running Future. Ascend custom-pool direct override bypasses helper and early-returns on nonzero/wrapped SDK errors for either flag; defaultNone disables selector, actual Ascend pool eligibility/runtime unknown. No active DMA/UAF or performance proof, no candidate acceptance.

- `2026-10-19T02:23:31Z` 主控结论为 `INCONCLUSIVE`。Source/CPU control-flow contracts verified, including actual Ascend override and custom-pool selector qualification. SDK/physical DMA completion, selected pool/flags and native PD/free/output/performance remain unproven. R001failedSSH/R002invalid retained, no candidate accepted.
