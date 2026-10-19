# 执行日志

- `2026-10-18T23:26:30Z` Loop 已冻结。下一步：Execute frozen R001 stream inventory once, then collect actual scheduler/worker call-site closure.

- `2026-10-18T23:26:30Z` 为用例 `source-contract` 创建 Run `L026-R001`（review）。

- `2026-10-18T23:27:51Z` Run `L026-R001` 记录为 `pass`；正确性为 `not-applicable`。Three canonical stream/env/Spec dispatcher files match HEAD; plan stream inventory locates Eagle/frozenKV/multilayer/standalone/UNO workers. Actual worker/config selection unknown; no stream execution or native safety proof.

- `2026-10-18T23:27:51Z` 为用例 `source-contract` 创建 Run `L026-R002`（review）。

- `2026-10-18T23:30:04Z` Run `L026-R002` 记录为 `pass`；正确性为 `not-applicable`。Three canonical Eagle/common/frozenKV stream worker files match HEAD. Inventory locates spec_info factory. Cached scheduler source has conditional WAR event/forward-stream wait; requires canonical collection and call-position audit. No native stream execution.

- `2026-10-18T23:30:04Z` 为用例 `source-contract` 创建 Run `L026-R003`（review）。

- `2026-10-18T23:32:47Z` Run `L026-R003` 记录为 `pass`；正确性为 `not-applicable`。Canonical scheduler/spec worker factory matched HEAD. EAGLE/EAGLE3 source selects V2 (multilayer conditional); scheduler WAR is is_cuda OR opt-in env(defaultFalse), event/coarse wait is applied after run_batch in overlap loop. Host/NPU completion not inferred; actual task flags unknown. Read-done producer inventory saved.

- `2026-10-18T23:33:41Z` 为用例 `source-contract` 创建 Run `L026-R004`（review）。

- `2026-10-18T23:35:45Z` Run `L026-R004` 记录为 `pass`；正确性为 `not-applicable`。Four canonical read-done helper/modelrunner/graph files match HEAD. Persistent external event helper returnsNone on non-CUDA; eager prefill publishing has separate flag/mode/backend/algorithm gates. No blanket NPU event or safety conclusion; PD loops need direct canonical audit.

- `2026-10-18T23:35:45Z` 为用例 `source-contract` 创建 Run `L026-R005`（review）。

- `2026-10-18T23:37:11Z` Run `L026-R005` 记录为 `pass`；正确性为 `not-applicable`。Canonical PD prefill/decode loops and reader ownership files match HEAD. Both PD overlap loops call _apply_war_barrier after run_batch; gate remains conditional. EAGLE last reader is draft runner when owned, otherwise target (PP). PP/backend capability inventory saved; no device/PD safety proof.

- `2026-10-18T23:37:12Z` 为用例 `source-contract` 创建 Run `L026-R006`（review）。

- `2026-10-18T23:41:08Z` Run `L026-R006` 记录为 `pass`；正确性为 `not-applicable`。Nine canonical NPU attention backend/registry/PP files match installed HEAD; Ascend inherits base shared_read_ends default. PP paths have separate waits; no native proof.

- `2026-10-18T23:41:26Z` 为用例 `source-contract` 创建 Run `L026-R007`（review）。

- `2026-10-18T23:42:09Z` Run `L026-R007` 记录为 `pass`；正确性为 `not-applicable`。Eight canonical source files match HEAD. NPUGraphRunner overrides execute and does not call generic shared-read resolver/publisher, excluding generic PRE_REPLAY fallback from this NPU execute path. EAGLE draft extend retains inherited replay read-done publisher. Native unexecuted.

- `2026-10-18T23:43:28Z` 为用例 `source-contract` 创建 Run `L026-R008`（simulation）。

- `2026-10-18T23:43:46Z` Run `L026-R008` 记录为 `pass`；正确性为 `pass`。326 checks across 66 source-policy cases PASS: exact complete methods with explicit boundary fixtures; NPU override bypasses generic resolver/publisher, WAR enabled chooses owner event or coarse wait, disabled leaves mailbox. No Torch/NPU, graph/DMA, native lifetime, or performance proof.

- 2026-10-19T01:59:46.665991+00:00: R008 primary summary correction: canonical result/stdout contain 266 checks, not 326. Original finish record preserved; notes/L026-run-summary-correction.json records hash. No execution repeated.

- `2026-10-19T01:59:46Z` 暂存知识变化 `npu-execute-war-owner-source-contract`：32 canonical files and 266 CPU source-policy checks/66 cases: NPUGraphRunner.execute overrides generic resolver/publisher, excluding direct attribution of generic PRE fallback; EAGLE last-reader owner and enabled event/coarse mailbox selection verified with explicit fixtures. PD overlap loops call conditional WAR; PP separate. Native device/H2D/PD lifetime and performance unproven.

- `2026-10-19T01:59:46Z` 主控结论为 `INCONCLUSIVE`。Source path and CPU policy validated, generic early fallback not directly used by NPU execute override. Actual task flags/streams/H2D/PD/free lifetime/native performance remain unverified, so L025 candidate remains unaccepted.
