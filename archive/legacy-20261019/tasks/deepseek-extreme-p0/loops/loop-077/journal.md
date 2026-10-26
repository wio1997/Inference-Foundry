# 执行日志

- `2026-09-27T00:58:26Z` Loop 已冻结。下一步：Read-only source/trace audit of device snapshot lifecycle, count_history export, all-rank endpoint and collective join, then choose smallest reversible measurement

- `2026-09-27T01:01:16Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run374`（design-check）。

- `2026-09-27T01:01:31Z` Run `run374` 记录为 `pass`；正确性为 `not-applicable`。Existing acceptance is device-staged until cohort end; Draft seven-step Markov feedback and Host-scope tail verified; next low-impact route capture specified

- `2026-09-27T01:03:41Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run375`（profile）。

- `2026-09-27T01:05:55Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run376`（review）。

- `2026-09-27T01:06:03Z` Run `run376` 记录为 `pass`；正确性为 `not-applicable`。Seven serial Markov-bias tasks per rank-cycle span 0.642-2.516ms profiled, duration sum ~0.231ms; gap not removable or attainable floor

- `2026-09-27T01:12:14Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run377`（review）。

- `2026-09-27T01:12:28Z` Run `run377` 记录为 `pass`；正确性为 `not-applicable`。265 identical HCCL task signatures per Target rank-cycle, reported count*dtype 25.6512MB/rank-cycle; ABI and wire bytes unresolved

- `2026-09-27T01:18:04Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run378`（review）。

- `2026-09-27T01:18:27Z` Run `run378` 记录为 `pass`；正确性为 `not-applicable`。Exact 265-task HCCL order identical all 16 rank-cycles; conditional API input tensor sum115.108MB/rank-cycle, neither compulsory nor wire bytes

- `2026-09-27T01:21:29Z` Run `run375` 记录为 `pass`；正确性为 `pass`。48+12 exact clients, 40/40 rank/cohort Runtime FULL Graph pass, 40 cohort-end route/acceptance files, source restored/service stopped; no hot-path D2H/sync

- `2026-09-27T01:21:29Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run380`（review）。

- `2026-09-27T01:21:29Z` Run `run380` 记录为 `pass`；正确性为 `not-applicable`。80 selected rank-cycles matched to clipped useful acceptance; all43 layers sum576 routed tokens across TP8 on each of ten cycles

- `2026-09-27T01:21:49Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run379`（benchmark）。

- `2026-09-27T01:23:00Z` Run `run379` 记录为 `pass`；正确性为 `pass`。Exact 265 ordered HCCL chain passed eight-rank result checks; eager torch.distributed median latest device40.910ms with Host submit+wait~41ms, not comparable to FULL Graph capacity

- `2026-09-27T01:23:39Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run381`（test）。

- `2026-09-27T01:24:31Z` Run `run381` 记录为 `pass`；正确性为 `pass`。Four-class BF16/FP32 TP8 collective Graph capture/replay passed eight-rank output checks; latest-rank device median0.30057ms, diagnostic only

- `2026-09-27T01:24:57Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run382`（benchmark）。

- `2026-09-27T01:26:41Z` Run `run382` 记录为 `pass`；正确性为 `pass`。Exact 265 ordered HCCL Graph chain checked all8; latest-rank device median3.903ms, no model compute contention or producer arrivals; attained synthetic service only

- `2026-09-27T01:26:54Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run383`（benchmark）。

- `2026-09-27T01:26:54Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run384`（benchmark）。

- `2026-09-27T01:28:46Z` Run `run383` 记录为 `pass`；正确性为 `pass`。Independent exact-order 265-HCCL Graph replay all8 checks pass; latest-rank median3.842ms, isolated synthetic capacity

- `2026-09-27T01:28:46Z` Run `run384` 记录为 `pass`；正确性为 `pass`。Third independent exact-order 265-HCCL Graph replay all8 checks pass; latest-rank median3.925ms, isolated synthetic capacity

- `2026-09-27T01:29:28Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run385`（review）。

- `2026-09-27T01:29:35Z` Run `run385` 记录为 `pass`；正确性为 `not-applicable`。V3.6 joins selected-cycle useful output/current GMM arithmetic, current Markov dependency, conditional HCCL API tensor inventory and repeatable isolated Graph service; finite Product/Hardware/Scheduling ceilings remain null

- `2026-09-27T01:31:33Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run387`（benchmark）。

- `2026-09-27T01:31:59Z` Run `run387` 记录为 `invalid`；正确性为 `not-applicable`。Not executed: Astra High Bound review reprioritized original-path full-cycle device dependency join over another synthetic HCCL buffer variant; script remains a deferred method

- `2026-09-27T01:32:24Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run386`（profile）。

- `2026-09-27T01:39:27Z` Run `run386` 记录为 `invalid`；正确性为 `invalid`。Astra source audit found cycle-index gating and anchor bracket defects; stopped before clients; source SHA restored, health000; no Bound evidence

- `2026-09-27T01:39:28Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run389`（profile）。

- `2026-09-27T01:54:59Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run391`（review）。

- `2026-09-27T01:54:59Z` Run `run391` 记录为 `pass`；正确性为 `not-applicable`。265 tasks mapped to installed ABI and current source; API input115107840B/output145342464B per rank-cycle; compulsory/wire bytes unresolved

- `2026-09-27T02:06:32Z` Run `run389` 记录为 `pass`；正确性为 `pass`。48+12 exact clients; two eligible all8 FULL Graph cohorts, 32 sparse current-stream rank-cycles valid; three source SHA restored, health000; eligibility differs Run375, no E2E comparison

- `2026-09-27T02:06:32Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run390`（review）。

- `2026-09-27T02:06:32Z` Run `run390` 记录为 `pass`；正确性为 `not-applicable`。32 rank-cycle original-path event/acceptance ledger; same-rank begin64->begin65 57.163-65.690ms; finite Bound endpoints remain null

- `2026-09-27T02:10:56Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run393`（review）。

- `2026-09-27T02:10:56Z` Run `run393` 记录为 `pass`；正确性为 `not-applicable`。Run386 stale client overlapped Run389; 120 POST, combined inflight24; Run389/390 c12 timing invalid

- `2026-09-27T02:10:56Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run392`（review）。

- `2026-09-27T02:10:56Z` Run `run392` 记录为 `invalid`；正确性为 `invalid`。V3.7 candidate incorporated Run389/390 contaminated c12 timing; invalidated before model promotion; only Run391 ABI conclusion retained

- `2026-09-27T02:11:25Z` run389 initial pass superseded by Run393: frozen c12 timing INVALID under stale Run386 concurrent client load.

- `2026-09-27T02:11:25Z` run390 initial pass superseded by Run393: frozen c12 timing INVALID under stale Run386 concurrent client load.

- `2026-09-27T02:12:13Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run394`（profile）。

- `2026-09-27T02:12:53Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run395`（review）。

- `2026-09-27T02:19:03Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run397`（review）。

- `2026-09-27T02:19:03Z` Run `run397` 记录为 `pass`；正确性为 `not-applicable`。10-cycle TP8 conditional retained-row packed footprint interval5.675-69.521GB, no compulsory HBM/TPS inference; current GMM1246.614GFLOP/cycle

- `2026-09-27T02:32:10Z` Run `run394` 记录为 `pass`；正确性为 `pass`。Clean original-path sparse event diagnostic: 48+12 exact1024, 60 server POST, five cohorts x eight ranks FULL Graph pass, 40 event and runtime files, source hashes restored, service stopped; timings instrumented only

- `2026-09-27T02:32:15Z` Run `run395` 记录为 `pass`；正确性为 `not-applicable`。Exact60 and source-hash-gated Run394 analysis: 80 selected rank-cycles in five clean cohorts, 40 same-rank cycle64-to65 pairs; current-stream spans only, side-stream and A/A gates remain

- `2026-09-27T02:32:32Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run396`（review）。

- `2026-09-27T02:32:53Z` Run `run396` 记录为 `pass`；正确性为 `not-applicable`。V3.7 clean Run394/395 sparse Current scheduling ledger, Run391 source/ABI current HCCL tensors and Run397 conditional retained-route relaxation; finite Algorithm/Hardware/Scheduling/Product endpoints remain null

- `2026-09-27T02:34:57Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run398`（design-check）。

- `2026-09-27T02:34:57Z` Run `run398` 记录为 `pass`；正确性为 `not-applicable`。Source-pinned fail-closed per-token Target and Draft route capture design to narrow conditional retained-weight numerator; no NPU measurement

- `2026-09-27T02:37:01Z` 暂存知识变化 `PK-036`：Clean sparse current-stream Target/DSpark spans cannot be promoted to all-eight attainable Scheduling or Product bound without explicit side-stream and collective joins, clock calibration and marker A/A control

- `2026-09-27T02:37:01Z` 主控结论为 `INCONCLUSIVE`。Clean Current route, HCCL ABI and sparse phase evidence narrows conditional numerators and observed cadence, but complete compulsory work, mixed attainable capacity and all8 DAG remain missing; no finite Product ceiling or performance KEEP
