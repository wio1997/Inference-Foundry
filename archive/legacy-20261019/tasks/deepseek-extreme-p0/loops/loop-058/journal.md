# 执行日志

- `2026-09-25T16:11:20Z` Loop 已冻结。下一步：Build evidence matrix and replay-calibrated scenario model from existing formal cohorts; independently review assumptions, then choose minimal discriminating calibration

- `2026-09-25T16:12:00Z` 为用例 `mixed_32k_1024_c12` 创建 Run `replay-model-v0`（simulation）。

- `2026-09-25T16:12:48Z` Run `replay-model-v0` 记录为 `pass`；正确性为 `not-applicable`。Formal Run99 three pass durations reproduced exactly; cohort cycles 1217/1212/1206; conditional scenario median TPS 581.19-607.14 engineering and 616.32-681.52 aggressive, neither identified hardware bound

- `2026-09-25T16:14:37Z` 为用例 `mixed_32k_1024_c12` 创建 Run `tp8-collective-chain`（benchmark）。

- `2026-09-25T16:15:51Z` Run `tp8-collective-chain` 记录为 `pass`；正确性为 `not-applicable`。Exit 0; TP8 forty repeated BF16 49152-element plus FP32 3072-element all-gather pairs without per-call barriers: latest-rank chain median 21.420 ms, 0.53549 ms/pair, range 20.913-23.647 ms across 30 repeats; capacity only, no product E2E saving

- `2026-09-25T16:16:02Z` 为用例 `mixed_32k_1024_c12` 创建 Run `formal-wave-residual-audit`（design-check）。

- `2026-09-25T16:16:40Z` Run `formal-wave-residual-audit` 记录为 `pass`；正确性为 `not-applicable`。Formal Run99 wave client-minus-rank0-decode window rises from 2.545-3.148 s in run1 to 3.848-4.450 s in run2 and 3.996-4.521 s in run3; eight-rank decode wall spread <=0.019 s, so later E2E variation is outside measured runtime decode wall, but difference is not pure prefill

- `2026-09-25T16:18:23Z` 暂存知识变化 `loop058-formal-wave-residual`：Across Run99 three formal passes, rank0 decode wall totals 69.050/69.457/69.040 seconds while E2E is 80.188/86.600/85.978 seconds; per-wave client-minus-decode windows vary 2.545-4.521 seconds and eight-rank decode wall spread is at most 0.019 seconds. The window difference is not isolated prefill or removable idle.

- `2026-09-25T16:18:23Z` 暂存知识变化 `loop058-tp8-pair-capacity`：No-service eight-rank forty-pair BF16 49152 plus FP32 3072 all-gather chain measured latest-rank median 21.420 ms or 0.53549 ms/pair over thirty repeats, without per-call barriers; this is capacity calibration, not product E2E savings.

- `2026-09-25T16:18:33Z` 主控结论为 `PIVOTED`。Executable V0 replay closes Run99 formal samples exactly and preserves trajectory; TP8 chain and wave audit constrain communication capacity and locate E2E variability, but scenario savings and true hardware bound remain unvalidated
