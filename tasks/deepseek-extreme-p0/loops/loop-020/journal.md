# 执行日志

- `2026-09-21T03:16:08Z` Loop 已冻结。下一步：Inspect saved c12 host scopes, per-step DSpark acceptance counters and device event alignment; do not infer removable time from inclusive scopes.

- `2026-09-21T03:21:18Z` 为用例 `mixed_32k_1024_c12` 创建 Run `saved-c12-draft-scope-device-overlap-20260921`（profile）。

- `2026-09-21T03:21:18Z` Run `saved-c12-draft-scope-device-overlap-20260921` 记录为 `pass`；正确性为 `not-applicable`。TP0 saved c12: 170 draft_token host scopes median52.894ms/sum9.036s; concurrent device union median41.860ms/sum7.352s, compute median37.641ms, HCCL median5.272ms. Temporal co-occurrence only; async target tasks may spill into draft scope, so no removable framework time inferred.

- `2026-09-21T03:23:21Z` 为用例 `mixed_32k_1024_c12` 创建 Run `active-dspark-draft-stage-20260921`（profile）。

- `2026-09-21T03:35:55Z` Run `active-dspark-draft-stage-20260921` 记录为 `pass`；正确性为 `pass`。DP1TP8 no-profiler DSpark diagnostic passed short functional and12/12 c12x512. 160 no-prefill proposer steps/rank, graph disabled. _propose median36.11-41.38ms/rank: run_draft28.43-32.69ms, first attention metadata5.67-6.41ms, set_inputs1.47-1.66ms. Inclusive host spans only; no E2E candidate.
