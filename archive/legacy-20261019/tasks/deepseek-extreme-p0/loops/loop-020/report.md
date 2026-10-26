# Loop 报告：loop-020

- 结论：`PIVOTED`
- 决定时间：`2026-09-21T04:04:17Z`
- 模式：`optimization`
- 冻结目标是否满足：`yes`
- 可比性：`not-applicable`
- 因果结论：`partially-supported`

## 决定依据

TP0 trace flows causally join all 121038 async launches whose CPU origin is inside a draft_token scope to device X tasks at exact timestamps. Across 170 scopes, device-task union clipped to the host scope is median 51.661 ms versus 52.894 ms host scope, so proposer time is predominantly device-active rather than a removable host-only bubble. Draft acceptance advances only 3.54-3.64 tokens for 7 drafted tokens, but no safe material scheduling intervention is identified; DSpark ACLGraph is explicitly unsupported/eager.

## 证据

- `/data/wio/Inference_Foundry/evidence/20260921_dspark_audit/async_flow_attribution_tp0.json`
- `/data/wio/Inference_Foundry/evidence/20260921_loop020_draft_stage/draft_stage_summary.json`
- `/data/wio/Inference_Foundry/evidence/20260921_dspark_audit/acceptance_profile.json`

## 下一步

Attribute the highest-frequency eager DSpark device chains to exact CPU parents and source callsites; freeze one bounded fusion/reuse candidate only if its non-overlapped contribution exceeds baseline noise.
