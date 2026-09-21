# Loop 报告：loop-017

- 结论：`REJECTED`
- 决定时间：`2026-09-21T01:08:26Z`
- 模式：`optimization`
- 冻结目标是否满足：`no`
- 可比性：`invalid`
- 因果结论：`not-identifiable`

## 决定依据

mBase256 isolated candidate built, but first comparable exact-shape screen never completed after >8 minutes versus stock ~1.514ms/call; runtime/integration/tiling cause cannot be distinguished. No E2E or correctness evidence; operational gate failed.

## 证据

- `/data/wio/Inference_Foundry/evidence/20260920_loop017_compressor/candidate_observation.json`
- `/data/wio/Inference_Foundry/evidence/20260920_loop017_compressor/candidate_run.log`

## 下一步

Start Loop018: attribute warm c12 TP communication critical path and rank skew from saved profile, then select a falsifiable service-level candidate.
