# Loop 报告：loop-007

- 结论：`REJECTED`
- 决定时间：`2026-09-20T17:26:36Z`
- 模式：`optimization`
- 冻结目标是否满足：`no`
- 可比性：`valid`
- 因果结论：`falsified`

## 决定依据

DSA CP off alone regressed full warmed E2E performance: median output TPS -4.70%, TTFT +15.1%, TPOT +3.8%; no >5% gain. Functional gate passed; numerical equivalence not needed for rejected candidate.

## 证据

- `/data/wio/Inference_Foundry/evidence/20260920_dsa_cp_off/analysis.json`

## 下一步

Restore baseline FlashComm1/DSA CP on and profile target, draft, host and cold prefill critical path before another implementation
