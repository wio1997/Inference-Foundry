# Loop 报告：loop-006

- 结论：`REJECTED`
- 决定时间：`2026-09-20T17:05:15Z`
- 模式：`optimization`
- 冻结目标是否满足：`no`
- 可比性：`valid`
- 因果结论：`falsified`

## 决定依据

Coupled SP/DSA-CP-off path has no robust E2E benefit: median output TPS -1.61% vs baseline and inside observed noise; TTFT median worsened ~7.8%, TPOT ~0.7%. Functional check passed but numerical equivalence remains unproven.

## 证据

- `/data/wio/Inference_Foundry/evidence/20260920_sp_dsa_off_perf/analysis.json`

## 下一步

Restore FlashComm1 and isolate DSA CP toggle, or profile target/draft host gap if isolate also fails
