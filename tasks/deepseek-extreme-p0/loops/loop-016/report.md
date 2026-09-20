# Loop 报告：loop-016

- 结论：`REJECTED`
- 决定时间：`2026-09-20T22:41:58Z`
- 模式：`optimization`
- 冻结目标是否满足：`no`
- 可比性：`valid`
- 因果结论：`falsified`

## 决定依据

Captured-shape isolated kernel was faster, but actual DSA-CP fast path activated on eight ranks and same-prompt cold TTFT worsened by 70.19ms mean (+2.9642 percent), 0/8 pairs improved, zero prefix hits. No E2E gain.

## 证据

- `/data/wio/Inference_Foundry/evidence/20260920_loop016_direct_scatter/service_ab3/summary.json`

## 下一步

Start Loop017: attribute Compressor ratio4 cold prefill cost and test only a bounded candidate with correctness and same-prompt E2E benchmark.
