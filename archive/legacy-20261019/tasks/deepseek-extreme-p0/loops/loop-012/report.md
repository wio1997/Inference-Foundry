# Loop 报告：loop-012

- 结论：`ACCEPTED`
- 决定时间：`2026-09-20T19:47:44Z`
- 模式：`optimization`
- 冻结目标是否满足：`yes`
- 可比性：`valid`
- 因果结论：`supported`

## 决定依据

Runtime CPU/NPU QLI maxima parity passed on all eight ranks in Loop011. Prefill-only patch passed functional gate. Same 16 cold prompts all improved: 2626.03 to 2362.89 ms mean TTFT (-10.02%) with no prefix cache hits. Full mixed median output TPS 547.55 versus paired original 537.60 is within baseline noise; median TTFT 1143.58 versus 1262.74 ms shows no observed regression.

## 证据

- `/data/wio/Inference_Foundry/evidence/20260920_qli_prefill_only/comparison.json`
- `/data/wio/Inference_Foundry/patches/loop012_qli_prefill_only.patch`

## 下一步

Investigate the warm mixed decode critical path and find an intervention with measurable output TPS gain; retain prefill-only QLI patch.
