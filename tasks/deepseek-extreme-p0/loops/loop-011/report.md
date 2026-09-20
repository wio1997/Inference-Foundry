# Loop 报告：loop-011

- 结论：`PIVOTED`
- 决定时间：`2026-09-20T19:20:20Z`
- 模式：`optimization`
- 冻结目标是否满足：`no`
- 可比性：`valid`
- 因果结论：`partially-supported`

## 决定依据

All-step QLI CPU-max candidate passed >=192 per-rank parity checks and improved exact same cold prompts 8/8 by mean284.83ms (-10.81%), but paired mixed TPS +0.52% is noise and median mean TTFT worsened11.9%; refine to prefill-only rather than KEEP

## 证据

- `/data/wio/Inference_Foundry/evidence/20260920_qli_pair_baseline/comparison.json`
- `/data/wio/Inference_Foundry/evidence/20260920_qli_pair_baseline/cold/metrics_before.txt`
- `/data/wio/Inference_Foundry/evidence/20260920_qli_pair_baseline/cold/metrics_after.txt`

## 下一步

Loop012: apply CPU QLI maxima only when num_prefills>0; verify parity and matched mixed/cold workloads while leaving decode path original
