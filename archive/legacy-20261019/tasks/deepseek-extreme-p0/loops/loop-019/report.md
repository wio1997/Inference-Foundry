# Loop 报告：loop-019

- 结论：`PIVOTED`
- 决定时间：`2026-09-21T03:14:26Z`
- 模式：`optimization`
- 冻结目标是否满足：`no`
- 可比性：`valid`
- 因果结论：`partially-supported`

## 决定依据

QLI first NPU scalar read accounts for most measured first-builder host time, but the already-tested all-step CPU-max removal passed parity and gave no robust mixed TPS gain (+0.52% within noise, TTFT worse). Metadata op itself is only ~0.4ms/call. No new semantics-safe, high-value decode patch justified; host span is not a removable E2E bound.

## 证据

- `/data/wio/Inference_Foundry/evidence/20260921_loop019_qli_subphase/qli_subphase_summary.json`
- `/data/wio/Inference_Foundry/evidence/20260920_qli_pair_baseline/comparison.json`

## 下一步

Start Loop020: attribute DSpark proposer and target verification device/host critical path in warm mixed c12; quantify acceptance and exposed work before choosing implementation.
