# Loop 报告：loop-026

- 结论：`REJECTED`
- 决定时间：`2026-09-21T10:07:04Z`
- 模式：`optimization`
- 冻结目标是否满足：`no`
- 可比性：`valid`
- 因果结论：`falsified`

## 决定依据

Removing one of three trained draft layers saves23.21% proposer model time but destroys proposal quality: advanced tokens fall to1.363/cycle, far below3.15 break-even; output TPS drops55.85% to217.092, close to target-only208.047. Full three-layer semantics are necessary.

## 证据

- `/data/wio/Inference_Foundry/evidence/20260921_loop026_layer_bypass/comparison.json`
- `/data/wio/Inference_Foundry/evidence/20260921_loop026_layer_bypass/acceptance_delta.json`

## 下一步

Preserve all three trained layers and test the major framework gap: fixed-shape DSpark remains eager-only. Establish whether a segmented graph or standalone proposer replay can capture exact three-layer execution and reduce launch/runtime overhead without changing logits.
