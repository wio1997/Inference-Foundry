# Loop 目标：Remove per-cycle target metadata synchronization

- Loop ID：`loop-036`
- 模式：`optimization`
- 所属 Task：`deepseek-extreme-p0`
- 执行 Skill：`NONE`
- 冻结时间：`2026-09-24T10:53:45Z`

## 目标/假设

The fixed c12 target metadata updater performs a per-cycle NPU-to-host max().item synchronization to choose max_local_seq_len, contributing materially to the measured 8.673 ms metadata stage. A conservative cohort-fixed bound can remove this synchronization while preserving exact target metadata and long-run KV/DSpark semantics.

## 允许修改的路径

- `/data/wio/Inference_Foundry/runtime`
- `/data/wio/Inference_Foundry/bootstrap`
- `/data/wio/Inference_Foundry/diagnostics`
- `/data/wio/Inference_Foundry/scripts`

## 目标用例

- mixed_32k_1024_c12

## 接受条件

Under the frozen DP1xTP8 W4A8 DSpark7 c12 contract, exact native target metadata parity and continuous state/acceptance gates pass on all eight ranks through at least 256 live cycles; matched event timing shows a material metadata-stage reduction, and a formal same-mouth 48x32K-to-1024 E2E comparison is run only after correctness.

## 否定条件

A conservative bound changes metadata values or target outputs beyond Stock self-replay noise, increases device work enough to erase the stage gain, or profile proves the per-cycle sync is not material; pivot to the 47.702 ms target stage.

## 固定条件与禁止变更

执行前在此记录固定接口、测量口径和实现边界。
