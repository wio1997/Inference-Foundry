# Loop 目标：Test FlashComm1-off on TP8

- Loop ID：`loop-004`
- 模式：`optimization`
- 所属 Task：`deepseek-extreme-p0`
- 执行 Skill：`NONE`
- 冻结时间：`2026-09-20T16:32:38Z`

## 目标/假设

FlashComm1 sequence-parallel reduce-scatter and MTP all-gather create removable exposed communication time on this TP8 topology; disabling FlashComm1 improves full E2E throughput

## 允许修改的路径

- `/data/wio/Inference_Foundry/scripts/serve.sh`

## 目标用例

- mixed_32k_1024_c12
- correctness_short

## 接受条件

Golden4 outputs identical and three full warm-cache 48x32K-to-1024 c12 passes improve median output TPS by more than 5 percent over 543.65 without material TTFT or TPOT regression

## 否定条件

Correctness fails or robust E2E TPS does not improve beyond baseline noise, or TTFT/TPOT materially regress

## 固定条件与禁止变更

执行前在此记录固定接口、测量口径和实现边界。
