# Loop 目标：Isolate DSA CP off with FlashComm1 on

- Loop ID：`loop-007`
- 模式：`optimization`
- 所属 Task：`deepseek-extreme-p0`
- 执行 Skill：`NONE`
- 冻结时间：`2026-09-20T17:06:14Z`

## 目标/假设

The coupled-path no-gain result may hide a DSA CP tradeoff; with FlashComm1 kept on, disabling only DSA CP may reduce exposed decode cost or reveal which component offset communication savings

## 允许修改的路径

- `/data/wio/Inference_Foundry/scripts/run_flashcomm_candidate.sh`

## 目标用例

- mixed_32k_1024_c12
- correctness_short

## 接受条件

Functional suite and three full warmed 48x32K-to-1024 c12 passes succeed; median output TPS improves >5 percent over frozen 543.65 tok/s without material TTFT or TPOT regression, pending stronger numerical correctness

## 否定条件

Service or functional suite fails, or median TPS does not exceed baseline by >5 percent after three comparable passes

## 固定条件与禁止变更

执行前在此记录固定接口、测量口径和实现边界。
