# Loop 目标：Compressor ratio4 cold prefill critical path

- Loop ID：`loop-017`
- 模式：`optimization`
- 所属 Task：`deepseek-extreme-p0`
- 执行 Skill：`NONE`
- 冻结时间：`2026-09-20T22:42:42Z`

## 目标/假设

Compressor ratio4 costs material cold prefill critical-path time that a bounded change can reduce without harming correctness or mixed throughput.

## 允许修改的路径

- `vllm_ascend`

## 目标用例

- cold_32k_128_c1

## 接受条件

Attribute prefill-only Compressor cost on frozen case; implement smallest safe candidate, pass correctness, and improve paired cold TTFT beyond noise without mixed throughput regression.

## 否定条件

Compressor work is outside prefill critical path, change harms correctness, or paired E2E gain is absent.

## 固定条件与禁止变更

执行前在此记录固定接口、测量口径和实现边界。
