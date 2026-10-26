# Loop 目标：Remove QLI metadata NPU scalar synchronization

- Loop ID：`loop-011`
- 模式：`optimization`
- 所属 Task：`deepseek-extreme-p0`
- 执行 Skill：`NONE`
- 冻结时间：`2026-09-20T18:35:51Z`

## 目标/假设

Using already computed CPU local query/key maxima for DSA CP QLI metadata removes two NPU item synchronizations per step without changing numeric metadata and improves mixed E2E throughput or cold TTFT

## 允许修改的路径

- `/data/wio/vllm_ascend_26/framework/vllm-ascend/vllm_ascend/attention/context_parallel/dsa_cp.py`
- `/data/wio/Inference_Foundry/patches`
- `/data/wio/Inference_Foundry/evidence`
- `/data/wio/Inference_Foundry/scripts`

## 目标用例

- mixed_32k_1024_c12
- cold_32k_128_c1
- correctness_short

## 接受条件

Runtime parity assertions pass warm/cold, functional and numerical correctness hold, three frozen full-workload passes show robust >5 percent TPS gain or meaningful cold TTFT gain without regression

## 否定条件

CPU/NPU maxima differ, service fails, or same-condition E2E gain is absent/noisy or correctness fails

## 固定条件与禁止变更

执行前在此记录固定接口、测量口径和实现边界。
