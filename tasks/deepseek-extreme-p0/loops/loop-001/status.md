# Loop 状态

- Loop：`loop-001`
- 标题：Freeze current DP1 TP8 baseline
- 模式：`baseline`
- 目标用例：`mixed_32k_1024_c12`
- 执行 Skill：`NONE`
- 状态：`RUNNING`
- 结论：`PENDING`
- 下一步：执行 Run run-20260920T154106Z：docker exec dsv4ab python3 /data/wio/Inference_Foundry/scripts/bench.py --dataset /data/wio/vllm_ascend_26/datasets/GSM8K-in32768-num48-DeepSeek-V4-Flash-0731-w4a8-repeatRate0.9.jsonl --out /data/wio/Inference_Foundry/evidence/20260920_baseline/bench48.json --limit 48 --concurrency 12 --max-tokens 1024
- 更新时间：`2026-09-20T15:41:06Z`

## Run 记录

- `runs/mixed_32k_1024_c12/run-20260920T154106Z`

## 阻塞项

- 暂无

## 待归约知识

- 知识变化：`0`
- 基线变化：`0`
