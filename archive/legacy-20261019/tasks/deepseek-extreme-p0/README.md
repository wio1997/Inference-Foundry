# DeepSeek Extreme P0

## Task 契约

- Task ID：`deepseek-extreme-p0`
- 算子：`deepseek_v4_flash_w4a8`
- 开发框架：`vllm-ascend`
- 源码根目录：`/data/wio/Inference_Foundry`
- 创建时间：`2026-09-20T15:39:22Z`

## 任务目标

Correct full DP1 TP8 prefill and decode inference with E2E throughput materially above frozen baseline and closer to measured hardware bound

## 允许修改的路径

- `/data/wio/Inference_Foundry`

## 非目标

- SuperKernel or standalone runtime without E2E evidence

## 正确性与性能契约

实现前在此记录已经冻结的算子语义、用例、正确性判据、性能目标、硬件环境和交付边界。

## 恢复入口

执行 `taskctl.py resume --task-dir /data/wio/Inference_Foundry/tasks/deepseek-extreme-p0`，并继续其中记录的下一步动作。
