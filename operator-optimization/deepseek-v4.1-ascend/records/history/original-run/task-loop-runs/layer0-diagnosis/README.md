# 第零层 MC2 低活动原因

## Task 契约

- Task ID：`layer0-diagnosis`
- 算子：`DispatchFFNCombine / layer0`
- 开发框架：`vLLM-Ascend 910C`
- 源码根目录：`/Users/wio/work/169-vllm-ascend-reproduce-172/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/sources/vllm-workspace/vllm-ascend`
- 创建时间：`2026-10-28T10:59:40Z`

## 任务目标

按一致指标定位 NPU 低活动模块，尽可能区分调度、访存、计算、通信和数据依赖，保留无法识别的原因。

## 允许修改的路径

- `/Users/wio/work/169-vllm-ascend-reproduce-172/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/agents/diagnosis`

## 非目标

- 不改目标模型、框架或算子；不调参，不启动新 NPU 负载，不访问其他环境

## 正确性与性能契约

实现前在此记录已经冻结的算子语义、用例、正确性判据、性能目标、硬件环境和交付边界。

## 恢复入口

执行 `taskctl.py resume --task-dir /Users/wio/work/169-vllm-ascend-reproduce-172/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/task-loop-runs/layer0-diagnosis`，并继续其中记录的下一步动作。
