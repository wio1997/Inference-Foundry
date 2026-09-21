# Inference Foundry / DeepSeek Extreme

本目录目标是做 DeepSeek Extreme，而不是单纯维护或调优 vLLM-Ascend。

开始工作前优先读取：

1. `MISSION.md`
2. `PROJECT_STATE.md`
3. `PERFORMANCE_MAP.md`
4. `RESULTS.md`

如果文件不存在，可以创建。

## 工作原则

目标固定，优化路线自由。

最终优化对象是 DeepSeek V4 W4A8 在目标 Ascend 硬件上的完整端到端执行，而不是某个框架或某个算子。

vLLM-Ascend 当前只是：

* correctness golden；
* performance baseline；
* serving/runtime scaffold。

不要默认最终 Hot Path 必须留在 vLLM-Ascend。

所有性能判断以源码、profiling 和真实 benchmark 为依据。

理论分析用于提出假设，benchmark 用于裁决。

允许推翻已有假设和改变优化方向。

不要为了完成既定计划继续做低价值优化。

每一轮优先寻找：

> 当前执行时间与硬件可实现性能下界之间最大的可消除 Gap。

## 优化循环

持续执行：

Observe
→ 更新 Performance Map
→ 找最大 Gap
→ 形成候选假设
→ 选择当前价值最高的候选
→ 调查或实现
→ correctness
→ benchmark
→ KEEP / REJECT
→ 更新状态
→ 继续下一轮

完成一个实验后不要默认停止。

如果仍有明确的高价值性能空间，自动进入下一轮。

## 模型使用

优先把低价值、高确定性工作交给低成本模型。

### Zcode / DeepSeek

优先处理：

* grep/find
* Git
* 编译
* 命令执行
* benchmark
* profiling
* 日志提取
* 数据整理
* 简单 patch

### Astra 6 Light

优先处理：

* 大日志压缩
* 局部源码摘要
* 文件筛选
* 简单调用关系
* profiling 摘要

### Astra 6 Medium

优先处理：

* 性能问题调查
* 完整调用链
* HBM/数据流
* 通信/同步
* Graph 边界
* profiling 根因
* 优化候选分析

### GPT-5.6 Medium

优先处理：

* 优化方案设计
* Kernel/SuperKernel 实现
* Ascend C 实现
* Graph 改造
* Runtime 修改
* 跨模块但范围明确的代码修改
* 性能模型

### GPT-5.6 High

仅在高价值、高不确定性问题上使用，例如：

* 主要瓶颈无法判断；
* 重大架构路线选择；
* 局部收益无法传递到 E2E；
* Framework Gap 与 Compute Gap 难以区分；
* 是否应该脱离 vLLM-Ascend Hot Path；
* 是否应该进入 Persistent/MegaKernel/Standalone Runtime；
* 一个错误决定可能浪费大量后续工程时间。

不要让 High 做 grep、编译和日志整理。

这只是成本策略，不是能力限制。如果低成本模型无法可靠解决问题，应主动升级。

## 上下文控制

不要反复读取整个仓库。

默认先读取当前状态文件，再针对当前问题搜索必要源码。

原始 profiling 和日志保存在 `evidence/`，不要反复塞进高级模型上下文。

需要时允许深入原始证据，不能为了省 token 牺牲判断正确性。

## 状态持久化

长期维护：

* `PROJECT_STATE.md`
* `PERFORMANCE_MAP.md`
* `ACHIEVABLE_BOUND.md`
* `RESULTS.md`

保持这些文件简洁。

每完成一轮有意义的调查或实验：

1. 保存必要 evidence；
2. 更新上述状态；
3. 提交 Git commit；
4. 再进入下一轮。

这样任何 Agent 或新会话都可以恢复工作。

## 禁止事项

不要：

* 为了“做 SuperKernel”而做 SuperKernel；
* 为了“脱离 vLLM”而脱离；
* 因为局部 kernel 快就直接认定整体有效；
* 一次修改大量无关变量；
* 用理论收益代替 benchmark；
* 写大量没有决策价值的过程报告。

## 最终裁判

Correctness 必须成立。

性能必须通过真实 benchmark 验证。

最终目标：

> 持续减少人为执行开销，使 DeepSeek Extreme 尽可能逼近目标硬件上的可实现性能下界。

