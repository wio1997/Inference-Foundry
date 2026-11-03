# 169 vLLM-Ascend 性能排查：当前进度

更新时间：2026-10-28。此文件是正在进行的阶段记录，**任务还没有完成**。历史报告里如果出现“完成”，以此说明和当前 Task 状态为准。

## 你要求做什么

沿用已有 32K 输入、1K 输出、并发 8、约 90% 前缀命中的 vLLM-Ascend 基线，先扫算子融合和重组机会，再分析模块的活动与瓶颈，并建立有理论依据、不可超越的性能上界。未知数据不能编；本阶段只分析，不修改推理代码。GPT 主控，本机 ZCode 的 DeepSeek 执行简单工作。不能影响其他任务，远端只在自己的容器和任务目录内工作。

## 分工与结果

| Task | 当前发现 | 还不能得出的结论 |
|---|---|---|
| 融合扫描 | 已把既有 F01–F31 对照到 40 层主要源码路径，覆盖 attention、MoE、Engram 等；末层又发现 MTP 缓冲这一条件消费者。第 20 层的压缩 KV 按模型角色供第 20–39 层共享。 | 这不是“所有能融合的都已找完”；运行分支、算子内部消费者、数值等价及实际收益还需核实。 |
| 模块排查 | 在同一组可比较的 40 个目标解码层样本中，第 0 层平均采样窗口约 1419.59 微秒，所记录的 AIC MAC 活动代理约 3.640%，分别是该组最长和最低。 | 该数字不是整卡利用率或端到端占比；不能仅凭它认定通信是根因。现有记录缺请求关键路径。 |
| 算子极限 | 已证明一个严格限定的冷态矩阵乘例子和一个有条件的热态缓存容量公式；它们展示如何证明必要搬运量。 | 它们尚不能直接代入本次模型：真实热态必要读量、实际布局和硬件硬上限未证明，目前没有可信的本次数值 TPS 上界。 |

原有无 profiler 基线输出吞吐约 **519.84 token/s**，用于整体参照；带 profiler 的局部采样不能直接当作这项吞吐量的模块分摊。

目前另一个值得追查的是 Indexer：在 384 对同一采样窗口中，它均在同 stream 的 attention 之前结束。第 24、28、32、36 层单次耗时中位数约 493–494 微秒，第 2、8、14 层约 90 微秒；逐轮拆开后，后四层 run 1 中位数约 22 微秒，而 run 2、3 约 495 微秒。输入缓冲区形状相同，但有效长度和候选数没有记录，暂不能解释原因或当作端到端损失。详见 `agents/diagnosis/indexer-attention-findings.md`。

当前目标模式已重新创建并处于 active，四个 Task 都有恢复包；其实际进度以目标工具和各 Task 记录为准。

## 为什么还要继续

第 0 层一个融合 MoE 核心的原始事件在样本中出现 7 组 PMU 数据；它们与 7 次任务执行的时间十分接近。若只用任务 ID 匹配，会出现错误的多对多关联。当前还没有证明 profiler 的时间戳语义，也无法拆出内部通信和计算各花多久。下一轮继续核查数据来源，同时梳理热点融合候选的真实消费者和精度边界。

## 本轮继续排查

- 融合：冻结源码的一个 Indexer 特定分支存在把结果复制回同一存储位置的调用；实际是否产生设备搬运、能否安全删除仍未知。Compressor 的 pooling→归一化及 Engram gate 也是条件候选；旧 F24 把 Engram hash 和 lookup 笼统标为 CPU 的说法已更正。见 `agents/fusion/INDEXER-HELPER-REVIEW.md` 和 `agents/fusion/COMPRESSOR-ENGRAM-REVIEW.md`。
- 性能：在已采集的逐卡主流层窗口里，MC2 时长约占 37%，Indexer 在第二、三轮约占 6.8%；这不是端到端占比。见 `agents/diagnosis/mainstream-occupancy-findings.md`。
- 理论：新增 `task-loop-runs/ideal-scheduler-model`，建立结合分块依赖和资源争用的条件下界。独立审查发现初版公式遗漏非空条件，已修正并保留审查记录；适用于 169 的必要工作和硬件容量仍待证明。见 `agents/scheduler/RESOURCE-WINDOW-BOUND.md`。

## 如何查看和恢复

- 总恢复包：[PARALLEL-RESUME-PACK.json](PARALLEL-RESUME-PACK.json)，总状态：[STATE.json](STATE.json)。
- 四个 Task 位于 `task-loop-runs/fusion-coverage/`、`task-loop-runs/layer0-diagnosis/`、`task-loop-runs/theoretical-ceiling/` 和 `task-loop-runs/ideal-scheduler-model/`；各自的 `resume-pack.json` 是压缩上下文，`loops/` 按轮次保存假设、Run 和结论。
- 融合详细记录：`agents/fusion/LAYER-COVERAGE.md`。模块记录：`agents/diagnosis/module-priority-evidence.md`。理论记录：`agents/bounds/warm_fixed_analysis.md`。
- 不把一次 Run 的通过等同于整个 Task 完成；每一轮都明确记录已证实、条件和未知项。

记录已同步至 169 自有容器映射的 `/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41`，该工作目录位于用户指定的 `/data/tiankuan/wio/main` 下。
