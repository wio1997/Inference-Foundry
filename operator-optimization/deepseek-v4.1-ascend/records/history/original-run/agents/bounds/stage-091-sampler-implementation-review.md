# Stage 091：Ascend 实现的 5+1 形状边界

## 验证范围

ZCode `operator-ceiling169-v2-091-sampler-impl.exit-code` 为 0，verified 回执中模型为 `deepseek-flash`。`sampler-impl-snapshot-status.json` 记录 baseline 与 stage 的镜像 ID 相同、两个目标文件无覆盖挂载、源文件内容相同；本地快照重算 SHA 一致：

- `ascend_sampler.py`：`bb1d39de851540c16c2dd2b16df60c60a1fc4957cedb88bdce9417efc0d55eda`。
- `metadata.py`：`a6696487b3c8823c119f601de81504341bdefefe95bafe673c1f570ab2028da8`。

## 可以证明的单次采样器形状界

`metadata.py:26-27` 将 `metadata.max_spec_len` 设为本批 `num_draft_tokens` 的最大值。`ascend_sampler.py:378-...` 中 `rejection_sample` 的参数包含 `num_draft_tokens` 和 `max_spec_len`，`:467-472` 分配形状 `[batch_size, max_spec_len+1]` 的 INT32 `output_token_ids` 并初始化占位值；函数末尾返回这个张量。`:224-251` 的 `AscendRejectionSampler.forward` 使用这一函数的返回张量构建 `SamplerOutput`。

因此，在**该函数实际被调用、batch 每个请求的 `num_draft_tokens≤5`、不发生越界或对输出再次扩展**的条件下，一次结果中每个请求的张量行最多有 6 个 token 槽。这是张量形状上界，其中一部分可以是占位项；它不要求 6 个均被接受或输出。它同样不规定两次调用之间的时间。

Stage 089 的 patch 源码还显示 worker patch 会将 vLLM 的模块级 `rejection_sample` 名称指向该 Ascend 函数；这只是静态注册链，运行中的 patch 安装及后续覆盖还需对应 worker 启动和运行路径证明。

## 什么时候能够形成固定服务的条件吞吐界

只有在进一步证明以下所有条件时，才能把单次 sampler 的“每行不超过 6 槽”用于每 engine/iteration 的 `g=6`：

1. 本轮实际调用的是上述 sampler，所有 metadata 的 `num_draft_tokens` 不超过冻结配置的 5；启动参数与 metadata 之间的 runner/scheduler 构造和截断关系需要源代码闭环。
2. 一个 DP 的一次计数事件里，每条序列最多有一次上述 sampler 输出；若有多次调用、预填 token、缓存结果或客户端层合并/补发，需逐条纳入或修正分组上限。
3. 本轮 8192 个客户端有效 token 无遗漏无重复地对应到 2 个 DP 的事件中，事件被纳入端到端计时窗；每个事件最多 4 个有效序列的静态配置需与实际调度路径一致。
4. 每个 DP 的连续提交事件相隔至少 `δ`，且 δ 是对所限定执行域的所有合法调度成立的严格正下界；当前没有相应物理或语义证明。

若这些条件全部成立，则计数给出全体事件至少 `ceil(8192/(4×6))=342`，至少一个 DP 有 171 次，进而 `T ≥ 170δ`、`TPS ≤ 8192/(170δ)`。**这是仅限定当前服务与提交协议的条件公式，不是通用理想推理的无条件界。** 现有采样器源码只支持第 1 项中的局部形状关系，尚不能证明这些前提全满足；特别是 `δ` 的数值 UNKNOWN，因此不能报告实际 TPS 上限。

保留的 `oem-and-scheduling-symbolic.md` 给出不代入 `g` 的一般形式。后续检查调度器和 runner 产生 metadata 的代码，以及实时数据到输出的归属；不需要启动 NPU 或改服务实现。
