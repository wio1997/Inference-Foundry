# Stage 095：draft、metadata 与引擎输出的追踪

## 核验

ZCode `operator-ceiling169-v2-095-metadata-path.exit-code` 为 0，verified 回执显示执行模型 `deepseek-flash`。`metadata-path-snapshot-status.json` 记录 baseline 和 stage 容器镜像 ID 一致、目标文件无覆盖挂载、两容器源文件 SHA 一致；本地三个快照哈希已重新核对：

| 文件 | SHA-256 |
|---|---|
| `metadata-path-snapshot/ascend_runner.py` | `568c3049205b9caff5c656a7f5dd4c6e79fb358abe526f89fb9fb3f70a60a43c` |
| `metadata-path-snapshot/vllm_runner.py` | `87c29d08c0bbf66993d8b984811e7325e35e6436242a773feb55ec88f16b2c51` |
| `metadata-path-snapshot/vllm_scheduler.py` | `c1db45f3bbad3a875dd8638331b8870c4380ac1ab4734265ba5c863833a82884` |

以下结论是这些冻结代码的静态行为；没有对运行中的 Python 对象或每个请求执行路径作新的动态采集。

## 从 draft 列表到 sampler

1. `vllm_scheduler.py:263` 从 vLLM 配置读取 `num_spec_tokens`。`:812-827` 从请求中的 `spec_token_ids` 选择要调度的 token，必要时截短；`:1268` 的特定预填分支会根据配置数创建 placeholder。
2. `ascend_runner.py:1640-1675` 对 `scheduler_output.scheduled_spec_decode_tokens` 每个请求取列表长度，作为 `num_draft_tokens`；`:1773-1847` 将这些长度传给 `SpecDecodeMetadata`。没有 speculative tokens 的分支使用普通 sampler，并丢弃未完成 prefill 的无效采样。
3. `sampler-impl-snapshot/metadata.py:26-27` 将 batch 的 `max_spec_len` 设为这些长度的最大值。`ascend_runner.py:2764` 的 `sample_tokens` 在该调用中使用 `_sample` 一次；`:2973-2998` 在有 metadata 时调用 `self.rejection_sampler`，无 metadata 时调用普通 sampler。`:751-767` 在配置 speculative decoding 的相应 rank 上构造 `AscendRejectionSampler`。
4. 已核验的 `sampler-impl-snapshot/ascend_sampler.py:467-472` 将 speculative sampler 的结果分配为 `[batch_size,max_spec_len+1]`。因此，**如果该次调度的每个 draft 列表长度都不超过 5**，该次 sampler 每条序列最多返回 6 个槽，部分槽可能是占位符。

## 从采样输出到引擎输出

- 冻结配置启用异步调度。`ascend_runner.py:2926-2969` 将 sampler 输出关联至 `ModelRunnerOutput`，异步分支包装成 `AsyncGPUModelRunnerOutput`。
- `vllm_runner.py:345-372` 中 `get_output` 将采样张量搬到 CPU，单槽直接转列表，多槽调用 `RejectionSampler.parse_output`。`sampler-source-snapshot/sampler.py:268-307` 对多槽输出过滤占位符、越界词和被丢弃请求；该解析只筛选已有槽，不新增 token。
- `vllm_scheduler.py:1974-2036` 从对应 request 的 `sampled_token_ids` 取本步生成列表；`:2348-2365` 逐 token 更新 request，stop 时截断；`:2152-2162` 将结果放进 `EngineCoreOutput.new_token_ids`。这证明了**引擎内部**一条静态路径：sampler 输出槽 → 过滤/截断 → EngineCoreOutput，不证明 HTTP 客户端最终计数与每次引擎输出的对应关系。

## 剩余必要证据

1. **draft 长度上限**：启动配置和日志表明 `num_speculative_tokens=5`，但 `vllm_scheduler.py:2404-2424` 从 draft 结果更新 `request.spec_token_ids`；异步输出更新 `:2425-2459` 还会截断、按既定 placeholder 长度补位。另有可选的动态每 batch K (`:272-277,1392-1396`)。必须闭合 DSpark proposer 输出宽度、动态配置上限、所有异步 placeholder/更新入口及对应 runtime 路径，才能声称本次全部 metadata 的每行 draft 长度≤5；当前不能仅凭启动参数断言。
2. **客户端输出和迭代归属**：需要追踪 `EngineCoreOutput.new_token_ids` 经前端输出处理、流式响应及 benchmark 的 token 计数，证明每个 8192 个有效 token 恰好归属到指定计时窗内某个 DP 的一次事件。多次事件可合并为一个网络 chunk，不能把 chunk 数当 sampler 次数。
3. **时间界**：即便确认每步每序列输出≤6、每 DP 迭代最多 4 序列，也只能得到事件数量的约束；推导吞吐上界还需要证明同 DP 相邻有效事件的严格正最短间隔 `δ`。`δ` 不能从一次 trace 的最小观测间隔当作物理下界。

因此目前受限的 5+1 **单次 sampler 形状界有条件成立**；应用于全部服务事件、数字 TPS 上界及更宽融合/重排实现域仍为 UNKNOWN。下一轮只需针对上述缺口查具体 proposer、engine 前端和 benchmark 路径，无需重复宽泛搜索。本轮没有新设备工作或目标代码修改。
