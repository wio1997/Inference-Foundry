# 独立 Challenger Review — 2026-10-06

结论：**路线 KEEP，仅限一次有界、纯观测的 H1 区分诊断；性能实现 STOP，尚不能选择 drain-first。** 没有证据证明 H1 是最大性能 Gap，也没有证据支持 18ms/token 设备空泡。Current=None、本阶段没有新增代码级性能 KEEP。若今日重新开始，我仍先做这一个诊断，因为问题小、可被明确否定；不会继承 cap 扫描或据此选择 PD/Full Replica 架构分叉。

## 范围、身份与证据等级

本审查只读源码/规则并写本文件，未操作服务或设备、未跑性能实验。Git 分支 glm5-3-autonomous-20261001，HEAD b23845e4411632f0369a4ead7472b5f913f3ff6a，规则 GLM-RESEARCH-RULES-v2。亲读根/GLM AGENTS、MISSION、模型策略与 PLAN 相关条款；亲读此目录 EngineCore、executor、async scheduler、native scheduler、native/Ascend V2 runner、PP/async utilities、worker、v3/v5 scheduler 与 batch_queue_control。

对上述 source_identity.json 中十个关键源码快照重新计算 SHA，全部匹配；EngineCore SHA 为 6fdd067f54e5d42c57ff413e292685f5bdf6f343498d85c9262567f1eb746916，匹配本地控制插件校验值。额外亲读主 Agent 补取的 ascend_patch_spec_pp.py，SHA 704b4d52f8e8d8375cb1a83ae9d32aa9b78d591898970d41b4fa7e91e7433eed；审查时它尚未列入 source_identity.json，实际安装来源由主 Agent 提取说明支持，应补到来源清单。没有亲读 Run230 远端 raw 或现场 trace；历史数字仅为包内报告，不作为本审查已独立核验事实。

## 已核源码事实

1. v1_engine_core.py:652–687 先 schedule/execute/sample，再决定是否立即返回；696–714 才消费 FIFO 最老结果、处理 abort、更新 scheduler。存在推迟消费的控制顺序，但顺序本身不证明成本大小或负面效果，enqueue 也可能覆盖后续有效执行。
2. native_multiproc_executor.py:70–100 的 FutureWrapper 只有 result() 才按 RPC FIFO 调用 get_response 并置完成；done()==False 与设备/MQ readiness 无对应关系。354–416 的一个 result 可先排空此前 execute RPC。诊断必须标注每个 RPC，不能把整个 sample future.result 时间视为 sample 本身等待。
3. native_async_utils.py:AsyncOutput 在 copy_event 后才能读取正确 sampled tokens；native_model_runner.py:1528–1599 在创建 AsyncOutput 后才 postprocess/propose。Ascend 包装在 super.sample_tokens 返回后调用 broadcast_draft_tokens（ascend_model_runner.py:218–225）。ascend_patch_spec_pp.py:58–87 将 sampled broadcast 暂存，最终把 accepted 与 next-draft 合在一个 payload 发布。因此 **copy-ready、worker RPC 可回复、PP payload 发布、下一 Decode 所需状态就绪是不同事件**；发布调用完成也不等于通信设备完成。
4. native_async_scheduler.py 设置 next_decode_eligible_step=current_step+pp_size；native_scheduler.py:504 依据该计数限制同请求 Decode。占位符允许结果尚未 CPU commit 时继续原生调度。commit 并非每次 Decode 的先决条件。update_from_output 的 rejection rollback、stop/abort、placeholder/KV 更新仍有因果作用，需逐请求判定。
5. native_model_runner.py:1219–1228 在零 token 提前返回前仍调用 update_pp_decode_requests；native_pp_utils.py 通过 FIFO slot 与 main_stream.wait_event 消费结果并按 generation 过滤释放请求。**empty CPU step 仍可能推进必要 PP slot/state，不能删除或认成纯浪费。**
6. ascend_model_runner.py:842–905 在 MTP rejection 后 D2H num_computed_tokens，并在后续 metadata 消费处 synchronize。此 host 等待也是正确性依赖，不能直接计入设备 idle。v5 cadence 的 ready_decode predicate 只为保守候选，不是 KV/设备可执行性证明。

## 最大 Gap 排名（诊断价值排序，不是已测性能大小）

| 排名 | 可消除 Gap 问题 | 事实 / 推断 / unknown |
|---|---|---|
| 1 | 已可安全按 FIFO 获取的最老输出，被额外 schedule/RPC enqueue 延后 commit，是否延后有效 SSE 或下一有效 Decode | 控制顺序为事实；可能有 host 延迟为推断；频率、关键路径时间、可消除量、产品影响全部 unknown |
| 2 | MTP postprocess/draft→PP payload 与 metadata D2H 消费是否形成额外串行 | 消费链为事实；必要工作之外是否能重叠为 unknown；不是新 active 假说 |
| 3 | 组批/Prefill竞争/Graph与必要工作量变化是否解释历史 cap 方向 | 可改变运行轨迹是合理推断；现有材料无法排名或隔离，unknown；不激活参数/架构探索 |

无法给出“最大已证实可消除 Gap”。产品合同相对上限差距、稳定容量及两副本整体关键路径也 unknown。第 1 项胜在诊断便宜且可证伪，不是估计收益更大。

## Top 3 代码问题（非并行候选队列）

1. EngineCore 的额外 schedule/enqueue 发生时，FIFO 全部前置 RPC 是否已能安全读取；该延迟是否沿 scheduler/output_queue/SSE 或请求 Decode 依赖传播？这才是 H1。
2. 下一次 Decode 的真实门槛是 CPU commit，还是 PP accepted+draft slot、MTP state/seq_lens 或既定 PP cadence？必须观察最终 Ascend patch 后的消费者，不能以 upstream sampled broadcast 位置作结论。
3. 若未来 H1 阳性，怎样提供不取走回复、不改变 FIFO/error/aggregate 语义的真实 readiness，并只提交一次仍保留 enqueue overlap？先证明接口可实现及 abort、stale、grammar deferred、KV、跨 rank 行为，再写性能 patch。FutureWrapper.done() 不满足此条件。

## 最小区分诊断与决策表

固定现役 cap/resources/cadence/MTP 等条件，只选择一个短窗口，保留正常语义；使用有界缓冲观测，避免逐 token 同步打印和新增设备 synchronize。每条记录带 process/rank、request、scheduler_step、execute/sample RPC 序号、FIFO batch 序号和 slot generation。关联：schedule 入口/出口、RPC enqueue、worker sample 返回、copy_event 原有 synchronize 返回、get_output 返回、reply enqueue 完成、core 各 RPC dequeue 开始/返回、commit 开始/结束、输出入队与有效 SSE。PP draft 发布、消费者 wait_event 及 metadata 同步需另标，不能替代 copy-ready。跨进程/主机时钟给误差范围，先用同机单调时钟与因果序号。

`copy_event.synchronize` 返回是观察到已 ready 的保守时间上界，不是真实完成瞬间；reply enqueue 返回同样只证明“此时已发布”。若该上界已早于后续 schedule 入口且超出时钟误差，就可证明该次 FIFO reply 在 schedule 前已发布；反过来，上界较晚不能单独否定提前 ready。必须覆盖该 future 的所有先行 RPC/所需 reply ranks，避免 output_rank 单点伪阳性。

| 观察 | 裁决 |
|---|---|
| 前置 FIFO replies 已发布，额外 schedule 确有延迟，并传到有效 SSE；或逐请求证明 commit 阻断可执行 Decode | H1 阳性，评估延迟上界和频率，才允许一个有条件代码候选；这仍不是 PERF_KEEP |
| commit 提前只能让 host 较早拿到输出，PP/MTP 已独立推进且 SSE 关键链无收益 | STOP H1 产品收益主张，不以 host 微指标继续 |
| 可靠证明回复在该 schedule 窗口内尚未 ready，或可消除延迟不足以改变产品裁决 | STOP H1，依据已测关键路径再 PIVOT |
| 只有同步返回时间、缺 RPC/rank 完整性或 SSE/设备关联 | INCONCLUSIVE；不能冒充阴性，也不能进入大型 E2E |

如果只证明 SSE 延迟，允许称用户输出路径证据，不宣称 Decode 加速或设备空泡；若主张 Decode/吞吐收益，必须关联实际 target/draft/PP设备区间及等待依赖。一次纯观测能支持/否定必要条件，无法单独证明重排后的净收益；阳性后仍需 correctness、matched A/B、可重复完整 E2E。诊断本身未运行。

## Stop List 与最易误读证据

- 停止预先 drain-first、done() guard、cap/budget/cadence 扫描、删除 empty steps、绕过 draft/PP/metadata 同步、同时激活 H2 或新 Runtime 架构。
- 停止把 CPU future 等待、模型调用数、CPU empty/execute 比例和 PP scheduler current_step 当设备时间或利用率；调用“non_block”也不保证广播 enqueue 没有 host 成本。
- 停止把 cap3→cap2 历史差异或 231/234/235 TPS 当 matched code gain；包内指出 token-vector/salt/order/MTP 混杂，本审查未独立核验 raw。
- 最容易误读的是 **copy-ready 已早、所以设备等 core commit**。该实现特意让 D2H 与 postprocess/draft overlap，并延后 PP accepted+draft 发布；copy-ready 可能发生在必要后续工作尚进行时。其次是把 native empty call 当无工作：它仍推进 PP slot。

接受包内“先一个有界 H1 诊断、未知保持未知”的方向；拒绝把 H1 称为已确定最大 Gap，也拒绝将设备关联缺失一概解释为无任何用户输出价值。Review 是路线建议，不是性能 KEEP 或服务变更授权。
