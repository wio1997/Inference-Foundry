# 第17代：pending 接纳数的条件证明与证据边界

阶段完成，待 root 验收；整体未完成。只做离线源码和已保存记录审计，没有新模型/服务/NPU/SDK 工作，没有实施 Q06，也没有性能或质量等价结论。

重新读取 gen16 四个真实 SG rank 的原始 `records.json`，逐文件 SHA 与原 collector closure 一致。按进程启动身份、显式 launch、result 对象身份及 RID/pool epoch 关联 submit、preplan、原结果处理前后；没有用时间、相邻序号或数量猜关联。每个输入的完整选定语义记录 errors=0、dropped=0。序号只检查既有显式关联的因果顺序。

| 事后主机记录 | 每 rank | 四 rank 观察总数 |
|---|---:|---:|
| 成功关联的结果处理调用 | 42 | 168 |
| 输出列表长度增加 1 | 40 | 160 |
| 输出列表长度增加 0 | 2 | 8 |
| preplan 时 C=L−1 且未结束，之后实际增加 1 | 2 | 8 |

八个零增量都在处理前已经 finished、pool slot/generation 已释放，且 preplan 不再满足完整 allocation identity。不能填回释放后的 slot/generation。八个预算模式分别是每 rank 的 launch8（7→8）和 launch41（31→32）；另有 launch9/42 处理 finished 请求而不增加输出。这些是观察器自己的显式 launch，不冒充原 native step，也不将两个 engine RID 按数量/时间归给外部 API 请求。这里只证明 host 列表长度变化，不证明具体 token ID、step→token 设备因果或去掉 forward 后的输出等价。

冻结源码给出一个更窄的条件结论：成功进入并完成普通非 spec decode 处理器、存在对应 token row、不走 beam/finished/retracted skip、无 sampling-mask abort 时，`_normalize_decode_outputs` 把该行单个 ID 包为长度1的列表；处理器先 extend 输出，再调用 finish-state 方法。因此该条件路径接纳数为1。EOS/length 的结束判断位于 extend 之后，不能把未知 EOS 值当成零接纳的理由。但采样掩码拒绝分支明确令 `new_accept_len=0`，而 stash 返回一次并不排除它。

执行了冻结的完整 `process_batch_result_decode` 与 `_normalize_decode_outputs` AST，共128个 CPU stub 分支组合。finish/cache/stream/device/relay/spec resolution 是明确 stub；没有导入 torch 或服务。普通路径与 mask-abort 路径可以具有相同的 gen16 已记录标量投影，而主机输出列表实际增量分别为1和0。这是构造的源码反例，未观测到真实 mask abort 或部署错误。另7个输入破坏用例验证审计器拒绝错 result、错 launch、错 RID/进程、虚构 post-free generation、丢记录及篡改 pending 副本。

所以 gen16 不能提供规划前可用的正下界。缺少的事实包括实际处理器 class/function/source 的加载绑定、batch mode/spec、token-row coverage、beam/retracted/mask 分支及其在处理前的不变性。gen16 四份 loader 回执绑定了 scheduler 源码；没有为 batch-result processor 单独留下加载 SHA。本轮只验证它的冻结 manifest 和先前文件来源，不把文件存在升级成实际执行绑定。`pending_token_lower_bound` 仍 unknown，所有决策 fallback。copy 完成、GDN/cache 内容与引用生命周期、slot 复用、干预后的 TP 一致性、精确输出/greedy 等价也仍未知。事后观察增加1不能填充这些证明义务。

仅观察一次原 native pipeline：rank 状态95份 verified_archived，但同一状态文件的 submitted94/inflight4/pending106 尚不一致；记录原样保留，不把 submitted 当 completed，也不接受最终 archive SHA。assembler/sharder/autostage 仍在各自等待条件。原 SDK gen82 已 terminal、自有容器停止、slot resumed；没有新借用。原8卡归原 supervisor 所有，未释放；无外部进程被终止。精确外部身份及日志沿用并核对记录在 external-jobs-receipt.json。

本补充尚未写入 gen9 预览或实际204地图；需和 gen8–16 一起显式 overlay/repackage/browser QA。未分配 R，未发布，未标记优化采用或整体完成。源码范围、全文件/摘录 SHA、每个真实结果调用的计数、CPU反例与失败记录都在本代独立目录，证据包不含 transport/key。

下一步先看 native204 是否达到 final-assembly/COMPLETION_GATES.md；达到后优先做实际输入/导出/审计/归档 SHA 闭合及最终地图。未达到时，独立准备 gen8–17 补充的统一显式 overlay/索引与预览 QA，保留 pending 正下界与运行安全未知；不要为了再次看到相同尾计数而重跑 canary 或矩阵。
