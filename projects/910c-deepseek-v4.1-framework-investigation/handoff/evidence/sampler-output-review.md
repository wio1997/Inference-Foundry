# Sampler 分支与实际输出等待判断

**Goal ACTIVE，loop-014 EVALUATING；全链路未扫完。每个替换原语逐位一致。** 新 NPU 隔离进程 1、服务推理请求 0；只在自有 `/data/tiankuan/wio` 新建 a1/a2 补证目录，未改服务、权重、全局环境。定点读取原 C8 SQLite，未采集新 profile。阅读/补证日期为北京时间2026-11-10，Run与目录沿用UTC 2026-11-09日期。

## 介绍

从 target scores 的真实消费者出发，继续走 sampler/processor/RNG/verify、D2H、输出线程、实际回复 rank 与下轮退役。小 score parent 已低值，不重开 cast/clone/argmax 微方案；新的等待必须按不同交付范围重新判断。

## 到了哪里

[28 份实际源码身份](sampler-execution-identity.json)已核对，其中 27 份与既有本地件一致；缺失的 Ascend `worker/utils.py` 只读取回。源码函数导航、13 组条件族和具体候选见[前沿](sampler-output-frontier.json)。导航范围不是逐行覆盖声明，内建 CANN 算子内部没有借此标为已扫全。

- **真正可进入的处理器**：`build_logitsprocs` 在 speculative config 下只实例化 MinTokens，custom processors 抛错，min_p/logit_bias 给出不生效提示。不能把它们算成当前 DSpark 可省工作。bonus 与 target 自己的 allowed/badword/penalty/MinTokens 顺序不同；prefix history 和 mask 不能直接共享。
- **分数观察者**：raw 先于原地 processor；processed 有独立 logits/logprobs 语义。请求 topk/指定 token 分数或 rank 时，argmax 不足以交付。当前 raw-target clone 即使无请求分数也会做，但整段 budget 已包括它，结束同预算的小变体。
- **MinTokens 与 thinking**：structured-output 行在全被 mask 成 -inf 时，会恢复原有限 stop logits；简单合并 mask 会丢依赖。thinking 状态依据 committed outputs、draft prefix、prev length/countdown/force index 更新，bonus 先更新、target 后使用，不可跨代复用或调换调用。
- **cutoff 与随机数**：Ascend probability sort 的 TopP 基于未经过 TopK 改写的概率；tie 和 underflow 行为要保留。batch-invariant 回到 upstream sorted-logit 实现，并非逐位可交换。bonus、uniform、recovered 的全局生成、seeded override、形状与调用顺序不同；recovered 的 q 是每请求一行，在 K 个位置间共享。不能改成每位置新 q 或用统计等价采样替换。
- **验证分支**：greedy/K1/一般 K0、普通 random、entropy、block 及 PyTorch fallback 已走到实际实现。普通 verifier 首拒即停；block 会扫描完整 K、顺序累计并截断 pi、累计 uniform 后取最后合法位置，不能套首拒规则。recovered Triton 4096 分块及严格 `>` 跨块选择决定 tie 顺序。fallback 对 q=0/isinf 有特殊处理，不能视为与 Triton 逐位等价。synthetic+block 明确报错；synthetic 改接受语义，不能作性能替换。
- **共享计算的真实边界**：prompt mask/输出 prefix integer histogram 可作为有 penalty 时的大 producer，lazy recovery 可在保留完整 RNG 推进和原 verify 决策后只算最终所需恢复行；当前固定 greedy/no-penalty 调用没有这些工作，不为它们另测。histogram 与 logits 的跨 CTA 消费必须有 join，-1 先按原规则转 vocab padding。

随后查到不同 parent：`sample_tokens` 先提交 DSpark，再创建 `AsyncGPUModelRunnerOutput`，后者的 `wait_stream(default)` 包含整段 draft。输出 CPU IDs 其实早已由 target sampler 生产，DSpark 的 next-token helper clone 原 sampled IDs 后处理 discard，未改写原输出存储。

仅源码顺序仍不足以定价。本轮[定点 SQL 回执](sampler-output-handoff-receipt.json)绑定原 joint168 的 16 个数据库身份、4 个 eager 波次与各 rank 一个 B4 graph：

| 实际交付事实 | 证据 |
|---|---|
| 72 个 native 窗口均有唯一 late side-stream `MEMCPY_ASYNC→EVENT RECORD`，在第五次 Markov 输出之后 | task ID/stream/connection/API 全部保留于压缩原件 |
| 实际客户端回复是每 DP 的 TP0，即归档 rank 0/8 | Multiproc `unique_reply_rank`、`_get_output_rank` 与 Ascend patch 继承关系；9 个窗口有对应非主线程 synchronize |
| 63 个非回复窗口没有对应输出线程等待 | 不把所有 16 rank 的最大等待当客户端收益 |
| 9 个输出等待的结束与 late event 相差最大 35.025µs | 仍缺 event handle/copy byte count，按源码、流、顺序、唯一性和时间吻合匹配；保留 CPU/device offset 敏感性 |

原始 SQL 行保存在 `sampler-output-handoff-raw.json.zlib`；均为旧 profile 的只读查询，前后文件 size/mtime 不变。首个 schema 查询有 SQL 引号错误，修正后重新只读核对，不涉及 NPU 试验或覆盖失败结果。

## 结果

| 范围 | 状态 | 判断 |
|---|---|---|
| **实际回复 rank 提前 D2H，连同下一输入代次所有权** | **待确认，值得深入首 token 路线** | 已有真实消费者、72/9绑定、付原CPU返回/拷贝代价的条件净价。native证明不能依赖pinned源自动快照，必须扩大到staging lease/读完成事件；[代次协议设计](early-output-generation-protocol.md)。 |
| 仅前移D2H、保持原staging复用的直接补丁 | **已停止（此具体配方）** | [保留的设计记录](sampler-early-output-copy.patch)虽可解析且只改sample_tokens，但没有建立CPU下一代写与旧H2D读的关系，不能准入。 |
| 所有分支无条件提前 output，并把 data-ready 当所有工作完成 | **已停止（此具体配方）** | async KV 的 defer-free 会把 `update_from_output` 当 completed writes；EC/hybrid/EPLB/debugger 等也有后续状态。必须 fallback 或另有真正 completion fence；不能拿较早 copy event 退役这些代次。 |
| 独立 greedy score clone/cast/argmax/reject 小方案 | **已停止（原独立 1% budget）** | 前轮整个 head+verify 含 GEMM 全免费，最高个体响应仅 0.58204%、六普通情景 0.52940%。新的输出等待在该 parent 之外，不重开其低值子项。 |
| 固定负载中的 random lazy-recovery、penalty prefix/cache 变体 | **已停止（当前不执行）** | 没有当前可省工作；已保留合法条件、数学/状态边界，负载改变才重开，不采随机数或命中率证据。 |

提前 D2H 的价格**保留 target/sample readiness、原 CPU 输出线程到达、原 copy+record 时长、全部 draft 计算及提交**，没有擦除长达 80–100ms 的整个 event wait，也没有把每轮收益累计到客户端：

| 首次回复 | 保留上述代价后的条件净省 | 原 TTFT 条件比例 |
|---|---:|---:|
| 最早短请求 slot 7 | 15.239ms | 3.395% |
| 同批 slot 0/6 | 7.883ms | 0.699% / 0.702% |
| 同批 slot 2/3 | 7.332ms | 0.924% / 0.930% |
| 后到 slot 1/4 | 8.016ms | 0.539% / 0.537% |
| 后到 slot 5 | 10.089ms | 0.690% |

整批 P90 TTFT 条件净省 8.016ms（0.5373%）。最短请求额外付 1ms 不重叠代价后，仍约 3.172%；P90 同样付 1ms 后约 0.4703%。这是**有明确成本扣除的原时序敏感性**，不是服务实测加速，也不是整个模型物理上界。单个 B4 graph 的同机制约 4.68–4.71ms，只作形状参考；未据此外推六普通请求、吞吐或 response end。首次 ID 对首次非空客户端内容的传播仍需原负载 A/B 确认。

原直接补丁只改 `NPUModelRunner.sample_tokens` 且通过AST解析，**现已停止，不能应用**。它虽限定 PP1/DSpark/TP0、async greedy、无 history/请求分数/NaN、无 KV/EC、无 hybrid/EPLB/debugger；其它分支保留原路径。仍只避开部分外部异步退役语义，未保护CPU staging；不能据此宣布cache生命周期或实现已通过。

决定性补证见 [a2原件](native-pinned-h2d-a2/evidence.json)：四例在CPU覆写前后gate均pending；shared INT32[24]/INT64[16]分别24/16元素变成新值，独立代次控制均0字节差异。Torch2.10.0+cpu / torch_npu2.10.0.post7 / Ascend910_9382，匹配当前16 worker队列设置，health前后200。a1在启动Torch/NPU前因进程筛选含辅助进程退出，error完整保留；修正筛选和unset环境传播后a2独立运行。此结果只否定自动快照假设，未在未修改服务触发race、未作性能或完整模型验收。

当前 Ascend block zero 与 CoW 使用 current-stream 操作；CoW 先 snapshot 所有源页再写目的页，不能逐页替换而破坏重叠 source/destination。下一步继续核其 segment/alias/graph buffer 与 default-stream 代次，重点验 EOS/abort、同 ID 重入、新请求复用和 DP 变空。实际运行资格、可能的 DMA/host 争用、六普通及客户端首次非空传播是此路线的具体缺口。没有新增“已确认可实现”的 native 项。

全图继续：随后合并定价剩余 metadata/host submission 大交付、末层 attention/FFN row-demand producer；不围着已低值的 score 或当前 inactive random 分支补小证据。
