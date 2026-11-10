# Target 提示词分数与分块回传

## 介绍

从 OpenAI completions 的有效请求参数追到 target 隐藏行、词表、分块写回、输出归并和请求行回收。搜索的最大范围是隐藏行生产到客户端可见分数的完整交付。冻结环境为 910C、CANN 9.2.0、TP8/DP2/EP16/PP1、DSpark 五步和异步调度。每个替换原语必须逐位一致。

## 到了哪里

| 交付 | 已走的实际路径及条件 |
| --- | --- |
| 请求到 prefix cache | `CompletionRequest.to_sampling_params` 359–361 接收显式 `prompt_logprobs`，或从 `echo` 加 `logprobs` 推导。528–539 拒绝显式正 K 或 −1 配 stream，允许 0。`SamplingParams` 543–547、634–642、839–841、889–899、963–966 定义默认绕过 prefix cache 读取、合法 K、服务上限以及与 trace decode 互斥。`Request.get_skip_reading_prefix_cache` → `KVCacheManager` 252 控制复用。冻结配置快照为 `raw_logprobs`、上限 20，服务拒绝全词表 −1。放开上限时 worker 将 −1 展开为 V。 |
| 真实 caller 与状态 | `NPUModelRunner.sample_tokens` 2872–2938 → NPU `_bookkeeping_sync` 3130–3143 → 继承的 `GPUModelRunner._get_prompt_logprobs_dict` 5643–5750，输入为当前真实 scheduled 隐藏行。`_update_states` 1225–1240 先清 finished 状态、评分计数和持久请求行，再在 1335–1340 登记新代。计数空立即返回。抢占未调度和无 token ID 的 embeds 分支不进评分。 |
| 隐藏行到 TP 词表 | 请求已有 S 个计算 token，prompt 长 L。本轮读该请求在 `query_start_loc` 的连续隐藏行，写回从 S 开始的行，真值 token 从 S+1 开始。`compute_logits` 经多模态 wrapper 277 到 DeepSeek V4.1 语言头 1627–1636、1778–1783。实际注册的 `AscendLogitsProcessor._get_logits_normal` 592–610 走 STANDARD TP8 shard 投影、继承的 TP all-gather 和逻辑词表裁剪。TP1 replicated、PCP shard 和 fine-grained 头有各自 guard。 |
| 分数到三路 CPU 行 | `AscendSampler` 继承原 `compute_logprobs/gather_logprobs`。两种 logits 模式走 FP32 cast，两种 logprobs 模式走 FP32 log-softmax，提示词均跳过生成侧 processors。原 `torch.topk`、指定 token gather 和 `(score >= chosen).sum(-1)` 用同一 FP32 分数面。第 0 列是指定 token，随后为原 Top-K，可重复。K=0 仍有指定分数和含并列排名。宿主结果是 `INT32[L−1,K+1]` IDs、同形 FP32 scores、`INT32[L−1]` ranks。每个 active token-ID chunk 先 H2D 整份 prompt IDs，评分后把三个切片 nonblocking D2H。 |
| 跨 chunk 与跨轮发布 | 调度数等于 `L−(S+1)` 时完成评分并保留 buffer，下一轮可在零评分行时发布同一 buffer。大于剩余行时本轮发布，清计数和缓存引用。非空发布字典在 NPU 上调用 `torch.npu.synchronize`，位于普通 PP1 草稿提交之前。单 token 的结果 buffer 有零评分行。 |
| 客户端实际读者 | `Scheduler` 2107–2110、2149–2179 仅在有生成 ID、pooler 输出或 stop 时交 `EngineCoreOutput.new_prompt_logprobs_tensors`。前端 `LogprobsProcessor` 121–206 解码 ID 并按已累积上下文修正 UTF-8。普通 list 首位为 `None`，flat 存空区间且索引首位得到 `{}`。原有重复 ID 在字典读出时保留末次值及 rank。DELTA 一次取走提示词分数，`RequestOutput` 送 full response 或首次 echo 流式结果。Full response 保留原 −inf 到 −9999.0 的 clamp。 |

从原源码执行 9 组分块和模式控制，以及 2 组前端容器控制。对 26 份源码核对字节 SHA。控制使用替身记录 H2D、head、D2H 和同步，数值路径以原源码和归档反例约束。[源码 SHA、请求字段及控制结果](target-prompt-logprobs-controls.json)。

## 结果

- **已停止 · 冻结负载下单独优化提示词评分。** 原 v168 C8 的 client 与 PLAN SHA、同形状框架 profile、平衡 C8 及六组普通负载的归档请求均无有效评分参数。计数表为空，整段额外词表投影、通信、FP32 分数、三路 D2H 和发布同步共 **0 次**。局部可省设备工作和请求收益上界均为 **0**，覆盖同预算下的数学化简、共用扫描、分片摘要、只搬目标 ID 及同步后移。空分支本身的 Python 入口成本不在该设备预算。
- **待确认 · 较大的末层选行交付。** 在冻结请求，提示词评分读者已可从真实 guard 排除。它帮助定义末层 attention/FFN producer 的输出行需求，仍须保留 target 的其他行读者、DSpark 三层 aux、全量 KV/index/cache 写入及跨卡所有权。沿用原联合情景的 22.9921 ms P90 上界，收益仍取决于实际可略过行、原 M/tile、跨卡时序和逐位输出。[原联合预算与读者](终端实际读者与跨迭代交付判断.md)。
- **待确认 · 启用评分的其他合法负载。** 优先比较整段词表交付。可保留原投影与 TP 归集，尝试共用 FP32 归一化后的 Top-K、真值读取和 rank 扫描。跨卡方案另付摘要轮次，原归一化和 tie 选择必须复现。归一化后的 FP32 舍入能把不同原 logits 变成同分，已知反例令含并列排名从 1 变 2。分块后只上传真实 ID 切片可减少重复搬运。若要把全设备同步缩到消费事件，须跟踪每一代 CPU buffer、写入事件和最早序列化/读者。开启评分默认改变 prefix cache 读策略。原归档 K=5、冷 1,464 token 情景估算本地 0.79–2.20 ms，连分片摘要 2.42–4.82 ms。实际行数、chunk、评分开启时的端点分母和逐位证明齐备才有重新定价价值。[旧情景模型与反例](https://github.com/wio1997/Inference-Foundry/blob/2557e99b66657511e7eb0bd66d57cbeae6a1f346/projects/910c-deepseek-v4.1-framework-investigation/target-readout-and-prompt-scores.md)。
- **已确认可实现 · 本段没有新纳入的替换原语。** 原 Python 控制支持分支资格与状态交付。替换实现和设备逐位证据待形成。

下一段从 AsyncModelRunnerOutput 完成后的实际 worker 消息队列和 EngineCore 进程边界追输出对象、复制和缓冲区回收。按相同负载的原有 profile 先估整个交付的暴露时间。[结构化结果](target-prompt-logprobs-review.json)。
