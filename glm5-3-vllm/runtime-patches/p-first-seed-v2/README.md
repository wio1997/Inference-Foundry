# P 已有 MTP1 草稿交接

已完成有限设备验证的机制原型，完整一般质量/容量门槛未通过，不默认开启。query-only v1 在两道正确性预选输入的干净配对测量中没有净收益，不作为默认实现；原件保留。

P 仍 K1、D 仍 K5，保留既有模型、精度和算子。每请求 `_glm53_export_p_draft` 才导出 P 已算出的一个未验证草稿。微小 pinned CPU copy 排在实际 MTP writer stream 上、原 Async 构造函数的 copy-stream dependency 之前；读取复用原完成事件，不加全局同步、clone 或 draft 计算。新 copy 已在16rank profile中将80条D2H逐条关联到PCIE_DMA_SQE；该P本地完成profile不替代完整PD路径验证。

复用镜像已有兼容字段 `ModelRunnerOutput.spec_token_ids`，聚合器要求 16 个 rank 的请求 ID、采样结果和草稿一致，否则清空草稿。控制 metadata 带内部请求 ID、N、确认 y1 及 rank agreement。D 的 `_glm53_use_p_draft` 受全局 adoption 和每请求显式开关共同保护；元数据不合格保持 native 路径。历史只确认 y1，computed=N，y2 只填入已有五个初始 spec slot 的第一个，由原生拒绝验证器验证。

`build.py` 对固定实际镜像的五份源码校验 SHA，生成不可变 overlay、diff、stager 和 manifest；未替换的 NPU 原生函数保持原样，helper 默认关闭。`run_pairs.py` 在同一 P/D 实例上以 cold salt 比较 control/adopt/seed，支持错误草稿注入并分别保留原 P metadata 与 D 实际请求。P 导出失败也先用 native D 消费该请求，再报告失败，避免试验错误遗留 P KV 所有权。

CPU harness 使用原生 Request、SamplingParams、CachedRequestState、实际 MQ enqueue/recv 代码与真实 KVOutputAggregator；兼容函数从已安装 Ascend 文件提取原样执行。socket recorder 不等于真实 IPC，未初始化完整 Scheduler/KV manager/InputBatch。设备正确性、整请求实际 target 调用数、干净端到端时延及新 D2H 的 SDMA 路径均以 evidence/20261018-p-first-seed-device-v2 的后续收据为准。

`profile_request.py` 在性能计时之外捕获实际 P seed 请求。`export_profile.py` 先保留停止的 rank0 原始 profile，再用 CPU 导出；单 rank 观测不能宣称新 copy 已覆盖全部 rank。

当前有限两case C1中seed/control last倍率0.962158、平均TPOT0.953850，TTFT1.002904。实验CPU代理的提前确认首token模式在同case TTFT0.706434、last0.963221，但平均TPOT1.014819且N256中段位置更晚；不是逐token支配、默认或容量验收。24题只首答案通过，seed/control完整64序列14/24一致；native control自身也随并发变化，见分析文档与recovery-v2。

`build.py --divergence-probe`只在新overlay追加默认关闭的`divergence.py`。显式env gate+两history SHA白名单，每case/rank最多6次。捕获既有target输入/位置、sample_hidden与logits，writer流pinned CPU copy在native sampler之前，原Async等待保留、自有copy event只query。`capture_divergence.py`提交匹配上限的native D C1/cold C4请求；不改算子、模型精度或正式计时路径。输出bin/jsonl由既有finish/backup保留，必须结合真实batch/query范围解释。

首次快照12请求×16rank=192份通过：实际query IDs/positions、native event及API输出均匹配，C1 hidden/logits重复一致，C4五个请求在最终hidden已不同；部分物理B1同shape仍变化，C4后第一条C1 Top5也改变、随后恢复。它定位到LM head之前，尚不能证明具体故障或无害舍入。

`--divergence-probe --shared-overlap-probe`另追加默认关闭的`shared_overlap_probe.py`，用同实例ABBA检验已有shared-expert overlap分支。只选择原执行器的布尔，target/MTP都覆盖，SP-only因初始化会选不同op而拒绝。必须从16rank真实native forward调用证明开关生效，未知inventory保留native并拒绝试验。该模式每history/mode/rank上限8次，`probe_shared_overlap.py`提交48请求/32重点请求，含C1-before/C4/C1-after；以对应冻结合同解释，诊断不作性能验收。
