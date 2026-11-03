# 整体覆盖台账入口

**整体未完成；范围闭包仍在扩展。** 文件/条件条目是定位索引，不是已审阅或运行覆盖。

| 模块 | 已关联源码数 | 分支站点关联数（可重复） | 结论性缺口 |
|---|---:|---:|---|
| M00 注册、平台替换、模型与算子选择 | 84 | 2520 | 真实模型/LogitsProcessor/LMHead类及权重分片模式；vllm.models与legacy DSpark导入重定向；已注册与实际使用的算子/硬件能力 |
| M01 请求、tokenization与采样契约 | 2 | 160 | 冻结请求到SamplingParams/metadata的逐字段映射；server/model默认值和条件processor |
| M02 前缀复用、KV分配与回收 | 6 | 999 | cache命中后真实计算域；块共享/回收/跨步所有权与hash状态 |
| M03 EngineCore、调度与异步队列 | 8 | 836 | 提交/回复/准备的实际因果边；队列容量、同次batch身份和阻塞点 |
| M04 worker输入准备与持久batch状态 | 2 | 1159 | 6956个scope的具体生产者/消费者；上一批读取完成与下一批覆盖；未隐藏准备与等待的拆解 |
| M05 embedding、模型wrapper和图像条件 | 3 | 75 | 文本请求的vision执行/驻留条件；embedding分片及token行域；wrapper到language模型的完整值链 |
| M06 mHC初始化、pre/post与跨层残差 | 2 | 379 | 系数/残差的全部消费者与alias；精度、归约与跨层状态依赖；内部复合核的primitive/effect展开 |
| M07 Q/KV投影、norm、量化与RoPE | 2 | 229 | query/cache行域与权重/scale所有权；所有indexer分叉与持久cache消费者；流/事件/原位更新身份 |
| M08 compressor、ring状态与latent分叉 | 4 | 261 | 跨迭代ring写读与C2有效性；两个latent消费者及BF16舍入；不同压缩比/空行/边界分支 |
| M09 Indexer评分、候选与跨层索引 | 2 | 192 | 真实key长度/候选覆盖率/mode；精确TopK、tie、sentinel与回退；层20到后续所有索引消费者 |
| M10 SWA、压缩KV与稀疏attention | 3 | 198 | 所有cache平面和mask/sink；A3与packed等条件分支的适用证据；kernel内部及跨流完整资源账 |
| M11 attention输出、head恢复、两级投影与归约 | 2 | 318 | 行/head/token恢复与padding；块ready是否可发布及collective顺序；两级投影精度与全体消费者 |
| M12 MoE路由、dispatch、专家计算和combine | 175 | 3999 | 同调用专家行数/容量/mask；内部组级ready与共享workspace；EP组、二进制身份及通信暴露 |
| M13 共享专家、路由分支汇合与scaling | 25 | 801 | shared/routed的量化域和输入版本；归约域、合并舍入与资源竞争；fused MC2选路与双分支并行条件 |
| M14 Engram hash、host/UVA lookup、投影与门控 | 8 | 173 | 完整host/UVA访问与ready链；跨rank表布局、scale和无效ID；投影/gate/残差与跨步hash状态 |
| M15 最终mHC collapse、norm及MTP辅助状态 | 1 | 254 | 最终hidden与aux/MTP缓冲的全部消费者；归一化与舍入、SP/PP恢复；跨迭代覆盖与图地址生命周期 |
| M16 目标/草稿词表投影与跨卡重建 | 3 | 131 | STANDARD/FINE_GRAINED/PCP_X_TP真实分支；全部logits消费者与处理顺序；局部选择的全局tie、padding和ID映射 |
| M17 目标采样、拒绝/接受、bonus与logprobs | 25 | 776 | 全部greedy/random/synthetic/block/entropy路径；bonus与target原值副本的条件消费者；实际metadata、RNG、输出token/回退状态 |
| M18 DSpark上下文投影、共享KV及输入发布 | 2 | 98 | context与query独立行域/分片；历史KV与当前请求身份；publish/overwrite事件和跨卡信息 |
| M19 DSpark decoder、mHC及共享计算 | 1 | 59 | 三个draft block身份与配置fallback；与目标层复用的所有语义差异；跨block隐藏状态/残差和buffer |
| M20 DSpark base logits、Markov和confidence | 3 | 598 | seed与逐位置Markov依赖；raw_logits的所有原位/概率消费者；词表分块与跨位置可交错的边界 |
| M21 draft/target metadata、slot与索引准备 | 3 | 268 | global/local请求长度与设备权威状态；持久地址的全体消费者；target/draft、空行、padding与skip-update |
| M22 graph捕获、replay、更新与多流生命周期 | 2 | 1101 | 每次graph/replay到任务/请求映射；静态地址、更新handle与event身份；跨流收敛、异常和回退条件 |
| M23 DP/TP/EP/PCP通信与分区所有权 | 6 | 422 | 参与组/序号/真实传输量；全流全卡暴露关键路径；重排/分块后的顺序与共享资源 |
| M24 跨迭代状态、接受回滚、预取和重用 | 67 | 3966 | NEXT_INVOCATION依赖与request重排；KV/概率/hash/RNG的版本和生存期；可提前的准备与必须等待的状态 |
| M25 采样结果回传、响应、流式输出与终止 | 2 | 204 | 设备结果到客户端chunk/输出token归一化；stop/EOS/ignore_eos和1K长度；主机队列、异步D2H与输出消费者 |

每个模块必须逐项关闭范围/dispatch、分支、值与全部消费者、alias/状态、跨流/卡/迭代依赖、变换规则、资源/性能/极限与文献复析八类门槛。
[机器可读台账](module-coverage-ledger.json)包含站点ID与260个规则检查槽；它们目前均未关闭。
[源码底账](coverage-ledger/summary.json)覆盖选取的冻结源，不宣称完整传递依赖闭包。
