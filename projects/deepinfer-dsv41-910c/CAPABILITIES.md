# DeepInfer 910C 优化能力与验证状态

记录时间：2026-10-30。适配参照 DeepInfer 提交 `553741557a57df44a46fb3ab72b6aafeaf945337`，部署条件为 172、DeepSeek-V4.1-Flash W8A8、16 个逻辑设备、DP2×TP8/EP16。[单项性能汇总](optimizations/RESULTS.md)列出已测指标、适用条件及未测能力。表中状态针对[当前保留版本](CURRENT.md)，不表示其他硬件或工作负载的支持范围。

| 能力 | 当前状态 | 证据与边界 |
| --- | --- | --- |
| 学习式 DSpark、Markov、confidence 与目标验证 | 已运行 | 使用检查点中的草稿网络；提示查找关闭。910C 采用专用执行和图池，已验证固定负载，不等同所有采样或动态批次均受支持。 |
| confidence 变长验证 | 部分适配 | 当前图池覆盖预算 4、7、24；一次 4K 请求在预算 7 与完整图的校验中出现差异。独立 Engram 形状候选可通过该请求，但仍存在长会话数值异常及跨条件差异，未通过完整验收。[图形状与异常记录](optimizations/20261030-dspark-shape/README.md) |
| ZOS 设备状态及草稿预发 | 部分适配并启用固定路径 | 设备状态参与固定 4×6 草稿预发；CPU 仍参与历史、接受结果和状态提交。变长、动态 remap、完整 EOS 和连续批处理不能由该路径推断。 |
| ZOS 状态投票缓冲复用 | 已启用 | 保留全 rank 投票及阻塞回读；同服务 ABBA 固定负载中，合并吞吐由约 603.94 提升至 611.01 token/s，两轮逐项 TTFT、TPOT 未退化。结果仅针对该次负载与运行条件。[投票复用](optimizations/20261029-zos-vote-reuse/README.md) |
| ZOS 设备 C2 和 Engram 窗口 | 候选未采纳 | 两项均完成固定负载与 16 rank 验证；分别未证明稳定收益，当前关闭。[设备 C2](optimizations/20261029-zos-c2-device/README.md)、[Engram 窗口](optimizations/20261029-zos-engram-window/README.md) |
| Engram 设备 hash 与 CPU 表 | 已启用 | 使用 CPU 共享表、UVA 读取和设备 hash；主干 CPU prefetch 关闭。prefill UVA 已有 16 rank 和固定负载证据，不代表完整官方后端。[设备 hash](optimizations/20261028-device-hash-overlap/RESULTS.md)、[prefill UVA](optimizations/20261029-engram-prefill-uva/README.md) |
| 共享专家和路由专家的 decode 重叠 | 已启用局部路径 | 当前配置保留 decode 多流；不能据此称 prefill 同样完成重叠验证。[路径记录](optimizations/20261028-device-hash-overlap/RESULTS.md) |
| 共享专家激活融合 | 候选未采纳 | 两轮吞吐分别增加约 0.56% 和 0.27%，TTFT 分别高 16.0 和 15.2 ms；样本和启动顺序不足以确定因果，未替换原路径。[单项对照](optimizations/20261030-shared-expert-activation/README.md) |
| YOCO 有界重放 | 256 token 近似路径启用 | 官方示例的 140 与当前 256 不等价；先前自定义样本出现输出差异，七题 LongBench 配对结果相同，但质量与缓存语义尚未充分验证。官方元数据、缓存槽位和单设备算子已完成限定范围对照；16 rank 单前缀恢复复制检查通过，真实多卡完整缓存语义及模型质量仍待验证。短末块的[独立 mask 候选](optimizations/20261030-yoco-short-mask/README.md)已完成限定形状短诊断，尚未采纳。[官方对齐进度](optimizations/20261030-replay-semantics/OFFICIAL-ALIGNMENT.md)、[语义与范围](optimizations/20261030-replay-semantics/README.md)、[长文配对](optimizations/20261030-yoco-longbench/README.md) |
| 正常 EOS 的 YOCO 有界重放 | 候选未采纳 | 七题精确、140、256 输出及结束原因一致；两轮正常 EOS 长输出中 256 相较精确路径 TTFT 较低，TPOT 持平或略低。正常 EOS 使现有 DSpark 预发回退；样本和缓存边界不足，不能直接对照 ignore-EOS 基线。[短输出](optimizations/20261030-yoco-normal-eos/README.md)、[长输出](optimizations/20261030-yoco-normal-eos/LONG-OUTPUT.md) |
| YOCO 索引提前准备 | 候选未采纳 | 固定负载的正反顺序对照及 16 rank 输出检查通过；两次测试均有首组额外等待，未证明稳定综合收益，第二轮开启时 TTFT 和吞吐均略差。[原对照](optimizations/20261030-yoco-index/README.md)、[反向顺序](optimizations/20261030-yoco-index/REVERSE-ORDER.md) |
| 官方 FP4 索引器 | 当前由 INT8 路径替代 | 官方索引 Q/K 使用 Hadamard 与 FP4，当前 910C 使用 INT8；固定镜像 A3 索引接口文档及源码限定 INT8。不同格式的 top-k、质量及性能影响尚未测量；不能将该接口限制推广为 910C 无法实现 FP4。[索引接口审计](optimizations/20261030-replay-semantics/OFFICIAL-ALIGNMENT.md#索引器格式和当前接口限制) |
| QLI metadata 复用 | 候选未采纳 | 正反顺序对照中吞吐变化较小，部分 TTFT 退化；曾出现 EOS 后继续生成差异，原因未确定。[单项对照](optimizations/20261030-qli-metadata/README.md) |
| prefill KV 通信与 Q 投影重叠 | 候选未采纳 | 16 rank 路径及完整输出通过；两轮未证明稳定收益，设备侧重叠尚未确证。[单项对照](optimizations/20261029-prefill-kv-overlap/README.md) |
| lazy mHC 跨子层融合 | 候选未采纳 | 固定 24 行候选完成短诊断与固定负载对照，未测出性能收益；不代表完整官方实现。[单项记录](optimizations/20261029-mhc-crosslayer/README.md) |
| EPLB 静态副本及映射融合 | 候选未采纳 | 真实输出与固定负载已核验；静态融合首轮 TTFT 严重退化，动态长度候选相对原映射吞吐基本持平。在线统计和迁移未完成。[动态映射](optimizations/20261030-eplb-dynamic-map/README.md) |
| SparseFlashMla | 已使用 BF16 绑定；混合量化未接入 | 当前目标及草稿均使用固定镜像绑定。官方主线的 910 架构注册支持 TurboQuant 模式，接口约束及 KV 格式与当前路径不同；FP8 模式不能直接套用于本机。数值、质量及收益未验证。[接口评估](optimizations/20261030-sparse-mla/README.md) |
| Recipe 原始文本分词 | 已启用 | 固定冷分词工作负载两轮平均 TTFT 约从 756.35 降至 657.80 ms；不代表官方聊天模板或其他文本均已验证。[分词记录](optimizations/20261029-recipe-tokenizer/README.md) |
| 可选解码 token 黑名单 | 未接入当前贪心路径 | 官方 Sampler 提供可选屏蔽，评测入口有自己的默认配置；当前服务未设置该选项。七题样本的相关 token 只出现在 EOS 后，不能推断其他输入。[接口与证据](optimizations/20261030-token-blacklist/README.md) |
| 连续批处理、动态 remap、通用采样 | 未完成验收 | 固定 C8、greedy 和 ignore-EOS 数据不能证明通用支持；需按请求状态、输出顺序和模型语义分别验证。 |

本地探索的 [DSpark 输出事件合并写入](optimizations/20261030-event-batch/README.md)不属于已验证的官方能力：两轮实际写入调用减少，但吞吐略低、TTFT 略高，候选未采纳。各单项结果来自不同候选与测试时段，不可直接相加。整体对照及其限制见[同期比较](comparisons/20261030-910c/README.md)。
