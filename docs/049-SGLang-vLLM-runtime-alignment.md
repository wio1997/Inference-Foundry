# DeepSeek-V4.1 W4A8 运行配置对照

对照的是 SGLang G1317 当前运行版与 vLLM G630 四档正式基线。两端各 8 张 Ascend 910B1，aarch64。P、D 两台主机上的历史与当前实际加载清单均匹配同一组 95 项权重校验记录。SGLang 的 24 份输入清单逐项匹配 vLLM 四档实跑清单，覆盖每档 5 份独立预热输入和 1 份正式输入。每个正式输入在 vLLM 的冷测和重放中使用相同 SHA。SGLang 当前只完成 32K 首轮预热。 [实机配置、权重及数据绑定](../evidence/SG_usage_recovery_G1276/SG_G1317_native630_actual_profile_alignment.json) [vLLM 四档结果](https://github.com/wio1997/Inference-Foundry/blob/cff38e0021cd7f859dea4b8a403b06c3791eb7dd/docs/016-native032-four-tier-first-matrix.md)

| 项目 | SGLang G1317 | vLLM G630 |
|:--|:--|:--|
| 版本 | Ascend daily A2，`0.5.24.dev536+g8a0edcf79`，CANN 9.1.0，torch-npu post6 | `v0.32.0rc1` 原生 arm64，CANN 9.2.1，torch-npu post7 |
| 并行和长度 | TP8、EP8、PP1、DP1、CP1；并发上限 16，上下文上限 263296 | TP8、EP8、PP1、DP1、PCP1；并发上限 16，上下文上限 263296 |
| Prefill 配置 | P/D 分块上限 8192/1024，页面参数 128，内存比例配置 0.88/0.80 | P/D 批 token 上限 8192/1024，块参数 128，内存比例配置 0.88/0.80 |
| DSPARK 和图 | P eager，D target/draft full；D 8 个 rank 均捕获 target 宽度 6 和 draft 宽度 5 | P eager，D `FULL_DECODE_ONLY`，draft 候选数 5，target 验证包含 1 个 anchor 与 5 个候选 |
| 异步与重叠 | scheduler overlap 已配置。专家与通信流水仍待匹配 trace | async scheduling 已配置。D 的 `fused_mc2` 使 shared-expert 多流重叠实际关闭 |
| KV 传输与前缀缓存 | Ascend P/D 传输，P 和 D 的 radix cache 启用 | `MooncakeHybridConnector`，P 和 D 的 prefix cache 启用 |
| 量化实现 | 同一 W4A8 权重，FP4 索引 K | 同一 W4A8 权重，Ascend ModelSlim，INT8 索引 K |

SGLang 的 `speculative_num_draft_tokens=6` 应按实际 D 图宽度解释。vLLM 的 `num_speculative_tokens=5` 计的是候选数。配对比较时仍需按每步实际请求、query、KV 长度和 padding 归一化，同时记录预热前缀命中及调度布局。上表的页面参数、分块和内存比例来自运行配置，物理页布局、每步 token 编组和实际 HBM 占用需要分别测量。 [SGLang 源码索引](../code/SG_PD_wait_recovery_G1317/G1317_runtime_source_index.json) [vLLM DSPARK 固定源码和图归因](https://github.com/wio1997/Inference-Foundry/blob/cff38e0021cd7f859dea4b8a403b06c3791eb7dd/docs/021-native032-allrank-coverage-and-DSPARK-source-chain.md)

PR #12 提供适配源码和同版本来源参考。其 910B3 和 16K/C4 工况与本任务 910B1、C16、四档输入不同，本任务性能数字以实际运行版为准。vLLM 32K 重放两端均有本地命中，64K 为低命中，128K 和 256K 的本地命中均为零。两框架的热态对照需结合 P、D 本地命中和跨节点接收计数。 [PR #12 与旧镜像的源码对照](https://github.com/wio1997/Inference-Foundry/blob/cff38e0021cd7f859dea4b8a403b06c3791eb7dd/docs/004-PR12-source-comparison.md) [原生 0.32 现场源码及版本校验](https://github.com/wio1997/Inference-Foundry/blob/cff38e0021cd7f859dea4b8a403b06c3791eb7dd/docs/008-required-vllm032-and-native-source-refresh.md)
