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

vLLM 的实际快照已逐档复算 D 本地与连接器 logical token。SGLang 首轮独立预热的实际首尾快照给出 P rank 0 本地、主机和存储命中均为零，P 传输计数增加 16。当前快照没有与 vLLM 连接器逻辑 token 对应的 D 指标。热态对照将分别报告各自可验证的本地命中、连接器记账及传输证据。 [vLLM 四档连接器计数](https://github.com/wio1997/Inference-Foundry/blob/31850432f9534faf776c94384d620d707e81a0c8/docs/024-native032-PD-prefix-accounting.md) [SGLang 首轮缓存和传输快照](../evidence/SG_usage_recovery_G1276/SG_G1317_32K_W1_cache_and_P_transfer_metrics.json)

固定源码中 `prefill_effective_tokens_total` 由 Prefill 报告路径更新。Decode 在预分配时分别取得本地设备命中和可向 Prefill 承诺的总前缀，API `cached_tokens` 将两端复用合并计数。实机短重放原始快照中，P 的 `device_hit` 增加 256，D 的 Prefill 系列保持为零。D 的零读数只覆盖已注册的这条 Prefill 序列，独立的 D 本地命中须从 Decode 路径确认。 [固定源码位置](../evidence/SG_usage_recovery_G1276/SG_G1317_actual_prefix_cache_scope_source_chain.json) [短重放前后快照](../evidence/SG_usage_recovery_G1276/SG_G1317_short_replay_prefill_counter_scope.json)

正式门槛的辅助工具已在实际服务器部署，以原首轮快照和原正式 runner 的读计数逻辑做了兼容验证。读数输出保留 P rank 0 的验证结果，并把 D 的独立命中保留为待验证。原首轮中，P rank 0 的 sender 直方图记录 16 条传输，累计 1540 MB。跨框架报告将分别列出该直方图、连接器逻辑 token 和设备侧证据。 [门槛处理工具](../code/SG_formal_cache_scope_G1385/scope_SG_formal_prefix_evidence_g1385.py) [原 runner 兼容验证](../evidence/SG_usage_recovery_G1276/SG_G1317_formal_cache_scope_runner_compatibility.json) [P sender 直方图](../evidence/SG_usage_recovery_G1276/SG_G1317_32K_W1_prefill_sender_transfer_payload.json)

针对 D 本地命中，新增了独立采集用的 rank 0 profiler 标记。固定源码中，提交函数只有在传输轮询成功后调用，且会核对 bootstrap room。标记在提交成功后输出此次请求的 L1 设备、L2 主机、L3 存储匹配长度及声明前缀长度，区分已知零命中和缺少匹配信息。CPU 测试复用了镜像实际的 `DecodePrefixMatch` 定义，验证失败、模拟传输和其它 rank 不产生标记。代码已在服务器校验，实际 trace 尚待采集。第二轮 32K 同进程已处理 180224 个新增输入 token，完成 5 次 KV 传输。 [固定提交边界](../evidence/SG_usage_recovery_G1276/SG_G1317_decode_prefix_commit_boundary_source.json) [采集代码](../code/SG_D_prefix_trace_G1387/decode_prefix_annotation_g1387.py) [CPU 验证](../evidence/SG_usage_recovery_G1276/SG_G1317_decode_prefix_annotation_fixture.json) [服务器校验](../evidence/SG_usage_recovery_G1276/SG_G1317_decode_prefix_annotation_stage.json) [第二轮采样](../evidence/SG_usage_recovery_G1276/SG_G1317_32K_W2_fifth_KV_transfer.json)

固定镜像中的 aarch64 PyTorch 2.10.0+cpu 成功将 4 条测试标记导出到 Chrome trace。读取器识别到 2 条有匹配信息的新请求标记，其中 1 条为零命中，另有 1 条缺少匹配信息和 1 条重建标记。用 CPU 标记构造 SQLite 测试库验证了字符串关联，随后读取了原始 G981 昇腾 Decode SQLite。G981 采集早于该标记，读取结果为 0 条标记且请求覆盖不完整。新增标记的 NPU 导出以及与实测请求的逐条关联仍待验证。 [读取器](../code/SG_D_prefix_trace_G1387/audit_decode_prefix_trace_g1389.py) [CPU 原始导出校验](../evidence/SG_usage_recovery_G1276/SG_G1317_decode_prefix_actual_CPU_profiler_export.json) [G981 SQLite 对照](../evidence/SG_usage_recovery_G1276/SG_G981_original_D_SQLite_prefix_reader_control.json)

后续匹配采集用完整输入 token 序列的 SHA256，将 Decode 标记关联到冻结清单，再按 `data_id` 核对 AISBench 原始请求全文、成功状态及输入输出 token 数。vLLM 已归档的同批 16 条原始记录全部通过逐条核对。关联逻辑使用这些原始记录和构造的标记验证，缺失、重复和外来标记均无法通过完整性检查。新增标记和关联器已在 Decode 主机校验，SGLang 实际 NPU 标记仍待采集。 [固定请求路径](../evidence/SG_usage_recovery_G1276/SG_G1317_decode_prompt_identity_fixed_source.json) [原始请求校验](../evidence/SG_usage_recovery_G1276/SG_G1317_decode_prompt_identity_native_raw_fixture.json) [标记代码](../code/SG_D_prefix_trace_G1387/decode_prefix_prompt_annotation_g1391.py) [关联器](../code/SG_D_prefix_trace_G1387/bind_decode_prefix_profile_g1391.py) [服务器校验](../evidence/SG_usage_recovery_G1276/SG_G1317_decode_prompt_identity_stage.json)
