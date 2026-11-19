# DeepSeek-V4.1 W4A8 运行配置对照

对照的是 SGLang G1317 当前运行版与 vLLM G630 四档正式基线。两端各 8 张 Ascend 910B1，aarch64。P、D 两台主机上的历史与当前实际加载清单均匹配同一组 95 项权重校验记录。SGLang 的 24 份输入清单逐项匹配 vLLM 四档实跑清单，覆盖每档 5 份独立预热输入和 1 份正式输入。每个正式输入在 vLLM 的冷测和重放中使用相同 SHA。SGLang 已完成 32K 前三轮预热，第四轮已启动。 [实机配置、权重及数据绑定](../evidence/SG_usage_recovery_G1276/SG_G1317_native630_actual_profile_alignment.json) [vLLM 四档结果](https://github.com/wio1997/Inference-Foundry/blob/cff38e0021cd7f859dea4b8a403b06c3791eb7dd/docs/016-native032-four-tier-first-matrix.md)

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

32K 首轮预热的 16 条 SGLang 原始请求已在运行主机按 `data_id` 逐条与只读输入数据核对，并从原始结果复算服务端 token 数和 AISBench 指标。第二轮同进程仍在线时，复核器返回进行中，留待该轮真实终态再验收。 [首轮原始请求校验证据](../evidence/SG_usage_recovery_G1276/SG_G1317_32K_W1_exact_raw_dataset_binding_W2_live_control.json) [复核器](../code/SG_phase_audit_G1205/audit_SG_32K_warmup_request_binding_g1394.py)

32K 第二轮终态复核通过。16 条原始请求均与第二轮独立冻结输入匹配，服务端每条实际输入 32768、输出 1024，AISBench 耗时 7887.82 秒。随后在同一 P/D 实例启动第三轮，已核对前两轮被复核并跳过、第三轮使用独立输入及实际客户端进程。 [第二轮原始请求复核](../evidence/SG_usage_recovery_G1276/SG_G1317_32K_W2_actual_raw_request_binding.json) [第三轮实际启动](../evidence/SG_usage_recovery_G1276/SG_G1317_32K_W3_actual_input_and_worker.json)

## Prefill 索引路径的工作量差异

同一输入清单的首轮预热两端均完成 16 条请求，实际峰值并发 16，每条输入 32768、输出 1024。SGLang 耗时 7989.66 秒，vLLM 耗时 46.08 秒。这组尚未计分的样本用于界定问题规模。SGLang 的固定 Prefill 索引路径用 `einsum` 生成逐 head 稠密分数，聚合后再掩码和 top-k。vLLM 的独立诊断采集在 Prefill rank 0 实测到 520 次 `VllmAscendQuantLightningIndexerV2`，任务累计 1.3505 秒。两条证据支持索引实现不同，差距归因仍需在 SGLang 匹配 trace 中核对索引评分、其它算子、调度与通信的时间分布。 [SGLang 首轮审计](../evidence/SG_usage_recovery_G1276/SG_G1317_32K_W1_verified_raw_request_audit.json) [vLLM 首轮审计](https://github.com/wio1997/Inference-Foundry/blob/63ef19ee22ab0c7de4ed951931c4ce0348c3995b/evidence/native_first_warmup_G676/warmup1_32768_audit.json) [vLLM 设备算子统计](https://github.com/wio1997/Inference-Foundry/blob/63ef19ee22ab0c7de4ed951931c4ce0348c3995b/evidence/profile_capture_G804/derived/P0_indexer_operator_aggregate_G1395.json)

按实测首块形状、128 MiB 原生分块上限和无缓存连续输入推算，精确后缀裁剪可减少的逻辑评分位置在 32K、64K、128K 和 256K 分别为 19.35%、10.88%、5.80% 和 3.01%。在 32K，原路径、精确裁剪和按 1024 向上取整分别涉及 5、161 和 17 种 query 行数与 K 宽度组合，向上取整后仍可少算 14.99%。该数量供设备验证选择形状，实际编译行为和耗时尚待测量。即使把 SGLang 首轮的全部墙钟时间都按 19.35% 线性缩减，仍需约 6444 秒。只靠减少这部分评分位置不足以解释首轮差距。 [计算记录](../evidence/SG_usage_recovery_G1276/SG_G1317_causal_indexer_shape_and_warmup_gap.json) [计算脚本](../code/SG_causal_prefill_score_G1362/estimate_causal_score_shape_tradeoff_g1396.py)

后续 Prefill trace 优先对照约 8K 个 query 的设备工作，区分单请求和双请求调度。SGLang 采样会用同一逻辑步骤下的实际 query 行数及各请求前缀长度关联稠密评分标记，同时记录标记内实际 K 行数。若两个框架的调度拆分或压缩布局不同，保留各自形状及可见 K 的差异，按匹配的逻辑工作量对照。 [vLLM 逐任务形状及调度分布](https://github.com/wio1997/Inference-Foundry/blob/2eab02ef840d5844e2076eaab3def9d7c0b56cfe/evidence/profile_capture_G804/derived/P0_indexer_operand_shapes_G1418.json) [SGLang 评分标记](../code/SG_P_step_attribution_G1355/index_score_annotation_g1355.py)

SGLang 的设备归属分析支持通过 `CONNECTION_IDS` 找到实际的 PyTorch 算子，再用同线程 forward 区间核实归属。用封存的 G981 Decode rank 0 核验，72 个任务关联到四个实际步骤，其他任务保留各自的未归属原因。该历史库没有 Prefill score 标记。隔离的 SQLite 用例验证了跨线程 score marker、重复连接和归属冲突。运行分析器时，将 `code/SG_profile_readiness_G1218` 加入 `PYTHONPATH`，并传入冻结数据库路径、SHA256 和输出路径。 [实际历史库与测试结果](../evidence/SG_usage_recovery_G1276/SG_D0_G981_framework_connection_attribution_G1420.json) [分析器](../code/SG_P_step_attribution_G1355/attribute_SG_prefill_steps_g1420.py) [vLLM 逐步参考](https://github.com/wio1997/Inference-Foundry/blob/6045560f4d2ba1fc7db4ab3abbbea0cacfe77380/evidence/profile_capture_G804/derived/P0_indexer_scheduler_bridge_G1419.json)

已核对当前 P/D 容器内的详细步标记源码。纯 `EXTEND` 记录 query、KV、query 平方及 query 与 KV 乘积的总和，KV 以 query 分块末尾的原始逻辑序列长度计。单请求可直接还原，双请求 query 行数不同时可解出各自前缀长度。等长 query 的歧义与其他未覆盖的批次会在输出中标记。用 vLLM 封存的 65 个实际非零步骤先计算同口径汇总再还原，50 个单请求和 15 个双请求的结果均与原始逐请求长度一致。SGLang 实际 trace 待采集，压缩后的可见 K 及设备工作量仍需核对。 [实机源码和离线验证](../evidence/SG_usage_recovery_G1276/SG_G1317_prefill_query_KV_geometry_contract_G1421.json) [还原工具](../code/SG_P_step_attribution_G1355/derive_prefill_request_geometry_g1421.py)

固定 Prefill 评分源码按绝对位置计算压缩 K 可见范围。对连续 C2 路径，位置 `p` 可见 `floor((p+1)/2)` 个压缩条目。八个实际 rank 的首块回执均确认 NPU 专用评分预算为 128 MiB，包内默认值为 1 GiB。将 vLLM 封存的 65 步逐请求形状代入同一源码口径，单次索引遍历预计 2589 个评分分块。稠密评分覆盖约 53.58 亿 query–K 对，其中约 42.95 亿对在 top-k 前满足因果范围，均为每个索引头的几何计数。后续 trace 用这些形状核对实际分块和耗时，并分别记录 vLLM 的页容量与有效逻辑长度。 [实机上限与逐步模型](../evidence/SG_usage_recovery_G1276/SG_P_logical_causal_work_at_native_geometry_G1422.json) [计数工具](../code/SG_causal_prefill_score_G1362/prefill_logical_score_work_g1422.py)

在当前 G1317 的 P/D 各八个 scheduler 中，手动起停使用镜像默认的 profiler 路径，请求中的 `GPU` 活动映射到 NPU。两个 trace 分区的可用空间快照分别为 1.95 TB 和 1.86 TB。采集关闭后先清点每 rank 的 CPU op range、设备数据与结束记录，再检查派生数据库。封存的 SGLang G981 Decode 八个 rank 均检出原始文件；清单工具也能识别原始文件齐全、待离线解析的情形。实际 G1317 采集的空间需求和事件覆盖仍待该次导出核验。 [当前运行环境与历史文件核对](../evidence/SG_usage_recovery_G1276/SG_G1317_profile_export_and_raw_rank_inventory_G1423.json) [原始文件清单工具](../code/SG_long_profile_G1305/inventory_SG_raw_profile_g1423.py)

固定 A2 实机包的 `libopapi` 和 910B 算子目录均有 BF16 `LightningIndexer`。CANN 9.1.0 的 V2 文档列出的压缩比为 1、4、128。对已压缩的连续 C2 K，可按 query 位置奇偶拆分。每组内相邻 query 的可见 K 数递增 1，适用常规右下因果掩码。将 65 个已封存的 vLLM 步骤代入，80 个请求分段的逐行可见范围都与 C2 原公式一致，合计 160 组。候选保留现有 FP4 来源的 Q/K。实际 910B dispatch、top-k 集合及次序、物理槽位与耗时仍须上卡验证。 [实机 ABI 与离线几何核验](../evidence/SG_usage_recovery_G1276/SG_A2_lightning_indexer_abi_and_C2_geometry_G1425.json) [分组代码](../code/SG_causal_prefill_score_G1362/c2_lightning_indexer_geometry_g1425.py) [CANN 9.1.0 原生接口](https://github.com/hicann/ops-transformer/blob/9.1.0/attention/lightning_indexer/docs/aclnnLightningIndexer.md) [V2 压缩比说明](https://github.com/hicann/ops-transformer/blob/9.1.0/attention/lightning_indexer_v2/docs/aclnnLightningIndexerV2.md)

用固定 A2 镜像的无 NPU 容器确认 `npu::npu_lightning_indexer` 的实际 PyTorch 注册签名。组件探针复用固定 SGLang 的评分和写回源码，保留算子原始输出供无效索引检查。写回前将算子的负数哨兵映射到可见范围之外，由 SGLang 原有排序及槽位映射完成输出。CPU 夹具通过四种形状，并分别拒绝不合法索引和物理槽位变化。真实 fused 算子数值、耗时和模型结果仍待上卡。 [探针](../code/SG_causal_prefill_score_G1362/probe_c2_lightning_indexer_g1426.py) [CPU 夹具与实机注册](../evidence/SG_usage_recovery_G1276/SG_A2_LI_component_probe_CPU_wiring_G1426.json)

## 32K 正式冷测和重放的时间窗口

原 vLLM 冷测和重放的各 32 条原始请求均复算出客户端并发峰值 16，第 17 条在首条完成后才开始。两端使用同一固定 AISBench 镜像。SGLang 第二轮原始 SQLite 复算最长单请求用时 7887.70 秒，整轮 7887.82 秒。当前 Router 的 14400 秒为单请求上限，Decode 的 14400 秒为 KV 等待设置。正式 AISBench 整轮上限由独立阶段参数控制。 [vLLM 原始请求并发](https://github.com/wio1997/Inference-Foundry/blob/72a0cf3e22549a59bcdc0742c22c3a325abe2f4e/evidence/native32_first_G691/derived/formal_C16_HTTP_admission_G1408.json) [SGLang 第二轮请求复算](../evidence/SG_usage_recovery_G1276/SG_G1317_W2_raw_client_admission.json) [复算脚本](../code/SG_W1_latency_timeline_G1375/audit_32K_client_admission_g1408.py)

以两轮较慢的实测预热耗时估算，两个连续 C16 窗口约需 15979 秒。按每个正式阶段预留两个原 12900 秒预热窗口，冷测和重放合计需 52080 秒的实机租期，包含原有收尾余量。原 CPU 客户端租期无法覆盖这个预算。第五轮实际结束后须按现有 P/D、配对锁和续租后的 CPU 最早到期时间复核。重放时间估算保留零命中的情况，正式耗时和命中仍待实测。 [预算和实际租期来源](../evidence/SG_usage_recovery_G1276/SG_G1317_32K_formal_client_window_plan.json)

正式启动脚本按 CPU 容器的实际只读挂载校验输入。现场已核对 32 条冻结请求、C16 配置和两阶段共用的输入 SHA，原测试没有首次正式提交记录或计分产物。启动脚本同时纳入独立阶段上限和 Prefill 计数范围，Decode 命中保留待实测。组合验证使用模拟的第五轮终态门槛，现场仍在运行第三轮预热。 [只读输入和提交账本检查](../evidence/SG_usage_recovery_G1276/SG_G1317_32K_first_formal_mounted_data_precheck.json) [启动门槛组合验证](../evidence/SG_usage_recovery_G1276/SG_G1317_32K_scoped_formal_launch_composition.json)

正式续跑会在第五轮原始请求及稳定性校验通过后，按封存 SHA 读取第三轮和第五轮的同一 Prefill 实例快照。比较范围为持久编译缓存清单的路径、大小和修改时间，Ascend 运行目录另行记录。对比器已部署到 Prefill 主机，隔离演练验证了拒绝预热未完成、缓存元数据改变或剩余租期不足的情况。 [编译快照比较器](../code/SG_phase_audit_G1205/compare_SG_prefill_compile_g1414.py) [Prefill 主机部署回执](../evidence/SG_usage_recovery_G1276/SG_G1317_P_compiler_comparator_physical_stage.json) [正式续跑演练](../evidence/SG_usage_recovery_G1276/SG_G1317_first_formal_event_controller_fixture.json)

第三轮 16 条原始请求均与该轮冻结输入匹配，每条实际输入 32768、输出 1024，耗时 7884.90 秒。空闲时采集了同一 Prefill 实例的第三轮快照，封存清单包含 211 个持久编译缓存文件。第四轮在原 P/D 实例和原 CPU 客户端启动，使用独立的第四轮输入，并复核跳过前三轮。 [第三轮原始请求复核](../evidence/SG_usage_recovery_G1276/SG_G1317_32K_W3_verified_raw_requests.json) [第三轮编译缓存快照](../evidence/SG_usage_recovery_G1276/SG_G1317_32K_P_compile_snapshot_after_W3.json) [第四轮实机启动](../evidence/SG_usage_recovery_G1276/SG_G1317_32K_W4_actual_input_and_worker.json)

全量 GSM8K 已挂接到原 32K 正式 runner 的终态。续跑进程先核对两阶段各 32 条请求、同一冻结数据及原始输出 SHA，再核查 P/D 空闲、同一 CPU 客户端和至少 14740 秒的资源窗口。实机只读检查确认预热期间质量启动被共享工作负载锁挡住。当前只是等待链路已就位，实际 GSM8K 请求与成绩仍待正式阶段终态。 [终态核对工具](../code/SG_phase_audit_G1205/verify_SG_formal_terminal_g1427.py) [等待与实机互斥记录](../evidence/SG_usage_recovery_G1276/SG_32K_to_GSM8K_event_continuation_G1427.json)

## 64K、128K、256K 的连续运行窗口

现有 G1317 的配对锁进程源码上限为 108000 秒。此前为 128K 和 256K 规划的首次冷测加重放分别需要 136480 和 272480 秒，均超出该上限。此前的长 CPU 租期从正式阶段起算，预热续跑则会校验客户端身份。新的候选租期从预热开始计算，覆盖五轮独立预热、冷测及重放，并为 P/D 初始化留出原规划时间。三个档位的配对锁和 CPU 客户端源码已在实机预置，CPU 只读校验确认三份配置使用同一镜像、CPU 限额和挂载合同。配对锁互斥已在隔离文件验证。真实长档阶段耗时和租期门槛须在各档启动前核验。 [租期预算和实机预置](../evidence/SG_usage_recovery_G1276/SG_wide_full_campaign_lease_preflight_G1424.json) [配对锁](../code/SG_wide_PD_window_G1348/workload_guard_wide_g1424.py) [CPU 配置生成](../code/SG_wide_CPU_window_G1351/prepare_SG_CPU_full_campaign_g1424.py)
