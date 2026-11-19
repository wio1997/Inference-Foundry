# DeepSeek-V4.1 W4A8 运行配置对照

对照的是 SGLang G1317 当前运行版与 vLLM G630 四档正式基线。两端各 8 张 Ascend 910B1，aarch64。P、D 两台主机上的历史与当前实际加载清单均匹配同一组 95 项权重校验记录。SGLang 的 24 份输入清单逐项匹配 vLLM 四档实跑清单，覆盖每档 5 份独立预热输入和 1 份正式输入。每个正式输入在 vLLM 的冷测和重放中使用相同 SHA。32K 正式测试进度见下文。 [实机配置、权重及数据绑定](../evidence/SG_usage_recovery_G1276/SG_G1317_native630_actual_profile_alignment.json) [vLLM 四档结果](https://github.com/wio1997/Inference-Foundry/blob/cff38e0021cd7f859dea4b8a403b06c3791eb7dd/docs/016-native032-four-tier-first-matrix.md)

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

当前 SGLang G1317 的 P/D 启动日志都将 decoder SWA bounded replay 解析为关闭，固定源码的自动启用条件未覆盖 Ascend。vLLM G804 的后段层尾窗已有实测设备形状，后续比较要报告这个运行配置和实际 query 行数的差异。SGLang 模型源码已有尾窗路径。若接入 Ascend，该路径会先在第 20 层（从 0 编号）为全量行写压缩 KV，再切换到尾窗元数据。验证要覆盖这次 KV 写入、SWA 窗口布局、DSPARK 状态交接以及需要完整行的 hidden state 和 prompt logprob 路径。当前 SGLang 尾窗的设备正确性和性能未验证。 [SGLang 两端日志与固定源码](../evidence/SG_usage_recovery_G1276/SG_G1317_P_D_actual_tail_gate_G1463.json) [核对代码](../code/SG_prefill_tail_G1463/audit_SG_tail_runtime_g1463.py) [vLLM 后段尾窗的实际核验](https://github.com/wio1997/Inference-Foundry/blob/5d5094d6c65990a23cad2abcafeb19876b8d4845/docs/018-native032-profile-capture-and-logical-work.md#prefill-后段层的尾窗工作量)

同一固定镜像的 V4.1 配置默认选择包含尾窗逻辑的 mHC 前向。用现役源码中的 CPU 行数函数和封存的 65 个 vLLM 调度步骤核对，普通 EXTEND 与 DSPARK 的 FULL 捕获都复现了vLLM 设备实测的尾窗行数。包含 prompt logprob 的用例只保留对应请求的完整 EXTEND，非 DSPARK 的完整 hidden-state 捕获保留全批，非纯 EXTEND 维持原长度。该检查覆盖行数选择，设备位置索引、压缩 KV、跨节点交接和输出仍待上卡核验。 [固定配置源码](../code/sglang/srt/configs/deepseek_v41.py) [行数函数复算](../evidence/SG_usage_recovery_G1276/SG_G1317_tail_reader_policy_CPU_G1464.json) [复算代码](../code/SG_prefill_tail_G1463/check_SG_tail_reader_policy_g1464.py)

G1317 两端启动日志均确认 target SWA 使用 paged 存储。若启用 decoder 尾窗，固定 Radix 代码会为共享命中保留一个 SWA 窗口用于重算，同一 streaming session 借用自身记录时保留原匹配上限。CPU 用例中，32K 输入和外部上限 32,767 时，生成的 Radix key 上限为 32,640，实际命中还会受页对齐和缓存状态影响。尾窗的固定几何公式显示，15-token 短分块接在 8192-token 分块后，首行在历史页完整驻留时可读 114 个因果位置。后续上卡要检查这些历史页的物理槽位、共享前缀重放和 P/D 交接。 [两端存储日志和 CPU 核验](../evidence/SG_usage_recovery_G1276/SG_G1317_tail_SWA_radix_boundary_G1465.json) [复算代码](../code/SG_prefill_tail_G1463/check_SG_tail_radix_limits_g1465.py)

当前 Decode 的实际后端为 dsv4，scheduler 环境确认静态 VERIFY。固定源码在有 CPU 长度镜像时给标记传入步前逻辑长度加本步验证宽度，镜像缺失时可退回页对齐的分配长度。两种来源写同名 g_sk 字段，缺少镜像的 draft 标记也可以没有 query/KV 字段。已用现役源码的 CPU 函数验证这三种输出。归一化要求逐步提供独立确认的边界，在 vLLM 906 个实际调度步上，两种逻辑边界的公式都重现 711,816,147 个未压缩因果 query–token 对。SGLang 同批实际标记的逐步来源待采集。 [运行源码与验证](../evidence/SG_usage_recovery_G1276/SG_G1317_verify_context_boundary_G1448.json) [逐步归一化工具](../code/SG_D_verify_work_G1448/normalize_verify_work_g1448.py) [vLLM 固定证据](https://github.com/wio1997/Inference-Foundry/blob/1cd4d843dcb21dbd4d063f5e55de2de2f88e36f8/evidence/profile_capture_G804/derived/D0_target_position_ledger_G1447.json)

PR #12 提供适配源码和同版本来源参考。其 910B3 和 16K/C4 工况与本任务 910B1、C16、四档输入不同，本任务性能数字以实际运行版为准。vLLM 32K 重放两端均有本地命中，64K 为低命中，128K 和 256K 的本地命中均为零。两框架的热态对照需结合 P、D 本地命中和跨节点接收计数。 [PR #12 与旧镜像的源码对照](https://github.com/wio1997/Inference-Foundry/blob/cff38e0021cd7f859dea4b8a403b06c3791eb7dd/docs/004-PR12-source-comparison.md) [原生 0.32 现场源码及版本校验](https://github.com/wio1997/Inference-Foundry/blob/cff38e0021cd7f859dea4b8a403b06c3791eb7dd/docs/008-required-vllm032-and-native-source-refresh.md)

vLLM 的实际快照已逐档复算 D 本地与连接器 logical token。SGLang 首轮独立预热的实际首尾快照给出 P rank 0 本地、主机和存储命中均为零，P 传输计数增加 16。当前快照没有与 vLLM 连接器逻辑 token 对应的 D 指标。热态对照将分别报告各自可验证的本地命中、连接器记账及传输证据。 [vLLM 四档连接器计数](https://github.com/wio1997/Inference-Foundry/blob/31850432f9534faf776c94384d620d707e81a0c8/docs/024-native032-PD-prefix-accounting.md) [SGLang 首轮缓存和传输快照](../evidence/SG_usage_recovery_G1276/SG_G1317_32K_W1_cache_and_P_transfer_metrics.json)

固定源码中 `prefill_effective_tokens_total` 由 Prefill 报告路径更新。Decode 在预分配时分别取得本地设备命中和可向 Prefill 承诺的总前缀，API `cached_tokens` 将两端复用合并计数。实机短重放原始快照中，P 的 `device_hit` 增加 256，D 的 Prefill 系列保持为零。D 的零读数只覆盖已注册的这条 Prefill 序列，独立的 D 本地命中须从 Decode 路径确认。 [固定源码位置](../evidence/SG_usage_recovery_G1276/SG_G1317_actual_prefix_cache_scope_source_chain.json) [短重放前后快照](../evidence/SG_usage_recovery_G1276/SG_G1317_short_replay_prefill_counter_scope.json)

正式门槛的辅助工具已在实际服务器部署，以原首轮快照和原正式 runner 的读计数逻辑做了兼容验证。读数输出保留 P rank 0 的验证结果，并把 D 的独立命中保留为待验证。原首轮中，P rank 0 的 sender 直方图记录 16 条传输，累计 1540 MB。跨框架报告将分别列出该直方图、连接器逻辑 token 和设备侧证据。 [门槛处理工具](../code/SG_formal_cache_scope_G1385/scope_SG_formal_prefix_evidence_g1385.py) [原 runner 兼容验证](../evidence/SG_usage_recovery_G1276/SG_G1317_formal_cache_scope_runner_compatibility.json) [P sender 直方图](../evidence/SG_usage_recovery_G1276/SG_G1317_32K_W1_prefill_sender_transfer_payload.json)

针对 D 本地命中，新增了独立采集用的 rank 0 profiler 标记。固定源码中，提交函数只有在传输轮询成功后调用，且会核对 bootstrap room。标记在提交成功后输出此次请求的 L1 设备、L2 主机、L3 存储匹配长度及声明前缀长度，区分已知零命中和缺少匹配信息。CPU 测试复用了镜像实际的 `DecodePrefixMatch` 定义，验证失败、模拟传输和其它 rank 不产生标记。代码已在服务器校验，实际 trace 尚待采集。第二轮 32K 同进程已处理 180224 个新增输入 token，完成 5 次 KV 传输。 [固定提交边界](../evidence/SG_usage_recovery_G1276/SG_G1317_decode_prefix_commit_boundary_source.json) [采集代码](../code/SG_D_prefix_trace_G1387/decode_prefix_annotation_g1387.py) [CPU 验证](../evidence/SG_usage_recovery_G1276/SG_G1317_decode_prefix_annotation_fixture.json) [服务器校验](../evidence/SG_usage_recovery_G1276/SG_G1317_decode_prefix_annotation_stage.json) [第二轮采样](../evidence/SG_usage_recovery_G1276/SG_G1317_32K_W2_fifth_KV_transfer.json)

固定镜像中的 aarch64 PyTorch 2.10.0+cpu 成功将 4 条测试标记导出到 Chrome trace。读取器识别到 2 条有匹配信息的新请求标记，其中 1 条为零命中，另有 1 条缺少匹配信息和 1 条重建标记。用 CPU 标记构造 SQLite 测试库验证了字符串关联，随后读取了原始 G981 昇腾 Decode SQLite。G981 采集早于该标记，读取结果为 0 条标记且请求覆盖不完整。新增标记的 NPU 导出以及与实测请求的逐条关联仍待验证。 [读取器](../code/SG_D_prefix_trace_G1387/audit_decode_prefix_trace_g1389.py) [CPU 原始导出校验](../evidence/SG_usage_recovery_G1276/SG_G1317_decode_prefix_actual_CPU_profiler_export.json) [G981 SQLite 对照](../evidence/SG_usage_recovery_G1276/SG_G981_original_D_SQLite_prefix_reader_control.json)

独立采集候选的关联器用完整输入 token 序列的 SHA256，将 Decode 标记关联到冻结清单，再按 `data_id` 核对 AISBench 原始请求全文、成功状态及输入输出 token 数。vLLM 已归档的同批 16 条原始记录全部通过逐条核对。关联逻辑使用这些原始记录和构造的标记验证，缺失、重复和外来标记均无法通过完整性检查。新增标记和关联器已在 Decode 主机做源码校验，现役 G1317 未装载该标记，SGLang 实际 NPU 标记仍待采集。 [固定请求路径](../evidence/SG_usage_recovery_G1276/SG_G1317_decode_prompt_identity_fixed_source.json) [原始请求校验](../evidence/SG_usage_recovery_G1276/SG_G1317_decode_prompt_identity_native_raw_fixture.json) [标记代码](../code/SG_D_prefix_trace_G1387/decode_prefix_prompt_annotation_g1391.py) [关联器](../code/SG_D_prefix_trace_G1387/bind_decode_prefix_profile_g1391.py) [服务器校验](../evidence/SG_usage_recovery_G1276/SG_G1317_decode_prompt_identity_stage.json)

32K 首轮预热的 16 条 SGLang 原始请求已在运行主机按 `data_id` 逐条与只读输入数据核对，并从原始结果复算服务端 token 数和 AISBench 指标。第二轮同进程仍在线时，复核器返回进行中，留待该轮真实终态再验收。 [首轮原始请求校验证据](../evidence/SG_usage_recovery_G1276/SG_G1317_32K_W1_exact_raw_dataset_binding_W2_live_control.json) [复核器](../code/SG_phase_audit_G1205/audit_SG_32K_warmup_request_binding_g1394.py)

32K 第二轮终态复核通过。16 条原始请求均与第二轮独立冻结输入匹配，服务端每条实际输入 32768、输出 1024，AISBench 耗时 7887.82 秒。随后在同一 P/D 实例启动第三轮，已核对前两轮被复核并跳过、第三轮使用独立输入及实际客户端进程。 [第二轮原始请求复核](../evidence/SG_usage_recovery_G1276/SG_G1317_32K_W2_actual_raw_request_binding.json) [第三轮实际启动](../evidence/SG_usage_recovery_G1276/SG_G1317_32K_W3_actual_input_and_worker.json)

## Prefill 索引路径的工作量差异

同一输入清单的首轮预热两端均完成 16 条请求，实际峰值并发 16，每条输入 32768、输出 1024。SGLang 耗时 7989.66 秒，vLLM 耗时 46.08 秒。这组尚未计分的样本用于界定问题规模。SGLang 的固定 Prefill 索引路径用 `einsum` 生成逐 head 稠密分数，聚合后再掩码和 top-k。vLLM 的独立诊断采集在 Prefill rank 0 实测到 520 次 `VllmAscendQuantLightningIndexerV2`，任务累计 1.3505 秒。两条证据支持索引实现不同，差距归因仍需在 SGLang 匹配 trace 中核对索引评分、其它算子、调度与通信的时间分布。 [SGLang 首轮审计](../evidence/SG_usage_recovery_G1276/SG_G1317_32K_W1_verified_raw_request_audit.json) [vLLM 首轮审计](https://github.com/wio1997/Inference-Foundry/blob/63ef19ee22ab0c7de4ed951931c4ce0348c3995b/evidence/native_first_warmup_G676/warmup1_32768_audit.json) [vLLM 设备算子统计](https://github.com/wio1997/Inference-Foundry/blob/63ef19ee22ab0c7de4ed951931c4ce0348c3995b/evidence/profile_capture_G804/derived/P0_indexer_operator_aggregate_G1395.json)

按实测首块形状、128 MiB 原生分块上限和无缓存连续输入推算，精确后缀裁剪可减少的逻辑评分位置在 32K、64K、128K 和 256K 分别为 19.35%、10.88%、5.80% 和 3.01%。在 32K，原路径、精确裁剪和按 1024 向上取整分别涉及 5、161 和 17 种 query 行数与 K 宽度组合，向上取整后仍可少算 14.99%。该数量供设备验证选择形状，实际编译行为和耗时尚待测量。即使把 SGLang 首轮的全部墙钟时间都按 19.35% 线性缩减，仍需约 6444 秒。只靠减少这部分评分位置不足以解释首轮差距。 [计算记录](../evidence/SG_usage_recovery_G1276/SG_G1317_causal_indexer_shape_and_warmup_gap.json) [计算脚本](../code/SG_causal_prefill_score_G1362/estimate_causal_score_shape_tradeoff_g1396.py)

保留原生 Top-K 和写回的因果后缀裁剪通过了固定 A2 镜像上的四个 CPU 用例。用例覆盖全掩码块、非单调可见长度、FP32 权重和 C2 连续输入，FP32 分数逐 bit、Top-K 与物理槽位逐项一致。原探针直接比较切片终点，遇到原生末块终点超过请求行数会误报，现改为核对实际行、token 和可见长度。用例输入为生成的 BF16 Q/K，实际 FP4 上游量化、NPU 裁剪正确性及耗时待验证。 [CPU 用例](../code/SG_causal_prefill_score_G1362/verify_causal_trim_CPU_g1455.py) [阶段证据](../evidence/SG_usage_recovery_G1276/SG_causal_trim_fixed_A2_CPU_gate_G1455.json)

后续 Prefill trace 优先对照约 8K 个 query 的设备工作，区分单请求和双请求调度。SGLang 采样会用同一逻辑步骤下的实际 query 行数及各请求前缀长度关联稠密评分标记，同时记录标记内实际 K 行数。若两个框架的调度拆分或压缩布局不同，保留各自形状及可见 K 的差异，按匹配的逻辑工作量对照。 [vLLM 逐任务形状及调度分布](https://github.com/wio1997/Inference-Foundry/blob/2eab02ef840d5844e2076eaab3def9d7c0b56cfe/evidence/profile_capture_G804/derived/P0_indexer_operand_shapes_G1418.json) [SGLang 评分标记](../code/SG_P_step_attribution_G1355/index_score_annotation_g1355.py)

SGLang 的设备归属分析支持通过 `CONNECTION_IDS` 找到实际的 PyTorch 算子，再用同线程 forward 区间核实归属。用封存的 G981 Decode rank 0 核验，72 个任务关联到四个实际步骤，其他任务保留各自的未归属原因。该历史库没有 Prefill score 标记。隔离的 SQLite 用例验证了跨线程 score marker、重复连接和归属冲突。运行分析器时，将 `code/SG_profile_readiness_G1218` 加入 `PYTHONPATH`，并传入冻结数据库路径、SHA256 和输出路径。 [实际历史库与测试结果](../evidence/SG_usage_recovery_G1276/SG_D0_G981_framework_connection_attribution_G1420.json) [分析器](../code/SG_P_step_attribution_G1355/attribute_SG_prefill_steps_g1420.py) [vLLM 逐步参考](https://github.com/wio1997/Inference-Foundry/blob/6045560f4d2ba1fc7db4ab3abbbea0cacfe77380/evidence/profile_capture_G804/derived/P0_indexer_scheduler_bridge_G1419.json)

已核对当前 P/D 容器内的详细步标记源码。纯 `EXTEND` 记录 query、KV、query 平方及 query 与 KV 乘积的总和，KV 以 query 分块末尾的原始逻辑序列长度计。单请求可直接还原，双请求 query 行数不同时可解出各自前缀长度。等长 query 的歧义与其他未覆盖的批次会在输出中标记。用 vLLM 封存的 65 个实际非零步骤先计算同口径汇总再还原，50 个单请求和 15 个双请求的结果均与原始逐请求长度一致。SGLang 实际 trace 待采集，压缩后的可见 K 及设备工作量仍需核对。 [实机源码和离线验证](../evidence/SG_usage_recovery_G1276/SG_G1317_prefill_query_KV_geometry_contract_G1421.json) [还原工具](../code/SG_P_step_attribution_G1355/derive_prefill_request_geometry_g1421.py)

固定 Prefill 评分源码按绝对位置计算压缩 K 可见范围。对连续 C2 路径，位置 `p` 可见 `floor((p+1)/2)` 个压缩条目。八个实际 rank 的首块回执均确认 NPU 专用评分预算为 128 MiB，包内默认值为 1 GiB。将 vLLM 封存的 65 步逐请求形状代入同一源码口径，单次索引遍历预计 2589 个评分分块。稠密评分覆盖约 53.58 亿 query–K 对，其中约 42.95 亿对在 top-k 前满足因果范围，均为每个索引头的几何计数。后续 trace 用这些形状核对实际分块和耗时，并分别记录 vLLM 的页容量与有效逻辑长度。 [实机上限与逐步模型](../evidence/SG_usage_recovery_G1276/SG_P_logical_causal_work_at_native_geometry_G1422.json) [计数工具](../code/SG_causal_prefill_score_G1362/prefill_logical_score_work_g1422.py)

将 vLLM G804 的 rank 0 索引器时间按精确调度步绑定到 C2 逻辑工作量。65 步中，工作量最低和最高四分组的 fused 算子任务累计耗时中位数分别为 12.27 ms 和 31.65 ms。仅用两框架共有的步级聚合字段即可对 57 步给出精确因果对数，其余 8 步的上下界相差 1 对。SGLang 可用同一逻辑口径处理无法逐请求还原的步。其计分标记和 vLLM fused 算子的计时边界不同，两边设备耗时比仍待同批实测及边界核验。 [逐步实测与推导](../evidence/SG_usage_recovery_G1276/SG_C2_work_by_native_P0_indexer_step_G1430.json) [分析工具](../code/SG_causal_prefill_score_G1362/audit_c2_work_by_trace_step_g1430.py)

固定 A2 镜像的 aarch64 CPU 探针执行原评分函数，核实 `aten::einsum` 子调用 `aten::bmm` 的输入为 `[1, Q×32, 128]` 和 `[1, 128, K]`，同一探针中另一种 einsum 的输入形状不同。按当前实例实测第一块代入，候选形状为 `[1, 16384, 128]` 和 `[1, 128, 4096]`。历史 Decode 数据库的 PyTorch 形状为空，设备任务形状仍可通过字符串表查询。CPU 模式的 Ascend 库导出未通过，须在本次 Prefill 实际采集核对形状字段与设备任务关联。 [形状实测和失败记录](../evidence/SG_usage_recovery_G1276/SG_G1317_score_trace_shape_controls_G1438.json) [固定镜像检查](../code/SG_P_step_attribution_G1355/probe_C2_score_trace_shapes_g1438.py)

在当前 G1317 的 P/D 各八个 scheduler 中，手动起停使用镜像默认的 profiler 路径，请求中的 `GPU` 活动映射到 NPU。两个 trace 分区的可用空间快照分别为 1.95 TB 和 1.86 TB。采集关闭后先清点每 rank 的 CPU op range、设备数据与结束记录，再检查派生数据库。封存的 SGLang G981 Decode 八个 rank 均检出原始文件；清单工具也能识别原始文件齐全、待离线解析的情形。实际 G1317 采集的空间需求和事件覆盖仍待该次导出核验。 [当前运行环境与历史文件核对](../evidence/SG_usage_recovery_G1276/SG_G1317_profile_export_and_raw_rank_inventory_G1423.json) [原始文件清单工具](../code/SG_long_profile_G1305/inventory_SG_raw_profile_g1423.py)

匹配采集结束后，验收工具先校对实际 driver、请求审计和 profiler 起停回执，再逐条核对 AISBench 的 16 条冻结输入并生成本次采集的校验和。两端各八个物理 rank 的原始文件按此校验和汇总。工具会报告缺少原始收尾文件以及派生库尚未生成的 rank。用 vLLM G804 实际 16 条请求和构造的 SGLang 控制及导出文件验证了这条链路，P/D 两端已通过加载检查。实际 SGLang 事件覆盖待采集后核对。 [验收工具](../code/SG_long_profile_G1305/intake_SG_matched_profile_g1431.py) [两端回执与离线验证](../evidence/SG_usage_recovery_G1276/SG_G1317_matched_profile_intake_readiness_G1431.json)

已确认当前 A2 镜像两端的 torch-npu analyse 接口一致。将历史 G981 Decode rank 1 的已关闭原始文件复制到隔离目录，用固定镜像、单 CPU 和 DB 模式重新导出，复核生成库中的任务数、设备时间区间并集及 step/replay 统计均与封存库一致。原库 8 个 rank 都有实际设备事件，每 rank 计算任务数为 10,889，rank 0 的通信任务比其余 rank 多 2,144 个。当前 32K 同批采集的全 rank 事件仍待核验。 [单 rank 导出代码](../code/SG_long_profile_G1305/export_SG_profile_rank_db_g1445.py) [固定镜像导出与八 rank 实测](../evidence/SG_usage_recovery_G1276/SG_A2_offline_rank_export_and_G981_events_G1445.json)

两端已部署逐 rank 事件检查。它按冻结的采集身份和各 rank 元数据摘要核对原始文件，检查派生数据库中的物理设备 ID、计算与通信事件、角色对应的 scheduler 标记和逻辑字段覆盖。历史 G981 的 Decode 八个 rank 均通过设备事件检查，每卡各有 2 个 DRAFT 和 2 个 VERIFY scheduler 标记，但没有对应的 query/KV 字段。错 rank、缺少派生库及元数据不匹配均有独立回执。 [逐 rank 事件检查](../code/SG_long_profile_G1305/audit_SG_profile_event_coverage_g1446.py) [两端部署与历史实测](../evidence/SG_usage_recovery_G1276/SG_32K_all_rank_event_coverage_stage_G1446.json)

匹配采集的事件检查现可接入离线导出的独立 rank 副本。它先校验原始与副本的采集文件路径、字节和 SHA256，再读取副本数据库并验证物理设备号及 scheduler 事件。首次检查包含 SDK 导出的分析产品和日志，两个导出方式产生了不同的文件。修正后，G981 Decode 的 rank 1 副本有 90 份采集文件与原始逐份一致，用该副本和其余原库完成八 rank 设备事件复核。两端新入口已经部署，G1317 的匹配采集尚未到达验收阶段。 [副本核验与现场回执](../evidence/SG_usage_recovery_G1276/SG_32K_offline_rank_event_intake_G1466.json) [事件检查](../code/SG_long_profile_G1305/audit_SG_profile_event_coverage_g1446.py)

固定 A2 实机包的 `libopapi` 和 910B 算子目录均有 BF16 `LightningIndexer`。CANN 9.1.0 的 V2 文档列出的压缩比为 1、4、128。对已压缩的连续 C2 K，可按 query 位置奇偶拆分。每组内相邻 query 的可见 K 数递增 1，适用常规右下因果掩码。将 65 个已封存的 vLLM 步骤代入，80 个请求分段的逐行可见范围都与 C2 原公式一致，合计 160 组。候选保留现有 FP4 来源的 Q/K。实际 910B dispatch、top-k 集合及次序、物理槽位与耗时仍须上卡验证。 [实机 ABI 与离线几何核验](../evidence/SG_usage_recovery_G1276/SG_A2_lightning_indexer_abi_and_C2_geometry_G1425.json) [分组代码](../code/SG_causal_prefill_score_G1362/c2_lightning_indexer_geometry_g1425.py) [CANN 9.1.0 原生接口](https://github.com/hicann/ops-transformer/blob/9.1.0/attention/lightning_indexer/docs/aclnnLightningIndexer.md) [V2 压缩比说明](https://github.com/hicann/ops-transformer/blob/9.1.0/attention/lightning_indexer_v2/docs/aclnnLightningIndexerV2.md)

用固定 A2 镜像的无 NPU 容器确认 `npu::npu_lightning_indexer` 的实际 PyTorch 注册签名。组件探针复用固定 SGLang 的评分和写回源码，保留算子原始输出供无效索引检查。写回前将算子的负数哨兵映射到可见范围之外，由 SGLang 原有排序及槽位映射完成输出。CPU 夹具通过四种形状，并分别拒绝不合法索引和物理槽位变化。真实 fused 算子数值、耗时和模型结果仍待上卡。 [探针](../code/SG_causal_prefill_score_G1362/probe_c2_lightning_indexer_g1426.py) [CPU 夹具与实机注册](../evidence/SG_usage_recovery_G1276/SG_A2_LI_component_probe_CPU_wiring_G1426.json)

C2 探针仍要求写回后的本地索引和物理槽位逐项相同。固定写回源码会先按本地索引排序，CPU 核验确认仅改变算子输出顺序时仍可精确一致。两个奇偶分组分别注入重复索引，两行都被判为选择与槽位不一致。首次演练把两组的失败行误算为一行，按分组修正后通过。历史 CPU 演练与本轮都未调用 stock NPU kernel，设备正确性和组件耗时仍需实测。 [C2 探针](../code/SG_causal_prefill_score_G1362/probe_c2_lightning_indexer_g1426.py) [固定镜像演练](../evidence/SG_usage_recovery_G1276/SG_C2_LI_canonical_writeback_diagnostics_G1449.json)

C2 短用例入口已在 Prefill 节点安装。实际启动需要同时取得 P/D 工作负载锁、0 号卡与专用 CPU 锁，核对上个 Prefill 控制器正常释放后只映射 0 号卡。短用例容器退出和卡上进程清空均有证据才释放登记。64K 入口保留原角色退出检查，并能识别可核对的单卡短测释放记录。真机预检拦截了当前占用，离线演练验证了锁冲突、释放及异常保留。 [单卡短用例入口](../code/SG_causal_prefill_score_G1362/run_owned_C2_short_probe_g1450.py) [真机预检和离线演练](../evidence/SG_usage_recovery_G1276/SG_C2_single_card_and_wide_handoff_G1450.json)

## 32K 正式冷测和重放的时间窗口

第五轮独立预热已验收 16/16。冻结输入、服务端用量及三份 AISBench 原始文件通过核对，实际耗时 7,879.52 秒。第三轮与第五轮的同一 Prefill 实例各有 211 个持久编译文件，核验范围内的文件无变化。接力复查并跳过五轮原预热，随后提交首轮 32K 正式冷测。控制端和 AISBench 冷测进程均已在现场通过命令核对。重放与首轮正式终态仍待完成。 [预热终态和正式冷测启动回执](../evidence/SG_usage_recovery_G1276/SG_G1317_W5_terminal_and_first_formal_cold_live_G1459.json)

首轮冷测期间利用已认证连接采样主机时钟。各节点有效样本 12 条，按收发时刻夹算，Prefill 主机时钟比 Decode 快约 19.9 秒。当前正式原进程已记账第一个 8,192-token Prefill 分块，本地命中增量为零。这次短窗时差仅用于解释同窗口主机日志，正式压测指标取自 AISBench 原始客户端计时，后续 trace 的设备时钟和采集窗口要独立核验。 [时钟比对工具](../code/PD_clock_G1460/sample_ssh_clock_bounds.py) [原始采样与冷测计数](../evidence/SG_usage_recovery_G1276/SG_G1317_P_D_host_clock_and_live_cold_G1460.json)

首轮冷测的前四次 Prefill KV 交接在同一 rank 的直方图中累计 480.61 秒，发送侧按源码记账 385 MiB。四次均超过 5 秒的最高有限桶。当前 Ascend sender 的计时从 Prefill 末块入队到发送侧轮询确认完成，覆盖队列阶段和轮询观察。链路耗时与实际带宽仍需独立的 SDK 或 trace 事件。Decode 另累计 20 次分配等待，其中 19 次在 25 毫秒以内、1 次超过 2.5 秒。两个角色的样本处于各自请求阶段，点对点对齐需用同一请求标识。 [原始指标差值与固定计时源码](../evidence/SG_usage_recovery_G1276/SG_G1317_first_cold_transfer_histograms_G1467.json) [复算代码](../code/SG_PD_transfer_G1467/audit_cold_transfer_histograms_g1467.py)

在原冷测日志中，23 条 Prefill 报告均对应单请求、8192 个新 token 和零缓存命中。前五段 Decode 活动各与一组连续的 Prefill 在途报告落在相同时间窗。按两次短窗实测主机偏差对齐，Decode 首次活动报告比每组 Prefill 第二次报告仍在途早约 80–120 秒。日志只有批次级字段，缺少 P/D 共用请求 ID 和逐次传输事件。现场差异提示需要沿同一请求检查发送侧轮询确认与 Decode 就绪时间，再区分调度等待和链路工作。 [双端原始日志索引及时钟校准](../evidence/SG_usage_recovery_G1276/SG_G1317_cold_PD_log_windows_G1468.json) [复算代码](../code/SG_PD_transfer_G1467/audit_live_cold_PD_windows_g1468.py)

核对现役 Prefill 源码确认，批次报告在结果处理阶段产生。该阶段先等待可用的复制完成事件，完成请求先调用末块 KV 发送，再打印报告。实际 64 条报告均为单请求 8192 个新 token，相邻报告间隔中位数 120 秒。按三次主机时钟采样对齐，前 14 个客户端首条 SSE 比每组第 4 条报告晚约 119–122 秒，与随后一条报告相距约 1 秒。该结果按时间顺序对齐，逐请求 KV 完成时间和设备执行区间仍需 trace。 [现役源码与客户端时序复算](../evidence/SG_usage_recovery_G1276/SG_G1317_cold_P_result_cadence_first_SSE_G1480.json) [复算代码](../code/SG_first_token_comparison/audit_prefill_result_cadence_g1480.py)

冷测第 16 次 Prefill rank 0 KV 交接观测落盘后，末块入队至发送侧轮询确认的均值为 120.15 秒。两端现役 Ascend 源码确认，发送请求交给后台线程，该线程处理传输并向 Decode 发布状态。Decode 按本地状态及元数据检查将请求加入等待队列。Prefill 调度器另行轮询本地状态，并完成缓存清理、指标和本端输出。后续需要逐请求记录后台发送、Decode 就绪、Prefill 观察与 Router 首条转发四个时点。 [半程直方图和两端现役源码](../evidence/SG_usage_recovery_G1276/SG_G1317_cold_halfway_KV_worker_poll_path_G1481.json) [复算代码](../code/SG_PD_transfer_G1467/audit_ascend_worker_status_path_g1481.py)

核对两端现役启动参数及已部署的匹配采集驱动后确认，逐请求时序日志和 OpenTelemetry 均关闭。后续原实例采集会打开 Torch profiler 的详细标记，用于算子、调度和设备事件；独立 Decode 请求标记仍处于源码校验阶段。CPU 用例确认，固定原生日志可随两端各自的请求 ID 输出 `bootstrap_room`，用于另行诊断关联。日志中的 Decode transfer duration 覆盖进入传输队列至进入执行等待队列，设备传输耗时仍需独立事件。 [两端功能核验及 CPU 用例](../evidence/SG_usage_recovery_G1276/SG_G1317_planned_profile_request_observability_G1469.json) [核对代码](../code/SG_PD_request_observability/audit_matched_profile_capabilities_g1469.py)

同轮冷测的 Rank 0 启动日志与占用读数一致。Prefill 全量 KV 池容量为 2,352,384 个槽，快照中活跃 24,576、可驱逐 2,321,664；SWA 池容量为 49,536，活跃 8,320、可驱逐 33,664。Decode 全量池容量为 617,600，活跃 525,312、可驱逐 67,584。按冻结输入各自使用独立的首个 128-token 页计算，Decode 即使把全部全量池用于前缀，也至多同时容纳 18 条完整 32K 前缀。本轮有 32 条请求。这个容量上限用于解释后续重放，逐请求的完整和部分命中需要单独核验。 [容量、占用和冻结输入依据](../evidence/SG_usage_recovery_G1276/SG_G1317_live_cold_KV_pool_residency_G1471.json) [复算代码](../code/SG_KV_pool_observability/audit_cold_KV_pool_residency_g1471.py)

首轮冷测的前 36 条 Prefill 报告均为单请求、8192 个新 token 和零缓存命中，报告中的等待队列有 14–15 条请求。现役启动配置的 Prefill 总输入上限为 16384，分块预算为 8192，未设置额外的 Prefill 请求数上限。固定源码先处理已有分块，再尝试加入等候请求。CPU 用例复现了一个 8192-token 分块耗尽本轮计算分块预算，即使分配器仍有容量也不能再加入 query；短分块在预算上可以继续申请，仍要满足独立的 KV 分配检查。由此可以解释该窗口的单请求批次，分块时长和改变分块后的端到端影响仍待实测。 [现役参数、批次报告和 CPU 用例](../evidence/SG_usage_recovery_G1276/SG_G1317_cold_prefill_chunk_admission_G1472.json) [复算代码](../code/SG_prefill_scheduler/audit_cold_prefill_admission_g1472.py)

首轮冷测已写出 9 条完整的 AISBench 临时结果。修正进度文件为列表结构后，观察器在读取前后均取得 9/32，并将 9 条临时记录逐条与冻结输入核对，均记录成功且每条为 32768 输入、1024 输出。该读取只使用完整换行记录，终态仍需核对完整的预测、性能文件和服务端用量。 [实机临时记录的验收范围](../evidence/SG_usage_recovery_G1276/SG_G1317_live_cold_client_first_nine_G1473.json) [临时读取器](../code/SG_formal_client_observability/audit_live_AISBench_snapshot_g1473.py)

从同一原冷测的稳定临时 SQLite 副本复算已返回的 9 条请求。它们在约 69 毫秒内发起，相邻首个 SSE 数据间隔为 480.216–480.264 秒，首到末 SSE 持续 66.58–122.04 秒。这段时序与每条 32K 输入占用四个约 120 秒 Prefill 分块相符，逐算子耗时仍待匹配 trace。读取器同时处理了 AISBench 状态文件短暂显示空列表的情况，临时记录可继续独立验证。整轮 TTFT、TPOT 和吞吐仍按终态文件验收。 [实际临时时序](../evidence/SG_usage_recovery_G1276/SG_G1317_live_cold_first_SSE_cadence_G1474.json) [时序复算代码](../code/SG_formal_client_observability/audit_live_AISBench_timing_g1474.py) [固定 SQLite 解码器](https://github.com/wio1997/Inference-Foundry/blob/5d5094d6c65990a23cad2abcafeb19876b8d4845/code/native_metrics_G751/raw_sqlite_timing_audit_g751.py)

以冻结数据中的请求编号与完整输入 SHA256 配对，已完成的 14 条 SGLang 临时记录在 vLLM 原始冷测中均属首批 C16。两端客户端从提交到首条 SSE 的平均时间分别为 3840.92 秒和 17.73 秒，首条到末条 SSE 分别为 92.14 秒和 10.79 秒。该子集平均端到端差距的 97.9% 在首条 SSE 之前。若保持首条 SSE 的时间不变，即使消除 SGLang 此后的全部时间，也只能缩小该子集平均差距的 2.36%。后续 trace 优先拆分首条输出前的请求准入、Prefill 算子、KV 交接和 Decode 工作，整轮成绩仍待验收。 [同输入客户端时序复算](../evidence/SG_usage_recovery_G1276/SG_G1317_32K_partial_first_SSE_gap_vs_native_G1479.json) [复算代码](../code/SG_first_token_comparison/compare_first_wave_g1479.py) [vLLM 首轮 32K 正式审计](https://github.com/wio1997/Inference-Foundry/blob/5d5094d6c65990a23cad2abcafeb19876b8d4845/evidence/native32_first_G691/audit_results.json)

原冷测随后写出 11 条符合冻结输入和用量的完整临时记录。与先前同进程的 9 条快照比较，临时 JSONL 的旧前缀逐字节一致，9 组旧时间戳的 SQLite 数组也逐条一致。进度文件在本次读取中有时为空，临时记录及独立副本可继续用于验证写入连续性。 [两次实际快照的比对](../evidence/SG_usage_recovery_G1276/SG_G1317_live_cold_temp_append_verified_G1475.json)

32K 冷测中途重新读取同一 Prefill 实例的缓存清单。与第 5 轮预热后的封存快照相比，211 个持久编译文件的路径、大小和修改时间一致，单独记录的 Ascend 工作目录中 73 个文件的元数据也一致。同一轮已独立核验 12/32 条临时请求。整轮编译缓存复核和冷测验收仍待完成。 [实机缓存清单和临时请求回执](../evidence/SG_usage_recovery_G1276/SG_G1317_live_32K_cold_prefill_compile_inventory_G1476.json)

已启动冷测和同输入回放的只读终点监听，现场核对了冷测客户端和父进程身份，并在两端试读。监听以原客户端进程退出为触发，采集 Prefill 缓存清单，再复查回放命令或 GSM8K 启动目录是否已出现。若下一阶段已开始，回执会标记采样时序，整轮性能仍以独立的正式请求审计为准。 [终点监听启动回执](../evidence/SG_usage_recovery_G1276/SG_G1317_formal_cold_replay_prefill_terminal_observer_G1477.json)

已为冷测和回放分别启动原始证据留存。每阶段的运行审计、编译和缓存记录落盘后，只读收集请求明细、SQLite 时间戳、AISBench 汇总与冻结配置，并独立复算指标和逐条比对输入。实机预检查验了同一客户端的第 5 轮预热文件布局及用量证据。两阶段的真实终态与成对比较仍待验收。 [独立证据留存启动回执](../evidence/SG_usage_recovery_G1276/SG_G1317_32K_independent_phase_raw_preservation_G1478.json)

四档正式对照表已用 vLLM G630 的八份实际冷热结果验证。表生成器检查冻结输入摘要、五轮独立预热、32 条 C16/O1024 正式请求、服务端用量和吞吐计量。重放同时列出 P/D 本地及连接器命中，只有 Prefill 的 SGLang 计数器无法证明 Decode 本地命中。当前无 SGLang 正式终态结果，比较栏为空。 [表生成器](../code/SG_formal_comparison_G1452/build_formal_PD_matrix_g1452.py) [实际 vLLM 基线接入及演练](../evidence/SG_usage_recovery_G1276/SG_formal_matrix_native_reference_admission_G1452.json) [原始 vLLM 四档结果](https://github.com/wio1997/Inference-Foundry/blob/a8c87daf33de03ed794be365cd742609fc3a48bb/docs/016-native032-four-tier-first-matrix.md)

第四轮 32K 独立预热的原 Prefill 实例已记录 51 个 8192-token 分块，相邻 50 次报告间隔均在 118–121 秒，中位数为 120 秒，与第一轮完整记录一致。这段日志的 Prefill 本地命中为零。当次两端设备快照的 AI Core 均为 99–100%，只覆盖采样时刻。第一轮实际全批耗时 7989.66 秒，最后一个首 token 在 7706.06 秒到达。若固定每条请求的首 token 时刻，只消除首 token 后耗时，整批最长可缩短 283.60 秒，即 3.55%。优先核对首 token 前的 Prefill、通信和交接。报告间隔尚无法分配到具体算子。 [分块记录、设备快照和收益上限](../evidence/SG_usage_recovery_G1276/SG_G1317_W4_prefill_cadence_and_W1_post_first_ceiling_G1432.json)

原 vLLM 冷测和重放的各 32 条原始请求均复算出客户端并发峰值 16，第 17 条在首条完成后才开始。两端使用同一固定 AISBench 镜像。SGLang 第二轮原始 SQLite 复算最长单请求用时 7887.70 秒，整轮 7887.82 秒。当前 Router 的 14400 秒为单请求上限，Decode 的 14400 秒为 KV 等待设置。正式 AISBench 整轮上限由独立阶段参数控制。 [vLLM 原始请求并发](https://github.com/wio1997/Inference-Foundry/blob/72a0cf3e22549a59bcdc0742c22c3a325abe2f4e/evidence/native32_first_G691/derived/formal_C16_HTTP_admission_G1408.json) [SGLang 第二轮请求复算](../evidence/SG_usage_recovery_G1276/SG_G1317_W2_raw_client_admission.json) [复算脚本](../code/SG_W1_latency_timeline_G1375/audit_32K_client_admission_g1408.py)

以两轮较慢的实测预热耗时估算，两个连续 C16 窗口约需 15979 秒。按每个正式阶段预留两个原 12900 秒预热窗口，冷测和重放合计需 52080 秒的实机租期，包含原有收尾余量。原 CPU 客户端租期无法覆盖这个预算。第五轮实际结束后须按现有 P/D、配对锁和续租后的 CPU 最早到期时间复核。重放时间估算保留零命中的情况，正式耗时和命中仍待实测。 [预算和实际租期来源](../evidence/SG_usage_recovery_G1276/SG_G1317_32K_formal_client_window_plan.json)

正式启动脚本按 CPU 容器的实际只读挂载校验输入。现场已核对 32 条冻结请求、C16 配置和两阶段共用的输入 SHA，原测试没有首次正式提交记录或计分产物。启动脚本同时纳入独立阶段上限和 Prefill 计数范围，Decode 命中保留待实测。组合验证使用模拟的第五轮终态门槛，当时现场运行第三轮预热。 [只读输入和提交账本检查](../evidence/SG_usage_recovery_G1276/SG_G1317_32K_first_formal_mounted_data_precheck.json) [启动门槛组合验证](../evidence/SG_usage_recovery_G1276/SG_G1317_32K_scoped_formal_launch_composition.json)

正式续跑会在第五轮原始请求及稳定性校验通过后，按封存 SHA 读取第三轮和第五轮的同一 Prefill 实例快照。比较范围为持久编译缓存清单的路径、大小和修改时间，Ascend 运行目录另行记录。对比器已部署到 Prefill 主机，隔离演练验证了拒绝预热未完成、缓存元数据改变或剩余租期不足的情况。 [编译快照比较器](../code/SG_phase_audit_G1205/compare_SG_prefill_compile_g1414.py) [Prefill 主机部署回执](../evidence/SG_usage_recovery_G1276/SG_G1317_P_compiler_comparator_physical_stage.json) [正式续跑演练](../evidence/SG_usage_recovery_G1276/SG_G1317_first_formal_event_controller_fixture.json)

第三轮 16 条原始请求均与该轮冻结输入匹配，每条实际输入 32768、输出 1024，耗时 7884.90 秒。空闲时采集了同一 Prefill 实例的第三轮快照，封存清单包含 211 个持久编译缓存文件。第四轮在原 P/D 实例和原 CPU 客户端启动，使用独立的第四轮输入，并复核跳过前三轮。 [第三轮原始请求复核](../evidence/SG_usage_recovery_G1276/SG_G1317_32K_W3_verified_raw_requests.json) [第三轮编译缓存快照](../evidence/SG_usage_recovery_G1276/SG_G1317_32K_P_compile_snapshot_after_W3.json) [第四轮实机启动](../evidence/SG_usage_recovery_G1276/SG_G1317_32K_W4_actual_input_and_worker.json)

第四轮 16 条原始请求全部通过冻结输入和用量校验，整轮耗时 7884.13 秒。空闲后 CPU 客户端完成同容器续租，容器 ID 和配置校验和均一致，随后复核跳过前四轮并启动独立第五轮。续租后的 CPU 窗口至北京时间 11 月 20 日 21:56:07。按现有保护预算，若要为 GSM8K 和匹配 profile 均留足时间，正式冷测与重放需在当天 13:33:57 前完成。第五轮终态和正式成绩仍待验证。 [第四轮审计、续租与第五轮启动](../evidence/SG_usage_recovery_G1276/SG_G1317_32K_W4_complete_W5_actual_start_G1436.json)

全量 GSM8K 已挂接到原 32K 正式 runner 的终态。续跑进程先核对两阶段各 32 条请求、同一冻结数据及原始输出 SHA，再核查 P/D 空闲、同一 CPU 客户端和至少 14740 秒的资源窗口。实机只读检查确认预热期间质量启动被共享工作负载锁挡住。当前只是等待链路已就位，实际 GSM8K 请求与成绩仍待正式阶段终态。 [终态核对工具](../code/SG_phase_audit_G1205/verify_SG_formal_terminal_g1427.py) [等待与实机互斥记录](../evidence/SG_usage_recovery_G1276/SG_32K_to_GSM8K_event_continuation_G1427.json)

G1317 的 GSM8K 复判工具已在 2204 校验，绑定 32K 正式审计、原 CPU 客户端、质量 worker 终态和同一组冻结请求，再与 vLLM 的 1319 条原始请求逐题复判。续租后，GSM8K 单项完整预算可保护到北京时间 17:50:27 启动，匹配 profile 自身则需要在 17:39:37 前开始准备。13:33:57 是保护两项完整预算的正式终态截止。历史 G1178 的 3258.89 秒仅供容量估算，该轮有一条传输失败。 [原始结果复判工具](../code/SG_G1317_quality_G1378/audit_G1317_GSM8K_terminal_g1437.py) [实际租期与校验回执](../evidence/SG_usage_recovery_G1276/SG_G1317_post_formal_renewed_window_and_GSM8K_terminal_stage_G1437.json)

32K 匹配 profile 的启动校验已按正式测试的角色计数范围更新。校验绑定冻结数据 SHA，要求 Prefill 冷测和重放均有实测计数，冷测命中增量为零。Decode 的两份回执须明确标记计数未观测。用已封存回执构造的离线校验通过，并拒绝了八组异常输入。更新后的校验工具已通过实机加载检查，仍待正式数据和实际 trace。 [校验工具](../code/SG_formal_cache_scope_G1385/verify_SG_matched_profile_cache_g1428.py) [离线与实机回执](../evidence/SG_usage_recovery_G1276/SG_G1317_matched_profile_cache_scope_G1428.json)

匹配 profile 已挂接到原 GSM8K 控制进程，要求实际 worker 终态回执和新的 P/D 空闲、CPU 租期及正式数据校验。实机已只读复核共用的 16 条 32K 输入、SGLang 配置 SHA 和输出挂载。GSM8K 和 profile 串行的合计预留为 30130 秒。此前按配对锁单独推算的截止为北京时间 11 月 20 日 14:41:53，实机启动取资源中最早到期时间。 [续跑回执与租期预算](../evidence/SG_usage_recovery_G1276/SG_G1317_quality_to_matched_profile_capacity_G1429.json) [终态观察器](../code/SG_long_profile_G1305/wait_SG_quality_terminal_g1429.py)

## 64K、128K、256K 的连续运行窗口

现有 G1317 的配对锁进程源码上限为 108000 秒。此前为 128K 和 256K 规划的首次冷测加重放分别需要 136480 和 272480 秒，均超出该上限。此前的长 CPU 租期从正式阶段起算，预热续跑则会校验客户端身份。新的候选租期从预热开始计算，覆盖五轮独立预热、冷测及重放，并为 P/D 初始化留出原规划时间。三个档位的配对锁和 CPU 客户端源码已在实机预置，CPU 只读校验确认三份配置使用同一镜像、CPU 限额和挂载合同。配对锁互斥已在隔离文件验证。真实长档阶段耗时和租期门槛须在各档启动前核验。 [租期预算和实机预置](../evidence/SG_usage_recovery_G1276/SG_wide_full_campaign_lease_preflight_G1424.json) [配对锁](../code/SG_wide_PD_window_G1348/workload_guard_wide_g1424.py) [CPU 配置生成](../code/SG_wide_CPU_window_G1351/prepare_SG_CPU_full_campaign_g1424.py)

128K 和 256K 的 P/D 源码已分别部署为候选 G14331 和 G14332，每档两端各校验 58 个源文件。模型启动参数沿用当前版本，Router 单请求超时和 Decode KV 等待分别配置为 36500 秒和 68500 秒。两档 P/D 生命周期分别预留 324130 秒与 620130 秒，已和各自配对锁及 CPU 租期计划核对。实际启动、KV 准入、每档五轮预热及正式冷测和重放仍待验证。 [候选源码校验和与预算](../evidence/SG_usage_recovery_G1276/SG_wide_128K_256K_actual_PD_source_preparation_G1433.json)

三个长档的首次预热校验按各自配置校验和绑定实际 P/D、Router、配对锁和 CPU 客户端。要求 P/D 每端八个 rank 的就绪证据、完成短请求功能检查，以及覆盖该档五轮预热和正式测试的剩余租期。前三轮后保留 Prefill 缓存快照边界。离线验收拒绝了混用实例、Decode rank 等待设置不一致和 CPU 租期不足。工具已在 CPU 主机核对三档实际配置，真实长档就绪及预热结果仍待实测。 [校验逻辑](../code/SG_wide_PD_window_G1348/wide_initial_gate_context_g1434.py) [部署回执和离线验收](../evidence/SG_usage_recovery_G1276/SG_wide_initial_gate_stage_G1434.json)

各长档的后两轮预热和首次正式阶段沿用同一 P/D 和 CPU 客户端，核对已完成预热、原始租期和正式输入提交账本。首次正式阶段比对同一 Prefill 实例的第三、第五轮编译清单，并按该档配置分别保护冷测与重放。三档离线演练覆盖了这三个边界，明确拒绝缓存变化和第五轮审计失败。工具已在实机加载，长档启动及测量仍待执行。 [续跑校验](../code/SG_wide_PD_window_G1348/wide_resume_context_g1435.py) [离线演练和部署回执](../evidence/SG_usage_recovery_G1276/SG_wide_resume_formal_gate_stage_G1435.json)

长档的只读观察器已在 P/D 两端安装。它核对当前实例的启动清单、每卡锁、权重加载、并行配置和队列空闲。Prefill 回执可用于启动 Router 前后的检查，Decode 回执包含八卡 target/draft 图及每个 scheduler 的实际等待值。两端均确认 64K、128K、256K 的候选目前没有启动句柄，观察器正确拒绝将候选文件判作就绪。 [观察器](../code/SG_wide_PD_window_G1348/observe_SG_wide_role_g1439.py) [部署回执和离线演练](../evidence/SG_usage_recovery_G1276/SG_wide_role_observer_actual_stage_G1439.json)

长档功能自检复用已在 G1317 实测的三请求逻辑，包括一次算术请求、一次短 SSE 和同输入重放。按各档实际 P/D 和 Router 绑定后，核对八卡 KV 传输和 Decode 图回放增量，保存原始请求、响应和指标校验和。工具已在 Prefill 主机校验。离线演练覆盖了实例错配、回执过期、原始文件变化和 rank 缺失，实际长档请求尚未发送。 [自检和终态复核工具](../code/SG_wide_PD_window_G1348/run_review_SG_wide_runtime_g1440.py) [实机加载与离线演练](../evidence/SG_usage_recovery_G1276/SG_wide_functional_runtime_tool_stage_G1440.json)

三档 AISBench 启动入口已在 CPU 主机校验实际 runner、用量证据和 CPU 配置。它先执行 runner 的无请求预检并核对已完成的预热，再检查队列与租期。初始三轮、第四轮、第五轮和首次正式阶段各保留独立启动句柄。离线演练覆盖了三档共 12 个阶段组合，包括拒绝旧 CPU、过期回执和错误重放时限。长档正式输入尚未提交。 [启动入口](../code/SG_wide_PD_window_G1348/launch_SG_wide_AISBench_g1441.py) [部署回执和离线演练](../evidence/SG_usage_recovery_G1276/SG_wide_AISBench_dispatch_stage_G1441.json)

长档切换前的只读检查在 P/D 两端分别校验了三档各 58 个部署文件。两端八张卡的登记与内核独占锁均对应 32K 实例，三档候选都拒绝继续启动。离线演练覆盖登记缺失、登记先于锁释放、正常释放和主机不符。后续实际认领仍由对应模型控制器在持锁后执行。 [检查入口](../code/SG_wide_PD_window_G1348/check_SG_wide_role_resources_g1442.py) [双端实测和离线演练](../evidence/SG_usage_recovery_G1276/SG_wide_role_resources_G1442.json)

64K 的角色启动入口已在 P/D 两端安装。执行启动前会核对旧控制器退出、原容器停止、八张卡的登记与进程清单，以及本轮独占工作负载锁。Decode 还需要本轮 Prefill 的已验证启动回执及 SHA256。离线 CPU 演练验证了重复启动拦截和回执时效；两端真机预检均确认 32K 仍占用资源，64K 未启动。 [角色启动入口](../code/SG_wide_PD_window_G1348/start_SG_wide_role_g1443.py) [双端预检和离线演练](../evidence/SG_usage_recovery_G1276/SG_wide_role_start_G1443.json)

32K 切换前的证据审查已接到实际采集队列，等待原采集进程退出。它核对 32K 冷热原始审计、同一轮 GSM8K 正式评分、冻结的 16 条采集输入和 P/D 两端各八个 rank 的原始收尾文件。离线演练确认缺失 raw rank 或质量终态绑定错误会阻止通过。解析后的事件覆盖需继续验收。 [证据核对](../code/SG_long_profile_G1305/review_SG_32K_exit_evidence_g1444.py) [实际监听和离线演练](../evidence/SG_usage_recovery_G1276/SG_32K_profile_intake_observer_G1444.json)

32K 的退出入口已在 P/D 两端安装。它读取封存的正式、质量和八加八 rank 原始采集验收，检查两端队列和当前配对锁。退出时先由原 Prefill 控制器关闭其容器和 Router，Decode 收到 Prefill 的实际释放回执后退出，最后核对双方释放再关闭配对锁。原 32K 测试链结束前，真机退出入口保持只读预检状态。 [退出入口](../code/SG_PD_retirement_G1453/retire_owned_G1317_role_g1453.py) [双端预检和离线流程](../evidence/SG_usage_recovery_G1276/SG_G1317_retirement_controller_prechecked_G1453.json)
