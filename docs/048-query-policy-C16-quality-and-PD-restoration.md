# DeepSeek-V4.1 查询头修正后的 C16 质量验证

在固定 SGLang Ascend daily A2 镜像、910B1 / aarch64、1017 W4A8 权重和 TP8 / EP8 上，D1163 关闭图与 DSPARK，通过 standalone 端点依次执行两组 C16 请求。16 条算术与记录绑定题全部精确匹配，GSM8K 官方测试集前 16 条也全部判对。32 个请求均返回 usage、stop 和 SSE DONE。短题请求与此前 C1 诊断逐字节相同。GSM8K 使用零样本、T0、seed0、thinking=false、2048 输出预算和官方严格评分器。每个结果已从独立的原始 SSE 重新构建。[逐条 SHA 与重建结果](../evidence/Q_head_norm_C16_G1175/G1179_parent_C16_raw_audit.json) · [原始请求和响应清单](../evidence/Q_head_norm_C16_G1175/raw/manifest.json)

完整 1319 条 GSM8K 已提交至同一 D1163，使用 vLLM 基线的非流式请求、并发 16、同一数据和评分逻辑。全作业上限放宽到 4 小时以容纳 standalone eager，单请求超时仍为 600 秒。作业启动已核对，首个现场快照为 10 条返回、10 条判对，04:22 UTC 核对到 199 条返回、194 条判对，零传输失败和零截断，完整结果等待终态。[启动回执](../evidence/Q_head_norm_C16_G1175/G1176_full_GSM8K_start_2204_parsed.json) · [首个进度快照](../evidence/Q_head_norm_C16_G1175/G1177_full_quality_observation1_2204_parsed.json) · [199 条进度快照](../evidence/Q_head_norm_C16_G1175/G1186_full_quality_observation3_2204_parsed.json) · [请求代码](../code/query_norm_mechanism_G1167/run_full_gsm8k_Q_policy_g1176.py)

P/D 配置已在两台 910B 上校验并落盘，基于以前实际运行的 P832 和 D974 部署。P1178 已就绪，八个 rank 均通过 target 1384 参数和 draft 73 参数严格装载，并记录查询头修正安装。Prefill 的 bootstrap 返回 6 个验证位置及实际传输描述。D1163 继续执行质量测试。恢复 DSPARK 1 步、6 个验证位置、Ascend DEVICE_RDMA、P eager、D target 和 draft full graph、前缀缓存和 overlap。两端 target 都带 D1163 的按配置归一化修正和压缩 KV 的 BF16 路径。SGLang 草稿的既有 Query 实现已判断同一配置标志，因此保留原实现。最终需以实际两端装载、物理缓存布局、图捕获和回放以及 PD 状态传输验证这些参数。正式 AISBench 和对比 profile 待运行。[双端源码和计划](../evidence/Q_head_norm_C16_G1175/PD_source_preparation_manifest.json) · [Prefill 就绪和布局回执](../evidence/Q_head_norm_C16_G1175/G1186_sg_P_observation3_2203_parsed.json)

四档 AISBench 的 24 个匹配配置已按当前 CPU 客户端目录调整并校验，同一套独立预热与计分数据的 SHA 仍需在服务就绪后复核。客户端配置对照仅有框架名、服务端口和容器内数据挂载路径的差异。实际 handler 的 usage 来源、两端编译缓存和计分条件将在运行前确认。[准备清单](../evidence/Q_head_norm_C16_G1175/aisbench_preparation_manifest.json)

## Decode 切换前的准备

已对 2204 实际 CPU 客户端做只读检查。AISBench 请求处理的两份源码和 tokenizer SHA 与原生 vLLM 基线一致。24 个数据文件都在预期的只读挂载中，28 个已落盘的脚本与配置均通过 SHA 复核。SGLang 端点的 usage 验证仍待执行。[实际客户端检查](../evidence/PD_runtime_preflight_G1193/G1191_SG_AISBench_client_readonly_2204_parsed.json)

Decode 验收分为八个 rank 的 target 与 draft 装载及图捕获、路由注册、真实请求的 Prefill KV 传输和 Decode 图回放。图日志与计数器解析已用历史原始证据核对。最终 PD 质量脚本的请求构造和评分逻辑与 vLLM 基线保持一致。当前这些新脚本处于准备阶段。[脚本索引](../evidence/PD_runtime_preflight_G1193/source_index.json) · [最终质量运行前检查](../evidence/PD_runtime_preflight_G1193/full_SG_PD_quality_preparation_g1192.json)

## 四档数据和 profile 输入验证

在当前客户端的只读挂载中实际核对了四档共 24 个 JSONL 文件、448 条记录、420250723 字节。数据和 tokenizer 的 SHA 与既有逐条分词审计一致，可以沿用确切输入 token 数和 448 个唯一首 128 token 前缀的证据。校验在独立 CPU 核上运行，未启动客户端推理请求，校验结束后释放了对应锁。后续 AISBench 仍需核对 SGLang 实际返回的 usage 和缓存命中。[当前数据校验](../evidence/SG_bench_inputs_G1198/G1195_dataset_bytes_2204_parsed.json) · [原始逐条分词审计](029-CPU-client-readonly-dataset-revalidation.md)

独立 profile 批次的 16 条请求与 vLLM 已采集批次逐条 SHA 一致，输入 32768 tokens、输出 1024 tokens，均使用 C16。两框架的客户端配置只差框架标签和服务端口。SGLang 采集脚本已准备，待 PD 和 32K 正式测量通过后运行。归因时按实测的 token、缓存命中、target 验证和 draft 工作量对齐，再比较算子、调度及通信。[当前 profile 挂载验证](../evidence/SG_bench_inputs_G1198/G1197_SG_matched_profile_inputs_2204_parsed.json) · [采集计划及源码指纹](../evidence/SG_bench_inputs_G1198/SG_matched_profile_preparation_g1196.json)

## Prefill 编译检查与同题长上下文验证

2203 Prefill 的两次空闲快照已用同一实例、镜像和缓存目录对照。常规缓存目录有 3 个文件，另有 73 个 Ascend 工作文件，后者为日志、锁和构建进程元数据。两次快照之间这些文件均未变化，Prefill 的设备前缀命中计数为零。正式预热和计分需在对应阶段边界再次取样。诊断日志与持久编译产物的变化分别归档。[现场快照](../evidence/SG_phase_audit_G1205/G1200_P_idle_compile_baseline_2203_parsed.json) · [两次快照比较](../evidence/SG_phase_audit_G1205/G1204_P_idle_compiler_comparison.json)

当前客户端可读取 vLLM 已验证的四个 32K、64K、128K、256K 长上下文检索请求，原始数据和 tokenizer 均通过 SHA 校验。SGLang 在最终 PD 路由就绪后将逐条核对实际输入 token 数和答案，随后重放同一请求。复用证据将分别记录 Prefill 的本地计数和 PD 响应的合计缓存 token 数。Decode 本地命中需要独立证据。[当前挂载复核](../evidence/SG_phase_audit_G1205/G1203_SG_long_inputs_verified_2204_parsed.json) · [准备记录和源码](../evidence/SG_phase_audit_G1205/source_index.json)

## P/D 联调准备与 profile 字段覆盖

固定镜像的 `/start_profile` 将 `detailed_annotations` 传入调度器 profiler。构建 CPU step span 时，生成路径从 CPU 长度镜像和 spec 元数据给出的 query 宽度得出 `g_sq/g_sqsq/g_sqsk/g_sk`，上下文路径从 chunk 和 prefix 长度得出 `c_*`。宽度字段缺失时源码默认取 1，采集后要按角色、rank 和模式核对字段覆盖以及 VERIFY、DRAFT 实际使用的宽度。响应端的 `cached_tokens` 汇总 PD 请求的令牌计数。P 侧设备、主机内存和存储的命中由 P 自己的指标确认，D 的已计算令牌包含从 P 传来的缓存状态。 [固定镜像源码摘录](../evidence/SG_profile_readiness_G1218/fixed_image_profile_and_usage_source.json)

在两份旧 D 短跑的真实 rank 0 数据库上，G896 的 6 次 `aclmdlRIExecuteAsync` 通过同线程和 CPU 时间范围关联到 4 次 DRAFT 调用和 2 次 VERIFY 调用。G981 的 4 次调用按 2 次 DRAFT 和 2 次 VERIFY 分布。10 次调用都有唯一的 step 和直接 CANN connection 对应的设备任务，后者的 model ID 均为占位值 `4294967295`。这两份 trace 的 query/KV 字段覆盖均为 0/4，支持调用归属复核，逐步逻辑工作量和设备图实例仍待实采。 [旧 trace 关联回执](../evidence/SG_profile_readiness_G1218/G1213_SG_historical_graph_join_2204_parsed.json)

关联脚本已用固定源码生成的 VERIFY、EXTEND、MIXED 标签验证，并在旧真实库上验证数值和文本两种时间戳格式。脚本保留没有唯一范围或设备关联的调用，单列字段覆盖率。当前 G1178 的匹配 profile 尚未采集。 [关联脚本](../code/SG_profile_readiness_G1218/parse_SG_step_graph_g1213.py) [CPU fixture](../evidence/SG_profile_readiness_G1218/G1213_parser_actual_label_fixture.json)

P/D 联调脚本和最终 GSM8K 客户端源码已分别在 P 和 D 校验 SHA。记录时 standalone D1163 的 GSM8K 仍在执行，已返回 859/1319，828 条判对，传输失败和截断均为 0。最终 P/D 质量与四档正式性能还需以切换后的原始输出验收。 [P 部署回执](../evidence/SG_profile_readiness_G1218/G1214_P_runtime_source_staged_2203_parsed.json) [D 客户端回执](../evidence/SG_profile_readiness_G1218/G1215_D_PD_quality_source_staged_2204_parsed.json) [单机进度快照](../evidence/SG_profile_readiness_G1218/G1218_full_quality_observation_2204_parsed.json) [归档 SHA 索引](../evidence/SG_profile_readiness_G1218/source_index.json)

旧库中每次 replay 的直接关联包含一条 `MODEL_EXECUTE`，另有设备同步任务。在各条 `MODEL_EXECUTE` 的设备时间窗内，开始执行的非占位 model ID 每次只有一个。G896 重复观察到 DRAFT 43/45、VERIFY 49，G981 重复观察到 DRAFT 44、VERIFY 46。该关联来自同设备的时间重合，可用于复核图族候选。完整 kernel 归属和跨 rank 覆盖仍需单独验收。DSPARK 在固定镜像中将 `draft_token_num` 传到 `DFlashVerifyInput.num_tokens_per_req`，部分 backend 可以不保留 draft 的 CPU 序列长度，因此 query 宽度的源码来源和实采字段覆盖须分别记录。 [设备任务类型](../evidence/SG_phase_audit_G1222/G1220_SG_historical_graph_schema_2204_parsed.json) [设备时间窗](../evidence/SG_phase_audit_G1222/G1221_SG_historical_graph_clock_2204_parsed.json) [固定 spec 源码摘录](../evidence/SG_phase_audit_G1222/DSPARK_query_width_fixed_source.json)

新版 AISBench runner 将实际 D 容器的 `ASCEND_WORK_PATH` 文件纳入编译边界快照。只读探针在 D1163 单机实例发现常规目录 1232 个文件，Ascend 工作目录 766 个文件，扩展快照额外覆盖 48 个 trace 文件，包含 GE Compile 和 FE CompileTd。请求、配置、判分及首轮登记等逻辑的 AST 与原 runner 一致。新版已单独落盘，仍需在 G1178 的预热和正式边界采集实际快照。 [扩展 runner](../code/SG_phase_audit_G1222/run_aisbench_cold_hot_g1222.py) [D 工作目录回执](../evidence/SG_phase_audit_G1222/D_compiler_work_path_stage.json) [归档 SHA 索引](../evidence/SG_phase_audit_G1222/source_index.json)

## D1163 单机 GSM8K 终态

固定 SGLang Ascend A2 镜像和同一 W4A8 权重，以 D1163 单机 TP8/EP8、eager、关闭 DSPARK 验证查询头归一化修正。全量 GSM8K 的 1319 个请求均有终态记录，1269 条判对，1 条传输失败，0 条达到输出上限。可读原始响应逐条复判，冻结文件数为 2642。客户端超时的请求、报错、原始结果和 SHA 已保留。 [终态复判回执](../evidence/SG_D1163_GSM8K_G1234/D1163_final_artifact_freeze.json) [超时证据](../evidence/SG_D1163_GSM8K_G1234/D1163_client_timeout.json)

两套全量 JSON 请求逐题匹配，其中 1319 条保存文件的字节一致。vLLM 原生 0.32 P/D 同题判对 1270 条。1318 对成功且未截断的响应中，双方均正确 1264，仅 vLLM 正确 5，仅 SGLang 正确 5，均错误 44。SGLang 最终 G1178 P/D 的 DSPARK、图回放和质量要以该实例的实际输出验收。 [逐题判分对照](../evidence/SG_D1163_GSM8K_G1234/native630_D1163_paired_grade_counts.json)

最终 P/D 上的 SSE 用量检查复用 vLLM 的同一份独立 32K 输入、1024 输出请求。脚本和请求字节已校验并落盘。AISBench 的服务门槛脚本保存实际配置、设备命中序列、预热和首轮数据 SHA。两项当前为运行准备。 [用量输入部署回执](../evidence/SG_D1163_GSM8K_G1234/G1227_usage_assets_staged_2204_parsed.json) [服务门槛部署回执](../evidence/SG_D1163_GSM8K_G1234/G1229_gate_source_staged_2204_parsed.json) [归档索引](../evidence/SG_D1163_GSM8K_G1234/source_index.json)

D1163 客户端和模型进程均退出、8 卡逐卡清空后，D1178 已按冻结源码启动控制器。控制器启动回执的范围为进程存活，拉起模型与图捕获仍需就绪检查。 [D1163 退役回执](../evidence/SG_D1163_GSM8K_G1234/D1163_retirement_verified.json) [D1178 启动回执](../evidence/SG_D1163_GSM8K_G1234/D1178_controller_launch_verified.json)

## G1178 最终 P/D 实测联调

Decode 的 8 个 rank 均通过 target 1384 项、draft 73 项严格装载。target 6-token VERIFY 和 draft 5-token full graph 均完成捕获，覆盖 bs=16。路由约 161 秒完成 Prefill 初始探测。同一进程复核为两个 worker 健康、负载为 0。 [Decode 就绪回执](../evidence/SG_final_PD_G1247/D1178_ready_graph_all8.json) [路由复核](../evidence/SG_final_PD_G1247/same_router_actual_two_workers.json)

三条实际请求通过。算术题返回 91，独立文本与同输入重放均产生 64 个输出 token。260-token 输入的响应缓存命中为 0 和 256。P 侧各 rank 的 KV 传输计数增加 3，D 侧各 rank 的 full graph 回放计数增加 31。原始请求、响应、指标的 SHA 均已复核。 [P/D 实测回执](../evidence/SG_final_PD_G1247/P_D_real_requests_and_runtime_counters.json)

最终 P/D 的 1319 题 GSM8K 已按 C16、2048 输出上限、temperature 0、seed 0 独立启动。首次启动检查中的目录拼写已修复。已复核同一个实际 worker 和首条正确答复。当前全量结果仍待收尾，四档 AISBench 和全 rank 同负载 profile 仍待执行。 [首次检查错误](../evidence/SG_final_PD_G1247/initial_launcher_path_error.json) [实际启动](../evidence/SG_final_PD_G1247/final_PD_GSM8K_actual_launch.json) [实际观察](../evidence/SG_final_PD_G1247/final_PD_GSM8K_actual_initial_observation.json) [归档索引](../evidence/SG_final_PD_G1247/source_index.json)

早期质量采样中，路由的两个 worker 仍健康且均记录负载 16，D 有 15 条在传输队列，1 条已运行。P 的 Triton 工作目录同期新增或更新了 10 个较晚的 `.npubin` 产物，包括 `alloc_extend_kernel`。后续同一质量 worker 已记录 22 条正确响应，D 在运行 12 条。这里保留启动阶段的队列与编译证据，稳定性在正式矩阵的独立预热中另行核查。 [质量初期队列和编译快照](../evidence/SG_final_PD_G1247/early_PD_quality_and_Prefill_compile.json)

## 长上下文和全 rank profile 准备

最终 P/D 已实测导出入口。长上下文和同负载 profile 脚本已在 D 落盘校验，P/D 的 profile 解析脚本使用同一 SHA。新解析程序在保留的旧 G981 Decode trace 上逐 rank 通过，8 张卡各读取 4 次 replay 和 10889 条已关联的计算任务。各类设备时间区间的并集与累计时长分别统计，匹配负载的归一化等待新 trace。 [D 工具部署](../evidence/SG_diagnostic_readiness_G1262/G1254_diagnostic_source_and_allocator_2204_parsed.json) [P 解析工具部署](../evidence/SG_diagnostic_readiness_G1262/G1258_SG_profile_readiness_2203_parsed.json) [历史 8 rank 验证](../evidence/SG_diagnostic_readiness_G1262/G1258_SG_profile_readiness_2204_parsed.json)

正在运行的 G1178 已验证 rank 0 的设备、主机内存、存储缓存命中序列均可读。保存的短 SSE 帧已逐个核对模型 ID、结束与用量字段，下一次独立 32K/O1024 用量检查仍待实际提交。P 的 allocator 源码以 `bs_upper` 和页大小为编译常量，extend 长度由设备运行时动态读取。当前发现 23 个 compiler cache 目录和 8 种 IR 文件 SHA，原始指纹保留，尚无证据将变体逐个归到请求长度或分摊端到端时间。 [P 指标和短流](../evidence/SG_diagnostic_readiness_G1262/G1257_live_SG_quality_and_gate_exports_2203_exports_parsed.json) [D 指标](../evidence/SG_diagnostic_readiness_G1262/G1257_live_SG_quality_and_gate_exports_2204_exports_parsed.json) [编译源码与缓存](../evidence/SG_diagnostic_readiness_G1262/prefill_allocator_fixed_source_and_cache.json) [归档索引](../evidence/SG_diagnostic_readiness_G1262/source_index.json)

匹配 profile 的输入检查已复核两个实际 manifest。24 份正式和预热数据的 SHA 逐项一致，共 448 条。独立 profile 的 16 个前缀锁定到既有 vLLM 采集，已校验为独立输入。profile 门槛脚本已落盘，等当前 32K 首轮数据和预热证据通过后再启用。 [实际资产核对](../evidence/SG_diagnostic_readiness_G1262/G1263_profile_gate_source_staged_2204_parsed.json)

## 最终 P/D GSM8K 终态

固定镜像 G1178 在同一对 910B1 主机运行完整 GSM8K，C16、输出上限 2048、temperature 0、seed 0。1319 个请求均有终态，1266 个判对，1 个客户端超时，0 个截断。独立复判验证同一数据集、grader 和所有可读响应，2642 个原始文件留在服务器并固定 SHA。超时请求保留原始记录，未重发。 [复判与冻结](../evidence/SG_final_quality_G1268/final_SG_PD_GSM8K_frozen.json) [超时证据](../evidence/SG_final_quality_G1268/D_transport_window.json)

与 vLLM 原生 0.32 P/D 的 1319 份请求逐字节一致。1318 对均成功且未截断的题里，双方均正确 1262，vLLM 单独正确 7，SGLang 单独正确 4，双方均错误 45。总分 vLLM 1270/1319、SGLang 1266/1319。计入全量分母的 SGLang 超时题在 vLLM 答对。 [逐题对照](../evidence/SG_final_quality_G1268/final_SG_PD_native_same_question_comparison.json)

两侧确认空闲后，开始独立 32K/1024 标准 SSE 探针。该输入沿用 vLLM 诊断的相同字节，首轮正式 AISBench 输入继续保留。 [SSE 探针启动](../evidence/SG_final_quality_G1268/actual_32k_sse_usage_probe_launch.json)

## 32K 独立用量检查与恢复

同 vLLM 请求字节的第一轮 32K/O1024 用量探针达到 600 秒总上限，客户端未得到完整 SSE 用量。P 记录了四个 8192-token 分块，前两段分块调度间隔约 120 秒。D 随后使用图回放生成。此次检查的 P 编译产物清单未观察到晚于阈值的新二进制，时间间隔的机制仍需采集实际算子和调度证据。该结果保存为独立协议检查失败。 [终态回执](../evidence/SG_usage_recovery_G1276/first_32K_probe_terminal_failure.json) [P 分块](../evidence/SG_usage_recovery_G1276/first_probe_P_scheduler_records.json) [D 调度](../evidence/SG_usage_recovery_G1276/first_probe_D_scheduler_records.json) [P 编译快照](../evidence/SG_usage_recovery_G1276/first_probe_D_linked_P_compile_snapshot.json)

P/D 复核空闲后，在同一实例对同一诊断输入启动独立复测。客户端逐帧落盘，记录首次内容、总时间和失败时的部分字节，单次总上限 1800 秒。第一次观察已保存 80 个完整 SSE 帧，最终用量仍待通过。两次探针均独立于五轮预热和所有正式首轮输入。 [复测源码回执](../evidence/SG_usage_recovery_G1276/G1273_SG_32K_usage_recovery_source_2204_parsed.json) [实际启动](../evidence/SG_usage_recovery_G1276/G1275_PD_32K_usage_recovery_actual_start_2204_usage_parsed.json) [实时 SSE 回执](../evidence/SG_usage_recovery_G1276/G1276_live_32K_usage_retry_2204_parsed.json) [归档索引](../evidence/SG_usage_recovery_G1276/source_index.json)

复测已收到完整 579 个 SSE 帧及 `[DONE]`。末尾用量确认 32768 输入、1024 输出，同输入重放命中 32640 token，首个非空内容在 2.746 秒，完整流 597.647 秒。5 项原始用量证据逐项校验通过，已形成正式 runner 可读取的用量口径。该请求为单并发重放，正式 C16 首轮尚待预热和测量。 [复测终态及用量校验](../evidence/SG_usage_recovery_G1276/32K_SSE_protocol_verified.json)

复测退出并确认双方空闲后，已启动同一套 32K、64K、128K、256K 长上下文检索输入。第一次实际观察已在运行 32K fresh，结果待验收。 [长上下文启动](../evidence/SG_usage_recovery_G1276/long_retrieval_actual_launch.json) [首个请求在飞](../evidence/SG_usage_recovery_G1276/long_retrieval_first_request_live.json)

32K 首次请求和逐字节相同的重放均精确找回三项指定值，分别返回 32768 输入、29 输出及 `stop`。Prefill rank 0 的设备缓存计数增量为 0 和 32640，与两次 PD 响应的 `cached_tokens` 一致。主机和存储计数均无增量，Decode 本地命中未独立计量。该 C1 检索快照已进入 64K 首次请求，后续档位仍待完成。 [两次实际用量及请求与响应 SHA](../evidence/SG_usage_recovery_G1276/long_retrieval_32K_actual.json)

## Prefill 与 Decode 优化候选

32K 首次检索在同一 Prefill 主机记录到两个 8192-token 分块调度事件，间隔约 120 秒，与前一轮独立 32K 探针的量级一致。优先检查两次调度之间的设备任务、CPU 调度与传输等待。当前日志只支持同主机事件间隔，仍需设备 trace 区分各项时间。 [本轮分块记录](../evidence/SG_usage_recovery_G1276/long_retrieval_32K_P_stage_sample.json) [前一轮分块记录](../evidence/SG_usage_recovery_G1276/first_probe_P_scheduler_records.json)

64K 首次请求执行期间，Prefill 三次采样均显示 1 条运行中请求、bootstrap 和传输队列为 0，八个 rank 的传输计数无增量。八卡 AICore 读数在三次采样中依次分布为 99–100%、25–26%、99–100%。设备在这些时点确有活动，仍需 trace 确定任务类型、区间长度以及对分块耗时的贡献。 [三次现场采样](../evidence/SG_usage_recovery_G1276/long_retrieval_64K_P_resource_samples.json)

64K 首次请求的客户端在读取响应状态时触发 900 秒 socket 超时，作业终止，保存了前两条 32K 通过记录。Prefill rank 0 此前已记录四次 8192-token 调度，分块间隔为 120、121、121 秒，第四次记录仍有 32768 token 待处理。64K 未取得可验收响应，64K 重放、128K、256K 均未提交。进程退出后，现场确认 P/D 运行、等待、bootstrap、预分配和传输队列都为零，路由的两个 worker 健康且负载为零。固定 AISBench CPU 镜像中的请求实现设置 `ClientTimeout(total=None)`，C1 检索脚本的 900 秒上限与正式链路各层的上限分别记录。 [客户端终态](../evidence/SG_usage_recovery_G1276/long_retrieval_64K_terminal.json) [四个分块](../evidence/SG_usage_recovery_G1276/long_retrieval_64K_P_chunk_records.json) [实际队列](../evidence/SG_usage_recovery_G1276/long_retrieval_64K_actual_drain.json) [AISBench 客户端源码指纹](../evidence/SG_usage_recovery_G1276/sg_aisbench_client_deadline_source.json)

32K/O1024 的 C1 缓存重放在首个内容之后仍运行约 594.90 秒。即便完全消除该请求的首发阶段，也只节省该次请求总时间的约 0.46%。Decode 优先复核 `_sparse_shared_kv`。旧 G981 设备 trace 的 rank 0 两次 VERIFY 和两次 DRAFT 窗口内，该算子 86 次任务累计 453.74 毫秒，paged FP4 索引累计 20.09 毫秒。当前 kernel 按 `(query row, query head)` 发程序，同一行的索引和 KV loads 不依赖 head id，因此值得验证跨头实际数据复用以及片上容量约束。算子累计值不用于估计可节省的端到端时间。需要同负载全 rank trace 才能判断应先验证哪种改动。 [SSE 时间](../evidence/SG_usage_recovery_G1276/32K_SSE_protocol_verified.json) [历史算子与工作量覆盖](../evidence/SG_usage_recovery_G1276/historical_decode_optimization_selection.json)

用 G981 原始 rank 0 数据库重算设备区间并集，86 次 `_sparse_shared_kv` 共占 453.74 毫秒。同一设备时钟下，这些区间未与其他已归类的计算或通信任务重叠，占该短采样中计算和通信区间并集 618.76 毫秒的 73.3%。原始表支持算子和任务区间核对。内存流量和当前 C16 的占比需分别实采。 [原始库指纹与区间统计](../evidence/SG_usage_recovery_G1276/historical_decode_G981_sparse_interval_union.json)

已准备独立候选内核，每个程序加载一次 KV tile，为两个相邻的 query heads 分别执行 FP32 在线 softmax。C1/C2 物理地址、候选顺序、BF16 布局、掩码和各头 sink 保持与 G1178 一致。以 8192 行、每行 640 个有效候选、每 rank 8 个头为例，源码中有效 KV 元素的逻辑读取量由 40 GiB 降到 20 GiB。硬件实际访存仍受编译器、缓存和片上空间影响。组件探针对同一张 910B、同一输入和固定原内核做 ABBA/BAAB 计时，并覆盖无压缩、C1/C2、高地址 bank、全掩码与三次变更输入图重放。当前完成源码核对，Ascend 编译、图重放和端到端收益等待实测。 [候选源码](../code/SFA_two_heads_G1300/lowratio_sparse_attention_two_heads.py) [组件探针](../code/SFA_two_heads_G1300/probe.py) [源码指纹和逻辑读取公式](../code/SFA_two_heads_G1300/source_pins.json)

## 32K C16 首轮独立预热

按 64K 首次检索观测到的 Prefill 分块间隔，原 Router 的 3600 秒上限存在截断同形状 C16 预热的风险。在两端队列排空后，将 Router 请求上限调到 14400 秒，复用原 P/D 进程、路由并发限制和请求策略。新 Router 上的三个实际短请求通过。八个 P rank 的 KV 传输计数各增加 3，八个 D rank 的图回放计数各增加 38，两个 260/64 SSE 请求的响应缓存计数由 0 增至 256。 [原 Router 配置](../evidence/SG_usage_recovery_G1276/G1178_router_previous_deadline.json) [切换回执](../evidence/SG_usage_recovery_G1276/G1178_router_deadline_actual_launch.json) [切换后短请求复核](../evidence/SG_usage_recovery_G1276/G1178_reconfigured_router_runtime_verified.json)

32K/C16/1024 已通过现场预检并启动第一轮独立预热，输入和用量审计逻辑沿用已校验版本。根据当前实测耗时放宽单阶段墙钟上限到 12900 秒，首轮后停在检查点，随后结合实际时长与资源有效期安排剩余四轮。现场已核对同一 runner 和 AISBench 进程，以及首轮 16 条数据的命令与 SHA。五轮预热、编译成熟度检查、计分首轮及同请求重放仍按原条件执行。 [首轮启动与脚本指纹](../evidence/SG_usage_recovery_G1276/SG_first_32K_AISBench_warmup_launched.json) [首轮实际运行](../evidence/SG_usage_recovery_G1276/SG_first_32K_AISBench_warmup_actual_inflight.json)

首轮预热的前两次现场采样中，Prefill 均有 1 条运行、15 条等待，Decode 尚未运行请求。当时 Router 的两端 worker 负载均为 0，因此队列排空须直接查 P/D 的角色指标。第三次采样时，Prefill 已记录 1 次新增 KV 传输，仍有 1 条运行、14 条等待，Decode 同时运行首条请求且转移队列里有 15 条。第一次到第三次采样相隔约 502 秒，Prefill rank 0 输入计数增加 32768，设备缓存计数无增量。该时间窗包含分块调度与 P/D 状态迁移，首轮端到端指标等待 16 条响应与 AISBench 原始记录。 [首条传输前](../evidence/SG_usage_recovery_G1276/SG_32K_warmup1_before_first_transfer.json) [Prefill 处理中](../evidence/SG_usage_recovery_G1276/SG_32K_warmup1_processing_first_prefill.json) [Decode 开始](../evidence/SG_usage_recovery_G1276/SG_32K_warmup1_first_decode.json)

约 26 分钟时，同一 AISBench 作业客户端的 `finish_count` 已到 2，`total_count` 为 16。原始响应、成功数和计时仍待首轮终态审计。现有匹配 profile 草案给同形状负载的上限为 900 秒，采集前需依据完整首轮的实际耗时重定请求超时与导出预算，保留原 16 条输入和输出长度。 [客户端现场计数](../evidence/SG_usage_recovery_G1276/SG_32K_warmup1_AISBench_progress_2_of_16.json) [原匹配采集计划](../evidence/SG_bench_inputs_G1198/SG_matched_profile_preparation_g1196.json)

## 长窗口匹配 profile 准备

采集控制已改为从预检记录读取负载与导出时限。组装预检和启动采集时都核对 P/D 服务、CPU 客户端及工作负载锁的到期时间，启动前再次读取两侧角色队列。未回复或未确认的采集操作保留实际状态和记录。原生匹配清单、完整 16 条请求、并发、输出长度与 P/D profiler 参数沿用原方案。 [预检代码](../code/SG_long_profile_G1305/build_SG_matched_profile_gate_g1304.py) [采集代码](../code/SG_long_profile_G1305/capture_SG_matched_profile_g1304.py) [源码指纹](../code/SG_long_profile_G1305/source_pins.json)

固定实例上的实际 Prefill 和 Decode 导出挂载均可写，观察时分别剩余约 1.95 TB、1.86 TB。新工具已在 CPU 客户端所在节点核对源文件 SHA 和 Python 编译。离线用例覆盖长负载时限、各资源到期以及保留原采集命令和输入指纹。实际 profiler 未启动，完整时限和导出耗时仍待实跑。同期首轮预热客户端进度为 3/16，终态才核对成功与实际 token 数。 [Prefill 存储](../evidence/SG_usage_recovery_G1276/long_profile_P_storage.json) [Decode 存储](../evidence/SG_usage_recovery_G1276/long_profile_D_storage.json) [服务端工具回执](../evidence/SG_usage_recovery_G1276/long_profile_tools_staged.json) [离线校验](../evidence/SG_usage_recovery_G1276/long_profile_budget_fixture_checks.json) [预热进度](../evidence/SG_usage_recovery_G1276/SG_32K_warmup1_AISBench_progress_3_of_16.json)

## 正式冷测与重放时限

正式 runner 已单独落盘。冷测与重放各有配置明确的时限，由同一个持有客户端锁的进程按顺序执行。启动前以实际 P/D 服务、CPU 客户端和工作负载锁的最早到期时间校验两个阶段的完整预算，重放前再检查剩余资源时间。五轮预热仍须在原截止时间内完成并重新审计。原首次提交登记、32 条冷测、同一份 32 条重放以及缓存增量审计保持一致。离线用例验证了已登记输入、缺失预热证据和预算不足均无法进入计分。正式负载尚未启动，需用预热实测耗时选择时限。 [runner 和启动模板](../code/SG_formal_timing_G1307/source_pins.json) [服务端安装回执](../evidence/SG_usage_recovery_G1276/formal_timed_runner_staged.json) [时限与首次提交校验](../evidence/SG_usage_recovery_G1276/independent_formal_deadline_fixture.json)

首轮预热仍在运行。最新 Prefill rank 0 计数较首个现场样本新增 4 次 KV 传输。此时 Prefill 有 11 条等待，Decode 有 1 条在运行、12 条在传输队列，客户端成功数由终态原始响应审计。 [现场队列与计数](../evidence/SG_usage_recovery_G1276/SG_32K_warmup1_four_KV_transfers.json)

## 资源租期与 CPU 客户端续用

首轮 32K 的 rank 0 Prefill 输入计数在约 2493 秒内增加 172032 token，期间新增 5 次 KV 传输。若该采样速度持续，32K 的五轮预热加首轮 32 条冷测，仅 Prefill 输入就约需 14 小时 46 分钟，比现有 CPU 客户端的有效期长约 102 分钟。该推算未计入最后的 Decode、重放与阶段间隔。把同一速度用于 64K、128K、256K 只得到资源排期情形，各档仍须独立测量。 [实测窗口、公式与各档情形](../evidence/SG_usage_recovery_G1276/SG_current_32K_P_rate_and_walltime_scenarios.json) [现场 CPU 租期](../evidence/SG_usage_recovery_G1276/SG_actual_CPU_original_deadline.json)

已部署同容器续用工具。阶段结束且请求队列排空后，工具使用原 CPU 锁交接持有者，并核对同一个容器 ID、固定镜像、CPU 配额、配置和挂载指纹。预热、正式计分和匹配 profile 的后续预检读取当前实际租期。现场三次只读检查得到同一配置指纹，离线用例覆盖锁冲突、变更挂载、容器启动记录不匹配和交接清理。当前首轮预热继续使用原客户端，现场交接尚未执行。 [源码与启动配置](../code/SG_cpu_lease_continuity_G1311/source_pins.json) [现场检查](../evidence/SG_usage_recovery_G1276/SG_CPU_lease_staged_current_proof.json) [离线用例](../evidence/SG_usage_recovery_G1276/SG_CPU_lease_continuation_fixture.json)

## Decode KV 等待超时与 32K 恢复

首轮 32K/C16 预热终态审计未通过。原始 16 条中，7 条同时达到 32768 输入和 1024 输出，另外 9 条输出为 0。AISBench 明细仍把这 9 条标为 `success=true`，token 数审计已将整轮拦下。Decode 原始日志记录了 9 个不同 bootstrap room 在 `WaitingForInput` 超时，每个 room 均见于八个 rank，耗时 3600.1–3601.0 秒。实际运行的 Ascend 传输继承接收方的等待检查，计时在发送接收方 KV 元数据后开始。八个 Decode rank 的实际环境变量均为 3600 秒。这层等待独立于已调到 14400 秒的 Router 上限。13 份客户端原始文件已逐项固定 SHA，P/D 原始日志继续保留。 [终态审计与冻结](../evidence/SG_usage_recovery_G1276/SG_32K_warmup1_terminal_timeout_freeze.json) [运行时配置与源码指纹](../evidence/SG_usage_recovery_G1276/SG_decode_actual_PD_transport_configuration.json)

两端已现场核实运行、等待与传输队列全部为零，原 runner 已退出，正式数据未提交。恢复实例将 Decode 的内部 KV 等待调到 14400 秒，并给 32K 恢复留出 30 小时的 P/D 服务窗口。原 P bootstrap 阈值、镜像、模型权重、推理参数和数据形状沿用。两个角色的 58 份源码均已在目标端校验。实际模型装载、图、双端运行时联调和新 epoch 的五轮独立预热仍需验收。 [构建程序](../code/SG_PD_wait_recovery_G1317/prepare_SG_PD_wait_recovery_g1317.py) [源码准备](../evidence/SG_usage_recovery_G1276/SG_P_D_wait_recovery_source_preparation.json) [Prefill 源码回执](../evidence/SG_usage_recovery_G1276/SG_PD_wait_recovery_source_staged_P.json) [Decode 源码回执](../evidence/SG_usage_recovery_G1276/SG_PD_wait_recovery_source_staged_D.json)

旧 Prefill 和 Decode 控制器均经现场核验退出，各自释放八张卡，随后启动修复实例的两个控制器。原始失败批和冻结索引保留。Prefill 和 Decode 的新服务就绪与请求复测等待各自运行时回执。 [Prefill 启动](../evidence/SG_usage_recovery_G1276/SG_PD_wait_recovery_actual_P_launch.json) [Decode 启动](../evidence/SG_usage_recovery_G1276/SG_PD_wait_recovery_actual_D_launch.json)

恢复后的 P/D 均通过 `/ready` 和各自 8 个 rank 的权重核对。Decode 的主模型和草稿模型图均在 8 个 rank 完成捕获。逐个读取 Decode 调度进程环境，8 个 rank 的 KV 等待均为 14400 秒。Router 使用 14400 秒请求时限，在同一进程完成两个 worker 的健康注册。 [Prefill 就绪](../evidence/SG_usage_recovery_G1276/SG_PD_wait_recovery_actual_P_readiness.json) [Decode 就绪及实际等待配置](../evidence/SG_usage_recovery_G1276/SG_PD_wait_recovery_actual_D_readiness.json) [Router 就绪](../evidence/SG_usage_recovery_G1276/SG_PD_wait_recovery_actual_router_readiness.json)

恢复实例通过 3 条真实 P/D 短请求。算术返回 91，两次 260 输入和 64 输出的请求均完整结束，前缀缓存命中从 0 增为 256 tokens。Prefill 8 个 rank 均记录到 KV 传输，Decode 8 个 rank 均记录到图回放计数增长。终态重新校验了 10 份原始文件，并记录了结果文件的 SHA-256。 [短请求终态及计数](../evidence/SG_usage_recovery_G1276/SG_PD_wait_recovery_short_functional_verified.json)

恢复实例重新启动原形状的 32K／C16／1024 输出预热。现场确认 AISBench runner 和容器内客户端均在运行，首轮使用原先锁定的 16 条预热输入和相同数据 SHA-256。单阶段上限 12900 秒，首轮终态后进入审计检查点。当前尚无该轮终态和正式冷测或重放数据。 [首轮启动](../evidence/SG_usage_recovery_G1276/SG_PD_wait_recovery_32K_warmup1_actual_launch.json) [客户端在飞证据](../evidence/SG_usage_recovery_G1276/SG_PD_wait_recovery_32K_warmup1_actual_inflight.json)

恢复后首轮预热继续在同一实例运行。Prefill rank 0 首两个 8192-token 分块日志相隔 120 秒，后续采样的输入计数已较短请求基线增加 32768。现场暂未看到本轮新增的 KV 传输计数或客户端终态。该记录用于识别长输入推进和阶段预算，正式端到端指标待终态审计。 [Prefill 分块原始摘录](../evidence/SG_usage_recovery_G1276/SG_PD_G1317_32K_W1_P_chunk_progress.json) [同一运行的队列采样](../evidence/SG_usage_recovery_G1276/SG_PD_G1317_32K_W1_D_observed_progress.json) [预热前冻结计数](../evidence/SG_usage_recovery_G1276/SG_PD_G1317_32K_W1_frozen_baseline.json)

下一次现场采样已观察到本轮第 1 次 KV 传输，距预热前快照约 13 分 27 秒。客户端仍在运行，逐请求输出及正式指标待终态审计。 [新增 KV 传输计数](../evidence/SG_usage_recovery_G1276/SG_PD_G1317_32K_W1_first_KV_transfer.json)

已在 Decode 节点部署并校验绑定 G1317 的续跑、正式冷测及重放脚本。匹配 profile 工具在同一节点校验了原先锁定的 16 条输入及两框架配置。后续预热、正式测量和 profile 仍待各自实际门槛通过。 [续跑工具回执](../evidence/SG_usage_recovery_G1276/SG_PD_G1317_AISBench_continuation_tools_staged.json) [匹配 profile 工具与输入回执](../evidence/SG_usage_recovery_G1276/SG_PD_G1317_matched_profile_tools_staged.json)

## 宽输入的阶段时限与分配约束

同一恢复实例的后续 472.46 秒窗口中，Prefill rank 0 新增 32768 个输入 token，区间速率约 69.36 token/s，累计已观察到本轮 2 次 KV 传输。上一组相邻 8192-token 分块给出约 68.27 token/s，作为固定节奏估算的输入。运行中采样只覆盖当前阶段。 [第二次 KV 传输与计数窗口](../evidence/SG_usage_recovery_G1276/SG_PD_G1317_32K_W1_second_transfer_and_rate.json)

| 输入 | 首轮 16 条仅 Prefill 估算 | 五轮预热及首轮冷测 32 条仅 Prefill 估算 |
| --- | ---: | ---: |
| 32K | 2.13 小时 | 14.93 小时 |
| 64K | 4.27 小时 | 29.87 小时 |
| 128K | 8.53 小时 | 59.73 小时 |
| 256K | 17.07 小时 | 119.47 小时 |

估算沿用 32K 已观察的分块节奏，假设预热与冷测输入全新，排除传输、Decode、上下文差异和重放。以该节奏，64K 起首轮即超过当前 12900 秒客户端阶段上限，且首轮整体 Prefill 时间超过 14400 秒单请求等待设置。32K 出现首个 KV 传输时，Decode 已有 15 条等待传输。单请求 KV 等待从 metadata 发布计时，宽档启动前须分别核对客户端阶段、Router、KV 等待及两端资源窗口。 [宽输入场景与原始证据哈希](../evidence/SG_usage_recovery_G1276/SG_PD_G1317_wider_shape_phase_budget_scenario.json)

当前容器两端安装的核心调度与传输源码 SHA 一致。调度器先处理已有分块请求，当该步的分块预算耗尽后停止从队列取新请求，这与现场多次采样中的 Prefill 1 条运行及其余排队相符。分块内部的算子、通信和同步耗时仍需匹配 trace 才能分解。Decode 实际采用 `early`，8 个调度 rank 的接收等待均为 14400 秒。镜像中的 `prefill_complete` 检查只允许 Mooncake、单 attention tensor rank 和受支持的 KV 池，同时拒绝 Decode radix cache。当前 Ascend、attention TP8、SWA/C128 KV 状态及 Decode radix 配置均需要保留，因此该模式无法直接接入本轮配置。 [运行参数、源码定位与 SHA](../evidence/SG_usage_recovery_G1276/SG_PD_G1317_live_admission_and_wait_source.json)

宽档 runner 已部署到 CPU 客户端所在节点。宽档预热在预检和实际提交前校验原始预热总期限、四项资源租约、Router 请求时限与 Decode 8 个 rank 的 KV 等待设置。每轮仍使用 16 条独立输入，正式阶段仍是 32 条冷测和同输入重放。冷测数据的登记安排在快照及阶段时限检查之后，紧接客户端提交。当前真实 Decode 的 14400 秒设置拦截了一个 18000 秒的宽档预检用例。离线用例通过了 64K、128K、256K 的预算组合及续跑总期限检查，实际宽档仍待具备足够时间窗口的实例。 [已部署源文件与依赖哈希](../code/SG_wider_warmup_G1341/source_pins.json) [runner 部署回执](../evidence/SG_usage_recovery_G1276/SG_wider_warmup_runner_actual_staging.json) [预算生成工具部署回执](../evidence/SG_usage_recovery_G1276/SG_wider_warmup_gate_helper_actual_staging.json) [离线检查](../evidence/SG_usage_recovery_G1276/SG_wider_warmup_runner_fixture_validation.json)

## 同负载 profile 的逻辑工作量参照

原生 vLLM 独立分支提交 `cff38e0021cd7f859dea4b8a403b06c3791eb7dd` 保存了 16 条 32K／C16／输出 1024 诊断的 CPU 和设备证据。以该提交的 P/D 各 8 个 rank 的原始解析汇总为依据，已生成可供 SGLang 匹配的逻辑工作量参照。Prefill 65 个有查询步骤中，50 步处理 1 条请求、15 步处理 2 条，每条请求横跨 5 个步骤，合计调度 32767 个 query token。这个分布支持对相同工作量的单步耗时做进一步归因。 [参照结果及输入 SHA](../evidence/SG_usage_recovery_G1276/PD_native_work_and_device_reference.json)

Prefill 匹配使用请求数及 query、extend 后 KV 长度的四个汇总量。Decode 验证步骤只用可统一的 query 工作量，分别保留 vLLM rank 0 的直接 CANN connection 和 `SparseFlashMla` 物理输入形状。设备窗口保留 rank 维度，WAIT 在全类型区间并集中保留。SGLang 对接检查同一输入集的 SHA、16 条完整响应和各侧 8 个 rank，只有详细注释的目标步骤进入匹配。该工具已复算 vLLM 实际证据，SGLang 对接目前只有离线标签与输入边界用例。算子、通信和长尾的时间对比仍待实际匹配采集。 [对齐脚本](../code/PD_logical_work_alignment_G1345/build_PD_logical_work_alignment_g1343.py) [离线输入边界检查](../evidence/SG_usage_recovery_G1276/PD_logical_work_alignment_fixture.json)

原 G1317 32K 首轮预热继续运行，Prefill rank 0 已观察到 139264 个新增输入 token 和 4 次新增 KV 传输。16 条输出的终态审计尚待作业结束。 [在飞采样](../evidence/SG_usage_recovery_G1276/SG_PD_G1317_32K_W1_fourth_transfer.json)

## 64K 宽档部署文件

64K 候选实例的 P/D 部署文件已在两台节点分别校验 58 项。模型参数、图、并行和缓存配置沿用 G1317，候选文件仅调整实例目录、服务生命周期及 Decode 等待。候选 Router 保持 C16 和禁用重试，连同 Decode 等待设为 18500 秒。候选 64K 预热单轮为 18000 秒，冷测和同输入重放各留 34000 秒，均基于先前的同模型 32K 观察估算。现场 G1317 原服务保持运行，宽档候选尚未启动。 [64K 参数及模型参数哈希](../evidence/SG_usage_recovery_G1276/SG_64K_wide_PD_candidate_source_plan.json) [Prefill 文件验证](../evidence/SG_usage_recovery_G1276/SG_64K_wide_PD_candidate_physically_staged_P.json) [Decode 文件验证](../evidence/SG_usage_recovery_G1276/SG_64K_wide_PD_candidate_physically_staged_D.json)

Router 的准备工具已验证会拒绝旧实例凭据、任一 Decode rank 的等待不一致以及资源持有者不匹配。128K 和 256K 也通过了同参数生成检查。若按这批规划值执行且重放阶段几乎无缓存命中，两档的冷测加重放分别需要 136480 秒和 272480 秒的连续 CPU 客户端可用窗口，均超过现有客户端 70000 秒的保守租约。宽档须按实际时限和服务余量做启动校验。 [Router 配置及离线检查](../evidence/SG_usage_recovery_G1276/SG_wide_Router_renderer_fixture.json) [三档配置生成检查及资源窗口](../evidence/SG_usage_recovery_G1276/SG_wide_PD_window_source_validation.json) [生成工具 SHA](../code/SG_wide_PD_window_G1348/source_pins.json)

128K 和 256K 的 CPU 客户端配置已在 Decode 节点完成现场预检，计划租约分别为 138280 秒和 274280 秒。创建参数引用现场已校验的同一镜像、挂载、32 GiB 内存和四核 CPU 配置，设备权限为空。两份候选均使用现有 CPU 锁。现场验证该锁仍由原客户端持有，第二个持有者无法取得。预检首次遇到 Docker 元数据省略可选的 `DeviceRequests` 字段，已兼容该格式并重新核对了设备权限。新客户端尚未创建，配置一致性将在实际创建后、启动前再按 Docker 元数据对比。 [现场预检](../evidence/SG_usage_recovery_G1276/SG_wide_CPU_client_source_actual_staging.json) [保留的首次失败](../evidence/SG_usage_recovery_G1276/SG_wide_CPU_client_first_preflight_compatibility_failure.json) [生命周期离线用例](../evidence/SG_usage_recovery_G1276/SG_wide_CPU_client_lifecycle_fixture_validation.json) [源码 SHA](../code/SG_wide_CPU_window_G1351/source_pins.json)

同一 32K 预热进程继续运行。最新采样较预热前新增 212992 个 Prefill 输入 token，记录到 6 次 KV 传输。16 条响应仍待终态逐条审计。 [原进程在飞采样](../evidence/SG_usage_recovery_G1276/SG_G1317_32K_W1_after_wider_CPU_source_staging.json)

## Prefill 索引评分现场核对

G1317 原实例的八个 Prefill rank 都保留了同一次大输入索引评分记录，query 为 8192×32×128，key 为 4096×128。已核对容器实际评分源码 SHA。当前 128 MiB 上限下，首片 512 个 query row，按源码公式整个形状分 16 片。逐 head 的 BF16 分数张量在这次调用中合计对应 2 GiB 逻辑中间量，随后才进行 head 归约和因果掩码。该数量按 rank 和单次调用计算。 [实际操作数、分块和源码工作量](../evidence/SG_usage_recovery_G1276/SG_G1317_P_prefill_score_operand_and_work_bound.json) [工作量复算脚本](../code/SG_prefill_score_work_G1353/audit_P_prefill_score_work_g1353.py)

29 条连续 8192-token 分块日志的相邻间隔为 118–121 秒，中位数 120 秒。两次现场采样的八卡 AICore 读数均为 99–100%，HBM 余量约 478–541 MB。采样时 Prefill 有 1 条运行、8 条排队、1 条传输中，bootstrap 队列为零。两次读数的范围有限，仍需要同负载设备 trace 量化评分、共享 KV attention 和通信的相对耗时。当前显存余量支持先量化各路径开销，再评估扩大评分分块的条件。 [设备采样与原日志定位](../evidence/SG_usage_recovery_G1276/SG_G1317_P_prefill_score_operand_and_work_bound.json)

按无缓存且连续四块的 32K 情景推算，理论上约 20% 的 dense 评分位置最终被因果掩码排除，按现有 query tile 截掉统一无效后缀约可少计算 19.35% 的评分位置。实际 A/B 还需复核分数位模式、原生 top-k 和端到端运行时间。G1317 原预热已记录本轮第 7 次 KV 传输，运行仍在继续。 [估算公式及边界](../evidence/SG_usage_recovery_G1276/SG_G1317_P_prefill_score_operand_and_work_bound.json) [在飞计数](../evidence/SG_usage_recovery_G1276/SG_G1317_32K_W1_seventh_transfer.json)

Prefill 逐步归因工具已部署至 CPU 解析环境，并用保留的 G981 rank 0 数据库核对了所需六张 SQLite 表的字段。设备任务以唯一 CANN connection ID 连接到同线程的 CPU 前向区间，计算与通信重叠在设备时钟域内求并集。重复连接、跨线程和多重前向包含分别保留。专用评分标记已在离线调用用例中验证，仅在启用详细注释时包裹原评分调用。当前服务尚未加载标记，普通 `einsum` 留作待确认线索。本轮只完成字段及边界核对，匹配的 Prefill trace 仍待采集。 [已部署回执](../evidence/SG_usage_recovery_G1276/SG_P_step_attribution_actual_CPU_staging.json) [连接与时间区间验证](../evidence/SG_usage_recovery_G1276/SG_P_step_attribution_metadata_fixture.json) [评分标记验证](../evidence/SG_usage_recovery_G1276/SG_P_score_marker_fixture.json) [源码和依赖](../code/SG_P_step_attribution_G1355/source_pins.json)

32K 原首轮继续运行，Prefill rank 0 已记录本轮第 8 次 KV 传输和 278528 个新增输入 token，完整响应及指标仍待终态审计。 [首轮在飞采样](../evidence/SG_usage_recovery_G1276/SG_G1317_32K_W1_eighth_transfer.json)

### G981 旧 Decode trace 的图启动边界

在物理 910B 的 G981 rank 0 封存库上，4 次 `aclmdlRIExecuteAsync` 各有唯一的同线程前向区间和直接关联的设备 `MODEL_EXECUTE` 任务，前向顺序为 DRAFT、VERIFY、DRAFT、VERIFY。直接按 CANN connection 关联计算和通信时，19083 条设备记录中有 18424 条缺少匹配的 CANN API，659 条的 CANN API 位于这些前向区间之外。图启动的设备锚点单独统计。 [直接计算与通信关联实验](../evidence/SG_usage_recovery_G1276/SG_G981_direct_compute_comm_attribution.json) [实际设备锚点回执](../evidence/SG_usage_recovery_G1276/SG_G981_model_execute_window_audit.json)

4 个设备启动区间为 8.02–8.82 微秒。区间内开始执行的非占位 model ID 依次为 44、46、44、46，每个区间均有 2 条候选任务在结束边界后继续运行。分别统计启动与完整包含后，这份旧 trace 仍只支持图启动和图族候选的复核。完整重放耗时和当前匹配 Prefill 的设备逐步归因仍待采集。工具使用上节的冻结库与步区间解析器，运行时将 `code/SG_profile_readiness_G1218` 放入 `PYTHONPATH`。 [窗口审计工具](../code/SG_P_step_attribution_G1355/audit_SG_graph_launch_windows_g1360.py) [重复连接与跨界校验](../evidence/SG_usage_recovery_G1276/SG_model_execute_window_fixture.json) [固定源码 SHA](../code/SG_P_step_attribution_G1355/source_pins.json)

G1317 原 32K 首轮的最新采样已记录 10 次 KV 传输和 335872 个新增输入 token。客户端原进程仍在运行，完整响应及正式性能指标待终态。 [首轮在飞采样](../evidence/SG_usage_recovery_G1276/SG_G1317_32K_W1_tenth_transfer.json)

### Prefill 索引评分的因果后缀裁剪

候选沿用原生 128 MiB 的 query row 分块，只将 GEMM 的 K 输入裁到各分块内的最大可见位置。裁剪后继续应用逐行因果掩码，并在调用原生 top-k 前将后缀恢复为 `-inf`，保留原宽度和槽位映射。主机校验覆盖 129 种可见长度排列和 8 个闸门形状，包括不可见整片、并列分数及奇数宽度。四个 32K 建模分块的逻辑评分量合计减少 19.35%。 [候选实现](../code/SG_causal_prefill_score_G1362/causal_score.py) [主机校验和形状账本](../evidence/SG_usage_recovery_G1276/SG_causal_score_host_tile_fixture.json)

固定镜像的 910B 闸门已准备并核对服务器侧源码 SHA。它复用原生 per-32 FP4 fake quant、评分、top-k 和写回，分别要求 FP32 分数位模式、原生选中索引及 raw/page 写回逐项一致。K 宽度变化可能触发不同的算子路径，设备数值仍待实测。后续接入服务需要从可信的主机侧元数据取得各行可见长度，并核对其与设备行顺序的一致性。当前 P/D 实例继续原首轮，新增输入累计 360448 token，已记录 10 次 KV 传输，另有 1 次传输在途。 [910B 正确性闸门](../code/SG_causal_prefill_score_G1362/probe.py) [服务器源码校验](../evidence/SG_usage_recovery_G1276/SG_causal_score_probe_source_staging.json) [原首轮采样](../evidence/SG_usage_recovery_G1276/SG_G1317_32K_W1_eleven_request_equivalents_input.json)

从正在运行的固定镜像核对了位置和请求行生成源码。普通连续 eager Prefill 的第 `i` 行位置为 `prefix_len + i`。按现有压缩比，分块内最大可见 K 宽度可由最后一行直接计算，每片只需主机整数运算。规划器校验前缀、扩展长度、原生 K 宽度和总行数，已经用 2139 个随机请求及四个目标长度的末段布局比对逐行参考。接入还需要证明当前 Prefill 的实际设备行来源。替换位置、CP、图填充和尾层路径必须排除在该验证范围外。固定的 G1317 Prefill 参数为 attention CP 1、Prefill 图后端 disabled。 [固定镜像和位置来源](../evidence/SG_usage_recovery_G1276/SG_G1317_indexer_host_source_geometry.json) [主机规划器](../code/SG_causal_prefill_score_G1362/host_geometry.py) [四档布局校验](../evidence/SG_usage_recovery_G1276/SG_causal_score_host_geometry_fixture.json)

首版服务器冒烟校验在 Python 3.9 遇到 `zip(strict=True)` 兼容错误。移除该参数并保留向量长度校验后，原主机冒烟验证通过。固定推理镜像为 Python 3.12，设备数值校验仍待执行。G1317 原首轮新增输入累计 401408 token，已记录 12 次 KV 传输，客户端仍在运行。 [初次失败](../evidence/SG_usage_recovery_G1276/SG_causal_score_host_geometry_initial_failure.json) [修正后服务器回执](../evidence/SG_usage_recovery_G1276/SG_causal_score_host_geometry_stage.json) [首轮在飞采样](../evidence/SG_usage_recovery_G1276/SG_G1317_32K_W1_twelfth_transfer.json)

### 原 32K 首轮的续跑条件

通过活进程和文件描述符核对 G1317 的 P、D 两端八卡锁、Router、原配对锁及 CPU 客户端。原配对锁截至 11 月 19 日 13:30 UTC，原 CPU 客户端截至 21:09 UTC。若原首轮通过原始响应审计且两端清空队列，已准备的配对锁交接可覆盖到两端服务最早到期前 600 秒。后续每轮仍按实际剩余时间重新核准。按当前 12900 秒单轮上限计算，原 CPU 客户端能接纳新预热的最后时间约为 17:28 UTC。评估到达该时间前最近的合格暂停点，再决定是否续期 CPU 客户端。 [现场租约和时限](../evidence/SG_usage_recovery_G1276/SG_G1317_32K_live_continuation_admission.json)

独立终态审计增加了 16 个请求 ID 各出现一次的要求，并在终态复算原始记录、输入及配置 SHA 和落盘指标。用历史失败首轮的 16 条真实记录复算得到 7 条满足输入和输出长度，另用重复 ID 用例验证拒绝条件。新审计器已在原 32K 服务器成功识别现有 runner 和活客户端。最新采样累计 434176 个新增输入 token 与 13 次 KV 传输，终态审计仍待完成。 [终态审计器](../code/SG_32K_resume_audit_G1368/audit_SG_32K_W1_requests_g1368.py) [历史和重复 ID 复核](../evidence/SG_usage_recovery_G1276/SG_warmup_terminal_request_ID_fixture.json) [服务器部署回执](../evidence/SG_usage_recovery_G1276/SG_G1317_warmup_terminal_request_auditor_stage.json) [首轮在飞采样](../evidence/SG_usage_recovery_G1276/SG_G1317_32K_W1_thirteenth_transfer.json)

### G1317 首轮通过，第二轮已启动

原 32K 首轮于 11 月 19 日 11:50:40 UTC 到达暂停检查点。独立复核确认 16 个请求 ID 各出现一次，全部满足实际 32768 输入和 1024 输出。输入数据、配置以及原始明细、汇总 JSON 和 CSV 均按 SHA 核验。AISBench 记录的预热耗时为 7989.66 秒，输入、输出和总吞吐分别为 65.6208、2.0507 和 67.6715 token/s，平均 TTFT 为 4328484.6 ms，平均 TPOT 为 105.7 ms。正式对比仍待完成。 [首轮原始审计和校验索引](../evidence/SG_usage_recovery_G1276/SG_G1317_32K_W1_verified_raw_request_audit.json)

首轮结束且两端队列归零后，原配对锁持有者退出，继任进程接管同一把锁。已确认在同一 P/D 实例上启动第二轮，继续使用原 campaign 的总期限和单轮上限。启动预检复核并跳过首轮，第二轮使用不同输入 SHA，实际容器客户端和 runner 均在运行。 [配对锁交接](../evidence/SG_usage_recovery_G1276/SG_G1317_32K_W1_verified_pair_guard_transfer.json) [第二轮进程和输入](../evidence/SG_usage_recovery_G1276/SG_G1317_32K_W2_actual_original_campaign_start.json) [首轮末段 Prefill 分块](../evidence/SG_usage_recovery_G1276/SG_G1317_32K_W1_actual_P_chunk_pacing.json)

### 首轮逐请求时间轴

从封存的 AISBench SQLite 读取 16 条请求的原始时间点，独立复算观测峰值并发、总耗时以及平均 TTFT、TPOT、E2EL，与落盘指标一致。请求起始集中在 120.06 毫秒内，相邻首 token 到达间隔中位数为 480.21 秒。首 token 前的时间占逐请求 E2EL 总和的 97.56%，最慢一条为 96.45%。这段时间覆盖完整的客户端到首 token 路径，包含排队、Prefill、KV 传输和首步生成。预热期的设备阶段占比仍待 profile。复算工具使用已归档的 NumPy SQLite 解码器，实际客户端的三份计时源码 SHA 也与该解码器的既有验证版本一致。 [逐请求时间轴与源库 SHA](../evidence/SG_usage_recovery_G1276/SG_G1317_32K_W1_raw_SQLite_latency_timeline.json) [复算脚本和依赖](../code/SG_W1_latency_timeline_G1375/source_pins.json)

Prefill rank 0 在该轮记录 64 个 8192-token 分块，累计覆盖全部 524288 个输入 token，缓存命中为零。63 个报告间隔中有 62 个在 118–121 秒内，另一个为 1 秒。报告间隔中位数的四倍与客户端首 token 阶梯一致。固定源码确认日志中的瞬时吞吐以两次报告间隔为分母，保留末尾短间隔供审查。两个时钟域仅比较间隔规律，具体请求和设备任务的对应关系仍需匹配 trace。 [完整 Prefill 日志计数及源码定位](../evidence/SG_usage_recovery_G1276/SG_G1317_32K_W1_P_full_chunk_report_cadence.json)

### 当前运行版本源码归档

用实际启动回执核对了 G1317 两端各 58 项部署清单。Prefill 的 32 项和 Decode 的 33 项模型适配及容器入口代码均有逐字节匹配的仓库路径，角色复用的源码由索引直接引用。本次补入 Prefill 集成入口和两端共用的容器启动脚本。含现场网络、目录及配置的编排文件在受控的原部署留存，公开索引记录 SHA。 [实际运行源码索引](../code/SG_PD_wait_recovery_G1317/G1317_runtime_source_index.json) [Prefill 集成入口](../code/SG_PD_wait_recovery_G1317/prefill/integration.py) [容器启动脚本](../code/SG_PD_wait_recovery_G1317/run_pd.sh)

已验证范围包括固定镜像下的 910B P/D 部署、短请求、KV 传输和 32K、C16 首轮。1319 题 GSM8K 的 1266 条正确与 1 条超时属于上一版 G1178。当前 G1317 整体质量、四档正式性能、匹配 profile 和优化收益仍待验收。 [上一版质量与恢复配置](#g1178-最终-pd-实测联调) [当前首轮请求回执](../evidence/SG_usage_recovery_G1276/SG_G1317_32K_W1_verified_raw_request_audit.json)

### G1178 质量证据与当前版本的关系

在两台 910B 主机分别核对两个版本各 58 项原始部署清单及文件，并确认镜像和推理启动参数逐项一致。两版同角色只有控制器、启动入口和角色配置文件变更。两端 95 项既有权重校验记录计算出相同目录 SHA。角色配置中除实例目录对应的 profiler 路径外，实际值变化为 Decode 的 KV 等待从 3600 秒增至 14400 秒。 [Prefill 实机源码与权重记录](../evidence/SG_usage_recovery_G1276/SG_G1317_G1178_quality_lineage_P.json) [Decode 实机源码与权重记录](../evidence/SG_usage_recovery_G1276/SG_G1317_G1178_quality_lineage_D.json)

旧 GSM8K 第 304 条在约 600 秒触发客户端超时。历史日志缺少可在客户端与服务端关联该请求的唯一 ID，超时的服务端原因仍待确认。当前版本的质量复测脚本保留同一 1319 题、C16、原始请求与判分逻辑，以及 600 秒客户端请求上限。语料、判分器和脚本均在服务器校验 SHA，等待当前测试的合适空闲边界。第二轮 32K 正按原进程执行，较第二轮自己的基线新增 49152 输入 token 和 1 次 KV 传输。 [旧超时回执](../evidence/SG_final_quality_G1268/D_transport_window.json) [当前质量脚本](../code/SG_G1317_quality_G1378/run_full_gsm8k_SG_PD_g1378.py) [服务器校验回执](../evidence/SG_usage_recovery_G1276/SG_G1317_quality_worker_actual_stage.json) [第二轮实际采样](../evidence/SG_usage_recovery_G1276/SG_G1317_32K_W2_first_KV_transfer.json)
