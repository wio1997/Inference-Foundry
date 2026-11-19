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
