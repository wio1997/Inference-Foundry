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
