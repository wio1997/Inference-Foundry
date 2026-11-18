# C2 实际分配一致仍不足以支持跨 DSPARK 模式的 PD

固定 SGLang daily A2 镜像不变：7da963869ee54984dfeea2445befdf181710ef846448466db2361247759904ec，910B1/aarch64、torch2.10+cpu/torch_npu2.10post6、CANN9.1。模型、W4A8 主权重、BF16 KV、FP4 index、TP8/EP8/PP1/CP1/DCP1、overlap 和前缀缓存沿用原实例。D1020 仅用于关闭 DSPARK 的隔离诊断，不能替代最终对齐配置或性能结果。

## 原生状态契约与实际八卡证据

原生 Mooncake peer registration 在传输前检查状态条目长度，所以“偶数前缀不发送 C2 payload”不能绕过全局登记约束。固定镜像 CPU 契约实际运行，未初始化 NPU：原生校验拒绝 32768/8192 字节条目、接受 32768/32768，原生计价在三个合成 request-slot 数量下与物理公式一致。[CPU 证据](../evidence/targetonly_ring8_G1036/G1017_ring8_CPU_contract_2204_parsed.json)只证明函数与元数据契约。

D1020 的 helper 保留原生 resolver 的函数对象身份，让分配与计价使用同一 ring8 几何；未伪造布局属性、修改模型数学或放宽传输校验。原生池及 planner 源文件均先核对 SHA。[实际八卡门槛](../evidence/targetonly_ring8_G1036/G1026_parent_D1020_allrank_ring8_gate.json)与[原始快照](../evidence/targetonly_ring8_G1036/G1025_pair_ring8_actual_2204_parsed.json)记录：

- 八 rank target 各 1384 参数严格装载，missing=[]；无 draft 账本。
- 八 rank target query1 full graph 捕获完成，bs=[8,16]。
- 每 rank 三层 request-scoped C2 FP32 池，ring8，单层 shape=[402,1024]，1646592 字节。
- 原生固定计价 4939776 字节，恰等于三层实际分配之和；三个 wire item 均为 32768 字节。
- 实际 API：speculative_algorithm=None、decode radix=True、TP8/EP8、max-running16、context263296。

## 首批真实请求触发更完整的原生布局拒绝

P832 继续使用 DSPARK6，D1020 为 no-SP。旧 guard 真实退出后，新 guard 绑定实际 P832/D1020 身份并持有 canonical workload flock，才提交与 G999 完全相同的 C16 输入、采样参数及 HTTP task AST。首次启动前检查在解析 Docker 纳秒时间戳时失败，未创建诊断文件或发请求；改用整秒解析后完成全部原有门槛才提交，没有重试任何模型请求。

2026-11-18 21:46:54UTC，原生 CommonKVManager.try_ensure_parallel_info 在 /route 后比较 dsv41_spec_layout，拒绝 P 的 DSPARK 布局字典与 D 的 None。异常列出 compression_ratios、kv_item_lens、kv_layer_ids、num_draft_tokens、state_item_lens、state_types。C2 条目长度匹配只解决其中较底层的 stride 前提，不能使跨模式的完整 target/draft 布局相等。校验保持原样。

[原始失败、请求与终态](../evidence/targetonly_ring8_G1036/G1035_failed_ring8_stop_raw_2204_parsed.json)显示 simpleC16 全部 16 条实际等待约 240 秒后 TimeoutError，零 response 文件、零生成内容；后续 GSM cohort 未提交。客户端自然结束，父级核查时没有发送 SIGTERM。D 的实际控制器退出、容器停止、八卡 owner 释放均已确认；容器 OOMKilled=False，不能仅以 exit137 推断 OOM。26 个原始文件逐项校验 SHA 后归档，旧实例和服务日志保留。

这次失败不构成生成语义对比，也不能解释原 P/D974 双端 DSPARK 的算术、代码复制和 GSM8K 错误。上一轮全量质量仍为失败；不得把 CPU、分配、图捕获或 API 就绪升级为完整质量通过。

## 下一步及完整目标

单端关闭 DSPARK 的隔离方式已被实际原生握手否定。下一步使用双端一致的 no-SP 诊断，保持真实 ring8、原生计价、前缀缓存及完整 PD 校验；helper 只扩展至 prefill/decode 两个角色。[双角色 CPU 契约](../evidence/targetonly_ring8_G1036/G1033_pair_ring8_CPU_contract_2203_parsed.json)已分别在 P、D 配置下实际通过，NPU_initialized=False；P832 的 58 个基线文件也与本地保留副本逐项 SHA 相等。helper SHA 为 23b7022f231e9218d3e712db87a06e6b230f87b257c1793c830f2d9f79f5ce5a。CPU fixture 只约束元数据，双端实际装载、图、PD 生成与语义仍需单独验证。双端 1038 [准备清单](../code/paired_targetonly_ring8_G1033/prepared1038/preparation_manifest.json)已生成，各角色 58 个源文件固定 SHA，P/D 内存预算仍为 0.88/0.80；尚未部署。切换 P 会同时结束其容器内的 router，必须核对并接管该独立生命周期，再在新 P 中恢复原生 /ready 路由；不能仅替换模型后假定旧路由仍在。该诊断完成后仍须恢复目标 DSPARK 配置并修复原有语义故障。

当前状态见[终态索引](../metadata/current_pair_P832_D1020.json)。完整 1319 题质量、SGLang 四档 AISBench 首轮冷/热矩阵、最终匹配 P8/D8 profile 与差异地图、端到端优化收益均未完成。本轮没有改写 vLLM 基线或将诊断数据计作性能结果。
