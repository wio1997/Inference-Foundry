# SGLang · DeepSeek-V4.1-Flash W4A8 · 910B PD 对齐

本分支维护固定 SGLang A2 daily 的适配、部署、压测和profile证据。vLLM 独立维护于 [vllm-deepseek-v41flash-w4a8-910b-pd](https://github.com/wio1997/Inference-Foundry/tree/vllm-deepseek-v41flash-w4a8-910b-pd)。历史跨框架组件与源码快照保留原始记录，后续各框架只更新自己的分支。用户当前合同为四档 32K/64K/128K/256K、C16、输出 1024，预热后首轮 cold32 和 hot32。每个 P/D 角色使用完整八卡，两个框架分时使用同一对机器。

95文件权重与target/draft装载、D图capture和多个组件已有证据。G603/G604的7个所选完整PD响应、usage和SHA已由父级核验，8191越过原OOM点；32K在客户端300秒读超时，无完整响应，后续缓存重放未提交。两端实际队列/PD请求清空后，原owned controller在09:41–09:42UTC释放八卡，保留全部原始数据以切换用户指定vLLM0.32。完整质量、四档AISBench与双方profile/差异地图尚未完成。组件通过不能代替整体验收。

[当前Ascend daily与1017/A2支持范围](docs/031-current-Ascend-daily-support-scope.md)：最新官方nightly只构建A3/950，现有A2日期镜像未发现可直接替代的原生V4.1构建；保留现有A2适配。vLLM已按用户更新为v0.32.0rc1，详情在其独立分支。

- [权重来源和固定文件](metadata/pinned_files.json)
- [配置](metadata/config.json)
- [量化格式](metadata/quantization_summary.json)
- [当前证据和未验证项](evidence/readiness.json)
- [镜像源码与 SHA](evidence/source_provenance.json)
- [接入缺口](docs/001-adaptation-contract.md)

原始权重、连接信息和大型原始 trace 不进入公共仓库。源码文件按固定镜像原样保存并保留原许可证头。历史 DeepSeek V4/DSpark7/C12 的仓库结论不代表此次 V4.1/PD/C16 测试。

[Engram 原生 API 和接入路径](docs/002-engram-native-path.md)

[双框架八卡 Host UVA 与图重放组件证据](docs/003-host-uva-components.md)

[普通 HCCL 与 MoE 缓冲分离的八卡组件观测](docs/004-HCCL-buffer-split.md)

[固定镜像 DSPARK 原生图路径与参数口径](docs/005-native-DSPARK-graphs.md)

[SGLang INT8 Engram 原生接口接入与图执行验证](docs/006-SGLang-INT8-Engram-integration.md)

[完整权重、首次PD启动失败与接入候选](docs/007-full-model-PD-readiness.md)

[全部权重覆盖、KV 字节预算根因与 C1/C2 接入边界](docs/008-KV-byte-budget-and-C1-C2-contract.md)

[C1/C2 sparse attention八卡FP64数值与图组件](docs/009-C1-C2-sparse-attention-component.md)

[大地址与混合 bank 的八卡数值及图验证](docs/010-large-address-sparse-attention.md)

[原生 C1/C2 压缩器、RoPE 与 FP4 路径验证](docs/011-native-C1-C2-FP4-math.md)

[原生 FP4 Indexer 池写入、读回和完整图验证](docs/012-native-FP4-index-pool.md)

[910B 原生低压缩比 target/draft 池组合与逻辑页字节](docs/013-native-lowratio-target-draft-pool.md)

[原生 V4.1 元数据与 NPU Attention 后端集成](docs/014-native-backend-metadata-attention-integration.md)

[固定依赖栈 partial RoPE 缺失与原生兼容组件](docs/015-native-partial-RoPE-compatibility.md)

[本分支范围与历史归属](metadata/framework_scope.json) · [分支迁移说明](docs/016-framework-branches-and-reference.md)

[mHC原生split/Sinkhorn平台缺口](docs/017-mhc-native-split-sinkhorn-gap.md)

[mHC H5120 原生组件门槛、失败和完整候选范围](docs/020-mHC-native-5120-interfaces.md)

[完整 D 图 MoE 路由缺口与 Ascend 数学调用候选](docs/021-native-router-CANN-and-math-lowering.md)

[NPU复数频率索引的逐位兼容门槛](docs/022-complex-frequency-gather.md)

[完整原生V4频率索引覆盖修正](docs/023-complete-frequency-index-coverage.md)

[完整图工作区显存与D端预算候选](docs/024-full-graph-workspace-memory.md)

[完整target图与草稿RoPE设备分发修正](docs/025-native-draft-rope-dispatch.md)

[主模型和草稿的原生混合通信dtype](docs/026-native-mixed-target-draft-dispatch.md)

[完整target/draft图与待解决的前缀布局契约](docs/027-full-target-draft-graphs-and-radix-layout.md)

[C1/C2原生前缀缓存：八rank80例组件契约](docs/028-native-lowratio-prefix-cache-contract.md)

[有界CPU客户端：24份同字节数据与448前缀复核](docs/029-CPU-client-readonly-dataset-revalidation.md)

[实际prefill两次峰值与Engram张量存活机制](docs/030-prefill-transient-memory-and-gate-liveness.md)

[2026-11-18上游A2适配范围复核](docs/031-upstream-refresh-20261118.md)：官方main/release已重新固定，现有A2镜像与适配保持。

[SG832完整质量失败与分页索引候选](docs/033-SG832-full-quality-failure-and-paged-indexer-candidate.md)：14条600秒传输超时，完整质量未通过；实际八rank短诊断已留证，候选严格门结果待审。

[分页FP4索引器八卡严格验收与D918启动](docs/035-paged-indexer-all8-strict-gate-and-D918.md)：96个数值用例+96次变输入图重放通过；完整模型正在装载，质量/四档性能/最终匹配profile/端到端收益待验证。

[完整模型频率wrapper失败与D941修复](docs/036-D918-complex-frequency-wrapper-failure-and-D941.md)：D918图捕获失败并释放，D941补回原有gather后实际启动；完整模型/质量/性能仍未验收。

[D941真实PD/SSE、C2分页命中与完整质量](docs/037-D941-PD-SSE-real-kernels-and-full-quality.md)：限定功能通过，但1319题尝试出现14个600秒超时后已停止；C1大位置表仍在，完整质量/四档SG性能/最终profile/收益未完成。
