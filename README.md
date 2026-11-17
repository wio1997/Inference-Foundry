# DeepSeek-V4.1-Flash W4A8 · 910B PD 对齐

独立分支记录固定 SGLang A2 daily 与 vLLM-Ascend 0.30.0rc2 的适配、测试和差异证据。用户当前合同为四档 32K/64K/128K/256K、C16、输出 1024，预热后首轮 cold32 和 hot32。每个 P/D 角色使用完整八卡，两个框架分时使用同一对机器。

两个节点的95文件权重均已校验，已通过两端全部 target/draft rank 的原生权重加载覆盖，当前在 KV 分配和 C1/C2 attention/PD 接入阶段；两套固定镜像的八卡 Host UVA/图重放组件验证已通过，尚未启动 DeepSeek 完整模型推理，没有正式性能或质量结论。已有服务器上的 FP8/FP4 模型不作为这次 W4A8 权重。

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
