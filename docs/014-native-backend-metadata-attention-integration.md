# 原生 V4.1 元数据与 NPU Attention 后端集成

固定 A2 镜像中，原生通用 DeepseekV4AttnBackend 已有 C1/C2 source、compressor、Indexer、candidate 和 DSPARK 控制，而既有 NPU 后端缺少低压缩比 source 入口，目标定制 SFA metadata 二进制也不可用。G268 继承完整通用后端，只在 NPU 硬件边界接入已验证的 BF16 banked Attention 和缓存存取；对实际 NPU/V4.1 的 page128 放宽原生 constructor 中唯一 page256 断言，仍保留真实页大小。入口在执行前核对固定镜像源码 SHA。

八个物理 rank 使用实际 checkpoint ModelConfig、ServerArgs/RuntimeContext 和完整 target 池，执行原生 prefill 元数据构造。测试 C16 下每请求 1/6 个 query，共16/96行，压缩比0/1/2，每个案例检查实际 causal positions、SWA 与 supplied physical KV selections。NPU Attention 与逐行独立 FP64 oracle 比较；共48案例、48次完整图捕获、144次变更 query 的图重放，最大绝对误差0.00024404376745224。controller 已终止并确认八卡无进程，所有本地候选文件与实际执行文件逐字节一致。

- [验收范围](../evidence/native_backend_G268/acceptance.json)
- [rank0 实测](../evidence/native_backend_G268/evidence/rank0.json)
- [执行入口与独立参考](../code/sg_native_backend_g268/probe.py)
- [原生后端继承与 NPU 边界](../code/sg_native_backend_g268/npu_backend.py)
- [controller 终态](../evidence/native_backend_G268/state.json)
- [完整 P/D 引擎候选接线](../code/sg_pd_g274/integration.py)

这些案例使用 runner 接口夹具与外部提供的物理 KV 选择，没有装载完整模型权重，没有执行原生 source/compressor/Indexer/candidate 选取，也没有网络 PD。各 rank 的 request_window 明确为 false，因此不能声称 BF16 request-window 改写已通过；其完整引擎路径仍待核验。小池不证明263296上下文容量或性能。

G274 将已验证的 G82 target/embedded FLOAT draft 加载适配、G226/G268 池与后端、G254 选定字节计价接入完整模型 P/D，保持 TP8/EP8、DSPARK5、BF16 KV、完整 D 图和原生优化开关。G274 目前仅作为待验收候选代码归档，完整权重覆盖、source/candidate 语义、DSPARK 接收回滚、跨机 PD 状态、图执行、前缀复用与四档性能均须后续实际证据。
