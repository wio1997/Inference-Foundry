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

## 完整引擎首次接线的校验失败

G274 两端在任何模型权重加载前被 attention_registry.py 的 SHA 校验拦截，随后 controller 清理并确认八卡无进程。新增校验值误取自已有 GLM 工作副本；从实际固定镜像的已停止容器取回源码后逐行核对，attention_registry 的差异是 GLM KDA 路由，kv_cache_configurator 的差异是 GLM draft 池分支，DSV4 分支没有这些修改；request-window 源码完全相同。保留严格校验，G280 改为使用实际固定镜像的 SHA，未删除或放宽指纹检查。

- [P 原始失败](../evidence/full_model_G274/2203/server.log)
- [D 退出和释放](../evidence/full_model_G274/2204/state.json)
- [实际固定镜像源码清单](../evidence/G279_fixed_native_sources/manifest.json)
- [registry 逐行差异](../evidence/G279_fixed_native_sources/attention_registry.py.diff)
- [G280 指纹](../code/sg_pd_g280/native_integration_sha256.json)

G280 为重新接线的完整候选，不能从启动请求推定模型加载、图、PD 或性能已通过。

## 完整权重加载通过后的分配器边界

G280两端各8个target与8个DSPARK draft的原生load ledger全部通过、missing为空，共32份。目标/草稿权重实际加载后，选定计价已执行，bytes_per_full_token=2798.25；P日志计算full=3283072、SWA=49536，D计算full=3406208、SWA=41216。该计算与target池构造已经越过旧的低估预算OOM路径，但不能据此声称整个服务容量已通过。

随后原生NPU分配器无条件为C128池创建allocator，并访问None.kernel_page_size而失败；checkpoint仅使用0/1/2压缩比，C128池本就不存在。两端完整失败与八卡释放已保留。G294限定在实际LowRatio池、DeepSeek V4.1/BF16/page128，复用原生FULL/SWA分配与请求状态清理；C1/C2物理地址继续由FULL/ratio派生，C4/C128 bundle为空，不创建假池或丢弃真实C1/C2 payload。真实C4/C128层仍走原生路径，意外非零C128 prefix会被拒绝。

- [P完整加载与分配器失败日志](../evidence/full_model_G280/2203/server.log)
- [D终态和释放](../evidence/full_model_G280/2204/state.json)
- [实际固定镜像 allocator/SWA/storage 指纹](../evidence/G293_native_allocator_sources/manifest.json)
- [G294限定的低压缩比分配器](../code/sg_pd_g294/lowratio_allocator.py)

G294已作为新的完整P/D候选启动；此时尚未验收完整引擎分配、目标/草稿图、source/Indexer/candidates、prefix、DSPARK接受/回滚和跨机PD。接线通过或无异常日志不能替代这些实际门槛。

## Draft 共享 allocator 的原生分支

G294两端再次通过32份完整权重ledger，并实际构造target低压缩比池及分配器（日志Memory pool end）。随后DSPARK draft传入已有target allocator，包装器错误地断言其必须为None而退出。原生factory的已有allocator分支不仅返回同一对象，还绑定draft的SWA映射；不能跳过或重新构造它。

G301直接取实际固定镜像factory函数，在CPU接口夹具中复现原断言失败，分别验证普通共享映射与full-capacity identity映射两条原生分支。修正包装器对已有allocator直接委托原生函数，返回身份与映射语义均保持。该检查没有创建torch张量、执行NPU、装载权重或做PD，不作为完整服务验收。G304使用这一修正重新启动完整P/D候选；原失败与释放状态保留。

- [原始D端失败](../evidence/full_model_G294/2204/server.log)
- [P端终态与释放](../evidence/full_model_G294/2203/state.json)
- [实际函数CPU控制契约结果](../evidence/shared_allocator_contract_G301.json)
- [控制契约复现代码](../code/shared_allocator_contract_G301/verify_shared_draft_allocator_g301.py)
- [G304保留原生draft分支的接线](../code/sg_pd_g304/lowratio_allocator.py)

完整目标/draft图、C1/C2 source/Indexer/candidate、跨机PD与prefix、DSPARK接受回滚和四档性能仍未验收。

## 完整图路径暴露的第二处页大小限制

G304两端各8个rank均通过target/draft严格装载并创建原生共享allocator和实际FULL/SWA池。P端完整引擎启动、原生SDK初始化并开放API。D端进入真实target-verify完整图捕获后，在PagedIndexerMetadata.__post_init__中的固定256页断言退出并释放全部8卡。P API开放不等于模型语义、跨机PD或性能验收；D失败不作为算子正确性结论。

原生低比率索引工厂已将FULL页表展开为实际64-slot索引页表，所有地址和max_seq_len均使用显式page_size/compressed_page_size。G314保留真实128/64几何，仅在V4.1 NPU backend索引工厂的上下文中，且ratio为1/2、TORCH logits路径、无DeepGEMM强制metadata、无topk-v2时放行原生128页。默认调用上下文及其它设备保持256页断言，未伪造页大小，也未切换图、量化、EP、DSPARK参数。原生metadata源文件SHA严格核对后才修改该断言。

CPU接口复现执行实际固定源码的__post_init__，原128页断言失败可复现；ratio1/2、rows_per_request1/6四组合通过，越界上下文/设备/页几何拒绝，256页路径保留。CPU夹具不证明真实张量、图或模型正确。G314仅重启D端，P保持G304；最新观察D仍在装载权重，完整图、source/indexer、跨机PD、奇数长度C2状态和DSPARK回滚仍待验证。

- [D端原始完整图失败](../evidence/full_model_G304_D/server.log)
- [D端释放状态](../evidence/full_model_G304_D/state.json)
- [P端完整引擎启动证据](../evidence/full_model_G314/observation_2203.json)
- [页几何原生方法CPU接口复现](../evidence/G314_paged_metadata_contract.json)
- [修正的严格上下文和几何约束](../code/sg_pd_g314/paged_indexer_scope_g314.py)
- [实际D端新候选观察](../evidence/full_model_G314/observation_2204.json)
