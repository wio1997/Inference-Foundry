# C1/C2 原生前缀缓存组件契约

G495使用固定SGLang镜像和真实八卡910B1，在完整目标/草稿池拓扑的小容量8192 token配置上完成80个组件用例。每rank四种长度128/256/512/1024执行原生FULL/SWA树插入、命中、加锁、驱逐，核对前后可用页数相等，共32例；六种奇偶/页边界长度127/128/129/8191/8192/8193核对请求私有C2状态清理和本机位置复制、相邻槽不变，共48例。全部结束后控制器确认八卡进程为空并释放占用。

同轮实际复现旧布局guard拒绝，以及无C128分配器时原生C128组件读取None.page_size的错误。G491第一次隔离检查因夹具缺少原生上下文字段退出，原始错误和释放状态保留；G495改用完整原生TreeCacheBuildContext。失败不归因模型或算子，两个阶段均未加载权重或发送跨机推理。

适配限定在真实低压缩比池、BF16/page128、FULL派生C1/C2页、原生FP4索引、分页SWA和请求私有C2 ring。保留原生UnifiedTreeCore与FULL/SWA组件，未伪造_unified_kv标志，未关闭前缀缓存。布局检查仅为满足全部严格拓扑条件的池扩展能力；hierarchical cache拒绝规则实际仍有效。注册器仅在此限定分配器确实没有C128池/allocator时省去C128组件，不删除请求表或真实C1/C2状态。

G501完整候选使用同一已通过模块，并启用两端原生metrics/cache report，便于随后核对真实cache hit与跨机状态。组件通过不等于真实PD状态传输、完整模型语义、实际263296长请求、C16性能或优化收益通过；这些验收仍待完成。G503只准备48份AISBench配置和绑定真实D容器/环境/挂载的采集源码，没有正式测量或server-usage通过结论。

- [八rank80例父级核验与范围](../evidence/native_radix_contract_G495/parent_acceptance.json)
- [rank0实际记录](../evidence/native_radix_contract_G495/evidence/rank0.json)
- [控制器终态及释放](../evidence/native_radix_contract_G495/state.json)
- [实际组件探针](../evidence/native_radix_contract_G495/probe.py)
- [保留的首轮夹具错误](../evidence/native_radix_contract_G491_failed_fixture/evidence/rank0.json)
- [G501严格限定接线](../code/full_model_candidates/sg_pd_g501/lowratio_radix_compat.py)
- [P候选](../code/full_model_candidates/sg_pd_g501/candidate_p.json)
- [D候选](../code/full_model_candidates/sg_pd_g501/candidate_d.json)
- [AISBench准备状态](../code/benchmark_preparation_G503/preparation_status.json)
