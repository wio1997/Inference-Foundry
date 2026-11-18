# 长输入prefill的临时张量峰值

G501完成四个真实原生PD检查：算术答案4以及127/129/131输入各32输出；无retraction，实际DSPARK verify计数为正，131请求报告128个cached tokens。这些是选定短例，尚不等于完整长输入或缓存一致性验收。D八rank完整target/draft图及八组PD内存注册通过并开放API。

随后8191输入在P的原生mHC `_mix_stats_impl` 的FP32 square申请632MiB时OOM，日志可用228–286MiB。P清理并释放八卡。G523只把P静态预算0.92改成0.88，保留chunk8192、EP8、DSPARK5、BF16KV、权重、前缀和已有数学helper，D501保留0.80及完整图。P池由3282816降为2353024 tokens，初始化日志可用量5.48→7.96GB；这是原生日志单位，不与OOM的GiB/MiB混算。

新P再通过算术，8191随后在适配的Engram `rotated_gate` 最终FP32输出求和申请642MiB时OOM，可用176–235MiB，55.90GiB active/56.71GiB reserved。预算调整越过了此前mHC位置，但没有完成长输入可用性，不能称性能改善。两轮失败和原始HTTP500均保留，P八卡释放。D后续日志显示旧请求deferred KV释放超时后释放；还须新P/D状态与功能验收。

源码显示门控计算后，key及original两份大FP32中间量已不再被读取，却仍保持局部引用到函数返回。G533候选只在最终输出表达式前删除两个引用，算式和张量形状不变。实际八rank64例零容差数值比较和分配门槛通过：8191/8193行的pre-output活跃分配各降低约1280MiB，PyTorch峰值分配各降低约640.4MiB。这只证明该独立算子的张量存活机制，不是总HBM、更大batch或端到端收益。

G533校验器的torch.equal只比较数值，不能区分正负零，原始记录的bitwise_equal字段表述过宽；原始文件保留，由显式更正覆盖其范围。G537随后在同一case/seed实际完成八rank共64例的BF16 int16位模式检查，包含符号零、mask、FP32 KV、非连续布局和8191/8193行；所有位模式与输入不变门槛通过。长形状pre-output分配降低1342177280/1342342144 bytes，峰值降低671516672/671599104 bytes，八rank一致；原始框架PyTorch分配计数不等于总HBM。组件结束已核验八卡清空。

G542 P-only完整模型于07:32:55UTC实际启动，保持G523的0.88/chunk8192/EP8/DSPARK5/缓存/权重，D501 0.80保留。源码AST核对仅增加已验证的del及每worker一次、无张量操作的大形状路径记录；完整模型长输入、缓存与功能验收尚未完成，没有恢复或端到端性能通过结论。

- [两次峰值的父级判断与范围](../evidence/prefill_memory_assessment_G535.json)
- [P501原始OOM](../evidence/full_model_G501/2203/server.log)
- [四个短例及8191失败](../evidence/functional_G506/functional/state.json)
- [P523原始Engram OOM](../evidence/full_model_G523/2203/server.log)
- [P523退出与八卡释放](../evidence/full_model_G523/2203/state.json)
- [保留D的图/注册/API实际证据](../evidence/full_model_G501/2204/actual_graphs_registration_API.json)
- [G533数值与内存原始rank0](../evidence/engram_lifetime_G533_numeric_only/evidence/rank0.json)
- [比较器范围更正](../evidence/engram_lifetime_G533_numeric_only/comparator_scope_correction.json)
- [G537严格位模式探针](../code/engram_lifetime_G537_strict_bit_gate/probe.py)

- [G537严格64例及释放的父级复核](../evidence/engram_lifetime_G537_strict_bits/parent_acceptance.json)
- [G537实际rank0](../evidence/engram_lifetime_G537_strict_bits/evidence/rank0.json)
- [G542实际P-only启动](../evidence/full_model_G542/2203/actual_start_parent_receipt.json)
- [G542部署源码与AST前置检查](../code/full_model_candidates/prepare_prefill_lifetime_g542.py)
