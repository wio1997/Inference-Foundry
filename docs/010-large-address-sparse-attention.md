# C1/C2 稀疏 attention：大地址与混合 bank 验证

G134 已完成小地址池的八卡数值和图组件验证；完整模型的物理 KV 池可能超过 4 GiB，因此必须另验大地址。G152 在每个 4312007680-byte 池使用高物理行地址，虽先把行号转为 int64，全部八卡仍出现 MTE invalid GM address。初始化使用异步 index_copy，进入 attention 前没有独立同步，因此该失败不能单独证明是 gather 或 index_copy 的根因；原始失败保留。

G160 使用已验证过的 int64 指针表 ABI，把池划为每 bank 2^20 行，单次 gather 相对地址小于 1 GiB。初始化改为连续视图 copy 并先同步；正确值在低 bank 偶数行和高 bank 奇数行，另外两组填不同 poison，避免错误别名因复制数据而误通过。指针表在图捕获前准备并由持久 pool Tensor 持有，捕获中不分配表。

真实八卡均通过 C1/C2 × decode16 / verify96 / prefill8192，共 48 个数值案例、48 次图捕获、144 次变更输入重放。decode/verify 全行 FP64 比较，prefill 每案例抽查 32 行 FP64、全部输出检查有限值；最大绝对误差 0.00024413317441940308。每个池 4312007680 bytes，物理行偏移 4194561，核对后所有 NPU 进程清理完成。

- [数值、图和各 shape 原始时间汇总](../evidence/SFA_operator_G160/summary.json)
- [大地址失败 rank0](../evidence/SFA_operator_G152/evidence/rank0.json)
- [大地址成功 rank0](../evidence/SFA_operator_G160/evidence/rank0.json)
- [实际算子](../code/SFA_operator_G160/lowratio_sparse_attention.py)
- [混合 bank 与 poison 验证输入](../code/SFA_operator_G160/probe.py)
- [固定镜像原生 FP4 语义](../code/fixed_native_FP4/sglang/kernels/ops/quantization/fp4_fake_quant.py)

第二次 warmed 单次调用在 C1 prefill 为 0.7486–0.7539 秒、C2 prefill 为 0.7027–0.7049 秒；这不是隔离 kernel 的测量，也不是端到端收益。尚无同臂噪声标定，不能据此判定相对 G134 的收益或退化。此结果只证明提供候选行的 attention 组件；原生压缩器、FP4 indexer、候选层语义、物理 pool 接口、PD 全状态、完整模型质量及四档性能仍未验证。
