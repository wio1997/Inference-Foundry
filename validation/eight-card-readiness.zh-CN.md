# 八张物理卡的准备边界（离线源码审查）

截至本记录，授权和实测仅覆盖两张物理卡的四个 die。八张 910C 物理卡通常涉及 16 个 die，但实际机器映射、可用性与权限必须在获得明确授权后核实；不能由本记录启动扩大规模的工作。

- **进程和通信**：`portable/runner.py`、`portable/worker.py` 固定四 rank；`rank_linear.py`、`coordinated_bank_lease.py` 和 `replicated_rank_moe.py` 对 `world_size=4` 的收集、广播和一致性路径有显式分支。需要先定义并实现目标 rank 数、物理映射、分区、HCCL 通信组和错误恢复，核对各权重切片和输出重组；简单提高设备数不能构成支持。
- **显存与权重**：当前 worker 是部分复制输入的实验路径，按 token 选择专家，并设置单 rank 专家读取/驻留上限各 512 MiB。round 361 的四个 rank `max_memory_allocated` 约 16.51–16.58 GB；这是该短请求的框架分配峰值，不是完整卡 HBM、长期驻留或新拓扑的容量结论。需分别测量持久权重、加载瞬态、KV/状态、通信缓冲、框架保留和整卡占用，并验证峰值余量。
- **执行路径**：当前诊断工厂启用 tokenwise KDA，DSA 选用显式 FP32 expanded 实验实现；worker 设置单 token、最多 128 上下文。必须建立受支持的 prefill/decode 与多请求批处理路径，再评估算子、访存和通信，不能根据现有短样本外推吞吐。
- **性能评测前提**：先固定经质量验证的模型、权重/源码/镜像、并行布局和完整服务或离线执行方式；定义 prompt/output 长度分布、并发、预热、总 token 吞吐、首 token 与后续 token 延迟分位数、显存与错误率，并记录原始输出及负载。实际硬件分配、长上下文和预算均需另行满足授权与资源协议。
- **阶段验收**：先完成多 rank 功能和状态一致性、当前检查点精度诊断；再做容量与稳定性验证，最后在受控负载下测基线和优化收益。现有四 rank smoke 不是八卡准备完成证明，也不能据此给出完成时间。

证据：`publication/Inference-Foundry/portable/{runner,worker}.py`、`DeepInfer/deepinfer/model_executor/models/glm5_next/{rank_linear,coordinated_bank_lease,replicated_rank_moe,hybrid_decoder,experimental_fp32_expanded_dsa}.py`、`evidence/round361/native-result.json`。
