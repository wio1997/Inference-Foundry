# 八张物理卡的准备边界（离线源码审查，更新至本次实测）

当前授权仅覆盖两张物理卡的四个 die。八张 910C 物理卡通常涉及16个 die，但目标机器的实际映射、空闲状态和额外权限尚未核实。此文只归纳源码和现有四die证据，不授权扩大分配、启动测试或运行服务。

- **rank 与控制**：普通文本 runner/worker 接受显式 world size 2、4、8、16；已有两rank与四rank固定短例，8/16 rank 的实际模型、HCCL 与设备映射未验证。实验 layer-major KDA、all-KDA、双请求和参考探针仍要求四rank，不能仅把当前命令的 world size 改成16。现有资源 helper 的四die工作策略也不构成更大规模许可。参考 [rank验证](rank-portable-round408.json) 和 [双请求验证](two-request-diagnostics-round505.json)。
- **分区与通信**：`rank_linear.py` 要求每rank等宽且16行对齐，并用 all-gather 重组；`rank_text_install.py` 安装 rank dense/expert 接缝并协调状态；`coordinated_bank_lease.py` 对本地失败做 all-reduce 协调；`replicated_rank_moe.py` 仍含路由及通信。注意力、路由、embedding 和 head 在当前worker中仍有复制路径，不能把它等同完整生产 TP/EP。扩展前需核验所有层的切分整除、组内身份、通信顺序、失败处理及权重/输出重组，再逐阶段做有授权的功能验证。
- **内存和权重**：当前worker对专家读取与驻留分别设512MiB界限，单独的可选缓存按配置限制；这些上限不是整卡峰值。固定 recall17、四die、全部34层KDA 40+1预填和128MiB缓存的两次点采样，`npu-smi`显示每die已用18731–19058 MB；框架最大 allocated 为16155277312–16172846080 bytes/进程，缓存关闭后为0且无活动租约。点采样不是全设备连续峰值，框架分配、设备总量、RSS和逻辑缓存不能相加，也不能据此推算16die容量。[数据和版本](all-kda-memory-round504.json)。
- **执行路径与边界**：标准worker的本地 token/context 上限取决于显式模式（当前CLI上下文不超过128）；实验 all-KDA 模式使用40 token块和单token递推，DSA仍是逐token的FP32 expanded诊断实现。扩大的KV/状态、注意力展开、加载瞬态、通信缓冲、图捕获、批处理和服务生命周期未被上述短例覆盖；需先建立受支持路径，再测分配峰值与可复现时延。
- **验证顺序**：先冻结源码、镜像、检查点、拓扑和输入；在额外授权后逐步验证rank组、切片、collective、状态一致性和失败清理，再测整卡容量与稳定性；独立精度范围须单独确定。最后在受控预热、输入/输出长度、并发、吞吐、首token/后续token延迟、资源和错误率口径下测基线。现有四die结果既不是八卡验证，也不支持缩放收益或完成时间估计。

源码入口：`portable/runner.py`、`portable/worker.py`，以及 `DeepInfer/deepinfer/model_executor/models/glm5_next/` 下的 `rank_linear.py`、`rank_text_install.py`、`coordinated_bank_lease.py`、`replicated_rank_moe.py`、`kda.py`、`experimental_fp32_expanded_dsa.py`。本文件未执行八卡作业或更改服务。
