# 910C 运行环境恢复：资源归属与首次真实 NPU 探针

本轮恢复了服务器连接，并为本任务取得 **44 Prefill / 46 Decode，各16个 Ascend 资源**。46节点真实 NPU API 正确性探针三项通过，探针容器已删除。**整体仍为 E0，完整 PD＋MTP 模型服务及性能基线尚未建立。**

## 实际完成

- 通过既有跳板访问集群，重新读取当前占用。33–37、41、43、45等可达节点有其他服务；未启停或复用这些服务。48节点为 Ready Unknown，排除。
- 44/46/47的只读观察没有发现 NPU 进程、活动加速卡资源请求或普通运行容器。随后再次检查44/46，以调度器创建本任务独占保留；空闲观察本身没有当成资源租约。
- 两个新 Pod 的 owner、UID、节点、16资源请求与 Running/Ready 状态均已核实。专用 PriorityClass 为 value=-1000、globalDefault=false、preemptionPolicy=Never，不修改任何默认类别。[资源摘要](resource-summary.json)、[实际保留回执](runs/L039-R006-reservations/stdout.json)。这些保留继续用于接下来的 P/D 准备。
- 后续测试位置固定为910C服务器。Mac仅作为主控、记录与提交端。之前 Mac 上的原始 C++ 哈希构建完成，但待跑的 L038-R005 已在派发前取消，未产生哈希语义测试结果。

## 真实设备探针的范围

[探针代码](scripts/probe_L039_npu.py)运行于46节点本任务独立容器，设备返回 `Ascend910_9392`。使用镜像 `sha256:c5cf2deb674cd1c8aba10a70d8dcb3e413d3739f6a54808c568e11d711d2e263`，镜像内 git HEAD 为 `ce0a8fea5f4245839d174b2b8c55692be1810a7f`，Python3.12.13，torch2.10.0+cpu / torch-npu2.10.0.post6。

| 检查 | 负载 | 实际结果 |
| --- | --- | --- |
| INT32 H2D → 索引 → 加一 → D2H | 64×64 输入，4×64输出 | 与独立 CPU golden 精确相同 |
| FP16 NPU 缩放与回传 | 16×16，乘2 | 与 CPU 精确相同 |
| BF16 NPU 缩放与回传 | 16×16，乘2 | 与 CPU 精确相同 |

[原始结构化结果](runs/L039-R007-npu/result.json)、[容器隔离与清理](runs/L039-R007-npu/remote-result.json)、[主控回执](runs/L039-R007-npu/receipt.json)。容器网络隔离，仅暴露一个计算设备与所需控制设备，driver/firmware挂载只读，不挂模型。结束后按本任务标签验证并回收容器。这是小规模设备 API 正确性检查，没有逐算子 profiler、权重加载、PD交接、MTP验证、服务输出或性能测量，不覆盖36机制的原生正确性。

## 保留的失败与恢复

1. L038-R003：网络可达，但原认证失效，SSH255；用户补充认证后新 Run 恢复。凭据没有进入执行器 prompt 或提交。
2. L039-R003：集群拒绝在 Pod 中直接指定 Never，零 Pod 创建。[失败回执](runs/L039-R003-reservations/stdout.json)。
3. L039-R005：专用 PriorityClass 创建成功，但 Kubernetes省略 false字段，主控脚本将其按必填读取，触发 KeyError；零 Pod 创建。[错误记录](runs/L039-R005-reservations/stdout.json)。L039-R006使用新冻结脚本，读取已存在对象并核实 owner/UID/策略，随后取得资源，没有覆盖或重建已有对象。

L038/L039共13条实际派发的冻结命令，各执行一次，前后输入哈希相同；内层失败不会因外层CLI0改判成功。[执行器元数据审计](zcode-audit.json)。原始模型I/O、认证材料、其他任务的挂载清单与大型日志留在本地，不公开。

## 仍需闭合的原生合同

原37节点恢复的 `config.json` SHA256 为 `ca8f2f47b07919a514c0ca223dc2ea2bc7445afaa5ac76c013a3784e096426ca`，解决了此前只有公开替代文件、哈希不匹配的问题。[源码复核](source-revalidation.json)核实六份文件与固定ce0版本相同。实际服务归属和模型权重仍按本任务合同独立建立。

44/46已有原GLM-5.3普通W8A8文件，其config/tokenizer可与37一致，但量化描述和权重索引的哈希不同，不能直接替代原W8A8C8；Flash模型也不采用。[模型身份差异](model-contract-difference.json)。44还缺46所用B103镜像，需先对齐镜像/source及原量化权重，验证容量，再冻结TP/EP/PP/DP/CP、KV dtype、传输、MTP参数及输出判据。[当前部分运行合同](runtime-contract.updated.json)。

容量与硬件上界尚未校准：设备总容量不能替代每rank权重、工作区、KV/草稿池与尾块的可行性证明。本轮没有带宽、FLOPs、TTFT、ITL、接受长度或有效token/s，所有性能字段仍UNKNOWN。完整基线继续保持Ascend、原GLM-5.3、PD分离和推测解码。

脚本与冻结spec用于审查真实命令和条件。新环境复现须先取得新的任务资源、核验镜像与模型，再登记新的Run/spec；不要重放旧资源创建命令或修改旧输入后沿用旧回执。
