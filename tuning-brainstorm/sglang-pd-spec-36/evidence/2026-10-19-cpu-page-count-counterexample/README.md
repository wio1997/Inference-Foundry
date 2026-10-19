# M005/M017：默认 CPU 页计数候选的差分与转向

本轮完成 baseline 与隔离候选的完整模块 CPU 差分：99 组常规对照、3 个预期 stale-mirror 反例，共 696 项检查。fresh mirror、显式 count 与回退用例按实验合同通过；三例说明 dtype/shape/device gate 不能证明镜像新鲜。**不推广修改 allocator 默认行为，转向在已核验 producer 处显式传递现有 count。** 原候选和反例全部保留，整体 serving 仍为 E0，原生 PD+MTP 基线为空。

固定源码为 `ce0a8fea5f4245839d174b2b8c55692be1810a7f`。R001 从 canonical NPU allocator 构造独立候选，只改变 alloc_extend 默认 count=None 的计算分支。AST 检查表明显式 count、原 device fallback 和 count 后的分配/kernel/free 代码未变。R002 完整导入 baseline 与候选模块，在任务 CPU 容器中执行。安装源码与参考服务没有应用该改动。

## 验证方式

两侧使用真实类的 CPU page/slot tensor，kvcache=None、debug_mode=False。带标记的 CPU Tensor 子类统计逻辑 device 输入的显式 item；这不是 NPU 等待耗时或实际同步次数。

低于 200 页的路径在真实 `sgl_kernel_npu.mem_cache.allocator` 模块导入请求处抛出观测异常，发生在 import/launch 之前。没有返回 fake kernel、没有伪造 loc。原生导入边界停止只验证分支与停止前状态，不能计成 kernel 成功。

达到 200 页的原有 Torch 分支真正执行，逐项比较 int32 loc、剩余 page tensor 与 available_size。容量拒绝也真正执行。两侧 debug 配置相同；其他 debug bool 可能带来的原生同步不在本轮耗时结论中。

## 结果

| 类别 | 组数 | 观察 |
| --- | --- | --- |
| fresh，0/1/199 新页 | 18 | 候选无带标记 item，baseline 有 1 次；两侧在 native import 前停止 |
| fresh，200/201 页 | 12 | 实际 Torch loc、dtype、页池与容量一致 |
| fresh，32768 输入 B2/3 | 12 | 按实际新页数选择 Torch 或 native-import 边界 |
| fresh，带 ps−1/ps/ps+1 前缀 | 12 | 对照 loc/页池或同一 native-import 边界 |
| 显式 count | 18 | valid/None/non-Tensor CPU 参数保持既有行为，无 tagged item |
| 回退 | 21 | None、非 Tensor、meta device、float/dtype/shape/ndim 不匹配均保留旧 tagged item |
| 容量拒绝 | 6 | 真实返回 None，页池一致 |

常规用例中 37 组执行了 Torch 成功返回，6 组实际容量拒绝，56 组在 native import 前停止。所有类别都不能证明原生小页成功或服务性能。page 512 时 raw 32768 的 B2/B3 只有 128/192 新页；本轮保留这些输入，按真实分支处理，没有把它们改成更易通过的 Torch 大页用例。

## 三个反例及其含义

反例的 device 长度需要 200 页，同 shape/dtype 的 CPU 镜像却报 201 页。两侧都执行真实 Torch 分支，返回 loc 完全相同，但候选多消耗一页：在 page 128/256/512 时，分别额外占用 128/256/512 个逻辑槽位。

这说明只比较返回 loc 会漏掉容量错误，metadata gate 也无法鉴定数据是否属于同一次调用。反例违反了“CPU 是权威新鲜镜像”的前提，不否定该前提成立时的数学等价性；它否定把这个 gate 当成通用新鲜度证明、直接推广默认方法的做法。没有证据表明当前任务服务正在传 stale 镜像，也没有据此宣称现有 serving 存在该 bug。

原 NPU alloc_extend 的默认分支不依赖 CPU 长度值；全局改为自动信任 CPU 值，会扩大默认行为的前置条件。下一版保留 allocator 原默认 API，在源头拥有数据的 ordinary prefill、PD preallocation 与 EAGLE/MTP producer 处计算并显式传递现有 num_new_pages。未知调用者、None 路径与其他 allocator 不应被自动改写。

## 未满足的接受门槛

本轮没有性能测量，没有原生 kernel 编译/launch，没有物理 KV、PD 网络或完整 scheduler。fresh CPU 差分通过也不证明移除默认 item 后的 stream/WAR/消费者生命周期安全；原 item 可能提供了隐含的主机等待。新方案仍需任务自有 Ascend PD+MTP 的同条件 trace、loc/output、页回收、传输完成、设备读结束及端到端噪声分析。

计数的额外主机计算、类型分支与内部参数传递也有成本。短路径或 device 已就绪时可能无收益；其他 allocator 或外部平台要保持原调用。任何 loc、dtype、页身份/容量、None 返回、输出正确性或生命周期不一致均否定候选。已有 optional count 是基线能力，不能重复包装为新优化。

R002 容器已删除，参考服务的 ID、启动时间与 restart count、四个只读 metadata/tokenizer 文件前后一致；无权重、accelerator 设备或目标模型推理请求。

Loop 结论为 PIVOTED，causal conclusion 为 partially-supported。实验协议通过与优化被接受是两件不同的事：原默认方法候选未被接受，下一轮处理显式 producer 传递，原生性能仍未决。
