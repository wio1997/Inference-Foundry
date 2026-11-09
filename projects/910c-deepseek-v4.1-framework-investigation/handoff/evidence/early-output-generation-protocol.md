# 提前输出与输入代次协议：待确认的大交付

**研究设计，未部署、未获 native 实现准入。** 不能单独应用 `sampler-early-output-copy.patch`；该原样配方已停止。实际输出等待约7–15ms的价值仍保留，范围扩大到下一输入的 staging 所有权，而非用保留原全完成屏障的小方案替代此目标。

## 2026-11-10 所有权范围收窄

后续源码审阅找到已继承的 `synchronize_input_prep`：target 准备前等待、准备后记录，涵盖 block table / zeroer / target CPU staging。**此前的全输入独立代次需求只是保守设计假设，不能当成已证明必须的成本。** 已知晚到持久源是 proposer backup（每卡16B容量），CP qsl/suffix 为本次新结果。详见[输入staging判断](input-staging-review.md)。

新增两条有条件边界：扩展现有复用栅栏并支付 lost host overlap；或在无 discard greedy 下利用 backup 臂无活读，仅消去其 CPU fill/H2D。后者源码条件已推导，实际批次资格与 native 逐位/完整末读仍缺。独立 ring/snapshot 保留为有活 fallback 时的选项。下文是更广所有权协议原则，不能据此预收全部输入复制成本，或声称没有原保护。

## 介绍

sampler IDs 已就绪时，DSpark 仍在 device/default stream 后续计算。提前只读 D2H 不改变原语数值，但会让 core 更早处理结果、推进队列和回收 request。因而输出 data-ready 与下一轮 CPU staging 可复用，不能共用一个“已完成”含义。

## 到了哪里

原 C8 的实际回复 rank 为0/8，9个输出线程等待已匹配；条件价值见 [Sampler判断](sampler-output-review.md)。native a2 在匹配16个当前 worker的 queue/allocation 环境下，证明 pinned H2D `copy_` 不自动快照：INT32[24] / INT64[16]在 event pending 时被CPU覆写，设备全部收到新值；独立代次控制逐字节保留旧值。**这不是未修改服务已经发生race的证明。**

实际源码存在持久可写的 CPU staging：`CpuGpuBuffer.cpu/np`、Ascend zeroer 的 `_ids_pinned`。后者直接异步 copy 后使用 device ID，前者经 `pin_memory()` 后异步 copy；需按具体存储别名逐一识别，不能只数Python对象。CoW 的GPU源页则先全部snapshot再写目的页，属于另一个已由current-stream顺序保护的关系。

关键次序：旧 draft 还占 default stream → 下一目标的 H2D可以已入队但未读源 → 过早回复腾出core队列位置 → 再下一代CPU准备可能写同一 pinned storage。device stream 顺序没有给这个 CPU store 建立 happens-before。

## 结果

| 交付 | 需要保持的所有权 |
|---|---|
| sampled IDs → CPU | 保存 sampler 原输出和 dtype；DSpark只读原IDs、对clone做discard处理。可以单独持有只读原输出引用。 |
| mutable CPU input/metadata → H2D | 每个 physical storage / byte range 取得独立 lease；只在该源的最后一次 H2D读完成后允许CPU覆写。相同 immutable cached plan 可共享，但必须证明key涵盖实际值、无原地更新。 |
| CPU staging slot 释放 | 记录真实 `H2D-read-done`；不能用“命令已入队”“data reply ready”或Python函数返回代替。环内无空槽时只等待所需源的读完成，记入净成本。 |
| device input/metadata buffer | GPU旧地址可沿用时仍须保持所有默认流及非默认读者的末读→下代写顺序；graph稳定地址不是CPU源存储可覆写的证明。 |
| KV/CoW/Engram/request slot | 依赖request ID+代次，不用队列索引作身份；保留CoW原源快照。新请求、重排、EOS/abort、相同ID重入、DP空转均要覆盖。 |
| KV/EC等外部异步退役 | 维持原完成屏障或独立work-done确认；不能把提前data-ready传给现有defer-free的processed-step逻辑。第一版限制无KV/EC/hybrid/EPLB/debugger，并保留原fallback。 |

可选实现是每帧不可变 pinned snapshot 或带读完成事件的 CPU staging ring。固定8192长度的额外INT32/INT64槽分别为32/64KiB，**只是尺寸示例，不是全部缓冲成本**；还需枚举block table、qsl/length、request映射和zeroer等实际源。GPU静态地址与CPU staging可分别管理，不能未经查证把所有metadata都复制多份。

当前最短请求的条件净省15.239ms；若以该请求TTFT改善1%作为参考，可额外支付约10.750ms非重叠代次成本。P90原条件收益仅0.5373%，本路线按短请求首token排序，未认领吞吐或响应结束收益。新lease引入的CPU等待可能改变原提交流水，不能在未绑定完整staging集合前把它的成本记成0。

下一步按实际 `CpuGpuBuffer.copy_to_gpu` / block-table commit / metadata缓存的物理来源追到最后H2D读者，形成可审查的完整lease集合和成本，再决定native实现门槛。已解决的底层“是否自动快照”不再重复探测；不是直接重跑已停止补丁。
