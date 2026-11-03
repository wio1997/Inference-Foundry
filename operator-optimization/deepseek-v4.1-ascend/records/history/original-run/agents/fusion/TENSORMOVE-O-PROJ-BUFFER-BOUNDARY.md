# EP8 TensorMove：投影写回的源码边界与条件性候选

## 历史记录直接支持什么

固定 EP8、L0 的 `TensorMove` 与 `aclnnInplaceCopy` 的唯一 flow、调用栈及 `[8184,5120]` BF16 形状，复用 [已有身份核查](../diagnosis/TENSORMOVE-EP8-IDENTITY.md)。调用栈精确指向 CP `_project_output` → 基类 `_forward_o_proj` 的 1798 行；这支持其与输出投影写回的关系。trace 没有绑定输入/输出设备指针、算子实参的 storage identity、实际运行二进制哈希与全部前后任务的张量关系，不能声称已绑定某个设备矩阵乘 producer 或消费 kernel。CSV 37.100 µs 是 profiler 内该次设备任务时长，不能当作线上节省时间。

## 冻结源码链路（与历史绑定分开）

1. `models/deepseek_v41/model.py:760–763` 创建 `output = torch.empty_like(hidden_states)`，作为实参传入 `torch.ops.vllm.dsa_v41_forward` 并返回该 output。`attention/dsa_v41.py:56–80` 将 custom op 注册为 `mutates_args=["output"]`，真实调用以该 output 交给 V41 attention 实现。源码语义要求调用者观察到传入 output 的写入；在此接口里只返回另一个 tensor、而不写原 output，不能满足原有调用约定。此处 `empty_like` 每次源码调用创建 output；不能仅凭源码断言历史运行跨迭代是持久物理地址，或没有被图执行管理复用。
2. `attention/dsa_v41.py:716–717` 以同一 Python 形参 `output` 作为 `projected` 传入 `_project_output` 并返回；`attention/context_parallel/dsa_v41_cp.py:209–225` CP 路径对 attention 中间结果可做 token 补零、`restore_tp_heads`、必要的尾部补零，再将 `projected` 传给 `_forward_o_proj`。`dsa_cp.py:77–92` 中 TP size 大于 1 时有 view/permute/contiguous、分配 recv 和 all-to-all；size 为 1 则直接返回。不能把这些条件分支无条件映射到选中历史任务。
3. `dsa_cp.py:1657–1664,1754–1799` 基类 `_forward_o_proj` 的 `projected_output = _apply_wo_b(...)` 后，在 `output is not None` 时执行 `output[...] = projected_output` 并返回该 output。`_apply_wo_b` 根据 `full_weight` 调 `self.wo_b(...)` 或 `self.wo_b.quant_method.apply(...)`；`model.py:438–445` 将 wo_b 设置为 RowParallelLinear，`reduce_results` 由上游配置指定。必须确认实际 quant 方法、归约路径及最后有效 producer，才能界定可直接写回的位置。这里没有从源码或 trace 证明两个 storage 相同或不同，实际 W8A8 数学路径也尚未绑定到此任务。
4. `model.py:899–902` 中源码将 attention 返回值传入可选的 `sp_reduce_scatter`，然后交 `hc_post`；仅在关闭 sequence parallel 分支时源码上的直接下游才是 HC post。另一源码条件设置 `wo_b.reduce_results=False`（`model.py:833–836`）。这些是可能的数据流和通信约束，不是对历史运行时选择和对应设备消费 kernel 的确认。

## 可以提出什么候选、还缺什么

候选 A：让最终产出已完成所需归约且正确 dtype/layout/shape 的投影结果直接写 custom op 传入的 output，消除独立写回中间节点；候选 B：在满足同步与归约规则后，以 output 的 tile 为接口与后续 HC post 分块衔接。两者只是源码层面的结构方案：需要证明当前量化 kernel/collective 是否支持 out 参数或等价写入、TP/CP 通信与 SP 分支的顺序和完成事件、dtype/舍入、所有可能读写别名与图捕获生命周期、输出 buffer 初始化及全部可观察结果。若无法直接输出，则需保留写回；不能以 trace 邻接证明融合合法或估算收益。不能把 `[8184,5120]` 的逻辑读写字节直接当 HBM 流量。

最有信息量的后续证据：从已有历史 capture 查是否保存 output 与 `projected_output` 的 storage/设备地址、匹配 wo_b 算法实例与完整通信链以及 consumer 的流事件；若材料没有这些字段，则应标记历史记录无法判定，而非重复扫描或使用新 NPU。源内证据使用本任务冻结快照的路径与行号；历史 loaded-file 哈希仍未绑定。

## 已存 EP8 L0 样本：前后调用、设备事件与可证伪条件

本节把已有 [host 绑定](../diagnosis/TENSORMOVE-EP8-PROJECTION-HOST-BINDING.md) 与 [设备邻域](../diagnosis/TENSORMOVE-EP8-NEIGHBOR-DATAFLOW.md) 接回当前融合边界，不把已有证据重复计作新采集。只限 EP8 L0 该条样本：host `vllm::unquantized_gemm → aten::linear → aten::matmul → aclnnMatmul` 的包含关系和唯一 flow 对应设备 stream 47 的 MatMul task 43440；host 记录两输入 BF16 `[8184,1024]`、`[5120,1024]`。另一个唯一 flow 对应该流 copy task 43441；后续 host `vllm::ascend_sp_reduce_scatter_impl` 包含 `c10d::_reduce_scatter_base_`，其输出、输入形状分别为 BF16 `[1023,5120]`、`[8184,5120]`。因此对该样本可确认 host 经过名为 unquantized_gemm 的调用并看到 copy 与 reduce-scatter 的调用，不能由此推断 MatMul 内部累加精度、其他层精度或两条调用的 storage 恒等。

设备流 47 记录 MatMul 后 copy，再出现 `EVENT RECORD` 和 `EVENT WAIT`（分别为 task 43442、43443）；各自 host flow 可关联相应 `record_event`/`wait_event` 调用，后两者处于 reduce-scatter 相关 host scope。**未绑定的是具体事件对象配对、其他流的生产者/消费者，以及 MatMul 输出→copy 源、copy 目标→reduce-scatter 输入的 storage、offset、stride 与生命周期。** 不能将 RECORD 和 WAIT 因相邻就当成同一事件，也不能将 WAIT 的持续时间解释成通信或可隐藏时间。各 host 范围与设备时间不在本节相减。

对“让投影直接写 output 并取消 copy”的候选，可明确两个否定条件：

1. 若后续补证表明该次 copy 的目标和 reduce-scatter 输入不是同一有效视图，或有其他消费者依赖 copy 对不同 storage 所作的发布，则仅把 MatMul 目标改成前述 output 不能保留原数据流；须按真实消费者另行设计，不可宣称只删一个 copy 即等价。
2. 若新的投影最终写入与 reduce-scatter 在不同 stream，而其实际事件图允许后者在该写入完成前读取，则该重排不满足原值契约。最小反例是目标单元初值 0、投影最终写 1，通信先读得到 0；一个更早发布的无关事件不能证明该写入已完成。这是一般性的反例，不声称本样本发生了错误，也不要求每个 tile 一定各自新建事件。可行设计须证明 producer 的完成对对应 consumer 读取形成 happens-before，并保留 group 中各 rank 的 collective 次序与原数值要求。

这里尚不能证伪或证明候选在历史样本成立：已知 flow 只连接**每一次** host 提交到对应设备任务，不连接彼此的 tensor；即使看到相同指针也还需确认 allocation 代际、offset、stride 和生命周期。最小有价值证据是同一次 MatMul/copy/reduce-scatter 三方的张量视图与 buffer 身份，以及实际 event record/wait 对象、流和提交顺序；若既有 capture 不含这些字段应保留 UNKNOWN，不凭更多同形邻接记录补齐。既存邻域 JSON SHA-256 `22c3cf7e9248ca2f11da46a2aac8856ac66165dde36db2c09248b6f04d71b14b`，host 绑定审计 JSON SHA-256 `6876ce541f70451926744e9a6509c7bde1c2c9e436dc6d9cb693716c1f9da651`。仅使用既存资料，无 NPU 工作或实现修改。
