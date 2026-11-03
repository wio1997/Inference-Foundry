# F04 → F20：FFN norm 双输出与 router 的跨边界消费者

本文审查的是 F04 的 norm 输出接入 F20 router 的跨边界消费者，不是 F20 内部 router GEMM → sqrtsoftplus/topk → route pack 的完整审计。

这是冻结源码级消费者审计，不证明历史调用实际进入哪条分支，也不计算性能收益。`model.py:904-920` 显式调用 `rms_norm_cast`，将 `x` 和 `x_fp32` 一起送入 MoE；函数 `844-852` 若设备复合接口不可用，先 `normalized = post_attention_layernorm(hidden_states)`，再返回 `(normalized, normalized.float())`。故在**此 fallback 分支**中，router 的 FP32 输入是在归一化结果 dtype 舍入之后再转的，不是可直接用未舍入 FP32 中间值替换；如果精度语义要求保持该路径，做投影融合要保留中间舍入的数值效果，不一定要求物化该 buffer。

另一个设备分支由 `device_op.py:65-73` 在 `HardwareCapability.RMS_NORM_CAST`、启用 custom op、`torch.ops._C_ascend.npu_rms_norm_cast` 存在时返回该复合调用的双输出。冻结 A3 profile 具备该 capability，旧 trace 有 `RmsNormCast` 名称，但这两项只支持分支有可能/采样中有同名工作；它们不能单独证明这次模型的每次实际对象走该分支、该算子的逐位语义与 fallback 等价、buffer 物化或 task 一一对应。准确的内核源码及运行对象/等价验证未取得时保持 UNKNOWN，不把源码 docstring 当数学证明。

`model.py:269-298` 在指定路径把 rounded `hidden_states` 交给 `self.experts`，将 `hidden_states_fp32` 交内部 router 或用于 `F.linear(router_input, self.gate.weight)` 生成 logits；sequence parallel 分支可能对两者分别 shard。即使将 norm、router 或量化重组，也不能只保留一个近似版本并忽略 expert 的输入、router 的不同精度或 SP 关系。`model.py:220-259` 的 hash layers 还提供 `tid2eid` 和 input_ids；`line 247–250` 明确 top-k 权重归一化之后再缩放 routed 输出；不能把 hash 当作永远无需 gate 权重或擅自交换归一化与缩放。tuple/shared 和 TP 归约条件在 `300-326`，是后续数据/通信语义；不会因为 router 被融合而消失。

已知源码条件候选是 **FFN norm → router 输入准备 → router 投影/选择** 的按行组合，同时仍产生或等价提供 expert 所需 rounded 结果。适配需补充原复合算子内部语义、真实 router 对象、top-k/hash 的实际消费、所有分片与并行流；否则不能声称既有融合不完整、必省读取、精度等价或瓶颈大小。
