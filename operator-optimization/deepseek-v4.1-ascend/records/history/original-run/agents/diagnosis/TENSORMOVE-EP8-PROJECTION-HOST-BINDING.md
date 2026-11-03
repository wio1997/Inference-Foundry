# EP8 单样本输出投影：host 调用与设备任务的绑定

## 结论及范围

已提取且校验过的单个 EP8 L0 trace 窗口，可以把同一 host 线程内的一次 `vllm::unquantized_gemm` → `aten::linear` → `aten::matmul` → `aclnnMatmul` 的调用，与设备流 47、task 43440 的 `aclnnMatmul_MatMulV3Common_MatMulV3` 关联起来。依据是：每一级 host 事件完整时间区间逐层包含下一层，PID/TID 相同；`aclnnMatmul` 的起点与全 trace 唯一的 `torch_to_npu` flow 的 s 端严格一致，flow 的 f 端匹配设备事件的 PID/TID/起点。host 调用栈同时包含 `dsa_cp.py:1663 _apply_wo_b` 和 `:1795 _forward_o_proj`。这些证据是该条样本的 host 调用关联，不能自动推广到其他 layer、请求或所有投影。

host `vllm::unquantized_gemm` 和 `aten::linear` 均记录两个 BF16 输入，shape 分别 `[8184,1024]`、`[5120,1024]`；其内 `aten::matmul` 的两个输入为 BF16 `[8184,1024]`、`[1024,5120]`。所以该次 **host 路径确实经过名为 unquantized_gemm 的自定义调用，输入记录为 BF16**，无需仅靠权重描述条件推断。这里的 BF16 是 profiler 记录的 host API tensor 类型，不是设备底层指令、内部数据格式或累加精度的证明，也不是整模型全为 BF16 的结论。

以下时刻均是同一个 host 线程内、相对 `aclnnMatmul` host 起点的微秒；这些是 host scope 而非设备执行时间：

| 调用 | 开始 | 结束 |
|---|---:|---:|
| `vllm::unquantized_gemm` | -1091.850 | +96.650 |
| `aten::linear` | -1014.010 | +87.470 |
| `aten::matmul` | -789.030 | +85.610 |
| `aclnnMatmul` | 0 | +78.490 |
| `aten::copy_` | +111.190 | +245.830 |
| `vllm::ascend_sp_reduce_scatter_impl` | +285.330 | +1482.180 |

接下来的 `aten::copy_` host scope 完整包含 `aclnnInplaceCopy`，后者通过另一个全 trace 唯一 flow 关联到设备 task 43441。`vllm::ascend_sp_reduce_scatter_impl` scope 完整包含 `c10d::_reduce_scatter_base_`，其 host 输入形状分别为输出 `[1023,5120]`、输入 `[8184,5120]`，BF16；第一维的比例恰为 8，不能仅根据这一比例认定全局 TP 或其他并行组配置。

## 分支条件与边界

这条 host 栈包含 `ops/linear.py:391 forward` 和 vLLM 基类 `linear.py:1773 forward`。如果运行时文件确与现存冻结源码对应，则冻结 `ops/linear.py:391` 是 `custom_op is None` 时的基类回退路径，冻结 `dsa_cp.py:1663` 对应 `full_weight=False` 分支；后续 `model.py:901` 的 SP reduce-scatter 对应 `use_sequence_parallel=True`。上述变量的历史运行值和实际加载文件哈希尚无直接绑定，因此这些是**源码一致条件下的分支推断**，不是实际配置独立实测。运行时 OTP、TP 数值、参数对象身份和完整量化 mapper 仍未知；同名函数和行号提供强关联，但不足以证明二进制与冻结源码完全相同。

关联信息没有 tensor/storage 指针和 stride、offset、生命周期，不能证明 Matmul 输出是 copy 的源，也不能证明 copy 目标与 reduce-scatter 输入是同一 storage；只凭时间和 shape 不可得出融合正确或可节省端到端时间。设备端 matmul/copy 的流顺序与其他限制见 [邻接分析](TENSORMOVE-EP8-NEIGHBOR-DATAFLOW.md)。`EVENT WAIT` 不能直接解释为通信时间或芯片空闲。

## 复现与校验范围

运行 `python3 agents/diagnosis/audit_tensormove_projection_host.py`。脚本核对输入 JSON SHA-256、选中两条 flow 的全 trace s/f 唯一计数、端点和线程、host 完整区间包含、调用栈及 copy 与 reduce-scatter 的 host 区间关系，并输出 [结果](tensormove-projection-host-audit.json)。输入 [邻域提取](tensormove-ep8-neighbors.json) 之前由 [提取脚本](tensormove_ep8_neighbors.py) 校验 trace SHA-256 `53ec20cf6d8b8f79b67a7f2f4ddaabf823b4c0305e88e2dbf9cf62d02e7c6f23` 生成；本脚本不重新读取原始 trace，flow 的全局唯一性依赖该提取器。host 同名事件唯一性只在提取的相应 host 起点窗口内检查，不能声明全 trace 所有同名调用唯一。

复算脚本 SHA-256 `c2bea28c8a9cde20777120fdeae847bf52615b228c2c2b537c2cad22d3cc5ed6`；结果 SHA-256 `6876ce541f70451926744e9a6509c7bde1c2c9e436dc6d9cb693716c1f9da651`。仅对已有数据作离线分析，没有启动 NPU 负载，也没有更改推理实现。
