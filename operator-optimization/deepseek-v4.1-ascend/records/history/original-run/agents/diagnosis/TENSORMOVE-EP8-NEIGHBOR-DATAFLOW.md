# EP8 单样本：投影、复制和 reduce-scatter 的邻接证据

## 范围和方法

只读已有 EP8 层 0 trace，设备流 47 的 task 43441，即已在 [TENSORMOVE-EP8-IDENTITY.md](TENSORMOVE-EP8-IDENTITY.md) 中绑定的 `aclnnInplaceCopy_TensorMoveAiCore_TensorMove`。这是一条样本，不代表其他层、其他请求或整个运行。trace SHA-256 为 `53ec20cf6d8b8f79b67a7f2f4ddaabf823b4c0305e88e2dbf9cf62d02e7c6f23`，脚本先逐字节校验。

在设备起点前后各 500 µs 内截取该设备流的 X 事件，在对应 host 提交时间前后各 1500 µs 内截取对应线程的 X 事件；对选中设备事件的 `torch_to_npu` flow，统计整个 trace 内每个 id 的 s、f 数量。窗口只限制事件**起点**，不能据此推断窗口内其他流没有任务。大时间戳用十进制定点计算。记录 [tensormove-ep8-neighbors.json](tensormove-ep8-neighbors.json) 含内部路径与时间，不直接公开。

## 已观察到的序列

| 设备事件 | 起点相对 copy，µs | 时长，µs | 主机侧绑定 |
|---|---:|---:|---|
| task 43440 `aclnnMatmul_MatMulV3Common_MatMulV3` | -320.933 | 320.8935 | flow 的 s、f 全 trace 各唯一；s 精确匹配 host `aclnnMatmul` 起点，调用栈包含 `dsa_cp.py:1663 _apply_wo_b`、`dsa_cp.py:1795 _forward_o_proj` |
| task 43441 `aclnnInplaceCopy_TensorMoveAiCore_TensorMove` | 0 | 37.0995 | flow 唯一，s 精确匹配 host `aclnnInplaceCopy` 起点，在 `aten::copy_` 中，形状 BF16 `[8184,5120]` |
| task 43442 `EVENT RECORD` | +37.120 | 0.020 | flow 唯一，host s 匹配 `record_event` 起点，位于 `c10d::_reduce_scatter_base_`、`vllm::reduce_scatter` 和 `vllm::ascend_sp_reduce_scatter_impl` CPU scope |
| task 43443 `EVENT WAIT` | +37.160 | 493.130 | flow 唯一，host s 匹配 `wait_event` 起点，位于 `vllm::reduce_scatter` 和 `vllm::ascend_sp_reduce_scatter_impl` CPU scope |

同流上 Matmul 结束与 copy 开始的间隔为 0.0395 µs。`vllm::ascend_sp_reduce_scatter_impl` 的 host 输入记录为 BF16 `[8184,5120]`。这些 flow 分别将设备事件和 host 提交点关联，**不将前后操作的 tensor/storage 身份绑定起来**。CPU scope 包围关系也不是设备耗时；不能把主机与设备相差约 1.3 秒的时间戳当成排队时长或调度延迟。

冻结的 [dsa_cp.py](../../sources/vllm-workspace/vllm-ascend/vllm_ascend/attention/context_parallel/dsa_cp.py) 第 1663、1795–1799 行分别调用 `self.wo_b`、以 `_apply_wo_b` 结果写 `output[...]`；[model.py](../../sources/vllm-workspace/vllm-ascend/vllm_ascend/models/deepseek_v41/model.py) 第 760–763、899–902 行描述 attention 输出 buffer 和 `self_attn` 之后按条件进行 `sp_reduce_scatter`。这些源码加上 host 栈、形状与同流邻接，支持“wo_b matmul → copy → 条件 reduce-scatter”的**候选数据路径**。源码快照 SHA-256 分别为 `7e8dd93eeda988866138342523ecfa7cc9c42ce67669d4b1507ee8c5f3452305` 和 `101f96b38e9fc47e7be147ddfe56976aee896867c73a33a7d95c3e3c833a6e91`，但历史运行时载入文件的 hash 尚未绑定。

## 未确认内容与后续证据

- trace 这些事件没有提供能连接 matmul 输出、copy 两端及 reduce-scatter 输入的 tensor/storage 指针。相同形状、同流顺序和源码路径不能代替张量身份和别名证据，也不能证明可直接写入 buffer 且保持正确性。需要该次调用中三者指针、offset、stride、dtype、生命周期及允许原地写入的接口语义。
- `EVENT WAIT` 的配对 record / 事件 ID 与另一流执行情况未绑定，493.130 µs 是该条 trace 中的事件区间，不能直接当作通信时间、整个芯片空闲、可隐藏时间或用户侧收益。需要事件关联、另一流窗口、采集范围完整性和事件计时语义。
- 两个 flow 的唯一性只证明各自 host 到设备事件的关联，不证明 matmul 输出就是 copy 的源，也不证明 copy 目标就是 reduce-scatter 的输入。融合候选仍依赖以上约束、算子输出 buffer 接口与消费者时序的核实。

复算脚本 [tensormove_ep8_neighbors.py](tensormove_ep8_neighbors.py) SHA-256 `16bf9a7ecb8bb4f9b786137ca234fa26abb5de9691f216aefab9e05ec179aa1c`；输出 SHA-256 `22c3cf7e9248ca2f11da46a2aac8856ac66165dde36db2c09248b6f04d71b14b`。仅使用任务容器内已有 trace，无新 NPU 负载、无推理实现修改。
