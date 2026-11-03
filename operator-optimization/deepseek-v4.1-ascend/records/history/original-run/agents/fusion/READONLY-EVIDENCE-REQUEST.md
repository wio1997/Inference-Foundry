# 动态量化、MoE 返回值和 tensor 身份：具体只读取证清单

## 已从本地快照确定的源码路径

1. F05 的具体包装已定位：V4.1 `attention/dsa_v41.py:440-495` 使用 `v1_impl.cv_wq_b.quantize(qr)` 和 `.matmul(...)`；`attention/context_parallel/dsa_cp.py:47,1466-1468` 导入并创建 `CVLinearWrapper(self.wq_b)`；真实模型的 `wq_b` 在 `models/deepseek_v41/model.py:407-415` 使用 ReplicatedLinear 或 ColumnParallelLinear，具体取决于 CP。需要快照 **`/vllm-workspace/vllm-ascend/vllm_ascend/ops/cv_linear.py`**，然后按其 imports 找对应的量化方案、线性方法（预计目录 `vllm_ascend/quantization/methods/`，先只读文件名和 import，不猜具体模块名）和 **`/vllm-workspace/vllm-ascend/vllm_ascend/ops/linear.py`**。
2. **注意区分旧版和 V4.1**：`attention/context_parallel/dsa_cp.py:1836-1853` 的另一段代码确实包含 `npu_rms_norm_dynamic_quant`，但 V4.1 的 CP 类 `attention/context_parallel/dsa_v41_cp.py:170-188` 继承 `AscendDSAV41Impl`；V4.1 自己的 `dsa_v41.py:352-384,440-495` 使用上述 wrapper。**不能看到旧版文件中有 fused norm+quant，就断言 V4.1 当前层执行了该 fused 路径。**是否运行多流分支仍待实际配置及 trace 绑定。
3. F23 的 `models/deepseek_v41/model.py:29,70,236,283-328` 导入 `vllm.model_executor.layers.fused_moe.FusedMoEFactory`，并只在相应 tuple/shared expert 条件下调用 `vllm_ascend.ops.triton.mul_add.muls_add_triton`。需要只读快照 **`/vllm-workspace/vllm-ascend/vllm_ascend/ops/triton/mul_add.py`**，以及从 **`/vllm-workspace/vllm/vllm/model_executor/layers/fused_moe/`** 的 `__init__.py` 导出开始，以 grep `FusedMoEFactory`、`MoERunner`、`return_shared`、`shared_output` 搜索少量明确文件，沿 factory、runner、forward 返回值路径取必要源码；目录存在性/实际导入可能随版本变更，先列文件，禁止凭路径存在性臆测执行分支。补齐有关 Ascend 适配类和 FusedMC2 的对应 wrapper 时同样先按 imports 定位。
4. 以上每文件请记录 frozen baseline 容器和已授权 CPU stage 的 SHA-256、镜像 ID、容器源码覆盖情况，再复制到任务目录 sources/；不同内容要分别保存并报告。读取和校验不运行被导入源码、不启动 NPU。

## 本地 trace 是否带 tensor/storage ID

本地只有现成的 schema 摘要 `evidence/trace-schema-inspection.json`、window CSV/JSON 和少量解析结果，没有本地 profiler SQLite 原始文件。摘要列出的 `COMPUTE_TASK_INFO` 有 input/output 的 shape、dtype、format 和 `attrInfo` 字符串引用；`PYTORCH_API` 有 input shapes/dtypes、callchain、connectionId；`TASK` 有 streamId、modelId、taskId 和 connectionId。**这些表结构没有显式的 tensor ID、storage ID 或地址字段。**但 `attrInfo`、`STRING_IDS` 值、原始 trace JSON 的其他事件内容没有在本次本地核验中全面展开，不能据此断言整个原始 trace 都不存在地址或 ID；connectionId 也不是 tensor identity。

## 对主控的精确只读请求

- 利用已有 169 任务原目录和已授权 CPU stage，只选一个已归属 layer0 内部窗口（同 rank/model/stream，参考 `captured-layer-attribution/module-call-metrics.json`）及其临近 `HcPre/RmsNorm/DynamicQuant/QuantBatchMatmulV3/DispatchFFNCombine/HcPost`：在原 profiler SQLite **以只读 URI 打开**，检查 `COMPUTE_TASK_INFO.attrInfo` 对应 `STRING_IDS.value`、`PYTORCH_API` 关联的 shape/dtype/connectionId 与 trace JSON 同一事件 args。限定返回最多 30 条、字符串截断 1024 字符；仅报告列名/是否存在非空 tensor/storage/address 字段及可追溯路径，不输出完整原始 trace 或用户数据。最好选一条已知可关联的 task，若未找到对应行记录 UNKNOWN，不得把缺失当作零。
- 若存在可复用的明确 tensor/storage ID，验证 F02/F05/F23 的同调用生产者和消费者。若只有 shape 与时间，不宣称数据依赖、物化或 cache identity；连接 ID 仅用于 API/task 对应。检查一个样本最多证明该样本的字段情况，不能推论全部 16 rank 或所有事件。
- 全部结果保存于本任务证据目录，主控负责同步，绝不操作 172 或启动设备负载。
