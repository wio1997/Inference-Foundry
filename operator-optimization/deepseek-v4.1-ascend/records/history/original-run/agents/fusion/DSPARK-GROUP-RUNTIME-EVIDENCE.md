# DSpark cache group 数量与共享缓冲：已有证据核对

**当前不能确定这次 DSpark 的 draft attention group 数量，也没有实测 tensor storage alias。** 因而上一份“多 group 时公共输出可能只需生成一次”的候选仍是条件分析；不能声称本基线确有重复或预计收益。

## 已找到的配置与日志

- `evidence/frozen-serve.sh:28` 显式请求 DSpark、`num_speculative_tokens=5`、`enforce_eager=true`；保存的 `benchmark-20261028/raw-results/service-log-tail.txt:24,100` 分别显示 API 非默认参数和 DP0 引擎配置中 method dspark、5 speculative tokens。日志记录的是启动和配置，不能直接证明每次运行的调用及所有对象状态。
- 同一保存日志 `:1445,1447` 中，DP0 与 DP1 的 engine 日志各有一条 `kv cache group sizes` 列表，均为 12 个 `128`。这只是日志报告的**engine cache group size 列表**；没有给出每个 group 的层名、DSpark draft 层映射或 `draft_attn_groups`。它不等于“DSpark 有 12 个 group”，也不能排除 1 个或其他数量。
- 启动命令没有显式 `dynamic_spec_config`，冻结 `ascend_config.py:1303–1314` 默认动态方法 None。若无运行时覆盖，dynamic DSpark confidence 路径关闭；本地日志没有相应 resolved object dump，因此它的实际启用状态仍未逐对象确认。

对同一份保存日志逐行查询的脚本 `inspect_dspark_group_evidence.py` 生成 `dspark-group-evidence.json`，列出文件 SHA、匹配行号及表结构。该文件是记录下来的 log tail，匹配不到 `draft_attn_groups` / `kv_cache_groups` / `dynamic_spec_config` 字样，仅说明**这份文件**中无匹配，不说明其它日志或运行进程不存在信息。请勿把该日志里 DP、TP 或 Engram group 的 size 当成 draft KV group 数量。

## 共享输出：源码传参关系与实际存储是两回事

冻结 `dspark_proposer.py:240–307` 中 `token_indices_to_sample` 在 for 循环外创建，同一个 `self.input_ids`、`self._context_positions_buffer`、`self.positions`、局部 `token_indices_to_sample` 被每个 group 的扩展 kernel 调用重复传入；这是**源码同一对象引用**的关系。`_per_group_context_slot_mapping_buffers[gid]`、`_per_group_query_slot_mapping_buffers[gid]` 则按 gid 传入，`_context_slot_mapping_buffers` 再按 layer 映射到其所属 gid（`:207–215,308–311`）。若多个 attention group 引用同一 gid，按代码索引它们引用同一 gid 对应 buffer；不同 gid 是否在历史运行中共享底层 storage、或公共输出是否与任何输入 alias，现有记录没有对象指针/存储 ID 证明。

已保存的 `evidence/trace-schema-inspection.json` 所列 `TASK`、`COMPUTE_TASK_INFO` 和 `PYTORCH_API` schema 具有时间、名称、部分 shape/connection/callchain 等字段，**这些列本身没有 storage 地址或 group-to-layer 映射字段**。因此该概要不能用于确认“这些 buffer 实际不 alias”或某条 kernel 记录属于某一 group。其他未检查的 profiler 表/完整调用栈和 unrecorded logs 不在本结论范围；即使在同一窗口看见多个 kernel 名称，也不能直接推导 group 数，尤其此前已发现 globalTaskId 会因 replay 复用。

## 若需要收窄

需要同一次冻结负载的运行时 `draft_attn_groups` 个数与每项 `kv_cache_group_id`、layer names 和 kernel block size，以及对应缓冲的 storage identity/offset/shape、时序和事件。现有保存记录不满足这些要求；只在确有可用且不会扰动服务的已有日志/trace 中检索。没有这样的数据就保留 UNKNOWN，无需新增 NPU 测试。
