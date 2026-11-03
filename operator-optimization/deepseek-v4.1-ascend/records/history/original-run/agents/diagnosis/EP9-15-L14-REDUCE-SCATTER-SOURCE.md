# EP9–15：L14 窗口中 reduce-scatter 的主机来源和组元数据

沿用 `ep8-15-l14-l39-aicpu-overlap.json` 中 EP9–15 各卡第 14 层 MC2 窗口与其他流 `reduce_scatterAicpuKernel` 的精确匹配：EP9 有 2 条（task 3993、3999），EP10–15 各有 1 条（task 3999）。各卡独立以 device、PID、stream、task、startNs 查询原 DB、检查 endNs 和 connectionId，再查 `CONNECTION_IDS` 关系和同一 DB 中的 `PYTORCH_API`，随后在该 host 线程内找区间包含的调用范围，并用同卡 sidecar 的 call ID 精确核对层标记。

| 卡及 TASK | 匹配的 host API / sidecar 层 | 关联通信 op 导出名 | 组元数据类别 | 组内 rank |
|---|---|---|---|---:|
| EP9 / 3993 | `HcclReduceScatter` / 22 | `hcom_reduceScatter__097_206_1` | TP | 1 |
| EP9 / 3999 | `HcclReduceScatter` / 23 | `hcom_reduceScatter__097_209_1` | TP | 1 |
| EP10–15 / 各自 3999 | `HcclReduceScatter` / 23 | `hcom_reduceScatter__097_209_1` | TP | 2–7，依卡递增 |

八条中，每条所选 TASK 连接集合均有且仅有一条查到的 `PYTORCH_API` `HcclReduceScatter`；它的同线程嵌套路径含 `HcclReduceScatterBase`、`c10d::_reduce_scatter_base_`、`vllm::reduce_scatter`、`vllm::ascend_sp_reduce_scatter_impl` 和一个 decoder 层标记。每个层标记在同卡 sidecar 有一对 begin/end；EP9 的 task 3993 对应第 22 层，其余对应第 23 层。**层号是主机提交范围标记，不证明张量、模型实例、请求或因果依赖。**

每条记录的通信 op 来自与所选 TASK 不同、但同一个 host ID 关联的 connection；op 名称仅为导出字段，不证明这两个 connection 一定是同一物理事务。EP9 task 3999 的 TASK connection、EP11 task 3999 的 TASK connection 在 `CONNECTION_IDS` 中均关联两个不同 id；EP10、EP13 的通信 op 所在 connection 也关联两个不同 id。EP11 的关联集合还含一个 `aclrtRecordEvent`，因此不能把连接集合当作只含两个 launch 的唯一事务。查询列出了这些反向关联；这里“一个 host API”仅指在所查 `PYTORCH_API` 表中匹配到一个，不代表其他类型的 id 不存在或连接天然唯一。

所选通信 op 在每个 DB 中各自记录 `group_name_5` 和 `rankSize=8`。其 group 字符串与该库唯一的 `parallel_group_info` JSON 字典键精确匹配，解析未发现重复键；匹配条目按严格名称规则标为 TP，成员列表均含 8 个不重复整数，各卡 `group_rank` 对应成员索引与本卡 `RANK_DEVICE_MAP` 的唯一行一致，七卡记录的成员列表相同。EP8 对应记录已有同样类型的内部一致性核查，见 `EP8-L14-COMM-NOTIFY-SCOPE.md`。完整组成员只保存为本地 JSON。上述验证提高了对应组元数据的一致性，但采集器版本、通信组生命周期及实际运行时 communicator 与记录的契约尚未完全闭合；不能以此证明跨卡同一次 collective、相同请求、同一物理事务、张量依赖或 L14 变慢的原因。

复算：在任务根目录执行 `python3 agents/diagnosis/run_ep9_15_l14_rs_source.py`。wrapper 固定前一阶段结果和锚点文件摘要，查询核对 DB 上一阶段首次观察摘要、sidecar 冻结摘要，以及读取前后 DB 摘要不变且无 WAL；检查 stage 的身份和挂载，`mode=ro`、`query_only`、CPU/IO 低优先级。结果为 `ep9-15-l14-rs-source.json`，含逐卡原始字段、反向连接、所选主机范围及本地组元数据。不访问 NPU，不启动新负载，不修改服务或推理实现。
