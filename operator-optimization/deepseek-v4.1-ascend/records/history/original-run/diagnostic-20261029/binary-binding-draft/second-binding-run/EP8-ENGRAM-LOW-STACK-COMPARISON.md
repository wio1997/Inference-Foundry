# EP8 旧 profiler 与 Engram copy：可比较到哪里

找到一次更早的 EP8 采集，配置 `with_stack=false`、`record_shapes=false`。在其已导出 DB 中定向查询，观察到 14 条 `Slice_Tiling`，时间范围 **2.81–13.02 微秒**。其中 8 条各被同线程的一个 `aclnnInplaceCopy` 和一个 `Dequeue@aclnnInplaceCopy` 严格包含；这些 `Slice_Tiling` 为 **5.09–13.02 微秒，中位数 7.575 微秒**。另 6 条未满足本查询的唯一父 copy 条件，不能据此解释为不存在父调用或相同操作。

这 8 条在已导出 DB 的 queue correlation 各关联唯一 enqueue：4 条的上层名称为 `aten::to → aten::_to_copy → aten::copy_`，4 条为 `aten::contiguous → aten::clone → aten::copy_`，enqueue 时间早于相应 dequeue。**旧记录没有这 8 条的 shape、dtype 或调用栈，无法证明它们就是新采样中的 Engram gate 或 gathered copy，更无法建立逐次配对。** 即使名称链相似，也不能把 5–13 微秒与新采样的 20.756–21.785 毫秒直接相除，或把差异归因于 profiler 开销。没有可比较的线上算子耗时或收益结论。

| 条件 | 旧 EP8 | 新 EP8 |
| --- | --- | --- |
| 记录线程与位置 | rank 8、PID 727；整个已导出 DB 中满足查询条件的 8 条 | rank 8、PID 746；两次第 14 层候选队列等待区间中的 12 条 |
| CPU/NPU、级别 | 两者均记录；Level1 / PipeUtilization | 两者均记录；Level1 / PipeUtilization |
| PyTorch stack / shape | 两项均 false；所查上层行的栈、shape、dtype 均为空 | 两项均 true；12 条有 Engram 栈和形状 |
| 其他采集配置 | data_simplification=true、record_op_args=false、msprof_tx=false；启动配置 max_iterations=8、ignore_frontend=true | data_simplification=false、record_op_args=true、msprof_tx=true；max_iterations=12、ignore_frontend=false；另加载诊断 sidecar 和 C interposer |
| 负载 | 保存的旧采集恢复审计记录 8 请求，输入 262144、输出 8192、prefix hits 235904；原客户端有 `KeyError('data_id')`，随后从已保存结果恢复了匹配核验 | 客户端完成，记录同样的 8 请求、输入/输出总量和 prefix hits；不证明每个 copy 的 token 数、请求或缓冲一致 |
| 来源版本 | 保存的旧只读 inventory 与启动记录报告相同镜像；已冻结 `common.py` 和 `embedding.py` 哈希分别为 `4797b8…eb58d`、`a7741e…c4c9`，与当前同镜像源码一致；历史实际加载实例并未独立逐次验证 | 本次容器与 stage 的镜像 SHA、只读 rootfs、无源码目录挂载及文件完整哈希见 `engram-source-readonly-evidence.json` |

两边的时长均是各自 DB 内同线程 start/end 的差值，不需要假设两个会话的绝对时间同步；若要逐次跨会话比较则仍缺乏事件、输入、时钟和生命周期的严格映射。旧采样并非未插桩线上基准，其 profiler 扰动未知；新版除 stack/shape 外还有多项配置差异。DB 行是否漏失、同名 ID 在记录外是否复用、设备/请求身份、capture 迭代覆盖都未证明。新版离线导出还出现 SQLite I/O error。旧 DB 的总行数不是每请求频次，也不能外推 16 卡或完整线上负载。

## 复算

`python3 run_inspect_low_stack_ep8_copy.py` 在本人 stage 容器中以低 CPU 优先级只读固定旧 EP8 DB；输入 DB SHA-256 `67d14f7f386658f8c8c5d3c9059dff9b0f1decb200eba621a3432fcf1e23de58`，`profiler_info_8.json` 已核验 SHA-256 `a249b01a52dc6d9cbbd34953f2573b52d1d9f7280ff2d561b228b4a9417eb544`。输出 `low-stack-ep8-copy-audit.json` SHA-256 `8ff2294f51b8171cac59f1361c00bc21bb9dca905c9c4346a67bd3c23eff258b`，列出嵌套及 correlation 记录。新版 12 条依据 `EP8-ENGRAM-COPY-QUEUE-PROVENANCE.md` 和其可复算输出。未进行新 NPU 采集或修改目标代码。
