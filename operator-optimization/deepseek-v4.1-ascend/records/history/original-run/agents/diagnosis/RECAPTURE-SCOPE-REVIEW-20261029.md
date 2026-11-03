# 2026-10-29 重采：配置核验及 scope 归属状态

在离线导出进行中，对两个指定 rank 的 `profiler_info` 做了只读检查；该会话配置不是实际导出事件覆盖的证明。EP0 的目录是 `dp0_pp0_tp0_dcp0_ep0_rank0_736_20261029023525651_ascend_pt`，EP8 是 `dp1_pp0_tp0_dcp0_ep8_rank0_737_20261029023525648_ascend_pt`。两个 `profiler_info` 一致报告：

- `config.common_config.record_shapes=true`、`with_stack=true`、`with_modules=true`，CPU 与 NPU activity 同时启用；
- `config.experimental_config._profiler_level="Level1"`、`_aic_metrics="ACL_AICORE_PIPE_UTILIZATION"`、`_msprof_tx=true`、`_data_simplification=false`、`record_op_args=true`；
- `config.experimental_config._op_attr=false`。这与请求开启 op_attr 的意图不同，不能声称本次已开启。

检查时两者均已有 `kernel_details.csv` 和约 537 MB / 533 MB 的 `trace_view.json`，但当时离线导出尚在运行，文件存在不意味着导出结束。此后按主控要求暂停读取这些文件。先前试用 `ijson` 的 `traceEvents.item` 在两文件上报 `IncompleteJSONError`、0 条事件；随后检查表明这些文件采用顶层 JSON 数组，而不是 `{traceEvents:[...]}`。因此这次错误既不能证明 scope 不存在，也不能证明文件损坏或尚未写完。尚无可靠的 `wio_diag` 事件数量，也没有 scope 到设备任务的有效 flow 证据；禁止用 CPU scope 持续时间作为 NPU 模块耗时。

下一次只在 CPU 离线导出完全结束、文件稳定后，先对 EP0/EP8 使用支持顶层数组、Decimal 时间的本地 `diagnostic-20261029/summarize_profile_scopes_v2.py` 逐个流式读取，核对 scope 与 sidecar `call_id`、label、同线程关系、缺失和重叠；再对确实命中的 scope 选择有限样本，读取导出中可匹配的 flow start/end 和设备 task 标识，核验流方向、时间域、ID 唯一性与重放复用。如果没有完整且唯一的因果链，只报告 scope 存在及候选，不能推断模块 NPU 耗时、请求身份或计算瓶颈。v2 脚本同时识别 profiler_info 前导下划线键，保留精确字段路径；其尚未对新 trace 运行。

以上仅只读已有文件和配置，未发起 NPU 工作、未修改服务或用户模型；未检查未完成的其他 rank。

## 导出完成后：EP0 / EP8 与 EP8 层级闭合

主控确认 16/16 导出结束后，对 EP0 与 EP8 的 `trace_view.json` 做流式读取：分别有 792806、786948 个事件和各 144 个 `wio_diag:` 事件，读取前后文件大小及修改时间不变。`736-5` 与 `737-5` 的 `NPUModelRunner.execute_model` X scope 在各自文件均唯一；二者虽然各报告 3 请求、8180 调度 token，但摘要不同，不能当同一批请求对照。EP8 `737-5` 的 scope 时间是 trace 内的 `1793241326863420.820–1793241329118981.900 µs`；这是 CPU 包围范围，不是设备或端到端耗时。

EP8 目标 scope 的唯一 `torch_to_npu` start/finish 流对共 3978 个，finish 的 `(pid,tid,ts)` 均唯一匹配设备 X 事件，223 个导出名称。该范围的任务时间可以互相重叠，不能把它们的时长和解释为墙钟。`DispatchFFNCombine` 40 个的导出时长和约 1,141,418.534 µs，`SparseFlashMla` 40 个约 34,417.812 µs；`EVENT WAIT` 530 个约 2,581,901.558 µs，它们不是互斥的框架阶段。采集参数中的 stack 等会扰动运行，不能把这些值视为正式无 profiler 基线或实际速度上界。

进一步以 `DeepseekV41DecoderLayer.forward` scope 的 call ID、同 PID/TID、sidecar `layer_idx` 为锚，限定 scope 完全落于 `737-5` 内；每个 `torch_to_npu` start 均需位于对应层 CPU scope、flow 起止 ID 唯一、finish 精确对应设备 X，再按起点、model、task 和 physical stream 查 CSV。得到 40 个 layer 0–39 scope 各一次，并将 40 个 `DispatchFFNCombine` 提交事件各归入一层；无未归属或重叠歧义，40/40 CSV 唯一命中。trace 和 CSV 的 duration 差最大 0.00075 µs；CSV 为更低显示精度。设备任务可以在对应 CPU 层 scope 结束之后执行，所属层指的是**提交时的层**。

完整每层时长、cycles、MAC/vector 时间和报告比例保存于 `diagnostic-20261029/ep8-737-5-layer-mc2.json`（SHA-256 `afaeab84b04bc49ff371a00c61c07eaf4abb9ea0f7626b0f93bc48f818b2c795`）；只读查询脚本 `query_ep8_737_5_layer_mc2.py` SHA-256 `717dadefd18d8dcaceea5e659e2e8714b97e6000b01da81e6d2ccf7563994f83`，结果亦保存 trace、CSV、sidecar SHA。EP8 的 737-5 调度摘要是 `1b826883b782ba77814d5bd40fcd3e46c759e380069db2e066ff0e9cb4bce89d`，这一内容摘要不是已证明唯一的调度步 ID。记录中 CPU 核/向量 cycles 与比率是导出字段，不能直接当作整卡利用率或 MC2 等待原因；不能将已绑定提交的层当作真实 tensor 依赖、计算执行阶段或无 profiler 基线的瓶颈。

## 诊断 hook 是 profiler 对照中的额外混杂因素（只读核查）

对 L14/L39 的同次重采集，已存 sidecar 聚合 `diagnostic-20261029/sidecar-hierarchy-20261029.json`（SHA-256 `e4ffaf3cacaeebddb6beb249d9f48d76a3800dc5f02528cdc2ae46e6c3a8a023`）分别列出 EP8 `737-20`、`737-45` 两个 conditional_target 层 scope；与本报告所引用 L14/L39 scope 身份一致。按该聚合所定义的 CPU sidecar 墙钟区间，分别为 73.130740 ms 和 43.264720 ms。**它们是主机包围范围，不是同一时段内设备工作、单条 MC2 持续时间、hook 执行时间或 profiler 开销。**异步下发会使设备任务与主机调用范围不重合。

保存的 hook 源码 `diagnostic-20261029/instrumentation/sitecustomize.py`（SHA-256 `6e9bbadd7205a08f95de0c63a022645accbfe9fbae8a74e37d7859e3b4c8ef08`）的 active 分支会先发送 `call_begin`，进入 `torch.profiler.record_function`，调用原函数，再发送 `call_end` 并退出上下文；`_emit` 内在锁内生成时间戳和 JSON，执行文件 open/write/close。假设该保存源码确为历史加载版本，则 begin→end 时间包括 begin 时间戳后该条日志的后续序列化及写入、上下文进入和原函数调用等；不包括 end 时间戳之后该条日志的后续写入和上下文退出，且其间其他工作、等待及采集开销未拆分。不能将两层 sidecar 时间差归为 hook 成本、MC2 或框架调度开销。

另存聚合 `diagnostic-20261029/metadata-summary-early.json`（SHA-256 `2e322b98e2f972e35d74cb2d5fc98dd7071a0ff48f38f8b47a62d91a2b48538a`）列有 2757 条 begin、2757 条 end、16 条 profiler start 与 16 条 stop anchor，及 0 条 instrumentation_error；该数字仅是汇总覆盖，0 个已记录错误不证明没有扰动或信息丢失。历史代码加载身份、文件写行为耗时和范围未单独度量。当前无 profiler 基线与此诊断属于不同进程及请求，无法将差异归因于 profiler，且该诊断还同时包含额外 hook，不能忽略其潜在扰动。要区分影响，未来需在获准条件下保持负载/硬件/服务条件并验证取数覆盖，以无采集、仅 hook、仅 profiler、两者同时的对照来拆分交互；重复样本及专家路由等动态工作量仍需校验。此轮没有发起采集或设备任务。
