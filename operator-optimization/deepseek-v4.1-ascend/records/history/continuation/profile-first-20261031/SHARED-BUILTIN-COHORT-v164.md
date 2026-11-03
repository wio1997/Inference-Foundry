# 171 共享完整负载采集：已准备，尚未运行

目标与资料均在 `/data/tiankuan/wio`。原主服务保持运行、启动时间未变。诊断容器 `20b61d30c3aa03932bd094ca500c09e572630f113f0e7c226b328e5d86d47915` 已创建，状态 CREATED/PID0；没有新模型或设备执行，没有授权文件和提交状态文件。

批准后只运行一次控制流程：原服务先做完整 C8 参照（2条前缀预热＋8条并发正式请求），参照有效且无其他请求/NPU进程时停止原服务；启动隔离诊断容器，加载相同模型和TP8/DP2/EP16、DSpark5及原graph配置；采集每worker一次capture_model期间CPU原始记录，以及正式完整C8的两个短窗口（每个最多3个worker execute_model调用）；结束后停止诊断容器并恢复原服务、核对health200。两个正式pass总计16条请求、4条前缀预热。

每个正式请求必须32768输入/1024输出、temperature0/ignore_eos；每pass统计要求prefix queries262144/hits235904、generation8192、success8、零抢占。不满足则该次负载无效，不自动重跑。正式窗口仍标initial和later_decode_candidate，不能用客户端可见首字代替设备phase证据。

采集只启用已准备的builtin字段及Python图/guard/scheduler标记；关闭native interposer及TorchDispatchMode。startup为CPU-only profiler，formal为CPU+NPU；record_shapes/profile_memory/with_stack均开启，关闭自动离线解析。原1352/全reader/数值与物理上限等缺口不会因此自动闭合。

执行阶段45分钟超时，之后恢复服务另有40分钟上限。目录16GiB是轮询停止阈值，可能有短暂超出，不称硬总额上限；单文件256MiB为硬限制，Docker日志另有30MiB轮转限制。发现其他NPU进程时不抢占、不kill；恢复池仍被占用则记录延迟恢复与实际PID。

21项新CPU合同检查通过，覆盖完整client算法不变、字段工厂、精确授权、占用拒绝、失败时恢复及不重提交。它们不证明真实Torch/CANN/编译兼容性。服务器15份执行输入pins、语法和默认describe已核对；默认命令不运行采集。

服务器目录：`/data/tiankuan/wio/migration-171-20261102/builtin-cohort-v164-228ad25fcf9b`。计划SHA256 `c07ab99fce77d7dd17f85836fd8c417cf1cdd3e791ef9c6fabdd2fd6768906f2`，准备凭据SHA256 `762883ae518373c59700270f6393bdace81151acffd98a1ab07be95eed1af285`。审批模板为PENDING，不能作为授权。

整体优化Goal保持active/未完成；本阶段是共享证据采集，并不缩减全部模块、分支及跨算子/层/卡/迭代的范围。
