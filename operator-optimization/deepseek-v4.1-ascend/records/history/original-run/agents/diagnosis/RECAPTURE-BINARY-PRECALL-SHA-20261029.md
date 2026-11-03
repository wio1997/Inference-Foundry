# 截断的 binding trace：已安装对象的预调用哈希候选

2026-10-29 04:36:49 UTC 只读检查任务专属 `binding-diagnostic/runtime/metadata/binding-trace.jsonl`，16,776,606 字节，SHA-256 `34685e37bfb0eec10d23cb243cb3b2f4ad718642d10f88186c2fc43f4db959f5`。脚本 `diagnostic-20261029/binary-binding-draft/observe_installed_sha.py`；ZCode 原始结构化输出 `zcode/operator-ceiling169-v2-sha-check.zcode.json`，exit 0。未访问请求或生成文本，未启动设备负载。

- 日志中 16 个不同 PID 各一条 `load_data` 返回成功，其预调用哈希与已安装 `DispatchFFNCombine` 对象 `4cf1474cb1009b96fc743f5a396a9c07a6980146c41818b60cdbf0d3453a43a9` 相同。另两预设对象哈希在本截断日志无命中。
- 限制在同一 PID、同一 bin handle，记录顺序上后续未见 unload 或同 handle 新 load 的候选窗口：14 个 PID 共 160 条成功 `get_function_entry`，entry 均为 1000010；由其返回的 func handle 继续可候选关联 13 个 PID 的 162 条 launch_begin 和 160 条 launch_host_args。这些是独立事件计数，尚未逐次配对；`get_function_entry` 未给出函数名，不足以证明实际目标函数身份。
- 所有窗口持续至日志截断，不能证明 handle 的实际生命周期。此 trace 含一个 16 MiB `trace_cap` 标记；`dropped_since_last` 累计至少 36,312，且截断以后及未最终报告的丢失不在其中。缺失记录不等于未发生；多线程的事件记录顺序也不一定与调用实际完成顺序一致。
- 预调用哈希仅是传入数据的候选摘要，不证明运行时最终加载、执行了相同字节。尚未绑定到本次 profile 的目标层、具体任务或请求；无证据说明这些调用落在本次正式采样负载中。不能以此推断性能、唯一的算子实现或 16 卡完整覆盖。
