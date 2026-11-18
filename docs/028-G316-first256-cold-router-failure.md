# 256K首次正式冷测失败与路由边界

G316完成五轮独立预热后，首次正式cold32于06:24:35UTC实际启动，06:57:41UTC结束。原始32行中18行成功且均为262144输入、1024输出；data_id18–31共14行返回Service Unavailable，记录0输入/0输出。AISBench进程退出0并不等于测试通过；外层原始明细验收拒绝失败请求，因此hot没有提交、原始first ledger和首次冷测保留，没有重发。

14次失败与06:30:32–33UTC原生路由日志的14次PD选卡错误相对应：No available decode workers (all circuits open or unhealthy)。之前P/D的/health存在超时，而已接收的真实请求继续生成并完成；采集时空闲workers重新报告healthy。固定镜像路由默认每60s检查一次、单次5s、连续3次失败标记unhealthy，实际启动参数未覆盖这些默认值。503的直接边界已定位为路由拒绝选择D；健康策略、生成健康检查排队和circuit具体作用仍需固定运行时源码进一步核对，不能仅凭/health超时断言设备或模型故障。

当前D411项编译缓存与五轮预热末态相同，独立P636项也与预热末态相同，新增/删除/大小/mtime变化为零。这些证据不能消除14次真实失败。AISBench成功子集输出吞吐9.5139 token/s是18成功请求口径，排除出完整32请求C16对比；没有有效cold/hot完整五指标结论，更没有收益判定。此前128K的D编译资格更正保持。

- [父级32行/503/缓存复核](../evidence/matched_SG_G316/256K/first_cold_failed_G541/G546_parent_256K_first_cold_failure.json)
- [首次失败记录与原始JSONL/SQLite](../records/runs.json)
- [实际外层退出与未提交hot](../evidence/matched_SG_G316/256K/first_cold_failed_G541/run/state.json)
- [实际原生路由日志](../evidence/matched_SG_G316/256K/first_cold_failed_G541/P/router.log)
- [实际路由命令](../evidence/matched_SG_G316/256K/first_cold_failed_G541/P/router_handle.json)
- [固定镜像默认健康策略源码](../evidence/matched_SG_G316/256K/first_cold_failed_G541/fixed_router_sources/sglang_router/router_args.py)
- [五轮独立预热与当前P/D证明](027-G316-256K-five-warmups-current-cache.md)

固定运行时HTTP源码进一步确认：`/health`创建1-token健康请求，在PD模式使用FAKE_BOOTSTRAP_HOST，并等待detokenizer/scheduler的last_receive_tstamp更新；路由只等待5s。原生`/ready`单独调用tokenizer_manager.is_ready报告接收就绪状态。该源码与原始日志给出健康探测与长请求无输出区间冲突的机制假设；更改探测端点后的长负载验证尚未完成。

- [实际固定镜像HTTP健康/ready源码](../evidence/matched_SG_G316/256K/first_cold_failed_G541/actual_fixed_health_sources/sglang/srt/entrypoints/http_server.py)
- [实际固定镜像tokenizer状态源码](../evidence/matched_SG_G316/256K/first_cold_failed_G541/actual_fixed_health_sources/sglang/srt/managers/tokenizer_manager.py)


G565于08:00:12UTC实际启动原生路由，唯一服务策略变化为health-check-endpoint=/ready；timeout生命周期包装另行有界。P/D容器和模型参数未改，保留circuit breaker、请求3600s超时与禁止重试。实际监听socket FD与固定launcher setproctitle身份共同核验，仅结束原来归属确定的18080路由。前两轮控制脚本失败原样保留：G553在变更前未匹配已改名进程，G560在精确旧路由终止后等待docker-exec退出确认存在异步时序，G565核验旧进程真正消失和端口可用才启动。08:01:39UTC新控制器存活、两worker healthy/load0，native ready均200。长负载恢复、正式冷热测量和性能收益仍未验证，首次256K失败没有覆盖或重发。

- [新路由实际就绪观察](../evidence/native_router_ready_G565/G566_native_ready_router_actual_2201_parsed.json)
- [有界路由控制器](../code/native_router_ready_G565/router_ready_controller_g565.py)
- [原身份匹配前置失败](../evidence/native_router_ready_G565/G555_native_ready_router_actual_2201_parsed.json)
- [旧进程异步退出确认失败](../evidence/native_router_ready_G565/G561_native_ready_router_actual_2201_parsed.json)
