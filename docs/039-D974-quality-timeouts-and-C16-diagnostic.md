# D974完整质量失败：16个实际600秒超时，转向并发语义诊断

分页C1/C2的[组件、完整图和短设备路径](038-C1-native-allrank-gate-and-D974.md)已经通过限定门槛，但完整质量988再次失败。20:39:40UTC[精确停止本任务客户端内Python](../evidence/quality988_terminal_G1000/G996_SG974_FAILED_QUALITY_STOP_JSON_parsed.json)并确认实际退出，终态20条完成记录、2条严格正确、16条传输失败、36份请求、4份响应。这不是1319题准确率，没有通过完整质量，也没有端到端收益结论。

## 原始失败与排空

[20行原始结果](../evidence/quality988_terminal_G1000/raw/sglang974_full_test/results.jsonl)中16个失败均为TimeoutError，实际HTTP耗时600.098–600.119秒；不是CLI观察超时，也没有重发或放宽预算。四个完整响应对应index3/18/22/23，生成52/94/89/57 tokens；index18/22数学末值为7但缺少####，按原官方严格grader仍为invalid。其余请求的完整内容未返回，不能推测它们的数学结论。

[全部49份原始文件和SHA索引](../evidence/quality988_terminal_G1000/G997_failed_quality988_drain_2204_parsed.json)保留请求、响应、日志、源码及停止回执。原worker state仍可能保留full_test_running，实际PID退出及parent terminal assessment为终态依据。停止后CPU客户端仅sleep，P/D各两次running/queue计数为零；未采集独立transfer gauge，因此不扩大为所有传输状态已验证。模型和canonical guard保留，没有因失败重启服务。

## 重新选择问题

[只读日志与冻结D0窗口成本](../evidence/quality988_terminal_G1000/G993_readonly_system_cost_2204_parsed.json)显示质量负载期间D运行16请求、transfer和queue为零、图执行开启，报告的生成吞吐约24–43 token/s，同时驻留token数持续增长。这是服务日志口径，不是AISBench，不能与已归档native性能矩阵直接相除。

原生分页打分源码的循环终点已经是当前设备可见长度n_vis，而非maxcapacity；因此[否决重复的容量循环优化](../evidence/quality988_terminal_G1000/G995_system_problem_selection.json)。在此前18-input/B1/two-step冻结窗口，rank0按名称聚合的dispatch、sparse attention、paged FP4 duration sums约0.602/0.454/0.020秒；它们含重叠，尚无关键路径和C16归一化，不能把这些和当墙钟，或直接推算收益上限。

[P只读日志](../evidence/quality988_terminal_G1000/G993_readonly_system_cost_2203_parsed.json)还包含反复连接D侧随机端口失败。端口属于哪个旧/当前epoch、是否影响当前数据及延迟尚未确定；不能仅凭日志归因或重启P/D。

## 下一项有界诊断

确认失败进程已停止、SDK无活动、两端就绪及队列为零，并[重新核实guard980实际FD](../evidence/quality988_terminal_G1000/G998_GUARD_ACTUAL_JSON_parsed.json)后，启动[C16流式语义诊断999](../evidence/quality988_terminal_G1000/G999_C16_DIAGNOSTIC_ACTUAL_START_JSON_parsed.json)：16条简单算术/唯一code绑定问题，随后原GSM前16题的流式前缀。前一组max128、后一组max256，C16/T0/seed0/thinking=false/零重试，保留逐事件原始SSE和实时文本；外层硬边界795秒。

缩短输出预算仅用于观察串题、异常生成或不终止前缀，不能替代原1319题/max2048质量门槛，不能计算新的完整准确率。该诊断不采新Profile、不跑AISBench、不改镜像/权重/服务配置。根据结果再判断是否需要隔离target-only与DSPARK；target-only若被采用也只是诊断配置，最终框架对齐仍须恢复并验证所要求的优化组合。

SG完整质量、四档首cold/replay、最终匹配P8/D8 profile、归一化差异地图及端到端收益继续未完成。

[后续C16语义故障及target-only隔离](040-C16-semantic-faults-and-targetonly-isolation.md)：32个流式请求已完成，错误数值/截断/重复生成已复现；D974已退休，D1004仅诊断启动。
