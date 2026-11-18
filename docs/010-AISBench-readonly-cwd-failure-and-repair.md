# AISBench只读工作目录失败与修复

G663仅无负载预检通过；随后G666错误地指定docker exec工作目录/task。CPU容器将/task根挂载为只读、client子目录挂载为可写，AISBench在相对路径tmp/创建临时配置时抛出Read-onlyfilesystem，监控子进程继续存活。进程存在不能代表负载在推进。

[原CLI错误](../evidence/client_frontend_G673/run/warmup1_32768_cli.log)与[收尾证据](../evidence/client_frontend_G673/G673_frontend_cleanup_2204_parsed.json)保留：P prompt/generation计数252967/1344、D254311/122052，与负载前完全相等，两端running/waiting0，没有details文件。这次启动没有发送模型请求，不能计作预热或性能结果。核对CPU容器归属和精确argv后，仅终止三个本任务前端进程，原host runner记录rc143/failed_requires_review。模型和CPU容器保持运行，原始失败未覆盖。

[G674新runner](../code/benchmark_pipeline_native_G674/run_aisbench_cold_hot.py)将cwd改到每个run的可写client挂载，增加RW挂载身份断言；数据、C16、1024输出、一次尝试、五个独立warmups、usage审计和首coldhot规则保持原协议。读取旧收尾证明并确认客户端已空闲后，在RW子目录创建/删除自身空目录的CPU探针通过，[新无负载预检](../evidence/client_frontend_G673/G674_benchmark_preflight_2204_parsed.json)通过。

随后新run保留旧失败；warmup1数据此前没有发送到服务，仍是未消费前缀。P初始编译快照534文件和[实际guard/八卡flock核验](../evidence/client_frontend_G673/G665_P_compile_baseline_2203_parsed.json)已保存。真实预热后仍需独立检查P编译缓存、完成其余预热、核验D编译/稳定性，才能放行首cold/hot。没有DS正式性能通过结论。

后续G674/G675真正完成首轮未计分预热：16/16成功，逐请求32768输入/1024输出，实际max concurrency16。父级重新解析原始details/JSON/CSV与usage契约，并核对P prompt/generation增加524272/16、D增加524288/16384，均匹配实际PD路径。[父级核验与原始SHA](../evidence/native_first_warmup_G676/G676_parent_first_warmup_protocol_audit.json)仅接受为unscored warmup，不计入正式矩阵。D编译缓存711文件前后无变化；P预热后快照534文件已保存，仍需比较完整元数据并完成后续独立预热，不以文件数相同宣称编译稳定。
