# G4 profiling：P通过，D设备窗口未通过

四档正式冷热已完成，G118另用16个新前缀32768→1024输入做C16诊断，客户端16/16成功，两端stop API均200。它不进入正式性能结果。

SDK解析后的P全部8rank通过正模型算子与物理NPU映射验证；D虽然8份trace存在且CANN解析返回0，但8份op_statistic缺失，独立trace审计显示每rank仅6–10个正硬件事件、模型事件0。不得把文件存在或控制API成功称为完整P/D profile通过。原始raw SHA、SDK来源、namespace PID/NPU身份、失败原始输出及P的实际算子统计均保留。

- [P八rank严格验收](../evidence/G4_profile_G118/CPU_p/review/role.json)
- [P rank0算子、通信和within-trace时间分析](../evidence/G4_profile_G118/CPU_p/review/rank_0.json)
- [D八rank设备模型事件审计](../evidence/G4_profile_G118/CPU_d/D_trace_scope_audit.json)
- [D原始失败](../evidence/G4_profile_G118/CPU_d/review/role.json)

固定Ascend worker在model_runner.execute_model之前无条件调用profiler.step，等待KV的调度迭代可能先消耗8步记录窗口。G138保留相同G4容器和服务配置，使用与全部既有输入前缀不同的新C16诊断：P先start，启动负载后轮询D实际running和generation/spec-draft计数，只有已经有模型进展时才start D；这仍需新的正模型设备事件验收来证明修正有效。不得仅按推测认定根因已完全验证。

- [原生worker机制](../code/G4_profile_native/worker.py)
- [原生profiler计步与自动stop状态复位](../code/G4_profile_native/wrapper.py)
- [G138实际生成计数触发的采集器](../code/vllm_G4_profile_g138_package/collect_profile.py)

原始大型NPU二进制trace留在任务服务器，仓库保存其SHA清单与脱敏定位指针。算子duration相加存在重叠，不是端到端walltime份额；跨主机绝对时间尚未做clock offset校准，不能据此相减定位通信延迟。完整框架性能地图仍需同机SGLang矩阵与匹配设备profile。
