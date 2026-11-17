# G4 同一诊断窗口的 P/D 十六 rank 模型事件验收

G138 使用新的 32768→1024 输入做 C16、16 请求诊断，全部成功。它保留 G4 服务和四档正式结果，诊断负载不计分。P 先开始采集，D 等待真实 running 请求和 generation/spec-draft 计数增加后才开始；触发时 running=1，不声称 D 在记录窗口内为 batch16。

G148 选择每角色恰好八个 G138 新 capture，排除 G118 旧 capture；原始文件逐项 SHA 与冻结副本相同。SDK 在另一份副本上解析，原始二进制 trace 保留在服务器。全部 16 rank 均核对 namespace PID、物理 NPU 和正模型设备事件：P 每 rank 8200–8206 个，D 每 rank 7384 个。此前 G118 的 D 零模型事件失败仍在原记录中。

- [本次验收](../evidence/G4_profile_G138/acceptance.json)
- [P 八 rank](../evidence/G4_profile_G138/CPU_p/review/role.json)
- [D 八 rank](../evidence/G4_profile_G138/CPU_d/review/role.json)
- [P rank0 的算子与通信事件分析](../evidence/G4_profile_G138/CPU_p/review/rank_0.json)
- [D rank0 的算子与通信事件分析](../evidence/G4_profile_G138/CPU_d/review/rank_0.json)
- [采集控制和原始客户端审计](../evidence/G4_profile_G138/job/state.json)
- [原始失败和触发机制](011-vllm-G4-profiler-device-gate.md)

修正后设备窗口有效，但本轮没有重复性测量或新的收益判定。算子时长相加包含重叠，不能作为端到端墙钟时间份额；跨主机时钟偏移未校准，不能直接相减推断通信延迟。最终同机 SGLang 四档矩阵与匹配 profile 仍待完成。
