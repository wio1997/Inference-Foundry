# GLM — HANDOFF

更新时间：2026-10-01，默认模型与节省额度策略更新。

- **状态**：规则/工具已建立，本地模拟交接与模拟CLI检查通过；[新对话启动/演练证据](docs/START_NEW_CHAT.md)。未访问现役服务器、修改推理代码或执行benchmark/profile，无后台优化实验。
- **执行默认**：单个Sol high主Agent；medium做明确日常工作，ultra临时解难题，Astra暂不安排。Zcode/DeepSeek CLI负责执行、服务/监控与大日志归约，subagent用[Job/Result协议](docs/ZCODE_PROTOCOL.md)交接；主Agent只收摘要/关键证据。详见[模型策略](docs/research/AGENT_MODEL_STRATEGY.md)；未切换当前会话模型或实际调用现场CLI。
- **Current**：尚无当前DP1/TP16/EP16、0.27组合的有效E2E基线；数值界unknown。
- **活动点**：[GLM-OPT-0001](records/points/GLM-OPT-0001/point.md)，characterization/PLANNED，尚无Run。恢复现场与Current，而非预定V2或固定PD比例。
- **已知**：两台相同910C；每台8卡16芯是上下文计划解释；GLM-5.2性能阶段/MTP/目标W8A8。实际身份见MISSION待恢复项。
- **历史入口**：[REUSE](REUSE.md)、根performance_knowledge/entries.jsonl。已有133条知识；root DeepSeek Current/PID仅历史。
- **旧GLM资料**：本地7月16/17日profile标GLM52、P1/prefill、TP8/DP2、约1536输入/max_tokens1；缺版本/量化及关联D元数据。可作线索，不能充当当前0.27基线，原trace未解析。
- **下一步**：在实际运行环境只读核验P/D/proxy与Git/安装、artifact、rank、effective config、负载及已有日志；查询相关历史，选最小缺失证据。缺机器连接/源码目录时明示缺项，保护活跃服务。

恢复时核对实际Git和进程，不机械重跑旧next_action。后续保持本页短，链接活动点、Current证据和关键未知；历史事实与Run存于索引。
