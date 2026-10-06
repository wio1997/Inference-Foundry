# 新对话启动与模拟交接

工作目录使用真正的Inference-Foundry checkout。2026-10-06从checkpoint158（父commit `e378d459e08e9f12300118e158bba16db6c4d04f`）之后采用新研究规则：GPT-6.1 Sol high是性能架构师，Astra按需独立Challenger，Zcode / DeepSeek是执行层。文档不会替用户设置模型。历史Run、raw evidence、裁决和本页已完成演练保持原样，不复制旧聊天或通读研究档案。

## 1. 按需模拟，零模型/服务器调用

已有有效演练无需重跑；仅在新执行环境或协议改变影响交接时按需验证，不以模拟阻塞源码研究。在仓库根执行，输出目录必须是新目录，建议放仓库外：

```text
python3 glm5-3/scripts/simulate_zcode_handoff.py --output-dir /absolute/new/simulation-directory
```

它创建合成输入和mock CLI，调用正式桥接器，实际走Job → CLI输出落盘 → Result → 校验 → 主Agent摘要。产物全部标simulation=true；model_calls=0、run_id=null，不访问网络、DeepSeek、NPU或现役服务。

预期结果是handoff=VALID、task_status=needs_decision、Current=unknown。模拟证明协议传递、原始输出隔离和缺项识别；不证明CLI后端可用、服务就绪或推理性能。模拟原始日志保留在指定目录，不提交Git。

## 2. 可粘贴的新对话启动指令

```text
接续 https://github.com/wio1997/Inference-Foundry 的glm5-3任务，研究分支glm5-3-autonomous-20261001；核对包含checkpoint158之后研究规则commit的Git状态。
先读根AGENTS、glm5-3/AGENTS、HANDOFF和当前优化点（由points.jsonl定位；checkpoint158为GLM-OPT-0002），按需读PLAN、RECORDING、CODE_PERFORMANCE_LEDGER和模型策略，不复制全部历史。
GPT-6.1 Sol high作为性能架构师亲自负责目标/最大Gap/因果/架构/关键源码/复杂Runtime与Scheduler重构/实验设计/matched A/B/KEEP、REJECT、Current裁决。
HANDOFF、历史Run、Zcode Summary、point下一问题和上一轮next_action只是证据；允许推翻假设、停止无Product Gain路线、删除不必要Runtime层和改变执行组织，不继承V2或既定路线的优先权。
恢复后先执行Goal Review：最终产品、时间去向、最大可由代码消除的Gap、最近实验是否缩小Gap、今天重启是否还选这条路线。checkpoint158无性能KEEP，不能机械接续grammar/compatibility或参数扫描。
出现模型策略所列架构分叉、持续无代码KEEP、跨模块根因不闭环、高成本重构、Current停滞或偏离最大Gap时触发独立Astra Challenger Review；直接读关键源码/diff/profile/trace/matched Run/raw evidence，不只复述HANDOFF。Astra不常驻、不做日常Run或每轮审批，Sol以真实代码和E2E最终裁决。
Zcode/DeepSeek只负责服务器、服务启动/停止/恢复、benchmark、monitor、profile采集、日志归约、身份核验和边界明确的机械修改，不决定长期研究主线。按Job/Result协议只读核验现场/现有controller；保护活跃服务，未知入口明示。涉及Gap、根因、架构、关键代码、KEEP，Sol必须亲读必要源码/关键diff/profile/决定性raw，不能只依赖Summary。
优先Observe→最大可信Gap→具体代码路径→代码假说→Patch→Correctness→Matched A/B→真实完整E2E→KEEP/REJECT。配置只服务baseline/代码假说/可运行性/资源辨因，配置/部署/代码收益分账；标准PD与Full Replica / Complete-request Placement严格区分。
无matched A/B不算性能成果，无真实完整E2E不得提升Current；账本只以代码级KEEP计主要进展。重要checkpoint报新增代码KEEP、E2E Gain、最大剩余Gap、下一最高价值代码问题；没有新增时写“本阶段没有新增代码级性能 KEEP。”
已有历史证据/模拟足够不重跑。保留全部历史Run/raw/裁决，PID/hash/epoch/controller/ownership记录只做到支撑真实性、安全、可复现所需；不实现复杂自动模型路由。
```

若新对话暂时只做模拟，在指令末尾加：“本轮仅模拟交接，实际环境恢复留到我提供现场入口后。”

## 3. 现场接入最少信息

能从新对话所在机器发现的不用重复填写；不能发现的补以下位置即可，凭据用既有SSH/本机配置，不贴聊天或Git：

- 真实仓库目录及vLLM/vLLM-Ascend源码或安装位置；目录未知可由现场只读查找。
- 服务器连接方式/已连接host，以及当前允许操作的实例/设备范围。
- Zcode CLI可执行路径或可用PATH，实际后端model id/版本按现场核验。
- 当前P/D/proxy启动入口与已有controller身份；未知可提供配置/日志位置，先只读恢复，不能按旧PID盲重启。

需要现场恢复时，第一个真实Job只读核验现状，不预设重启。已有证据直接复用；Sol先做Goal Review并直接研究必要源码/关键diff/profile/raw，再判断缺失证据与下一代码问题。协议VALID、功能成功或Summary均不能替代matched A/B与完整E2E，也不能提升Current。

## 4. 接续规则

Job≠性能Run，VALID≠性能KEEP，running≠completed；next_check_at不是已建立的定时任务。subagent按[同一协议](ZCODE_PROTOCOL.md)交接，不默认复制整个主会话；共享现场仍由唯一controller排程。

实际Git、HANDOFF、points.jsonl及真实Run用于恢复事实，服务器state.json用于核验执行状态；研究路线以当前AGENTS和最大可信代码Gap选择，不受历史next_action绑定。checkpoint158是已有事实索引，不能以本页早期演练PLANNED状态恢复；模拟/功能测试不填入性能基线，不继承旧DeepSeek Current或7月GLM profile。连续无代码KEEP等Goal Review信号出现即暂停惯性Run，否定的低价值路线立即停止。

## 5. 已完成的本地演练

2026-10-01执行`simulate_zcode_handoff.py`：Job/CLI/Result联检通过，合成CLI原始输出288058字节留文件，父Agent交接返回1624字节。任务正确返回needs_decision、Current=unknown、run_id=null；真实模型调用为0，未访问现役服务器。该字节数只描述本次合成输入，不代表真实负载或额度节省比例。

桥接源码SHA256：`783f73371b0810492d41ad52c61375cb980aaafcd25afc6a02a4c8692786a94a`；模拟器SHA256：`5ad6d894289ea1f306115ec72593fb9d6eaad9c19307b053bab11c0e18414ff1`。本地保留`work/handoff-dry-run-20261001/simulation-report.json`及其引用的Job/Result/bridge和合成日志，原始产物不发布到Git。源码/现场改变影响接口时再重验，无需重放本轮对话。
