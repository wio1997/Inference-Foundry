# 新对话启动与模拟交接

工作目录使用真正的Inference-Foundry checkout。2026-10-06从checkpoint158（父commit `e378d459e08e9f12300118e158bba16db6c4d04f`）之后采用新研究规则：GPT-6.1 Sol high是性能架构师，Astra按需独立Challenger，Zcode / DeepSeek是执行层。文档不会替用户设置模型。历史Run、raw evidence、裁决和本页已完成演练保持原样，不复制旧聊天或通读研究档案。

当前规则版本为`GLM-RESEARCH-RULES-v2`，基于`df7399c28791a72241f47b5c7474b6e9ac46cdb9`增量完善。版本核验与Performance Research Reset是性能Run入口门槛；本页历史演练不替代它们。

## 0. 先核验branch / HEAD / rule version

实际研究分支是`glm5-3-autonomous-20261001`。在真实checkout先只读核验Git，必要时fetch远端最新研究分支，再定位最后修改AGENTS的规则commit：

```sh
git status --short --branch
git fetch origin glm5-3-autonomous-20261001
git branch --show-current
git rev-parse HEAD
glm_rule_commit=$(git log -1 --format=%H origin/glm5-3-autonomous-20261001 -- glm5-3/AGENTS.md)
git show "$glm_rule_commit:glm5-3/AGENTS.md"
git merge-base --is-ancestor "$glm_rule_commit" HEAD
git diff --exit-code "$glm_rule_commit" -- glm5-3/AGENTS.md
```

确认branch为实际研究分支、HEAD包含最新规则commit、工作区AGENTS内容与最新规则一致；rule version从该AGENTS读取，而不是只记住v1 SHA或一直固定为v2。出现分支/HEAD/版本不匹配，先保护dirty工作并恢复正确研究分支和规则版本，禁止在旧main/旧AGENTS研究；无法确认最新版本则先解决入口缺项，不开始性能Run。不要盲reset、force覆盖并发工作。

首次恢复摘要必须报告 **branch / HEAD / rule version与rule commit / active optimization point**（由points.jsonl定位）。这是一次必要入口核验，不是持续PID/hash/Git巡检主线。

## 首次实际研究输出：Performance Research Reset

新Sol会话或强制Goal Review后，**第一项GPU/NPU性能Run前必须输出**以下紧凑Reset，字段内容对应[AGENTS的十项要求](../AGENTS.md#0-启动身份与performance-research-reset)，记录在当前点的新恢复/评审记录中，不另建流程系统：

```text
Performance Research Reset
Rule identity: branch / HEAD / rule version / rule commit / active optimization point
最终产品: 一句话
当前产品Current: 最可信完整E2E；产品代码commit / 功能合同 / 标准workload / E2E指标 / 剩余Gap；未知明写
当前 active performance KEEP stack: CURRENT_PERFORMANCE_STACK引用，空则明确无active PERF_KEEP
最大代码 Gap: 仅一个主问题
决定性证据: 3～5条关键源码/profile/trace/matched Run/raw locator
源码入口: 文件 / 类 / 函数
候选 patch: 改什么代码；哪部分工作不是模型必要工作
预计消除的工作: 减少 / 删除 / 重叠什么，新增成本是什么
最小 matched A/B: 最小对照、correctness、噪声/重复或预定义稳定服务窗口验收
Astra: YES / NO，触发信号或无需理由；YES时引用紧凑Review Package
```

Gap证据不足时先写unknown/缺项，**仅允许一个最小补证据diagnostic/profile动作**，事先写Hypothesis、Distinguishing evidence和结果A/B的Decision table；随后更新Reset。不允许直接大规模参数扫描、多个正式E2E或并行新方向。默认1个active performance hypothesis + 1个区分该假说的diagnostic；切换方向必须先关闭/park/显式停止旧方向。大型正式E2E等待诊断支持、correctness通过、patch值得产品裁决。

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
先fetch核验实际研究branch、HEAD包含最新rule commit、工作区AGENTS为最新rule version（本版GLM-RESEARCH-RULES-v2，不固定沿用旧SHA）；不匹配先恢复，禁止旧main/旧AGENTS研究。首次摘要报告branch/HEAD/rule version与rule commit/active optimization point。
先读根AGENTS、glm5-3/AGENTS、HANDOFF和当前优化点（由points.jsonl定位；checkpoint158为GLM-OPT-0002），按需读PLAN、RECORDING、CODE_PERFORMANCE_LEDGER和模型策略，不复制全部历史。
GPT-6.1 Sol high作为性能架构师亲自负责目标/最大Gap/因果/架构/关键源码/复杂Runtime与Scheduler重构/实验设计/matched A/B/KEEP、REJECT、Current裁决。
HANDOFF、历史Run、Zcode Summary、point下一问题和上一轮next_action只是证据；允许推翻假设、停止无Product Gain路线、删除不必要Runtime层和改变执行组织，不继承V2或既定路线的优先权。
恢复后先执行Goal Review：最终产品、时间去向、最大可由代码消除的Gap、最近实验是否缩小Gap、今天重启是否还选这条路线。checkpoint158无性能KEEP，不能机械接续grammar/compatibility或参数扫描。
全新会话或强制Goal Review后，第一项GPU/NPU性能Run前必须输出本页Performance Research Reset，最大代码Gap只留一个；证据不足只做一个最小补证据动作再更新Reset，不扫描或批量正式E2E。
出现模型策略所列架构分叉、持续无代码KEEP、跨模块根因不闭环、高成本重构、Current停滞或偏离最大Gap时触发独立Astra Challenger Review；直接读关键源码/diff/profile/trace/matched Run/raw evidence，不只复述HANDOFF。Astra不常驻、不做日常Run或每轮审批，Sol以真实代码和E2E最终裁决。
Zcode/DeepSeek只负责服务器、服务启动/停止/恢复、benchmark、monitor、profile采集、日志归约、身份核验和边界明确的机械修改，不决定长期研究主线。按Job/Result协议只读核验现场/现有controller；保护活跃服务，未知入口明示。涉及Gap、根因、架构、关键代码、KEEP，Sol必须亲读必要源码/关键diff/profile/决定性raw，不能只依赖Summary。
优先Observe→最大可信Gap→具体代码路径→代码假说→Patch→Correctness→Matched A/B→真实完整E2E→KEEP/REJECT。配置只服务baseline/代码假说/可运行性/资源辨因，配置/部署/代码收益分账；标准PD与Full Replica / Complete-request Placement严格区分。
无matched A/B不算性能成果，无真实完整E2E不得提升Current；账本只以代码级KEEP计主要进展。重要checkpoint报新增代码KEEP、E2E Gain、最大剩余Gap、下一最高价值代码问题；没有新增时写“本阶段没有新增代码级性能 KEEP。”
默认1个active性能假说+1个区分diagnostic，每个真实性能Run前记录Hypothesis/Distinguishing evidence/Decision table；A/B都导向同一决定则不执行。大型正式E2E只在诊断支持、correctness通过、patch值得产品裁决后执行。
按Type分类；只有PERFORMANCE+matched A/B+correctness+噪声感知重复完整E2E Product Gain得到PERF_KEEP。维护当前active stack，确认旧stack+新patch整体成立，累计收益重新测量而不相加；旧patch失效/冲突/被覆盖/回归在新状态中标superseded/regressed，不改原裁决。
功能与性能双轨，非阻塞correctness问题进backlog；模块新增先说明替代谁、为何不能现有实现、REJECT如何退休。按AGENTS的5有效Run/3candidate/3INVALID等阈值强制Goal Review，可更早触发；Current绑定产品代码/stack/功能合同/workload/E2E/Gap，按阶段完成条件交付。
已有历史证据/模拟足够不重跑。保留全部历史Run/raw/裁决，PID/hash/epoch/controller/ownership记录只做到支撑真实性、安全、可复现所需；不实现复杂自动模型路由。
```

若新对话暂时只做模拟，在指令末尾加：“本轮仅模拟交接，实际环境恢复留到我提供现场入口后。”

## 3. 现场接入最少信息

能从新对话所在机器发现的不用重复填写；不能发现的补以下位置即可，凭据用既有SSH/本机配置，不贴聊天或Git：

- 真实仓库目录及vLLM/vLLM-Ascend源码或安装位置；目录未知可由现场只读查找。
- 服务器连接方式/已连接host，以及当前允许操作的实例/设备范围。
- Zcode CLI可执行路径或可用PATH，实际后端model id/版本按现场核验。
- 当前P/D/proxy启动入口与已有controller身份；未知可提供配置/日志位置，先只读恢复，不能按旧PID盲重启。

需要现场恢复时，第一个真实Job只读核验现状，不预设重启。已有证据直接复用；Sol核验规则身份、亲读必要源码/关键diff/profile/raw并输出Reset，再选择一个代码假说或最小补证据动作。协议VALID、功能成功或Summary均不能替代matched A/B与完整E2E，也不能提升Current。只做规则维护的会话不启动服务或性能Run，也不假装已完成现场Reset/架构Review。

## 4. 接续规则

Job≠性能Run，VALID≠性能KEEP，running≠completed；next_check_at不是已建立的定时任务。subagent按[同一协议](ZCODE_PROTOCOL.md)交接，不默认复制整个主会话；共享现场仍由唯一controller排程。

实际Git、HANDOFF、points.jsonl及真实Run用于恢复事实，服务器state.json用于核验执行状态；研究路线以当前AGENTS和最大可信代码Gap选择，不受历史next_action绑定。checkpoint158是已有事实索引，不能以本页早期演练PLANNED状态恢复；模拟/功能测试不填入性能基线，不继承旧DeepSeek Current或7月GLM profile。连续无代码KEEP等Goal Review信号出现即暂停惯性Run，否定的低价值路线立即停止。

## 5. 已完成的本地演练

2026-10-01执行`simulate_zcode_handoff.py`：Job/CLI/Result联检通过，合成CLI原始输出288058字节留文件，父Agent交接返回1624字节。任务正确返回needs_decision、Current=unknown、run_id=null；真实模型调用为0，未访问现役服务器。该字节数只描述本次合成输入，不代表真实负载或额度节省比例。

桥接源码SHA256：`783f73371b0810492d41ad52c61375cb980aaafcd25afc6a02a4c8692786a94a`；模拟器SHA256：`5ad6d894289ea1f306115ec72593fb9d6eaad9c19307b053bab11c0e18414ff1`。本地保留`work/handoff-dry-run-20261001/simulation-report.json`及其引用的Job/Result/bridge和合成日志，原始产物不发布到Git。源码/现场改变影响接口时再重验，无需重放本轮对话。
