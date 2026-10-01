# 新对话启动与模拟交接

工作目录使用真正的Inference-Foundry checkout；本次本地outputs/github只是发布包，不是完整推理源码。无需复制旧聊天或通读研究档案。后续主Agent默认6.1 Sol high，文档不会替用户设置模型。

## 1. 先模拟，零模型/服务器调用

在仓库根执行，输出目录必须是新目录，建议放仓库外：

```text
python3 glm5-3/scripts/simulate_zcode_handoff.py --output-dir /absolute/new/simulation-directory
```

它创建合成输入和mock CLI，调用正式桥接器，实际走Job → CLI输出落盘 → Result → 校验 → 主Agent摘要。产物全部标simulation=true；model_calls=0、run_id=null，不访问网络、DeepSeek、NPU或现役服务。

预期结果是handoff=VALID、task_status=needs_decision、Current=unknown。模拟证明协议传递、原始输出隔离和缺项识别；不证明CLI后端可用、服务就绪或推理性能。模拟原始日志保留在指定目录，不提交Git。

## 2. 可粘贴的新对话启动指令

```text
接续 https://github.com/wio1997/Inference-Foundry 的glm5-3任务。
在真实仓库checkout恢复Git状态，先读根AGENTS、glm5-3/AGENTS、HANDOFF和GLM-OPT-0001；资料按需读，不复制全部历史。
查看HANDOFF与START_NEW_CHAT.md的模拟检查；已有证据足够则不重跑。新执行环境或协议改变时按需重验，模拟不创建性能Run、不启动实际服务。
随后让Zcode/DeepSeek按Job/Result协议只读恢复实际安装、P/D/proxy、rank、有效配置、已有controller和证据。缺关键连接入口时集中说明；能从现场发现的自行核验。保护活跃PD服务。
主Agent用Sol high处理关键研究/复杂代码/裁决，明确日常工作可用medium；ultra临时解难题，Astra暂不安排。服务启动/等待/监控和大量日志归约交Zcode及现场脚本，只返回紧凑结论与可定位证据。
恢复真实Current，按机制查历史，只补缺失证据；按RECORDING记录优化点、代码和Run，再自主选择当前两台可行域内的高价值优化。不因缺数值界停在写文档。
```

若新对话暂时只做模拟，在指令末尾加：“本轮仅模拟交接，实际环境恢复留到我提供现场入口后。”

## 3. 现场接入最少信息

能从新对话所在机器发现的不用重复填写；不能发现的补以下位置即可，凭据用既有SSH/本机配置，不贴聊天或Git：

- 真实仓库目录及vLLM/vLLM-Ascend源码或安装位置；目录未知可由现场只读查找。
- 服务器连接方式/已连接host，以及当前允许操作的实例/设备范围。
- Zcode CLI可执行路径或可用PATH，实际后端model id/版本按现场核验。
- 当前P/D/proxy启动入口与已有controller身份；未知可提供配置/日志位置，先只读恢复，不能按旧PID盲重启。

第一个真实Job是只读现场恢复，不预设重启或重写路线。足够已有证据直接复用；不足部分再设计最小真实测量。主Agent收到Result后核对执行身份、关键unknown和历史适用条件，决定下一点；不是收到一句“成功”就提升Current。

## 4. 接续规则

Job≠性能Run，VALID≠性能KEEP，running≠completed；next_check_at不是已建立的定时任务。subagent按[同一协议](ZCODE_PROTOCOL.md)交接，不默认复制整个主会话；共享现场仍由唯一controller排程。

以实际Git、HANDOFF和真实Run为准。模拟与测试不填入GLM性能基线，不继承旧DeepSeek Current或7月GLM profile。2026-10-01已自主恢复现役服务并形成真实Run；当前点/执行以HANDOFF、points.jsonl和服务器state.json为准，不继承本段历史演练时的PLANNED状态。

## 5. 已完成的本地演练

2026-10-01执行`simulate_zcode_handoff.py`：Job/CLI/Result联检通过，合成CLI原始输出288058字节留文件，父Agent交接返回1624字节。任务正确返回needs_decision、Current=unknown、run_id=null；真实模型调用为0，未访问现役服务器。该字节数只描述本次合成输入，不代表真实负载或额度节省比例。

桥接源码SHA256：`783f73371b0810492d41ad52c61375cb980aaafcd25afc6a02a4c8692786a94a`；模拟器SHA256：`5ad6d894289ea1f306115ec72593fb9d6eaad9c19307b053bab11c0e18414ff1`。本地保留`work/handoff-dry-run-20261001/simulation-report.json`及其引用的Job/Result/bridge和合成日志，原始产物不发布到Git。源码/现场改变影响接口时再重验，无需重放本轮对话。
