# 新对话最小恢复

当前目标为**GLM-5.3 W8A8标准P/D分离**，权重`/data/tiankuan/wio/GLM-5.3-w8a8`，新服务模型名`glm-53`。用户2026-10-06正在上传；完成并核验config/index/tokenizer/分片前不加载或测试。旧版本的模型结构、fit、功能及性能结论不自动成为5.3证据；历史Run保留实际身份。

启动只读下列身份核验、恢复顺序和Reset模板；其余链接按当前动作展开，历史演练不是必读。读过的内容不重复加载，不为缩短上下文省略决定性源码/raw。

## 0. 先核验branch / HEAD / rule version

在真实checkout执行一次；规则版本从AGENTS读，rule commit取入口及其权威细则的最新修改，不固定旧SHA：

```sh
git status --short --branch
git fetch origin glm5-3-autonomous-20261001
git branch --show-current
git rev-parse HEAD
glm_rule_commit=$(git log -1 --format=%H origin/glm5-3-autonomous-20261001 -- glm5-3/AGENTS.md glm5-3/PLAN.md glm5-3/RECORDING.md glm5-3/docs/START_NEW_CHAT.md glm5-3/docs/research/AGENT_MODEL_STRATEGY.md)
git merge-base --is-ancestor "$glm_rule_commit" HEAD
git diff --exit-code "$glm_rule_commit" -- glm5-3/AGENTS.md glm5-3/PLAN.md glm5-3/RECORDING.md glm5-3/docs/START_NEW_CHAT.md glm5-3/docs/research/AGENT_MODEL_STRATEGY.md
```

实际研究branch须为`glm5-3-autonomous-20261001`；身份不匹配先保护dirty工作并恢复正确规则，不盲reset/force覆盖并发工作。无法确认最新规则先解决缺项，不进入性能Run。

## 恢复顺序

1. 核验branch / HEAD / rule（上节）。
2. 读[AGENTS](../AGENTS.md)短规则。
3. 先查[ENVIRONMENT_RECOVERY](../ENVIRONMENT_RECOVERY.yaml)所需的project/local_access/deployment字段，再查[RECOVERY_INDEX](../RECOVERY_INDEX.md)目标路径行；其他节及verify/discovery命令仅在相关缺项或漂移时读取，不全量展开或逐项执行。
4. 读[HANDOFF](../HANDOFF.md)首段、`当前现场`和`下一研究问题`；详细Run/功能证明按证据链接再读，旧建议不绑定路线。
5. 查[CURRENT_PRODUCT_MAP](../CURRENT_PRODUCT_MAP.md)的Current/路径缩写、现役入口与目标组件行；只沿相关真实import/argv深入。
6. 读[CURRENT_PERFORMANCE_STACK](../CURRENT_PERFORMANCE_STACK.md)当前状态/active表，及[ledger候选表](../CODE_PERFORMANCE_LEDGER.md#后续工作与代码候选)的相关条目；分类和判据在准备裁决时读。
7. 从[points.jsonl](../records/points.jsonl)仅提取id/status/path/latest_run/current，用HANDOFF确定本次active point。多个ACTIVE不等于并行假说；不展开runs数组，point与HANDOFF重复的段落不重读。
8. 按导航只读核验相关现场动态状态与所有权，未知写unknown；不重启、不运行会写状态的observer、不发生成请求。
9. 输出下方Performance Research Reset，转入一个最大可信代码Gap的源码/证据分析；细则按AGENTS的动作表读取，实验前满足对应门槛。

启动不默认读PLAN、RECORDING、模型策略全文、研究档案、协议演练、全部runtime或全部Run；需要据以裁决的具体源码/diff/profile/raw仍必须亲读。

## 首次实际研究输出：Performance Research Reset

新会话或强制Goal Review后、首个GPU/NPU性能Run前输出一次，存当前点的新恢复/评审记录；不重写历史。模板是字段定义的唯一入口：

```text
Performance Research Reset
身份: branch / HEAD / rule version / rule commit / active point
产品与Current: 目标一句话；可信完整E2E及产品代码/功能合同/workload/指标/剩余Gap引用，未知明写
Active KEEP stack: CURRENT_PERFORMANCE_STACK引用
最大代码Gap: 一个主问题
决定性证据: 3～5条源码/profile/trace/matched Run/raw locator
代码假说与patch: 文件/类/函数；非模型必要工作；改什么；减少/删除/重叠什么及新增成本
最小matched A/B: correctness、比较条件、噪声/重复或稳定窗口验收
Astra: YES/NO及触发信号或无需理由；YES引用Review Package
```

Gap缺证据先写unknown，只允许一个最小diagnostic/profile补证据后更新Reset；具体[Run决策价值](../RECORDING.md#入口reset与run决策价值)在执行前读取。只做规则维护时不启动实验、不假称已完成现场Reset。

## 可粘贴的新对话指令

```text
接续 https://github.com/wio1997/Inference-Foundry 的 glm5-3，研究分支 glm5-3-autonomous-20261001。目标是GLM-5.3 W8A8标准P/D分离优化，权重/data/tiankuan/wio/GLM-5.3-w8a8，新服务模型名glm-53；先确认上传完成和artifact完整，不回退旧权重。按 glm5-3/docs/START_NEW_CHAT.md 最小恢复顺序执行，遵循 AGENTS；细则按需读，不继承旧 next_action。完成只读恢复与 Performance Research Reset 后，直接研究一个最大可信代码 Gap。
```

协议/执行环境变化需要模拟时，查[按需模拟](ZCODE_PROTOCOL.md#6-按需模拟零模型服务器调用)；已有证据足够不重跑。

## 5. 已完成的本地演练

2026-10-01执行`simulate_zcode_handoff.py`：Job/CLI/Result联检通过，合成CLI原始输出288058字节留文件，父Agent交接返回1624字节。任务正确返回needs_decision、Current=unknown、run_id=null；真实模型调用为0，未访问现役服务器。该字节数只描述本次合成输入，不代表真实负载或额度节省比例。

桥接源码SHA256：`783f73371b0810492d41ad52c61375cb980aaafcd25afc6a02a4c8692786a94a`；模拟器SHA256：`5ad6d894289ea1f306115ec72593fb9d6eaad9c19307b053bab11c0e18414ff1`。本地保留`work/handoff-dry-run-20261001/simulation-report.json`及其引用的Job/Result/bridge和合成日志，原始产物不发布到Git。源码/现场改变影响接口时再重验，无需重放本轮对话。
