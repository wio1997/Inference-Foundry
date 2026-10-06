# GLM Extreme

为GLM构建完整功能的专用推理框架，研究现有算子下多机多卡、动态负载及可变PD的调度极限。当前用GLM-5.2、两台910C；版本/资源口径见MISSION。

**从checkpoint158之后以[AGENTS](AGENTS.md)恢复研究规则，再读HANDOFF+当前活动点/Git恢复事实。** HANDOFF/历史Run/Summary/next_action仅作证据。Sol high是性能架构师，Astra按需独立Challenger，Zcode / DeepSeek是执行层；恢复后先检查Goal Review，不机械继承旧路线。

| 入口 | 按需用途 |
|---|---|
| [MISSION](MISSION.md) / [PLAN](PLAN.md) | 目标/合同、极限标准与逼近逻辑 |
| [RECORDING](RECORDING.md) / [优化点索引](records/points.jsonl) | matched A/B、完整E2E、Git提交与不可改写的Run证据/裁决 |
| [代码性能账本](CODE_PERFORMANCE_LEDGER.md) | 代码候选Baseline/Patched/Gain/Correctness/Verdict，代码KEEP进展 |
| [REUSE](REUSE.md) / [复用索引](records/reuse.jsonl) | 旧成果适用性、失败原因与去重 |
| [研究索引](docs/README.md) | GLM特性、源码、prof、历史迁移与模型成本 |
| [新对话启动](docs/START_NEW_CHAT.md) | 短交接指令、模拟演练与实际环境接入 |

交给执行Agent的短prompt：

```text
在glm5-3/按当前AGENTS.md接续checkpoint158之后的研究，HANDOFF和next_action仅作证据。Sol high作为性能架构师先做Goal Review，亲读必要源码/diff/profile/raw，选择最大可信代码Gap；Astra按需独立挑战，Zcode只做执行。代码优化经Correctness、matched A/B与真实完整E2E重复裁决，配置/部署/代码分账，无完整E2E不提升Current。维护代码性能账本，无新增时写“本阶段没有新增代码级性能 KEEP。”保留所有历史Run/raw/裁决，当前两台可行域内收敛。
```

当前事实由HANDOFF与points.jsonl恢复（checkpoint158活动点为GLM-OPT-0002，Current=None）。本次规则更新没有新增代码级性能KEEP或GPU/NPU实验。标准PD包含P Prefill→KV Transfer→D Decode；各副本完整Prefill+Decode必须标Full Replica / Complete-request Placement。
