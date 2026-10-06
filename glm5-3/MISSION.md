# GLM Extreme — MISSION

checkpoint158之后的研究优先级以[AGENTS](AGENTS.md)为准：代码优化优先，Sol high承担性能架构师职责，旧HANDOFF/next_action仅作证据；不重写历史事实与裁决。

目标：功能完整的GLM专用框架；现有正确算子下，多机多卡、动态请求与可变PD的合法调度最优。一次超过Stock是中间结果，最终报告当前最佳、剩余空间、适用条件和无法解决的限制。

## 当前现场

- 用户确认两台同规格、卡配置相同的Ascend910C服务器；结合前文按每台8卡16芯理解，实际rank/实例映射待核验。
- 版本标识vLLM-Ascend0.27，PD已运行，参考DP1/TP16/EP16，GLM采用MTP。各端effective config未采集。
- 当前已有GLM-5.2权重，W8A8是目标口径，实际artifact/cache dtype、MTP深度、connector和Graph待恢复。目录glm5-3不构成5.3精度验收要求。
- 本阶段当前两台资源内收敛；未来1P3D等只是配置例子，不等于固定目标或物理机数。额外资源未具备的方向给当前结论。

## 性能与功能合同

完整接口与行为从实际框架恢复并持续补充；清单只用于覆盖，不限制能力。真实负载下比较有效吞吐、TTFT/TPOT/尾延迟、失败/拒绝、队列稳定性和资源成本。沿用既有SLO；无SLO时报告折中曲线。正式比较固定相关条件，代码收益必须matched A/B；配置、部署和代码收益分开记账，没有真实完整E2E不得提升Current。只有代码级KEEP算主要性能进展。冷启动、实验迭代成本与稳态服务性能分开。

授权改调度、执行组织、状态/buffer、Graph、通信发起/等待、部署和Runtime边界，允许成套重构。当前不开发新计算kernel/fusion，不自动切换算子实现或减少模型必要工作。

## 后续交付与路线选择

已有现场/Current以真实Git、HANDOFF和Run恢复，不从初始发布状态重做。恢复后检查Goal Review信号，Sol直接研究代码与决定性证据，选择最大可信可消除Gap；不预定V2、grammar/compatibility或配置扫描路线。允许停止高投入但无Product Gain方向，删除不必要通用Runtime层并重构调度/状态/执行顺序。

标准PD为`Request → P Prefill → KV Transfer → D Decode → Output`；各Replica完整执行Prefill+Decode称Full Replica / Complete-request Placement，历史调度是否进产品需新裁决。后续交付matched A/B、correctness、可重复完整E2E Gain和[代码性能账本](CODE_PERFORMANCE_LEDGER.md)，无新增KEEP明写“本阶段没有新增代码级性能 KEEP。”不重跑足够历史证据；必要连接入口未知则写缺项，不猜旧脚本/PID为现役。
