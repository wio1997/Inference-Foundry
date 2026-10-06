> 历史迁移审计，非当前模型分工或优先级规则。checkpoint158之后采用[AGENTS](../../AGENTS.md)的Sol high性能架构师 / Astra独立Challenger / Zcode执行层与代码优化判据。以下旧Astra owner、GPT-6 Sol、优先主线、next_action是历史描述，不约束后续；来源审计和历史事实保留，SOURCE_AUDIT.json也仅作历史来源记录。

> 研究快照：整理于2026-10-01，随后作为按需资料发布；文中未提交/未运行陈述描述研究采集阶段，不代表后续Git发布状态。实际运行状态以局部HANDOFF和Git为准。

# 原仓库Markdown职责与GLM迁移清单

检查日期：2026年10月1日。已通过直接HTTP核验GitHub main仍为commit `064f4474b1eafd5985469bad1f0bab3233a8f417`，并检查根目录12份Markdown的职责、结构、最新追加记录和相关引用。本次没有修改原仓库、访问模型运行机器或提交GitHub。

原仓库已经有完整的“规则、目标、方法、状态、调度界、证据、Runtime设计与执行”体系。GLM建议沿用同名文件职责，保留Agent自主性；共享方法与历史证据通过明确引用复用，GLM的运行身份、当前结论和状态独立维护。

用户明确DeepSeek使用DSpark、GLM使用MTP。文档职责与研究方法可沿用；历史调度机制需按GLM条件核验，DSpark执行状态机不能整体迁移。旧proposer、辅助hidden输入、7-token组织、state/count mirror、Graph序列、cycle定义及性能换算保留为DSpark历史合同；GLM的MTP输入/输出、候选消费、提交/位置/KV推进、结束补位、PD与Graph依赖按当前实现重建。迁移索引标注通用方法、待适配机制、DSpark专属实现及GLM验证结果，不把旧KEEP当GLM当前成果。

## 1 根目录12份文档

| 原文件 | 主要职责 | GLM处理方式 |
| --- | --- | --- |
| [AGENTS.md](https://github.com/wio1997/Inference-Foundry/blob/064f4474b1eafd5985469bad1f0bab3233a8f417/AGENTS.md) | Agent目标、自主性、证据纪律、协作、恢复与持续推进原则 | 编写GLM局部版。保留自主选择路径，替换DeepSeek身份，明确当前只研究现有算子下的框架调度极限 |
| [MISSION.md](https://github.com/wio1997/Inference-Foundry/blob/064f4474b1eafd5985469bad1f0bab3233a8f417/MISSION.md) | 为什么做、产品目标、性能裁判、当前阶段与成功条件 | 编写GLM目标：910C、W8A8、已有PD链路的调度极限。当前并行参数是参考点，实验可比性与搜索自由分开 |
| [FOUNDRY_METHOD.md](https://github.com/wio1997/Inference-Foundry/blob/064f4474b1eafd5985469bad1f0bab3233a8f417/FOUNDRY_METHOD.md) | 跨模型的Contract、DAG、Necessary Work、Resource、Bound、Gap、E2E、Re-bound方法 | 共享引用。DeepSeek和Phase A/B/C属于历史验证路线，不成为GLM必须先完成的步骤 |
| [HANDOFF.md](https://github.com/wio1997/Inference-Foundry/blob/064f4474b1eafd5985469bad1f0bab3233a8f417/HANDOFF.md) | 新会话恢复入口：身份、Current、当前问题、现场、证据、下一候选动作 | 新建GLM现场文件，核验真实运行状态；不复制DeepSeek PID、端口、模型、旧next_action |
| [PROJECT_STATE.md](https://github.com/wio1997/Inference-Foundry/blob/064f4474b1eafd5985469bad1f0bab3233a8f417/PROJECT_STATE.md) | 项目进展、阶段、实验与状态演进历史 | GLM独立维护，首屏为当前摘要，历史通过任务和证据索引组织 |
| [PERFORMANCE_MAP.md](https://github.com/wio1997/Inference-Foundry/blob/064f4474b1eafd5985469bad1f0bab3233a8f417/PERFORMANCE_MAP.md) | 已观测耗时、关键路径、机会、假设、反例与下一区分性问题 | 建GLM的P/transfer/D/Host/rank/服务边界图；旧耗时只作旧环境证据 |
| [FRAMEWORK_SCHEDULING_BOUND.md](https://github.com/wio1997/Inference-Foundry/blob/064f4474b1eafd5985469bad1f0bab3233a8f417/FRAMEWORK_SCHEDULING_BOUND.md) | 现有算子条件下的框架调度界专支，必要依赖、保守合法图、成本、资源竞争、区间与缺失证据 | GLM当前最重要的技术主线文档。迁移方法与范围，重建PD、TP16/EP16的图和当前成本，不照搬旧数值或实验步骤 |
| [ACHIEVABLE_BOUND.md](https://github.com/wio1997/Inference-Foundry/blob/064f4474b1eafd5985469bad1f0bab3233a8f417/ACHIEVABLE_BOUND.md) | 更广的可实现性能界与研究历史，包含资源、硬件、算法、算子、调度和产品边界 | 原文作为历史证据库；GLM本地只汇总当前调度界、假设、区间、Gap和未知，链接到专项Framework文档 |
| [RESULTS.md](https://github.com/wio1997/Inference-Foundry/blob/064f4474b1eafd5985469bad1f0bab3233a8f417/RESULTS.md) | 实验结果与裁决，包含正式结果、诊断、无效结果、KEEP/REJECT及限制 | GLM结果独立记录，明确证据类型；失败和未转化到E2E的局部成果仍保留 |
| [SPECIALIZED_RUNTIME.md](https://github.com/wio1997/Inference-Foundry/blob/064f4474b1eafd5985469bad1f0bab3233a8f417/SPECIALIZED_RUNTIME.md) | DeepSeek专项Runtime的设计、控制权边界、实现模式与演进证据 | 保留为设计模式和反例来源。GLM同名文档记录实际选中的执行组织，内容可迭代，不预先指定固定cohort或独立decode架构 |
| [ZCODE_OPERATIONS.md](https://github.com/wio1997/Inference-Foundry/blob/064f4474b1eafd5985469bad1f0bab3233a8f417/ZCODE_OPERATIONS.md) | 机械执行调用、服务操作、controller、超时、结果验收与清理经验 | 引用通用验收原则，GLM命令、机器、路径、端口、P/D服务与设备范围按现场重填；Agent分工采用用户实际配置 |
| [README.md](https://github.com/wio1997/Inference-Foundry/blob/064f4474b1eafd5985469bad1f0bab3233a8f417/README.md) | 仓库入口；目前仅有标题，缺少文档导航 | GLM README提供目标、真实配置、恢复入口和上述文档索引；不堆实验历史 |

文件职责可以保持完整，实际内容按研究产生的事实填写。一次实验不必更新全部文档，新会话也不必全量通读。

## 2 当前最重要的Bound区分

`FRAMEWORK_SCHEDULING_BOUND.md`明确研究现有primitive实现下的合法执行安排，允许执行顺序、buffer、prefetch、Graph、异步编排、pipeline、overlap、请求调度和Host/Device控制。它不要求先完成compulsory-work证明或严格整卡硬件容量C⁺证明，可以通过乐观时间下界L与正确可执行调度的实测时间上界U逐渐收紧区间。数值证据不足则保持未知，不阻塞有价值的研究。

其中`E_must`表达真实值/版本、buffer生命周期、通信和请求依赖；`E_safe`暂保留尚未证明可删除的当前执行顺序。当前Python或stream顺序不自动成为永久依赖。历史未来轨迹可作审计材料，实际Runtime不能读取尚未产生的token、路由或logits。

`ACHIEVABLE_BOUND.md`约233KB，含多个研究分支和大量旧checkpoint。它适合保存历史约束与证据；GLM当前研究不把其中整套硬件/算法证明或算子研究当作前置工作。GLM当前摘要应以Framework调度界为中心，链接详细证据。

## 3 原文中需要明确处理的冲突

- 自主性原则已经清楚：AGENTS把流程、顺序、候选、工具视为默认框架与经验，允许跳步、并行、重排和推翻假设。GLM应保留这一原则，旧Next Run和gate属于当时证据状态。
- 部分文档头部摘要较旧，尾部追加记录更新。PROJECT_STATE首屏仍在Loop034，PERFORMANCE_MAP和ACHIEVABLE首屏仍有初期基线；它们尾部已包含Run675/678。不能将首屏直接复制成GLM当前状态。
- FRAMEWORK文档仍写Run99 571.681 tok/s；HANDOFF及ACHIEVABLE后续记录已写Run675当前保留配置660.309426，Run674历史配置676.824631。更新版本、配置和证据范围必须一起核对；两者都没有得到可信数值框架上限。
- AGENTS定义Astra Light为owner，并写不使用Sol作为主Agent；`docs/agent_orchestration.md`仍写GPT-6 Sol为owner。GLM协作配置应统一，不能原样复制互相冲突的角色说明。[协作文档](https://github.com/wio1997/Inference-Foundry/blob/064f4474b1eafd5985469bad1f0bab3233a8f417/docs/agent_orchestration.md)
- 原AGENTS允许未来主动转算子，MISSION、SPECIALIZED_RUNTIME和Foundry方法包含kernel/fusion扩展。用户当前范围明确覆盖这些建议：只研究调度，不自动转算子，不做5.2/5.3精度比较。

## 4 与根目录文档配套的资料

`performance_knowledge/README.md`、`entries.jsonl`、`sources.json`负责历史机制、适用条件、反例与来源；`tasks/deepseek-extreme-p0/`保存Task/Loop/Run和恢复包；`runtime/README.md`、`runtime/TARGET_CONTRACT.md`描述已有实现合同；`serving/README.md`描述缓存集成。它们通过本地迁移索引按当前问题读取，避免把旧67组KV、固定96token等DeepSeek合同当成GLM合同。

建议恢复入口为本地HANDOFF、当前Task恢复包和真实Git/机器状态；本地AGENTS/MISSION定义稳定目标，再按当前问题打开方法、FRAMEWORK调度界、性能图和必要历史证据。此顺序是建议，Agent可按信息价值调整。
