# GLM调度极限 — 执行逻辑

从checkpoint158之后按[AGENTS](AGENTS.md)生效；本方案是后续研究方式，不改写历史实验与裁决。以模型语义为约束，以资源和条件成本为模型，以matched A/B及真实完整E2E为裁判。旧HANDOFF、point下一问题、next_action和方案候选仅作证据，无预设V1/V2或架构优先权。

v2入口门槛：核验实际研究branch、HEAD/最新rule commit与AGENTS版本，首次报告active point；新会话或强制Goal Review后先[Performance Research Reset](AGENTS.md#0-启动身份与performance-research-reset)。默认仅1个active性能假说+1个区分diagnostic，切换前关闭/park/停止旧方向；强耦合组合解释无法独立测试。每个真实性能Run前写Hypothesis、Distinguishing evidence、Decision table，结果A/B若不改变决定就不执行；大型正式E2E等待诊断支持、correctness通过、patch值得裁决。Gap unknown只做一个最小补证据动作，不扫描或批量E2E。

1. **定义问题**：当前硬件可行域、完整功能、负载/到达/cache条件、既有SLO与性能对象。单次比较清楚，后续探索可扩展。
2. **恢复必要关系**：GLM主干/Full-Shared indices/MTP/提交/KV/通信/输出的producer-consumer和生命周期；区分必要依赖与当前额外顺序。模型共享机制须映射实际实现。
3. **建立条件成本**：shape、batch、上下文、MTP、placement及竞争会改变现有算子成本。HCCL可能包含peer wait，并发不天然等于max(独立耗时)。
4. **辨因与干预**：沿有效输出反查最大可信Gap或路线未知；短profile、少量成本测量或结构原型区分解释，方法可重排。D空闲先分清供给/传输/提交/容量/rank等待。
5. **实现并裁决**：`Observe → 最大可信 Gap → 具体代码路径 → 代码假说 → Patch → Correctness → Matched A/B → 真实完整 E2E → KEEP / REJECT`。说明具体浪费/源码/非必要工作、改动/理论节省、correctness、baseline/patched、E2E Gain和重复性，缺证据写INCONCLUSIVE/unknown；无matched A/B不算性能成果，无完整E2E不提升Current。关闭重型prof测正式收益，微测/模拟/内部TPS/功能VALID单独标记。

## 极限标准

同完整工作包和有效合法类，有乐观耗时下界L与可执行耗时U，则L≤T*≤U；固定有效输出Q时TPS*∈[Q/U,Q/L]，剩余可能提升≤U/L−1。成本/时钟/覆盖的不确定性一并报告，经验条件界与严格界分开。

持续服务用已达到的达标稳定容量R和有效乐观容量上界B，R≤R*≤B，剩余可能提升≤B/R−1；有限wall不自动证明在线容量。界不完整可以继续原型，明确unknown，不用固定失败次数或高利用率宣布到顶。

固定Stock节点/shape/K的界只约束该结构。允许合批、K、调用组织、PD/并行或整体重构后，必要工作与成本必须覆盖这些选择。跨配置界只针对当前可行域；未测但可行的方向不能忽略，需要额外服务器的方向记当前不可实施，不阻塞收敛。

## 工程候选与成本

从最大可由代码消除的Gap选择架构和改动，V1/V2资源适配与GLM动态控制都只是候选。Sol亲读必要源码/关键diff/profile/决定性raw，可以删除不必要通用Runtime层、重排调度和状态/执行顺序或大改控制层。不要求先重写整仓库或先算完完美界，不因历史投入继续无Product Gain方向。

GMU/batch/token budget/KV/HCCL buffer/端口/并行比例/MTP深度/Graph开关/实例数只用于公平baseline、验证代码假说、可运行性和资源辨因，禁止无限扫描。优先追查Scheduler/admission/batching、P/D竞争与P→KV→D→Decode、KV ready/commit/overlap、MTP metadata、Graph dispatch/padding、不必要同步/collective、CPU/tokenizer准备、framework串行和buffer/state生命周期。

标准PD为`Request → P Prefill → KV Transfer → D Decode → Output`；各Replica完整执行Prefill+Decode必须标Full Replica / Complete-request Placement。prefill-aware/prefill-work/work-seconds历史路线是否进产品另作新裁决，不追溯改写旧Run。配置、部署、代码收益分账，只有代码级KEEP计主要进展，汇总到[代码性能账本](CODE_PERFORMANCE_LEDGER.md)。

尽量预加载候选策略并合法驻留对照，减少长启动；地址/ABI/Graph或拓扑变更按依赖重建。多机prof覆盖实际实例/关键rank，关联request/step/transfer/collective与时钟误差，CPU enqueue不冒充设备完成。

完整功能保留，动态逻辑与稳定物理buffer分离；request、slot代际、KV有效版本、tentative/committed、Graph行映射和传输回收保持一致。接口名和实现语言不固定。

具体来源按需查[研究索引](docs/README.md)；记录与避免重复见RECORDING/REUSE。

后续代码工作按Type分类；只有PERFORMANCE、correctness、matched A/B及噪声感知重复完整E2E Gain能PERF_KEEP。至少两次可比较matched结果或预定义足够长稳定服务对照窗口，gain接近波动用A/B/A或B/A/B，无法隔离影响标MIXED / INCONCLUSIVE。[当前active stack](CURRENT_PERFORMANCE_STACK.md)只列仍生效的正式性能KEEP；验证旧stack+新patch整体，累计收益完整stack重测不相加。Current绑定产品代码commit/stack/功能合同/标准workload/E2E/剩余Gap，旧patch适用状态另记active/superseded/regressed而不改原裁决。

## Goal Review与checkpoint

连续5个有效性能Run无新代码KEEP、3个代码candidate无Product Gain、3个INVALID/driver failure、新增两个以上模块而Current不变、跨两个以上新领域而原问题未关闭、非目标功能问题占多个Run或最大Gap无法清楚描述时，必须按[AGENTS](AGENTS.md#8-强制goal-review)Goal Review。INVALID比例升高、方向扩散、参数扫描主导或关键链关系不清可提前触发，不能等跑满配额。必要时触发[独立Astra Review](docs/research/AGENT_MODEL_STRATEGY.md#astra-challenger-review)，Sol记录接受/拒绝与原因；不值得继续立即停止，Review后重新Reset。

功能与性能双轨：非阻塞correctness问题记backlog，明确阻塞产品完整性/性能裁决才暂停性能lane。新增模块回答替代谁、为何不能现有实现、REJECT如何退休；无效/过时实验退出默认路径，最终Runtime入口明确最小。阶段完成按AGENTS的最佳Runtime/active stack/重复E2E/主要功能合同/已知剩余Gap/无高置信巨大未验Gap/最终Astra或无需理由验收，报告当前与baseline、重测累计Gain、有效KEEP、限制和低价值backlog，不以Run失败或预算耗尽完成。

checkpoint先报新增代码KEEP、E2E Gain、最大剩余Gap与下一最高价值代码问题；无新增时写“本阶段没有新增代码级性能 KEEP。”工程证据足够支撑性能归因/correctness/KEEP、真实性、安全和可复现即可，不以Run、文档或模块数量评估进度。
