# GLM调度极限 — 执行逻辑

从checkpoint158之后按[AGENTS](AGENTS.md)生效；本方案是后续研究方式，不改写历史实验与裁决。以模型语义为约束，以资源和条件成本为模型，以matched A/B及真实完整E2E为裁判。旧HANDOFF、point下一问题、next_action和方案候选仅作证据，无预设V1/V2或架构优先权。

入口与Reset见[START_NEW_CHAT](docs/START_NEW_CHAT.md#恢复顺序)；单假说、Run决策价值和实验门槛见[RECORDING](RECORDING.md#入口reset与run决策价值)。本页仅在选择代码问题、Goal Review或交付时读取。

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

GMU、batch、`max-num-batched-tokens`、KV大小、HCCL buffer、端口、TP/DP/PP/DCP比例、MTP深度、Graph开关、实例数量和单纯部署调整，本身不算代码性能优化成果。只能用于建立公平baseline、验证代码假说、判断代码是否能运行、区分资源与代码瓶颈。禁止无限参数扫描制造优化进度。

优先研究可由代码消除的执行浪费：Scheduler/admission/batching、Prefill/Decode竞争、标准PD的`P → KV → D → Decode`关键链、P端Prefill Scheduler、D端Decode Scheduler、KV生产/发送/接收/ready/commit、P/D通信计算overlap、MTP draft/accept/commit/metadata、Graph dispatch/padding/shape、不必要collective/barrier/peer wait、Host ↔ Device同步、CPU request preparation、tokenizer重复工作、framework额外串行、buffer/state生命周期和GPU idle gap（本项目设备为NPU，同样追查设备空闲间隙）。热点或利用率不是极限证书。

标准PD为`Request → P Prefill → KV Transfer → D Decode → Output`；各Replica完整执行Prefill+Decode必须标Full Replica / Complete-request Placement。prefill-aware/prefill-work/work-seconds历史路线是否进产品另作新裁决，不追溯改写旧Run。配置、部署、代码收益分账，只有代码级KEEP计主要进展，汇总到[代码性能账本](CODE_PERFORMANCE_LEDGER.md)。

尽量预加载候选策略并合法驻留对照，减少长启动；地址/ABI/Graph或拓扑变更按依赖重建。多机prof覆盖实际实例/关键rank，关联request/step/transfer/collective与时钟误差，CPU enqueue不冒充设备完成。

完整功能保留，动态逻辑与稳定物理buffer分离；request、slot代际、KV有效版本、tentative/committed、Graph行映射和传输回收保持一致。接口名和实现语言不固定。

具体来源按需查[研究索引](docs/README.md)；记录与避免重复见RECORDING/REUSE。

Type、噪声感知matched A/B与PERF_KEEP判据只见[RECORDING](RECORDING.md#4-提交与裁决)；active stack与Current身份见[同页第6节](RECORDING.md#6-当前stack交互回归与产品current)。

## Goal Review与checkpoint

以下任一情况必须暂停惯性Run并Goal Review（可更早触发，不能把阈值当必须跑满的配额）：

- 连续 **5个有效性能Run** 没有新的代码KEEP / PERF_KEEP。
- 连续 **3个代码candidate** 都无法形成Product Gain。
- 连续出现 **3个INVALID / driver failure**。
- 新增两个以上Runtime模块但Current没变化。
- 主线连续跨越两个以上新领域仍未关闭原问题。
- 一个非目标功能问题占据多个Run。
- 当前最大Gap已无法用证据清楚描述。

原有早期信号同样有效：INVALID比例明显升高、方向扩散、参数扫描主导、与最终性能关键链关系不清、边缘API/driver/compatibility长期占主线。有效性能Run按真实有效性能测量计数，不能靠更换Type、点ID或会话重置连续无收益事实；CPU功能测试不冒充性能Run。

重新回答：

1. 最终产品是什么？
2. 当前时间实际花在哪里？
3. 当前最大的可由代码消除的Gap是什么？
4. 最近实验是否真正缩小这个Gap？
5. 今天根据现有证据重新开始，还会继续当前路线吗？

在当前优化点的新评审记录或新checkpoint简短记录证据、unknown、继续/转向/停止决定及下一代码问题，必要时触发Astra Challenger Review。若不值得继续，立即停止低价值路线；不可把旧next_action或已投入Run数量当继续理由。Review后必须重新产出Performance Research Reset才进入新的性能Run。Review先做源码/证据研究，不自动启动新GPU/NPU实验，也不重写历史。

checkpoint字段见[账本汇报规则](CODE_PERFORMANCE_LEDGER.md#重要checkpoint汇报)；功能阻塞、backlog和模块退出按[RECORDING](RECORDING.md#7-双轨backlog模块约束与完成)。

## 阶段性完成条件

至少满足：明确当前最佳Runtime与最小入口；active performance KEEP stack清楚；同口径完整E2E可重复；主要功能合同通过；最大剩余Gap已知；没有明显高置信、可实施、尚未验证的巨大代码Gap；Astra做过最终架构Challenger Review或明确记录无需的理由。

交付当前性能、Stock/声明baseline性能、完整stack累计Product Gain、每项仍有效KEEP、剩余限制及未解决但低价值backlog。不要求数学绝对极限，须用证据说明继续投入很可能只获得小于当前测量/业务意义的收益，或需要超出现有资源条件。Run跑不动、预算耗尽或多次REJECT不等于项目完成。
