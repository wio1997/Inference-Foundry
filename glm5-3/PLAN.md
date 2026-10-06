# GLM调度极限 — 执行逻辑

从checkpoint158之后按[AGENTS](AGENTS.md)生效；本方案是后续研究方式，不改写历史实验与裁决。以模型语义为约束，以资源和条件成本为模型，以matched A/B及真实完整E2E为裁判。用户2026-10-06明确限定产品为标准P/D分离；旧HANDOFF、point下一问题、next_action和方案候选仅作证据，不预选V1/V2 runner，不继承完整副本路线。

入口与Reset见[START_NEW_CHAT](docs/START_NEW_CHAT.md#恢复顺序)；单假说、Run决策价值和实验门槛见[RECORDING](RECORDING.md#入口reset与run决策价值)。本页仅在选择代码问题、Goal Review或交付时读取。

## 当前执行方案：标准PD优化

2026-10-06早前“先调整方案”的仅资料阶段已结束；用户已授权在新对话继续必要代码实现、验证和实验。已进入实际源码/raw研究、两机artifact结构核验和CPU诊断实现，见[checkpoint160](records/points/GLM-OPT-0002/research/pd_critical_path_20261006/CHECKPOINT160.md)。权重上传/完整性、现场资源与操作条件仍须先核验；上传未终态前不加载/测试，独立源码与CPU工作继续。以下是研究顺序，不是旧controller执行队列。Current=None，active PERF_KEEP为空；[目标纠正与源码证据](records/points/GLM-OPT-0002/research/engine_commit_20261006/PD_SCOPE_CORRECTION.md)。

**产品固定为GLM-5.3 W8A8的 `请求 → P Prefill → KV Transfer → D Decode/MTP → 有效输出`。** 两机目标权重`/data/tiankuan/wio/GLM-5.3-w8a8`，新服务模型名`glm-53`。2026-10-06用户正在上传；配置已切目标不代表artifact完整或已加载，完成并核验前不启动模型或测试。P池、D池分别组织DP/TP/EP域；不能将任一角色改成完整请求副本来代替PD。保持现有正确算子、全部功能、有效计数与SLO；基线与patched固定同一5.3模型身份，旧版本数据不构成新模型基线。

### 1. 一个参考部署，不开展配置扫描

本方案执行的是[GLM-5.3专用框架极致优化目标](MISSION.md#专用化范围与现场动态能力)。标准PD限定请求执行语义，不限定必须保留vLLM的通用控制架构。原生链用于建立可信参考和复用正确算子；由关键路径证据决定删层、替换或成套重构。动态请求/完整外部功能以及[既有SLA](MISSION.md#性能与功能合同)始终是验收约束。

| 角色 | 参考几何 | 用途与证据边界 |
| --- | --- | --- |
| 166：P | DP1 / TP16 / PP1 | 使用该机16个逻辑NPU做P；历史P89/Run101证明过部分真实PD请求 |
| 167：D | DP1 / TP16 / PP1 | 使用该机16个逻辑NPU做D；历史D100/Run101证明过KV接收及后续输出 |

这是旧模型可恢复性参考，不是GLM-5.3已确认fit、最优配置或当前驻留部署。两端历史DCP16/EP16、Mooncake producer/consumer及独立通信域从[Run100 manifest](records/points/GLM-OPT-0002/runs/GLM-RUN-0100/manifest_record.json)和实际安装源码定位，5.3需依据新config/量化/cache与加载路径重验；不重放旧PID、epoch、端口、环境或脚本。DP2/TP8仅作为**角色内部**的条件部署候选：现有瓶颈使其比较能改变决策、实际fit和正确性成立后，才选一个有价值对照；不会与参考基线并行扫描。配置收益单列，不计代码PERF_KEEP。

恢复采用最小原生PD服务链，逐项确认当前软件/必要兼容修复。历史MTP K、Graph/SP、KV/batch参数和AtomicMQ/BudgetScheduler控制均不是自动继承的PERF_KEEP；PP专用插件、完整请求placement和可选PD fallback不进入默认参考路径。任何可运行性调整明确记录，基线与patched使用相同条件。

### 2. 先建立真实PD研究基线，区分发布验收

- 复用旧模型Run101的3个真实PD请求及其raw调查协议/控制机制；GLM-5.3的tokenizer、prompt/KV布局、状态、外部KV复用、输出预算及输出语义必须重新核验。不继承其token数量、cache命中或功能正确性，不把该Run全部public输出算成PD覆盖，也不把有限功能数据当作5.3 matched性能baseline。
- 原生proxy/connector/scheduler是首要调查入口。保留原生P端内部max1引导语义；内部token只计成本、不计有效输出。先追实际消费者，不能凭外观删除必要工作。
- **研究基线**可以是功能已验证、真实走PD的明确请求集，用于定位瓶颈与否定代码假说；结果必须标明覆盖范围。短/长prompt、cold/prefix-cache条件和到达顺序采用既有工作包，并在比较前冻结；未统一条件的旧TPS不参加排名。
- **产品Current/发布验收**仍要求完整原有功能，包括tools/structured、Responses/续接、stream/background、取消、拒绝/错误、状态与KV回收。未覆盖项进入现有correctness backlog；若影响当前负载、候选合法性或完整产品验收则阻塞对应阶段。不能用D本地完整请求fallback把它们算成PD成功，也不能在尚未证明必要时把整套API重构变成第一主线。

### 3. 第一阶段交付：真实PD关键路径判断

先归约已有证据；只在缺少的观测会改变判断时，冻结**一个最小PD诊断**，不先跑大型完整输出压测。最小诊断需回答：哪段必要工作或等待决定有效输出，最大可由代码消除的等待是否确实存在？

| 因果位置 | 必须关联的证据 | 不能替代它的数字 |
| --- | --- | --- |
| 请求准备与P入队/执行 | 同一请求的resolved prompt、schedule、实际设备Prefill | API接收时间、scheduler调用数 |
| KV导出、传输与D接收 | request/engine/rank/block身份，真实transfer与recv完成 | helper HTTP结束、发送enqueue、外部cache命中计数本身 |
| D具备执行条件与实际Decode | KV/状态已合法ready、scheduler eligibility、设备target/MTP依赖 | future wait、CPU阻塞、collective接收端耗时本身 |
| 输出与终态释放 | MTP accept/commit、有效用户输出、完成/取消/拒绝及KV归还 | 内部helper输出、轮询、取消请求、总NPU利用率 |

跨机时间需要校准/误差说明，重叠区间不能相加当耗时。区分P供给不足、必要计算、真实通信、rank依赖和可消除控制等待；先由完整PD关键路径选最大Gap，再选代码。当前最大可消除性能Gap及可节省时间均unknown，旧Full Replica的PP profile不能补成PD结论。

### 4. 证据支持后只做一个代码候选

| 观察结果 | 决定 |
| --- | --- |
| 必要KV/状态/设备依赖已满足，但控制顺序或框架抽象成本仍阻挡D执行或有效输出 | 提出与已证明因果边界相称的代码改造，可局部修正，也可删除/替换框架层或成套重构控制链；保留必要FIFO、并行重叠、KV/状态、MTP提交和多rank语义 |
| 等待属于必要计算/传输、P供给或资源竞争 | 否定该控制等待解释，按实际最大Gap换题；不把等待全部计成可消除时间 |
| 身份、设备完成或因果关系不足 | 仅补能改变上述决定的最小证据；不实施猜测patch |

`AsyncLLM.add_request`只是已找到的候选接入边界，**不是预选patch**。只有证明现有proxy→connector→scheduler边界阻挡必要能力或关键路径，才改接入点。其P等待期间取消、n>1、工具续轮及streaming input必须按真实消费者设计，不以一个hook或新Runtime层代替分析。原始EngineCore提交顺序假说也只能在真实PD证据支持时重入；不继承旧drain-first、PP phase、cap/budget/cadence路线。

“一个代码候选”指一个可证伪的因果假说，不限定一个文件或微小patch；强耦合的专用控制/状态设计可整体交付，并说明为何不能独立验证。高成本大范围重构按既有规则触发Astra Challenger Review。删除通用兼容机制时保留真实GLM消费者及完整外部语义，不能将内部结构兼容要求误当必须保留的产品功能，也不能反过来删除现场合法能力。

### 5. 完整PD E2E裁决

每个候选先通过与改动风险相称的correctness和状态/多rank检查。A/B固定真实PD拓扑、物理资源、模型/算子、完整功能合同、有效输入输出、请求到达/顺序、cache与salt、采样/MTP/Graph、SLO和观测开销；不可避免差异明确标为混杂。

先确认局部关键路径减少确实传导到有效输出，再做无重型profile的完整E2E。按[既定重复/噪声门槛](RECORDING.md#4-提交与裁决)比较有效吞吐、TTFT/TPOT及尾延迟、错误/拒绝、队列稳定性和资源成本。未保持PD覆盖或完整功能、工作移出测量窗口、输出计数改变、仅微测改善、收益落在噪声内，均不PERF_KEEP、不更新Current；缺证据为INCONCLUSIVE，反证成立为REJECT。

### 6. 历史数据与分工

Full Replica/PP部署的TPS、容量和性能候选对PD产品收益为NOT_APPLICABLE，原Run/raw/裁决保留。仍匹配实际版本的原生源码、局部机制证据仅作调查线索；历史真实PD数据按具体覆盖复用。沿用现有point、ledger、controller和Git，不另建大账本。

Sol亲读源码/raw，决定Gap、唯一假说、patch和最终裁决；Zcode通过Job→Result→bridge执行明确的现场工作，Astra负责重大路线/架构挑战。本次Astra方案复核已完成，Sol接受5项意见：首阶段交付PD关键路径判断、不预选hook、区分限域研究与完整验收、保留原生引导及核验生命周期、固定参考几何且不把DP2变成第二主线。

## 通用研究步骤

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

标准PD为`Request → P Prefill → KV Transfer → D Decode → Output`；各Replica完整执行Prefill+Decode必须标Full Replica / Complete-request Placement，已被用户排除为本产品路线。prefill-aware/prefill-work/work-seconds历史Run保留，完整副本收益不用于PD产品裁决。配置、部署、代码收益分账，只有代码级KEEP计主要进展，汇总到[代码性能账本](CODE_PERFORMANCE_LEDGER.md)。

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
