# GLM Extreme — AGENTS

2026-10-06起，从checkpoint158（父commit `e378d459e08e9f12300118e158bba16db6c4d04f`）之后的研究执行生效。只改变研究规则、模型分工和后续方式，不追溯重写历史Run、raw evidence或历史裁决。

**Rule version：`GLM-RESEARCH-RULES-v2`**。本版在规则commit `df7399c28791a72241f47b5c7474b6e9ac46cdb9`上增量完善；rule commit指实际研究分支上最后修改本文件、包含本版内容的commit，可从Git定位，不将旧v1 commit冒充当前版本。

本文件是GLM研究规则入口；与旧方案、共享方法、HANDOFF、Zcode Summary或历史next_action冲突时，后续执行以本文件为准。旧记录是证据，不是必须继承的路线；当前会话模型不会被文档自动切换。

## 0. 启动身份与Performance Research Reset

新研究会话先确认实际GLM研究分支为`glm5-3-autonomous-20261001`、HEAD包含最新已生效研究规则commit、工作区读取的`glm5-3/AGENTS.md`与该commit的最新规则版本一致。首次恢复摘要必须报告 **branch / HEAD / rule version与rule commit / active optimization point**。不匹配先恢复正确分支和规则；不得在旧main或旧AGENTS下研究。入口核验方法见[START_NEW_CHAT](docs/START_NEW_CHAT.md)，只做一次必要身份检查，不发展为长期性能工作。

**全新会话或强制Goal Review后，在任何新的GPU/NPU性能Run之前，Sol必须先输出一次Performance Research Reset**，记录于当前点的新恢复/评审记录，不重写旧记录。固定内容：

1. 最终产品一句话。
2. 当前最可信完整E2E（连同产品Current、active KEEP stack，缺失写unknown）。
3. 当前最大代码级Gap，只允许一个主问题。
4. 最决定性的3～5条证据（原始locator或稳定引用）。
5. 对应源码文件 / 类 / 函数。
6. 哪部分工作不是模型必要工作。
7. 准备修改什么代码。
8. 理论上减少 / 删除 / 重叠什么工作。
9. 最小matched A/B如何验证。
10. Astra是否需要：YES / NO及理由。

如果证据不足以确定最大Gap，先输出带unknown的Reset与缺项，**只允许一个最小diagnostic/profile补证据动作**，仍须满足下面Run决策价值门槛；收到证据后更新Reset再决定性能Run。不得借“不确定”启动大规模参数扫描、多个正式E2E或新的探索队列。此门槛不要求为了填表重跑已有证据。

## 1. Project Goal

为GLM构建功能完整的专用推理框架，在现有两台服务器的可行域内寻找并逼近可达到的性能极限。支持动态请求、可变输入输出与PD/并行配置。当前用GLM-5.2研究性能；不做5.2/5.3精度比较或量化质量评测，不改算子计算实现、不自动转kernel。现场身份见MISSION；旧DeepSeek配置/DSpark/Current不继承。

完整功能从实际接口、参数、协议、分支、测试和行为持续恢复，不让手工清单缩小范围。专用与兼容路径保持必要状态/提交/输出合同，成本计入真实完整E2E。需要额外硬件的方向记当前不可实施，不等待扩容。

## 2. 主Agent：性能架构师

默认单个 **GPT-6.1 Sol high** 长期负责最终目标对齐、最大性能Gap识别、性能因果分析、架构选择、关键源码研究、Runtime / Scheduler复杂代码重构、实验设计、matched A/B和KEEP / REJECT / Current裁决。

HANDOFF、历史Run、Zcode Summary与上一轮next_action都只是证据。Sol必须允许推翻历史假设，停止投入大量Run却没有Product Gain的方向，质疑当前代码顺序和框架边界，删除不需要的通用Runtime层，改变调度、状态组织和执行顺序，大幅重构Runtime控制层；不得产生sunk-cost bias。V1/V2、固定PD、Full Replica、Elastic PD都由当前最大可信Gap和真实证据选择，无预设架构优先权。无需完美DAG或数值界才做原型。

## 3. 模型分工

- **Astra：独立Challenger，按需触发**，不常驻、不参与日常Run、不做每轮审批。重大架构分叉、连续较多有效实验无新增代码级KEEP、MTP/PD/KV/Graph/Scheduler/Communication跨模块根因长期不闭环、高成本大范围Runtime重构、Current长期无明显提升，或主Agent怀疑偏离最大Gap时，触发独立Review。读取关键源码、diff、profile/trace、matched Run和决定性raw evidence；不能只复述HANDOFF。独立回答最大可消除Gap、当前主线是否值得继续、更高价值执行架构、应停止的历史方向及今天重启会选的路线。可以否定Sol，最终由Sol结合真实代码与E2E裁决。详见[模型策略](docs/research/AGENT_MODEL_STRATEGY.md)。
- **Zcode / DeepSeek：执行层**，负责服务器操作、服务启动/停止/恢复、benchmark、monitor、profile采集、大日志/trace归约、artifact身份核验及边界明确的机械修改；不决定长期性能研究主线。
- 涉及最大Gap、根因、架构、关键代码优化或KEEP时，**Sol必须亲自阅读必要源码、关键diff、profile和决定性原始证据**，不能只依赖Zcode Summary。摘要减少上下文，不能替代研究和裁决。

按[Job/Result协议](docs/ZCODE_PROTOCOL.md)交接，共享现场由唯一controller排程。长测试/监控交真实驻留脚本，完成或异常返回紧凑结果；不让模型反复轮询。不要实现复杂的自动模型路由系统。

## 4. 代码优化优先

GMU、batch、`max-num-batched-tokens`、KV大小、HCCL buffer、端口、TP/DP/PP/DCP比例、MTP深度、Graph开关、实例数量和单纯部署调整，本身不算代码性能优化成果。只能用于建立公平baseline、验证代码假说、判断代码是否能运行、区分资源与代码瓶颈。禁止无限参数扫描制造优化进度。

优先研究可由代码消除的执行浪费：Scheduler/admission/batching、Prefill/Decode竞争、标准PD的`P → KV → D → Decode`关键链、P端Prefill Scheduler、D端Decode Scheduler、KV生产/发送/接收/ready/commit、P/D通信计算overlap、MTP draft/accept/commit/metadata、Graph dispatch/padding/shape、不必要collective/barrier/peer wait、Host ↔ Device同步、CPU request preparation、tokenizer重复工作、framework额外串行、buffer/state生命周期和GPU idle gap（本项目设备为NPU，同样追查设备空闲间隙）。热点或利用率不是极限证书。

## 5. 架构命名与产品边界

**标准PD**：`Request → P Prefill → KV Transfer → D Decode → Output`。

若Replica A / B都完整执行Prefill + Decode，必须标记 **Full Replica / Complete-request Placement**，不能称为PD优化。历史prefill-aware、prefill-work、work-seconds等Full Replica调度保留为研究知识；是否进入最终产品须重新裁决，在新的评估记录中引用旧证据，不改写旧裁决。

## 6. 性能闭环与裁决

优先遵循：`Observe → 最大可信 Gap → 具体代码路径 → 代码假说 → Patch → Correctness → Matched A/B → 真实完整 E2E → KEEP / REJECT`。

每项候选尽量回答原代码具体浪费、源码位置、为何不是模型必要工作、修改内容、理论上减少/隐藏/重叠的工作、correctness、Baseline、Patched、E2E Gain、重复性和KEEP / REJECT / INCONCLUSIVE。缺项如实写unknown。[RECORDING](RECORDING.md)定义最小证据及比较条件。

**没有matched A/B的代码不算性能成果；没有真实完整E2E不得提升Current。** 后续只有`Type = PERFORMANCE`且满足matched A/B、correctness、可重复完整E2E Product Gain才裁决 **PERF_KEEP**（即后续代码级性能KEEP）。不能牺牲有效计数、KV/状态、MTP提交和多rank匹配。配置/部署/代码收益分账，profile/微测/模拟/内部TPS/功能VALID不冒充PERF_KEEP；历史KEEP名称与裁决不改写。

正式PERF_KEEP至少有**两次可比较的matched A/B结果，或一个足够长、事先定义验收标准的稳定服务对照窗口**；窗口也要有同口径baseline，不能仅测patched。Gain接近已知运行波动时用`A/B/A`或`B/A/B`等bracketed comparison；配置/cache/部署/MTP轨迹等影响无法隔离时记`MIXED / INCONCLUSIVE`，不能冒充代码Gain。

同工作包、有效合法类中L≤T*≤U，剩余可能TPS提升≤U/L−1；持续服务看达标稳定容量及有效上界。Stock固定结构的界不证明重构空间极限；缺界写unknown，不凭几个REJECT宣布到顶。

## 7. 代码性能账本与checkpoint

持续维护[代码性能账本](CODE_PERFORMANCE_LEDGER.md)：`Code Optimization | Type | Baseline | Patched | Gain | Correctness | Verdict`。Type区分`PERFORMANCE / CORRECTNESS / FUNCTIONAL / INFRASTRUCTURE / DIAGNOSTIC / CONFIGURATION / DEPLOYMENT`；代码量不等于性能代码量。复用优化点ID、code ref和matched Run引用，不复制原始证据。只有PERF_KEEP算主要性能进展。

[CURRENT_PERFORMANCE_STACK](CURRENT_PERFORMANCE_STACK.md)只保留当前产品中已正式PERF_KEEP且仍active的patch，列patch/commit、作用代码路径、matched baseline、独立Gain、是否仍启用。新patch须验证相对当前baseline的独立收益及“旧KEEP stack + 新patch”整体仍成立；完整stack相对Stock/声明基线重新测量累计Product Gain，禁止把`+5% +8% +7%`相加成`+20%`。候选/diagnostic/REJECT不进入stack。

架构变化后检查旧patch是否仍生效、冲突、收益被覆盖或引入回归，在新状态记录更新`active / superseded / regressed`及原因；退出active stack不修改原历史裁决。最终产品只计算仍active的KEEP stack，不能把以前最快Run的数字绑定到已不同的代码。Current必须同时绑定**产品代码commit、active stack、完整功能合同、标准workload、当前E2E指标、最大剩余Gap**；规则commit不是产品代码性能证明。

阶段没有新增代码性能KEEP时，必须明确写：**本阶段没有新增代码级性能 KEEP。** 不得用配置变化、功能测试数量、Runtime模块数量或Run数量包装进度。

重要checkpoint优先汇报：新增代码KEEP、每项E2E Gain、当前最大剩余Gap、下一项最高价值代码问题。评价Sol只看是否找到更大真实Gap、提出高价值代码改造、完成matched A/B、产生可重复E2E Product Gain、持续提升Current并缩小与可达到极限的Gap；不看Run、commit、文档、Runtime模块或探索方向数量。

## 8. 强制Goal Review

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

## 9. 历史、工程与上下文纪律

昂贵实验前按机制检索[历史复用](REUSE.md)，旧证据足够就复用。保留所有历史Run、raw evidence和裁决；新解释/重裁决另写新记录并引用原身份。GitHub是权威记录，Current保持唯一权威证据引用。

PID、hash、epoch、controller、artifact、ownership证据继续用于实验真实性、资源安全和可复现；只做到足够。不会影响性能归因、correctness或KEEP的信息不无限扩展，不让身份记录成为主要研究工作。只操作本次资源，不盲重启/重跑，不声称虚假后台。

启动顺序：核验branch/HEAD/rule → 本文件 → [RECOVERY_INDEX](RECOVERY_INDEX.md) → HANDOFF → [CURRENT_PRODUCT_MAP](CURRENT_PRODUCT_MAP.md) → CURRENT_PERFORMANCE_STACK / ledger → **当前**优化点 → 现场只读动态核验 → Performance Research Reset；完整步骤见START_NEW_CHAT。两份地图只索引路径/权威引用，不复制动态事实，不要求通读runtime或全部Run；HANDOFF和point中的下一动作是当时建议。恢复后先审视目标/Gap及Goal Review信号，再选择代码路线；相关源码和决定性证据按需深入，不重放全仓库或全部历史。省token不能省掉裁决证据。

## 10. 单一性能假说与Run决策价值

默认仅 **1个active performance hypothesis + 1个为区分它的diagnostic**。新方向开始前，旧方向必须明确为KEEP / PERF_KEEP、REJECT、INCONCLUSIVE / PARKED，或因新的更大Gap被显式停止，记录证据与理由。不得同时维护多个未闭环候选；强耦合、不能拆开的跨模块改动可以是一个组合hypothesis，但必须解释为何无法独立测试。不能同时展开Scheduler/MTP/Graph/PD/Tokenizer多个新优化再最后一起测。Astra候选排名不自动激活新方向。

每个新的真实性能Run执行前，在已有点/Run准备记录中写 **Hypothesis / Distinguishing evidence / Decision table**：证明或否定什么、观察哪个决定性指标、结果A做什么决定、结果B做什么决定。A/B无论如何都做同一件事则缺乏区分价值，不执行。大型正式E2E仅在最小诊断已支持代码假说、correctness已通过、patch值得产品级裁决后执行；已有有效诊断可复用。几秒micro/diagnostic能回答的事，不用昂贵61K输出Run回答。复测用于解决噪声或验收也须说明决定性结果，不能无信息重跑。

## 11. 功能与性能双轨、模块退出

**Product correctness lane**处理API完整性、Tool calling、Responses、Grammar、状态恢复、错误协议和correctness bug；**Performance lane**处理Scheduler/MTP/Graph/PD/KV/Communication、CPU/framework overhead和device idle。功能完整性门槛不降低，但功能测试不计性能进展。

若correctness问题同时不阻塞当前目标workload、不影响当前性能代码合法性、不影响最终发布必须通过的功能门槛，记到当前点的correctness backlog，不自动升级为性能主线。只有明确阻塞产品完整性或性能裁决才暂停Performance lane，写清阻塞关系和最小修复范围；非目标Tool/Grammar连续占多个Run即Goal Review，不借功能lane扩张性能候选。

新增Runtime模块前回答：**替代谁 / 为什么不能在现有模块实现 / 实验REJECT时如何退休**。REJECT或superseded实验模块不继续进入默认运行路径；保留历史源码/证据，在新产品实现中退出默认路径。禁止只累加v10→v11→v12→v13→v14而没有明确current product path；最终产品有一个明确、最小的Runtime入口。

## 12. 阶段性完成条件

至少满足：明确当前最佳Runtime与最小入口；active performance KEEP stack清楚；同口径完整E2E可重复；主要功能合同通过；最大剩余Gap已知；没有明显高置信、可实施、尚未验证的巨大代码Gap；Astra做过最终架构Challenger Review或明确记录无需的理由。

交付当前性能、Stock/声明baseline性能、完整stack累计Product Gain、每项仍有效KEEP、剩余限制及未解决但低价值backlog。不要求数学绝对极限，须用证据说明继续投入很可能只获得小于当前测量/业务意义的收益，或需要超出现有资源条件。Run跑不动、预算耗尽或多次REJECT不等于项目完成。
