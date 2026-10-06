# GLM Extreme — AGENTS

2026-10-06起，从checkpoint158（父commit `e378d459e08e9f12300118e158bba16db6c4d04f`）之后的研究执行生效。只改变研究规则、模型分工和后续方式，不追溯重写历史Run、raw evidence或历史裁决。

本文件是GLM研究规则入口；与旧方案、共享方法、HANDOFF、Zcode Summary或历史next_action冲突时，后续执行以本文件为准。旧记录是证据，不是必须继承的路线；当前会话模型不会被文档自动切换。

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

**没有matched A/B的代码不算性能成果；没有真实完整E2E不得提升Current。** 代码级KEEP须同口径、功能有效、可重复的完整E2E Product Gain；不能牺牲有效计数、KV/状态、MTP提交和多rank匹配。配置收益、部署收益、代码收益分开记账。profile/微测/模拟/内部TPS/功能VALID不冒充性能KEEP。

同工作包、有效合法类中L≤T*≤U，剩余可能TPS提升≤U/L−1；持续服务看达标稳定容量及有效上界。Stock固定结构的界不证明重构空间极限；缺界写unknown，不凭几个REJECT宣布到顶。

## 7. 代码性能账本与checkpoint

持续维护[代码性能账本](CODE_PERFORMANCE_LEDGER.md)：`Code Optimization | Baseline | Patched | Gain | Correctness | Verdict`。复用优化点ID、code ref和matched Run引用，不复制原始证据。只有代码级KEEP算主要性能进展。

阶段没有新增代码性能KEEP时，必须明确写：**本阶段没有新增代码级性能 KEEP。** 不得用配置变化、功能测试数量、Runtime模块数量或Run数量包装进度。

重要checkpoint优先汇报：新增代码KEEP、每项E2E Gain、当前最大剩余Gap、下一项最高价值代码问题。评价Sol只看是否找到更大真实Gap、提出高价值代码改造、完成matched A/B、产生可重复E2E Product Gain、持续提升Current并缩小与可达到极限的Gap；不看Run、commit、文档、Runtime模块或探索方向数量。

## 8. 强制Goal Review

以下任一信号出现即暂停惯性Run：连续多个有效实验无代码KEEP、INVALID比例明显升高、方向不断扩散、Runtime模块增加而Current不提升、参数扫描占据主要工作、当前工作与最终性能关键链关系不清、边缘API/driver/compatibility问题长期占据主线。

重新回答：

1. 最终产品是什么？
2. 当前时间实际花在哪里？
3. 当前最大的可由代码消除的Gap是什么？
4. 最近实验是否真正缩小这个Gap？
5. 今天根据现有证据重新开始，还会继续当前路线吗？

在当前优化点或新checkpoint简短记录证据、unknown、继续/转向/停止决定及下一代码问题，必要时触发Astra Challenger Review。若不值得继续，立即停止低价值路线；不可把旧next_action或已投入Run数量当继续理由。Review先做源码/证据研究，不自动启动新GPU/NPU实验，也不重写历史。

## 9. 历史、工程与上下文纪律

昂贵实验前按机制检索[历史复用](REUSE.md)，旧证据足够就复用。保留所有历史Run、raw evidence和裁决；新解释/重裁决另写新记录并引用原身份。GitHub是权威记录，Current保持唯一权威证据引用。

PID、hash、epoch、controller、artifact、ownership证据继续用于实验真实性、资源安全和可复现；只做到足够。不会影响性能归因、correctness或KEEP的信息不无限扩展，不让身份记录成为主要研究工作。只操作本次资源，不盲重启/重跑，不声称虚假后台。

启动读取本文件、真实Git、HANDOFF和**当前**优化点；HANDOFF和point中的下一动作是当时建议。恢复后先审视目标/Gap及Goal Review信号，再选择代码路线；相关源码和决定性证据按需深入，不重放全仓库或全部历史。省token不能省掉裁决证据。
