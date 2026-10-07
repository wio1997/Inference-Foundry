# 优化点、实验与Git提交规范

从checkpoint158之后按[AGENTS](AGENTS.md)执行；本次在`df7399c28791`上增量采用`GLM-RESEARCH-RULES-v2`。所有历史Run、raw evidence、历史裁决及旧point/next_action保持原样；新的解释、产品取舍或重新裁决另写新记录并引用原证据，不追溯改账。以下字段加入已有点/Run/恢复记录即可，不另建审批或自动路由系统。

GitHub保存权威代码/配置、事实摘要、裁决和可追溯证据。运行机器保存大raw artifact，Git记录稳定位置、大小、SHA256及关键归约；不提交权重、大trace或凭据。文件数量不作为进度。

## 1. 一点一身份，一次执行一个Run

- 优化点：`GLM-OPT-NNNN`，按问题/机制划分；默认仅1个active performance hypothesis + 1个区分它的diagnostic，相关实验顺序收敛，不代表可同时激活多个候选。强耦合不可拆分修改可作一个组合假说，须说明无法独立测试的原因。摘要放`records/points/<ID>/point.md`，索引`records/points.jsonl`。
- Run：`GLM-RUN-NNNN`，记录真实执行的代码、合同、命令、观察器、结果与证据。默认在优化点下`runs/<Run>/`，不同配置/采集不得伪装同一次执行。
- 沿用根`scripts/taskctl.py`的Task/Loop/Run、compare和resume机制；若采用它，其目录为唯一权威路径，点索引只引用，避免复制第二份状态。先恢复真实源码目录，勿用假路径初始化；其算子字段/固定Loop习惯不限制GLM研究。

初始点允许PLANNED且没有Run；未执行不填性能。索引仅保留ID、标题、类别、状态、权威路径和关键Run/Current引用。

切换性能方向前将当前假说明确为KEEP / PERF_KEEP、REJECT、INCONCLUSIVE / PARKED或因更大Gap被显式停止，并记录原因；不能通过开新点掩盖未闭环候选。

## 入口Reset与Run决策价值

新会话或强制Goal Review后，性能Run前先完成[身份核验与Performance Research Reset](docs/START_NEW_CHAT.md#首次实际研究输出performance-research-reset)，在当前点引用该记录，不复制模板。Gap证据不足只做一个最小补证据动作后更新Reset。

每个新的真实性能Run前用以下紧凑准备记录，不启动GPU/NPU也可完成源码/证据检查：

```text
Hypothesis: 想证明或否定什么（当前唯一active hypothesis）
Distinguishing evidence: 决定性指标/观察、窗口/覆盖与最小工作量
Decision table:
  结果A（事先定义的条件/阈值） -> 下一决定
  结果B（相反或其他可区分条件） -> 下一决定
```

若A/B都导向同一决定，则无区分价值，不执行。大型正式E2E只在最小诊断（可复用已足够证据）支持代码假说、correctness通过、patch值得产品裁决后执行；不用大型压测回答几秒micro/diagnostic能回答的事。当前正式workload以[MISSION](MISSION.md#性能与功能合同)的80K输入/0.6K输出/93%共享前缀为准，不继承旧60K输出合同。噪声复测也预先说明接受/拒绝/继续辨因的条件，不以执行本身当决策。

## 2. 优化点最小内容

问题与合同；必要依赖/资源及假设；预期改变的有效输出链；相关历史与条件差异；本次新增信息；修改范围/配置；验收与裁决。已有证据足够则标`REUSED`并引用，省去重复实验。原型前无需写长证明，可随证据修正。代码候选尽量回答：

1. 原代码具体浪费在哪里？
2. 哪段源码导致（文件/函数/必要消费者及code ref）？
3. 为什么不是模型必要工作？
4. 修改了什么（patch/diff ref）？
5. 理论上减少、隐藏或重叠什么工作，增加什么成本？
6. correctness是否保持，验证证据是什么？
7. Baseline是多少（同合同Run、指标/单位）？
8. Patched是多少（matched Run、指标/单位）？
9. 真实完整E2E Gain是多少，公式/范围是什么？
10. 是否重复成立，波动/噪声和限制是什么？
11. KEEP / REJECT / INCONCLUSIVE是什么，缺项为何？

缺证据写unknown，不能以Summary代填。涉及最大Gap、根因、架构、关键代码或KEEP，Sol必须亲读必要源码、关键diff、profile和决定性raw。

## 3. Run最小证据

`manifest.json`记录：

- ID/优化点、时间、kind（formal_e2e/diagnostic/profile/microbenchmark/simulation等）、实际执行状态；后续代码工作另记Type（与Run kind不同），关联Reset/active hypothesis及Run前决策表。
- 框架/source repo与commit；有未提交改动则记录base+patch artifact/hash及关键文件hash。代码、镜像、模型/量化、设备/rank/部署和effective config身份。
- 实际命令/cwd/内层exit/超时；负载、到达、采样/MTP/Graph/cache条件及相关有效工作量。比对Run和可比性说明。
- 风险相称的功能/状态验证、完整请求/有效输出、吞吐/延迟/错误与观察器开关。缺失填null/unknown，不臆造。
- artifact路径/大小/hash、关键日志归约、必要的清理/恢复与活动资源状态。

短`summary.md`只写结果、因果证据、限制、裁决及下一问题；无需重复manifest。已失效、超时或缺产物保留`INVALID`，不归为性能成功/失败。指标可增减，足以复现和裁决即可。

## 4. 提交与裁决

提交按可审阅变化组织，不每个编辑都commit；一个不可拆分组合假说可以跨模块，但不能并行积累多个未闭环性能候选：

```text
glm(opt): GLM-OPT-0007 describe final code/config change
glm(exp): GLM-OPT-0007 GLM-RUN-0012 record evidence and verdict
glm(knowledge): record applicability and revalidation conditions
```

代码提交与实验记录相互链接；Run固定被测code ref，记录提交固定该Run证据。推送前核对实际diff，不夹带他人改动；研究分支按任务选择，Git更新不force覆盖并发工作。

后续裁决使用`PERF_KEEP / KEEP / REJECT / INCONCLUSIVE / INVALID / REUSED / NOT_APPLICABLE`；PARKED是暂停候选状态，不是性能成功，MIXED是混合归因标记。**没有matched A/B的代码不算性能成果；没有真实完整E2E的结果不得提升Current。** 只有`Type = PERFORMANCE`、功能/状态有效、matched A/B及可重复完整E2E Product Gain才能得到PERF_KEEP。不能牺牲有效计数、KV/状态、MTP提交和多rank匹配。其他Type可保留功能/工程改动，但其KEEP不是代码性能KEEP；历史裁决/字段名不追溯转换。REJECT写原因、未知和重验条件。

Matched A/B至少匹配模型/算子/资源、负载与到达/有效工作量、功能与SLO合同、采样/MTP/cache/Graph相关条件及观察器，明确baseline/patched的code/config身份与目标patch。正式PERF_KEEP至少两次可比较matched A/B结果，或一个足够长、预先定义验收标准的稳定服务对照窗口（也需要baseline）。报告波动/噪声；gain接近已知波动时用`A/B/A`或`B/A/B`等bracketed comparison。实际接受/输出轨迹有波动时说明配对和可比性，不假称完全一致；不强制重型profile。配置/cache/部署/MTP轨迹等影响无法隔离时标`MIXED / INCONCLUSIVE`，不算代码Gain或PERF_KEEP。

收益类别分别记`code / configuration / deployment`，工作量或算法变化也单列比较边界。GMU、batch、max-num-batched-tokens、KV/HCCL buffer、端口、TP/DP/PP/DCP、MTP深度、Graph开关和实例数本身不计代码成果，只用于公平baseline、代码假说、可运行性和资源辨因；禁止无限参数扫描。

架构明确记录标准PD（`Request → P Prefill → KV Transfer → D Decode → Output`）或Full Replica / Complete-request Placement（各Replica完整Prefill+Decode），不可把后者叫PD优化。历史prefill-aware/prefill-work/work-seconds保留原裁决，最终产品采用与否在新记录中重新裁决。

后续checkpoint更新活动点索引、HANDOFF Current引用和必要复用知识；不每Run重写全部文档。旧Run保持原始身份，新归约写新文件并引用旧hash，不改写历史数据。Current只有一个权威证据引用。

## 5. 代码性能账本与Goal Review

[CODE_PERFORMANCE_LEDGER](CODE_PERFORMANCE_LEDGER.md)是后续工作结果的单一紧凑汇总，列为`Code Optimization | Type | Baseline | Patched | Gain | Correctness | Verdict`。Type为`PERFORMANCE / CORRECTNESS / FUNCTIONAL / INFRASTRUCTURE / DIAGNOSTIC / CONFIGURATION / DEPLOYMENT`；用现有优化点ID、code ref、matched Run和裁决引用，不复制raw。配置/部署单独归类，不能填PERF_KEEP；其他Type不能以代码量包装性能。只有PERF_KEEP算主要进展，无新增时必须写：`本阶段没有新增代码级性能 KEEP。`

checkpoint按[账本汇报规则](CODE_PERFORMANCE_LEDGER.md#重要checkpoint汇报)引用成果。出现[PLAN的Goal Review信号](PLAN.md#goal-review与checkpoint)即暂停惯性Run，按该节五问在当前点/新checkpoint记录证据、unknown与继续/转向/停止决定；必要时Astra Review，之后重新Reset。阈值和五问不在此重复。

PID/hash/epoch/controller/artifact/ownership记录只做到足够：实验真实性、资源安全、可复现及性能归因/correctness/KEEP所需。无判断价值的信息不无限追加；Run、commit、文档、Runtime模块和方向数量不衡量研究进展。

## 6. 当前stack、交互回归与产品Current

2026-10-07用户明确新增active research performance stack：correctness→matched A/B/A/B→可重复完整PD E2E通过的候选可以保留启用，并在该stack上继续研究。最终产品Current/PERF_KEEP另由完整API、正式SLA、稳定性、完整stack E2E裁决；未完成最终验收不要求清空研究stack或还原已验证有效patch。以下“仅正式PERF_KEEP”是正式产品表的口径。研究表引用原Run/代码身份、实际启用证据、独立比较与局限，不重写原裁决；新增patch比较旧研究stack对旧研究stack+patch，累计收益仍须重测，不能相加。

[CURRENT_PERFORMANCE_STACK](CURRENT_PERFORMANCE_STACK.md)仅列正式PERF_KEEP且当前产品仍active的patch/commit、代码路径、matched baseline、独立gain与启用状态。新patch比较“旧stack”对“旧stack+patch”的独立收益，同时确认完整stack的correctness/E2E；累计Product Gain须完整stack相对Stock/声明基线重测，不能将独立百分比相加。

旧patch在新架构失效、冲突、收益覆盖或回归时，在新状态记录`active / superseded / regressed`、原因与证据，退出active stack；原历史裁决保持不变。尚未验证新架构适用性时明确unknown，不能继续冒用旧Gain。REJECT/superseded实验模块退出后续默认路径，历史版本仍保留。

Current的唯一权威证据引用必须连同**产品代码commit（未提交差异另引用patch）、active stack、完整功能合同、标准workload、同口径E2E指标/噪声、最大剩余Gap**一起恢复。架构/功能/workload变化后旧指标适用性未验证则标Current unknown/待重验，不拿旧最快Run证明新代码。规则commit或功能VALID不建立产品Current。

## 7. 双轨backlog、模块约束与完成

Product correctness lane处理API/Tool/Responses/Grammar/状态恢复/错误协议/bug，Performance lane处理Scheduler/MTP/Graph/PD/KV/通信/CPU与framework/device idle。correctness问题若同时不阻塞目标workload、不影响性能代码合法性、不影响发布功能门槛，写到当前点的correctness backlog（问题/证据/影响/重入条件），不升级性能主线。明确阻塞产品完整性或性能裁决才暂停性能lane，记录最小修复范围；功能完整性仍是发布门槛。

新增Runtime模块前简答替代谁、为何不能现有模块实现、REJECT如何退休；记录唯一current product path与最小入口，不靠版本序列堆模块。交付时按[PLAN的阶段性完成条件](PLAN.md#阶段性完成条件)核验并引用证据。
