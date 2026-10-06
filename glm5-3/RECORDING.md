# 优化点、实验与Git提交规范

从checkpoint158之后按[AGENTS](AGENTS.md)执行。所有历史Run、raw evidence、历史裁决及旧point/next_action保持原样；新的解释、产品取舍或重新裁决另写新记录并引用原证据，不追溯改账。

GitHub保存权威代码/配置、事实摘要、裁决和可追溯证据。运行机器保存大raw artifact，Git记录稳定位置、大小、SHA256及关键归约；不提交权重、大trace或凭据。文件数量不作为进度。

## 1. 一点一身份，一次执行一个Run

- 优化点：`GLM-OPT-NNNN`，按问题/机制划分，允许成套重构与多个相关实验。摘要放`records/points/<ID>/point.md`，索引`records/points.jsonl`。
- Run：`GLM-RUN-NNNN`，记录真实执行的代码、合同、命令、观察器、结果与证据。默认在优化点下`runs/<Run>/`，不同配置/采集不得伪装同一次执行。
- 沿用根`scripts/taskctl.py`的Task/Loop/Run、compare和resume机制；若采用它，其目录为唯一权威路径，点索引只引用，避免复制第二份状态。先恢复真实源码目录，勿用假路径初始化；其算子字段/固定Loop习惯不限制GLM研究。

初始点允许PLANNED且没有Run；未执行不填性能。索引仅保留ID、标题、类别、状态、权威路径和关键Run/Current引用。

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

- ID/优化点、时间、kind（formal_e2e/diagnostic/profile/microbenchmark/simulation等）、实际执行状态。
- 框架/source repo与commit；有未提交改动则记录base+patch artifact/hash及关键文件hash。代码、镜像、模型/量化、设备/rank/部署和effective config身份。
- 实际命令/cwd/内层exit/超时；负载、到达、采样/MTP/Graph/cache条件及相关有效工作量。比对Run和可比性说明。
- 风险相称的功能/状态验证、完整请求/有效输出、吞吐/延迟/错误与观察器开关。缺失填null/unknown，不臆造。
- artifact路径/大小/hash、关键日志归约、必要的清理/恢复与活动资源状态。

短`summary.md`只写结果、因果证据、限制、裁决及下一问题；无需重复manifest。已失效、超时或缺产物保留`INVALID`，不归为性能成功/失败。指标可增减，足以复现和裁决即可。

## 4. 提交与裁决

提交按可审阅变化组织，不每个编辑都commit，也不强制一轮一个机制：

```text
glm(opt): GLM-OPT-0007 describe final code/config change
glm(exp): GLM-OPT-0007 GLM-RUN-0012 record evidence and verdict
glm(knowledge): record applicability and revalidation conditions
```

代码提交与实验记录相互链接；Run固定被测code ref，记录提交固定该Run证据。推送前核对实际diff，不夹带他人改动；研究分支按任务选择，Git更新不force覆盖并发工作。

裁决分`KEEP / REJECT / INCONCLUSIVE / INVALID / REUSED / NOT_APPLICABLE`。**没有matched A/B的代码不算性能成果；没有真实完整E2E的结果不得提升Current。** 代码级KEEP需要功能/状态有效、同口径matched A/B及可重复的完整E2E Product Gain；微测/profile/模拟/内部TPS/功能VALID只支持各自结论。REJECT写原因、未知与值得重验条件，不永久否定整个方向。

Matched A/B至少匹配模型/算子/资源、负载与到达/有效工作量、功能与SLO合同、采样/MTP/cache/Graph相关条件及观察器，明确baseline和patched的code/config身份与目标patch。实际输出/接受轨迹有随机波动时说明配对/重复及可比性，不假称完全一致；按噪声选择重复、顺序或交叉对照，不强制每Run重型profile。配置/部署随patch变化须补能隔离代码收益的对照；不能隔离就标混合收益INCONCLUSIVE，不算代码KEEP。

收益类别分别记`code / configuration / deployment`，工作量或算法变化也单列比较边界。GMU、batch、max-num-batched-tokens、KV/HCCL buffer、端口、TP/DP/PP/DCP、MTP深度、Graph开关和实例数本身不计代码成果，只用于公平baseline、代码假说、可运行性和资源辨因；禁止无限参数扫描。

架构明确记录标准PD（`Request → P Prefill → KV Transfer → D Decode → Output`）或Full Replica / Complete-request Placement（各Replica完整Prefill+Decode），不可把后者叫PD优化。历史prefill-aware/prefill-work/work-seconds保留原裁决，最终产品采用与否在新记录中重新裁决。

后续checkpoint更新活动点索引、HANDOFF Current引用和必要复用知识；不每Run重写全部文档。旧Run保持原始身份，新归约写新文件并引用旧hash，不改写历史数据。Current只有一个权威证据引用。

## 5. 代码性能账本与Goal Review

[CODE_PERFORMANCE_LEDGER](CODE_PERFORMANCE_LEDGER.md)是代码候选结果的单一紧凑汇总，列为`Code Optimization | Baseline | Patched | Gain | Correctness | Verdict`；使用现有优化点ID、code ref、matched Run与裁决引用，不复制Run/raw。配置和部署收益单列在对应点记录，不计代码KEEP。只有代码级KEEP算主要进展，无新增时必须写：`本阶段没有新增代码级性能 KEEP。`

重要checkpoint先报新增代码KEEP及各E2E Gain、当前最大剩余Gap、下一最高价值代码问题；unknown明确写。出现[AGENTS的Goal Review信号](AGENTS.md#8-强制goal-review)即暂停惯性Run，在新checkpoint/当前点简短记录五问、证据、unknown和继续/转向/停止决定，必要时触发独立Astra Review。旧next_action只作历史证据，不改写，也不机械继承。

PID/hash/epoch/controller/artifact/ownership记录只做到足够：实验真实性、资源安全、可复现及性能归因/correctness/KEEP所需。无判断价值的信息不无限追加；Run、commit、文档、Runtime模块和方向数量不衡量研究进展。
