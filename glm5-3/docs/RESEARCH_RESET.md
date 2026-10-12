# 2026-10-12 研究审计与重构

结论：历史研究有真实收益与有用反例，不能整体否定。需要改变的是候选空间、实验入口和证据恢复方式。本次未连接服务器、复测或重新裁决旧 Run。

## 来源与范围

- 用户给出的 main 已通过远端 ref 核对为 `a1282c94026e3c8406531fcbf31758310c4bd350`。
- 本地已有研究分支提交 `86dbb61c787dc2d8ed143abbeee040bf29addff5` 含更完整的 Run / 六节点材料。以下相关引用固定到该提交；不把它当作当前部署或本次 main 已具备的源码。
- 原 main 的入口文档完整保存在 [legacy](../legacy/README.md)。补丁、fixtures、manifest 和证据原件未修改。

## 五个具体问题

### 1. 恢复入口互相矛盾

main 的 AGENTS 仍写当前用 GLM-5.2；HANDOFF 主体是 10-01 的 OPT0001/PLANNED/无 Run；README 和 10-07 发布包已指向 GLM-5.3 与 Run0256—0284。研究分支又含后续六节点 4P2D。

这不是证明实验无效，而是新 Agent 容易恢复错误目标、重复已完成工作或混用拓扑。[原 AGENTS](https://github.com/wio1997/Inference-Foundry/blob/a1282c94026e3c8406531fcbf31758310c4bd350/glm5-3/AGENTS.md)、[原 HANDOFF](https://github.com/wio1997/Inference-Foundry/blob/a1282c94026e3c8406531fcbf31758310c4bd350/glm5-3/HANDOFF.md)。本次只保留一个当前入口，未知现场身份明确 unknown。

### 2. 曾用框架支持边界替换产品目标

既有纠偏报告追踪了 Run108 的 D PP1 限制之后，Run109 转到本地 D、Run113 P idle、Run114 完整请求分配；Run244 核验的 resident 实例无 KV connector。这一段路线不满足用户标准 PD 要求。不能据此说全部历史 PD 都是假的：同一报告也保留了 Run101 的真实 PD 证据。

来源：[PD_SCOPE_CORRECTION](https://github.com/wio1997/Inference-Foundry/blob/86dbb61c787dc2d8ed143abbeee040bf29addff5/glm5-3/records/points/GLM-OPT-0002/research/engine_commit_20261006/PD_SCOPE_CORRECTION.md)。新合同固定模型与产品语义，框架支持限制只影响候选可行性，不能悄悄改变验收工作。

### 3. 主线仍受既有框架热点牵引

Run284 后的 Reset 仍将最大可删除 Gap 标 globally unknown，并区分初始 target 与 FULL 稳态；旧 eager starvation 的解释不能继续搬到 FULL。局部源码调查确实存在，但 main 没有可独立表达、执行和比较计划的实验内核。

来源：[RESET_PD_INITIAL_TARGET](https://github.com/wio1997/Inference-Foundry/blob/86dbb61c787dc2d8ed143abbeee040bf29addff5/glm5-3/records/points/GLM-OPT-0002/research/decode_path_20261006/RESET_PD_INITIAL_TARGET.md)、[GOAL_REVIEW_AFTER_RUN284](https://github.com/wio1997/Inference-Foundry/blob/86dbb61c787dc2d8ed143abbeee040bf29addff5/glm5-3/records/points/GLM-OPT-0002/research/decode_path_20261006/GOAL_REVIEW_AFTER_RUN284.md)。本次将必要 DAG 与额外执行顺序分开；新增内核目前只覆盖固定操作集合的计划，不宣称已找到最大 Gap。

### 4. 局部结论与产品验收没有闭合

[发布 manifest](../patches/20261007/manifest.json) 明确 formal_product_KEEP=false、Current=null。H6/H5 有限定收益；H11 正确但无可重复收益；H10 在 actual async=true 下 NOT_APPLICABLE；H12/H13 存在混杂或无重复收益。这些状态不能合并成一个“所有 patch 已验证”的 stack。

新流程分别验收工具、片段、原模型和服务；缺产品 SLA 不抹除局部事实，局部事实也不提升 Current。

### 5. 后续配置筛选没有证明系统收益

研究分支六节点报告中，dense 图、P MC2/buffer 变更未得到期望收益；各轮不同时间/文本/状态，不能把全部差值视为因果加速/退化。4096MB buffer 依赖导致每 engine KV 容量减少 27.02%，这是应保留的资源约束。

来源：[graph-prefill-mc2 REPORT](https://github.com/wio1997/Inference-Foundry/blob/86dbb61c787dc2d8ed143abbeee040bf29addff5/glm5-3/pd-serving/baselines/4p2d/experiments/graph-prefill-mc2-20261008/REPORT.md)。不把这些配置继承为新执行器默认值。

## 必须保留的成果

| 证据 | 原记录证明的事实 | 边界 |
|---|---|---|
| H6 / Run256 | 同 worker、common rebuilt binary 两对相同 23-token PD，D wall 降低 14.36% / 14.09% | 对应历史条件；不是 stock production binary 或新拓扑收益 |
| H5 / Run261 | H6 基线上两对 D wall 降低 6.98% / 6.94% | 与 H6 不能相加；不是完整 stock→stack / 80K600 SLA |
| H11 / H12 / H13 | 实现身份、限域正确性和未重复收益/混杂反例 | 不自动进入 active 性能栈 |
| F1/F2 | KV logical/physical block 错误身份、空 SSE 终态修复经验 | 新状态机需重新证明，不盲目移植 |
| AISBench adapter | fresh salt、prefix/external 命中分账、有效输出/SLO 记账 | 用于后续服务验收，不替代独立执行器 |

H6/H5 数字来自[发布 manifest](../patches/20261007/manifest.json)及所引用的冻结 evidence。本次没有重新测量，也没有累计百分比。

另一值得保留的机制线索是 [q1→全局 NONE](https://github.com/wio1997/Inference-Foundry/blob/86dbb61c787dc2d8ed143abbeee040bf29addff5/glm5-3/pd-serving/issues/pd-q1-global-none/README.md)：该记录把本地 graph eligibility 与跨 DP 模式 min 耦合联系起来。它不是修复已成功的证据，也不能用差分时间直接作为可回收预算。

## 本次实际改变与未完成部分

1. 保存旧入口原件，重写当前目标/恢复/证据合同，保留标准 PD 和动态服务验收。
2. 新增与框架无关的执行计划内核：显式必要依赖、额外顺序、资源约束估计和 CPU callback 回放；非法依赖、容量与未知成本直接拒绝。
3. 用明确 synthetic 的例子测试工具；固定成本/贪心排程仅筛选，不称硬件下界或 NPU 实测。
4. 建立 [GLM-PLAN-0001](../studies/GLM-PLAN-0001.md)，将真实输入、拓扑、成本与替身校准缺项写清楚。

尚无真实 GLM 片段接入、Ascend adapter 或新的模型性能收益。后续应补真实片段并验证机制，而不是将本次 toy 的数字包装为成果或再次扩建代理/控制平台。

## 本地验证

- `python3 -m unittest discover -s glm5-3/tests -p 'test_*.py' -v`：19 项通过，包含 16 项新内核测试及 3 项既有交接工具回归。
- demo 与 JSON estimate 入口成功，结果明确标 synthetic / screening_only / CPU correctness_only。
- 独立审查以 1000 个随机 DAG 检查依赖和资源峰值；另修复可变 callback 输出 alias 掩盖差异的问题，并加入固定回归。
- 13 份当前入口文档的本地链接检查通过；9 份归档快照逐字节匹配原提交。
- 发布包 15 份 patch 和 fixtures 的 SHA256 保持一致；旧 records、adapter、runtime 和发布证据未改写。

这些是本地工具与迁移检查，不是模型正确性或性能验收。
