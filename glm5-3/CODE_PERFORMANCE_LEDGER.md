# GLM代码性能账本

当前产品目标为GLM-5.3 W8A8标准PD，权重`/data/tiankuan/wio/GLM-5.3-w8a8`、新服务名`glm-53`。2026-10-06用户上传中，尚无新模型完整E2E基线、Current或PERF_KEEP；历史模型候选/裁决保留实际版本，不迁移收益或正确性。

从checkpoint158之后建立（2026-10-06；父commit `e378d459e08e9f12300118e158bba16db6c4d04f`），只汇总后续代码候选的证据引用，不迁移或重写历史Run、raw evidence和裁决。记录规则见[RECORDING](RECORDING.md)，研究规则见[AGENTS](AGENTS.md)。

在`df7399c28791`上增量采用v2规则，增加Type与PERF_KEEP判据；仅改变后续记账方式，不追溯把历史guard/MQ/affinity/配置变化重分类成新的裁决。

本阶段没有新增代码级性能 KEEP。

checkpoint159已完成实际源码与原始证据研究；[Sol裁决](records/points/GLM-OPT-0002/research/engine_commit_20261006/H2_DEVICE_DECISION.md)与[Reset](records/points/GLM-OPT-0002/research/engine_commit_20261006/POST_REVIEW_RESET.md)。Current=None，stack为空；最大可由代码消除的Gap/time unknown，不能把阶段分离或必要广播等待填成Gain。

## 后续工作与代码候选

| Code Optimization | Type | Baseline | Patched | Gain | Correctness | Verdict |
| ----------------- | ---- | -------- | ------- | ---- | ----------- | ------- |
| GLM-OPT-0002 native FIFO/future/PP observer与机械归约 [code](records/points/GLM-OPT-0002/research/engine_commit_20261006/commit_trace.py) | DIAGNOSTIC | not applicable | Run242 host + Run244 offline device | not applicable；无产品Gain | 6项CPU native语义；4×64 token commit/SSE；16 rank真实事件；范围有限 | KEEP仅诊断工具，原服务恢复，未进stack |
| GLM-OPT-0002 H1 ready-useful commit优先级候选（未实现） | PERFORMANCE | Run242诊断，非matched E2E | none | unknown | [原始关联裁决](records/points/GLM-OPT-0002/research/engine_commit_20261006/H1_DECISION.md) | REJECT限已观测窗口，未实施drain-first |
| GLM-OPT-0002 H2 phase-cohort admission条件候选（未实现） | PERFORMANCE | Run244观察，非matched E2E | none | unknown | 实际stage/broadcast；并行合法性/拆批成本未证 | [INCONCLUSIVE/PARKED](records/points/GLM-OPT-0002/research/engine_commit_20261006/H2_DEVICE_DECISION.md) |
| GLM-OPT-0002 H3 PD completion host观测/reducer（opt-in，未接现役） | DIAGNOSTIC | none | 无性能patch/设备Run | unknown | [8CPU检查/真实安装类与166 Python](records/points/GLM-OPT-0002/jobs/PD-DIAGNOSTIC-CPU-20261006/result.json)；[源码/raw与限制](records/points/GLM-OPT-0002/research/pd_critical_path_20261006/CHECKPOINT160.md) | KEEP仅诊断；性能假说INCONCLUSIVE，不进Current/stack |

上表区分诊断工程保留、未实现性能候选与PERF_KEEP；诊断KEEP不算主要性能进展。每行用优化点ID/简短改动及code/diff引用、工作Type；性能Baseline/Patched引用同合同matched Run、完整E2E指标/单位；Gain写公式、重复/波动/范围；Correctness和Verdict链接验证及Sol裁决。非性能工作无matched性能数据时写not applicable/unknown，不发明Gain。缺性能对照或完整E2E标INCONCLUSIVE，执行无效标INVALID，归因混合标MIXED / INCONCLUSIVE；不能以局部TPS代填E2E。

| Type | 含义与分类示例（非历史重裁决） |
| ---- | ---- |
| PERFORMANCE | 减少产品执行浪费，例如Scheduler经对照证明减少实际wall |
| CORRECTNESS | 合法性/bug修复，例如PP empty token guard；不自动是性能优化 |
| FUNCTIONAL | 产品接口/功能能力，例如Responses affinity涉及功能和Runtime correctness，主类型按本次目标明确 |
| INFRASTRUCTURE | 执行/控制基础设施，例如Atomic MQ；不自动是性能优化 |
| DIAGNOSTIC | 观测、profile、根因区分；不是性能patch KEEP |
| CONFIGURATION | GMU/batch/token budget/Graph开关等变化 |
| DEPLOYMENT | 实例/并行布局/部署与资源变化 |

一个组合改动可注明关联类型，但性能主张必须单独隔离并满足PERFORMANCE判据，不把功能或工程KEEP混入性能进展。

对越小越好的完整wall，Gain=`Baseline / Patched - 1`表示吞吐提升（相同有效工作量）；若报告wall缩短则为`1 - Patched / Baseline`，明确口径。对达标稳定容量/有效吞吐，Gain=`Patched / Baseline - 1`。Baseline为0或unknown时不计算百分比。SLO与功能合同必须同时满足，有限包TPS不能替代稳定服务容量。

只有`Type = PERFORMANCE`、correctness保持、matched A/B与真实完整E2E Product Gain满足噪声感知重复标准才能得到**PERF_KEEP**。至少两次可比较matched A/B，或足够长且事先定义验收标准的稳定服务对照窗口；gain接近运行波动时用A/B/A或B/A/B。配置/cache/部署/MTP轨迹影响无法隔离时MIXED / INCONCLUSIVE，不能算代码Gain。无完整E2E不得提升Current。Full Replica / Complete-request Placement与标准PD分别标记；历史调度是否进产品写新裁决引用。

PERF_KEEP不代表永久有效：新架构的应用状态另记`active / superseded / regressed`及证据，不改原裁决。[CURRENT_PERFORMANCE_STACK](CURRENT_PERFORMANCE_STACK.md)只引用当前active KEEP；新的patch除独立matched收益，还要证明旧stack+新patch整体成立，累计Product Gain从完整stack对声明基线重测，不相加百分比。

## 配置与部署收益

配置/部署结果在对应点保留Run引用，账本Type标CONFIGURATION/DEPLOYMENT，不填PERF_KEEP。GMU/batch/token budget/KV/HCCL buffer/端口/并行比例/MTP深度/Graph开关/实例数量本身不算代码成果；随patch一起变化须隔离代码对照，否则MIXED / INCONCLUSIVE。功能、correctness与基础设施KEEP也不能包装成性能KEEP。

## 重要checkpoint汇报

先报新增代码KEEP与每项E2E Gain，再报当前最大剩余Gap和下一最高价值代码问题。没有新增KEEP必须逐字写“本阶段没有新增代码级性能 KEEP。”未知Gap写unknown及缺失证据。Goal Review结果引用当前优化点或新checkpoint，不修改旧next_action/裁决，也不以Run数、功能测试数或Runtime模块数包装性能进度。

默认一个active性能假说加一个区分diagnostic，切换前关闭/park/显式停止旧方向；新会话或强制Goal Review后先Reset，每个真实性能Run前先写决策价值。Goal Review阈值统一见[AGENTS](AGENTS.md#8-强制goal-review)；本账本不另造候选队列、计数服务或流程系统。Current同时引用产品代码commit、active stack、功能合同、标准workload、E2E指标和剩余Gap，不复用已不同代码的旧最快数字。
