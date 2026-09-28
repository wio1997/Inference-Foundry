# Inference Foundry Method

> 版本：Methodology V0.2（中文版）  
> 作用：定义一套可以跨模型、跨硬件复用的“极限推理 Runtime 生产方法”。  
> 第一个验证案例：DeepSeek V4 Flash W4A8 + 8×Ascend 910B3 + DP1×TP8 + DSpark7。  
> 当前校准 workload：32K 输入 → 1K 输出，concurrency 12。

---

# 1. Inference Foundry 到底是什么

Inference Foundry **不是一套万能推理框架代码**。

它真正要沉淀的是：

> **拿到一个具体模型 + 一套具体硬件 + 一个具体 workload 后，怎么系统地找出推理执行的理论/可实现性能上限，并一步步重构 Runtime 去逼近这个上限。**

因此：

- DeepSeek Extreme 可以非常 DeepSeek-specific；
- 以后 Qwen Extreme 可以重新写；
- GLM Extreme 也可以重新写；
- 不要求未来模型复用 DeepSeek 的 Runtime 代码；
- 但必须尽量复用同一套分析、建模、实验、验证和决策逻辑。

Foundry 的核心闭环是：

**冻结合同（Contract）  
→ 建立当前性能（Current）  
→ 恢复真实执行图（Execution DAG）  
→ 区分必要工作（Necessary Work）  
→ 建立资源模型（Resource Model）  
→ 计算理论边界（Bound）  
→ 找最大差距（Gap）  
→ 重构 Runtime  
→ 正确性验证  
→ 正式 E2E  
→ 重新计算 Bound**

---

# 2. 为什么先做 DeepSeek

DeepSeek Extreme P0 是 Inference Foundry 的第一个 **Calibration Case（校准案例）**。

当前先冻结：

- DeepSeek V4 Flash
- W4A8
- 8×Ascend 910B3
- DP1×TP8
- DSpark7
- 32K 输入
- 1K 输出
- concurrency 12

原因不是因为以后所有模型都必须这样。

而是因为第一轮必须先把变量控制住，才能：

1. 稳定复现 Current；
2. 准确恢复真实 Execution DAG；
3. 证明哪些依赖是真的、哪些只是框架写法；
4. 建立可信的 Resource Model；
5. 计算和收紧 Bound；
6. 做因果明确的 Runtime 改动；
7. 跑通完整的一次 Foundry 闭环。

因此：

> **32K→1K c12 是用来验证方法的第一个案例，不是 Inference Foundry 的永久架构假设。**

---

# 3. 最重要的原则

不要问：

> “哪段代码看起来最慢？”

而应该问：

> **在模型语义、当前 primitive 成本、硬件资源和冻结 workload 都不变的情况下，最好的合法执行方式是什么？当前 Runtime 为什么到不了那里？**

所有优化都应该能串成：

**观察到的浪费  
→ 因果解释  
→ 对 Bound 的影响  
→ 可验证的 Gap  
→ 实现  
→ correctness  
→ formal E2E**

如果一个局部优化无法证明它约束了完整 Product 的 critical path，就不能直接当成产品收益。

---

# 4. Foundry 标准流程

## Step 1：冻结测试合同（Freeze Contract）

先固定所有用于公平比较的条件。

至少包括：

### 模型合同

- 模型架构；
- 权重/commit/hash；
- 量化格式；
- speculative decoding 算法；
- sampling 语义。

### 硬件合同

- 加速卡型号和数量；
- 拓扑；
- 显存；
- Driver / CANN / CUDA / runtime 版本。

### 并行合同

- DP / TP / EP / PP；
- 通信组；
- 通信拓扑。

### Workload 合同

- 输入长度或数据集；
- 输出长度；
- concurrency；
- prefix cache 条件；
- request 数量；
- warm/cold 协议；
- sampling 参数。

### Correctness 合同

- 模型语义；
- state；
- KV；
- speculative acceptance；
- request completion；
- API/output contract。

### Performance 合同

- TTFT；
- TPOT；
- Output TPS；
- Product wall；
- 重复次数；
- noise floor；
- KEEP / REJECT / INCONCLUSIVE 规则。

### 产物

形成一个清晰的 `CONTRACT`。

以后换模型时，重新填这个合同，但方法不变。

---

## Step 2：建立可信 Current

优化前先回答：

> **现在到底有多快？**

要得到可重复的当前值，包括：

- Output TPS；
- TTFT；
- TPOT；
- Product 总耗时；
- Runtime cycle 数；
- useful tokens / cycle；
- acceptance；
- Host 时间；
- Device 执行；
- Communication；
- Graph；
- Metadata / State；
- KV；
- idle / bubble；
- request completion。

这个结果叫：

> **Current Witness**

不能拿一次偶然跑出来的高 TPS 当 Current。

---

## Step 3：恢复真实执行图（Recover Execution DAG）

不要只看源码模块名字。

要恢复真实发生的：

- Host preparation；
- enqueue / launch；
- Target；
- Attention；
- MoE；
- HCCL；
- Acceptance；
- State Advance；
- KV mutation；
- DSpark proposer；
- Metadata preparation；
- Output publication。

然后给依赖关系分类。

例如：

### 必须依赖

A 的结果不出来，B 就不能运行：

`A → B`

这是模型/状态语义要求。

### 当前框架顺序

A 和 B 没有真正数据依赖，只是代码里先调用 A 再调用 B。

这种边可能只是：

> 当前实现制造出来的串行。

持续区分：

- `E_current`：现在代码实际强制的顺序；
- `E_must`：真正不能打破的顺序。

两者之间的差，就是潜在 Scheduling Freedom。

---

## Step 4：区分 Necessary Work 和 Framework Work

所有重要操作都要问：

> **这是模型必须做的，还是当前框架额外制造出来的？**

### Necessary Work（必要工作）

例如：

- 必要 GEMM；
- 必要 Attention；
- 必要 KV read/write；
- 必要 TP/EP communication；
- 必要 speculative verification；
- 必要 State Transition。

这些不能因为耗时高就简单删掉。

### Framework Work（可能的框架额外成本）

例如：

- 重复 Python object；
- Host↔Device scalar sync；
- 重复 metadata rebuild；
- 多余 dispatch；
- 无数据依赖的串行；
- Scheduler 每 cycle 介入；
- 不必要 Graph boundary；
- 多余 copy；
- redundant state mirror。

注意：

> **昂贵 ≠ 可删除。**

必须有 dependency / resource / correctness 证据。

---

## Step 5：建立资源模型（Resource Model）

只知道“没有依赖”还不够。

例如：

A 和 B 没有数据依赖。

理论上：

`A || B`

可以并行。

但实际可能同时抢：

- AICore；
- Vector；
- HBM bandwidth；
- Copy Engine；
- HCCL；
- Workspace；
- Stream/Event 资源。

结果：

同时跑反而更慢。

所以必须测：

- primitive cost by shape；
- resource occupancy；
- overlap 是否真的成立；
- contention 情况；
- cost interval；
- uncertainty。

核心原则：

> **Dependency Freedom ≠ Resource Freedom**  
> 没有依赖，不代表一定可以免费并行。

---

# 5. 三个性能数字必须分开

这是 Foundry 最重要的纪律之一。

## 5.1 Current

当前已经真实实现并经过正式验证的性能。

例如：

`571.681 tok/s`

---

## 5.2 Optimistic Bound（乐观理论边界）

假设：

- 只保留必要依赖；
- 资源调度非常理想；
- 所有可删除框架顺序都消失；
- primitive 本身暂时不变。

数学上得到的最好值。

例如：

`Current 86s`

理论最乐观：

`64s`

那么：

`49152 / 64 ≈ 768 tok/s`

但这个 **768 不是承诺值**。

它只是：

> “数学上目前还没有证据排除的最好情况。”

---

## 5.3 Achievable Bound（可实现边界）

通过实验去验证 Optimistic Bound 中的关键假设。

例如理论认为：

`A + B = 15ms + 5ms`

完全 overlap 后：

`15ms`

但真实测试发现同时运行产生 HBM contention：

`18ms`

那么理论就要修改。

Bound 可能从：

`768 tok/s`

收紧成：

`710 tok/s`

这个更接近工程可实现极限。

---

# 6. Step 6：计算 Optimistic Bound

基于：

- `E_must`；
- 当前 primitive cost；
- 理想合法调度；
- resource lower bound；
- request release/output 约束；

计算一个最乐观 makespan。

如果：

- 总输出 token = `N`
- 最乐观时间 = `L_opt`

则：

`TPS_opt = N / L_opt`

这个数字的意义：

> **告诉我们“如果框架组织完美，理论上最多还有多少空间”。**

未知边、未知成本必须明确标记，不能偷偷当不存在。

---

# 7. Step 7：通过实验收紧 Achievable Bound

理论不是最终答案。

要针对最重要的假设做实验。

例如：

- Target 与 HCCL 能不能 overlap？
- overlap 会不会抢 HBM？
- metadata 能不能提前一 cycle？
- Host issue 能不能提前？
- state 能不能常驻 Device？
- Graph boundary 能不能消失？
- HCCL 长 duration 是真实通信还是 peer wait？
- DSpark→Target 是否真的必须串行？

每个实验都会：

> 修正 DAG 或 Resource Model。

最终把：

`Optimistic Bound`

收紧成：

`Achievable Bound`

这就是“理论和实践一起走”。

---

# 8. Step 8：找最大的 Causally Removable Gap

比较：

`Current`

和：

`Achievable Bound`

例如：

`Current = 86s`

`Achievable = 73s`

那么：

`Gap = 13s`

然后继续问：

> 这 13 秒主要在哪里？

可能发现：

- Host issue；
- rank arrival skew；
- Graph boundary；
- metadata preparation；
- communication wait；
- scheduler；
- output drain。

不要因为某个函数 profiler 里最大，就自动优化它。

必须是：

> **它确实限制 critical path，而且有可消除空间。**

---

# 9. Step 9：按照 Gap 重构 Runtime

这个时候才真正大改代码。

优化边界不应该被当前类结构限制。

可以跨：

- Scheduler；
- ModelRunner；
- Metadata；
- State；
- Graph；
- Communication；
- Target；
- DSpark。

可能的手段包括：

- staticization；
- stable buffer；
- Device-side state；
- graph replay；
- full / segmented graph；
- persistent execution；
- pipeline；
- prefetch；
- overlap；
- Scheduler hot-path removal；
- Host sync removal；
- cross-op fusion。

核心原则：

> **Runtime DAG 决定优化边界，不是原框架模块名决定优化边界。**

---

# 10. Step 10：Correctness + Formal E2E

任何候选都必须先过 correctness。

包括必要时检查：

- tensor parity；
- rank parity；
- KV；
- state；
- acceptance；
- model semantics；
- request completion；
- API/output。

然后再跑正式冻结 E2E。

最终只有三种结果：

- `KEEP`
- `REJECT`
- `INCONCLUSIVE`

以下不能独立决定 KEEP：

- microbenchmark；
- profile；
- diagnostic TPS；
- internal accepted-token TPS。

---

# 11. Step 11：重新计算 Bound（Re-bound）

一次优化成功以后，瓶颈会移动。

例如：

原来：

`Host → 最大瓶颈`

解决以后可能变成：

`Communication → 最大瓶颈`

所以不能一直沿着旧 Performance Map 做。

必须重新：

- Measure Current；
- Recover DAG；
- Update Resource Model；
- Recompute Bound；
- Find New Gap。

这就是：

> **Re-bound**

---

# 12. Step 12：循环

不断执行：

**Current  
→ Bound  
→ Gap  
→ Intervention  
→ E2E  
→ Re-bound**

直到：

1. 剩余 Gap 很小而且解释清楚；
2. 剩余主要是当前冻结 primitive/operator 成本；
3. 剩余被硬件/resource lower bound 限制；
4. 继续投入不值得；
5. 或者应该换下一个 calibration case。

---

# 13. 为什么不能只做理论

只做理论可以得到：

> Optimistic Bound

但是很多关键问题无法仅靠源码证明：

- overlap 是否真的成立；
- HBM 是否冲突；
- Host 是否真的 exposed；
- Graph 是否能覆盖；
- communication duration 是否包含 peer wait；
- Device-side state 是否真的可行。

所以：

> **纯理论只能给一个可能很松的上限。**

要得到可信的 Achievable Bound，必须做实验。

---

# 14. 为什么不能只做代码优化

如果只做：

- 这里快 3%；
- 那里快 5%；
- 再优化一个 kernel；

你不知道：

> 当前已经接近极限了吗？

例如做到：

`650 tok/s`

但不知道理论上是：

`670`

还是：

`900`

就无法判断该继续还是换方向。

所以：

> **没有 Bound 的优化，很容易变成无目标的局部调优。**

---

# 15. Theory 和 Implementation 必须一起推进

Foundry 的正确方式：

```text
Measure Current
      ↓
建立/修正理论模型
      ↓
找到最大 Gap
      ↓
设计最小决定性实验
      ↓
真实运行
      ↓
修正 DAG / Resource Model
      ↓
收紧 Bound
      ↓
做结构性改动
      ↓
Formal E2E
      ↓
Re-bound
      ↺
```

一句话：

> **理论告诉实验该测什么，实验告诉理论哪里错了。**

---

# 16. DeepSeek P0 最终要沉淀什么

不能只留下：

- `fixed_decode.py`
- c12 Graph
- DeepSeek metadata
- DeepSeek DSpark 逻辑

这些都是产品代码，未来模型未必能复用。

真正需要沉淀的是下面这些。

---

## 16.1 Contract Schema

统一描述：

- model；
- hardware；
- quantization；
- parallelism；
- workload；
- correctness；
- benchmark protocol。

以后新模型只需要重新填写合同。

---

## 16.2 Evidence Taxonomy

所有数字明确属于：

- formal E2E；
- matched internal A/B；
- diagnostic；
- profile；
- microbenchmark；
- simulation；
- theoretical bound；
- independent review。

防止：

`profile 10ms`

最后莫名其妙变成：

`产品可以节省 10ms`

---

## 16.3 Execution DAG Schema

统一定义节点：

- Host Issue；
- Device Compute；
- Communication；
- Copy；
- State Update；
- Graph Boundary；
- Output。

统一定义边：

- Value Dependency；
- State Version；
- Collective Order；
- Buffer Lifetime；
- Resource；
- Current-order-only；
- Unknown。

模型变了，DAG 内容会变。

DAG 的表达方法不变。

---

## 16.4 Resource Model Schema

统一描述：

- primitive cost；
- shape；
- resource occupancy；
- overlap；
- contention；
- uncertainty；
- confidence。

不同模型重新采数据即可。

---

## 16.5 Bound Solver

长期目标是让工具能输入：

- DAG；
- primitive cost；
- resource constraint；
- workload ledger；

输出：

- Current makespan；
- Optimistic Bound；
- Achievable interval；
- Critical Path；
- 最大 sensitivity；
- 最大 unknown。

这是 Foundry 最重要的跨模型工具之一。

---

## 16.6 Performance Knowledge

记录每个实验：

- 环境；
- shape；
- topology；
- hypothesis；
- local result；
- E2E result；
- correctness；
- 为什么成功；
- 为什么失败；
- 什么条件下值得重验。

以后不是直接复用结论。

而是：

> 复用 hypothesis prior。

---

## 16.7 Runtime Pattern Library

沉淀的是“模式”，不是万能 Runtime。

例如：

- Stable Buffer；
- Device-side State；
- Graph Replay；
- Metadata Prefetch；
- Host-free Decode；
- Persistent Loop；
- Communication Overlap；
- Continuous Slot；
- Output Drain Separation。

新模型根据自己的 DAG 决定用哪些。

---

# 17. 推荐验证顺序

## Phase A：DeepSeek P0

当前先只做一件事：

> **把第一条 Foundry 闭环完整跑通。**

至少完成：

1. Current；
2. Execution DAG；
3. Framework/Scheduling Optimistic Bound；
4. 实验收紧 Achievable Bound；
5. 找最大 Gap；
6. 做至少一个 Bound-guided structural change；
7. correctness；
8. repeated formal E2E；
9. Re-bound。

在这之前：

> 不要因为未来通用性主动扩大 workload。

---

## Phase B：同一个 DeepSeek，换第二个 workload

第一条闭环结束以后，换一个明显不同的场景。

例如：

- 更长 context；
- 不同 concurrency；
- 不同 prefix-cache 条件。

然后重新按 Foundry Method 跑一次。

目的不是单纯再提性能。

而是验证：

> **第一次总结出来的是 DeepSeek/Foundry 规律，还是仅仅是 32K/c12 的偶然规律？**

---

## Phase C：换第二个模型

之后再选：

- Qwen；
- GLM；
- 或其他目标模型。

不要要求复用 DeepSeek Runtime。

重新：

**Freeze Contract  
→ Current  
→ DAG  
→ Bound  
→ Gap  
→ Runtime  
→ E2E  
→ Re-bound**

应该复用：

- schema；
- tools；
- bound solver；
- evidence；
- knowledge；
- agent workflow。

不要求复用：

- DeepSeek class；
- DeepSeek Graph boundary；
- DeepSeek fixed shape；
- DeepSeek scheduling assumption。

如果还能跑通：

> Inference Foundry 才真正从 DeepSeek 项目变成跨模型方法。

---

# 18. DeepSeek P0 什么情况下算成功

P0 必须同时得到两个结果。

## A. 产品结果

一个：

- 正确；
- 可运行；
- 性能显著改善；
- 明显更接近 Achievable Bound；

的 DeepSeek Extreme Runtime。

---

## B. Foundry 结果

能够明确回答：

1. Current 为什么是现在这个值？
2. Optimistic Bound 怎么算出来？
3. 哪些 Bound 假设只是理论？
4. 哪些被实验否定？
5. Achievable Bound 怎么收紧？
6. Current 到 Bound 最大 Gap 是什么？
7. 为什么选择这个 Gap？
8. Runtime 改了什么？
9. Formal E2E 提升多少？
10. Re-bound 后还剩多少？
11. 哪些逻辑可以给下一 workload / 模型复用？

如果只有 A：

> 这是一个成功的 DeepSeek 优化项目。

如果 A + B 都有：

> **这是第一个成功的 Inference Foundry Case。**

---

# 19. 防止项目跑偏的规则

1. 不要把 32K/c12 变成永久架构真理。
2. 第一条 Bound-guided 闭环没完成前，不要急着扩通用 workload。
3. profiler duration 不等于 removable time。
4. microbenchmark gain 不等于 E2E gain。
5. 一旦某模块离开 critical path，就不要因为历史计划继续优化它。
6. 未来 Runtime 不要求复用 DeepSeek 代码。
7. 未来项目必须尽量复用 Foundry 的分析/证据/Bound 流程。
8. Bound 主要假设未闭合时，允许写“numerical bound unidentified”。
9. 超过 Stock 不代表项目结束。
10. 目标始终是缩小 Current 与 Achievable Bound 的 Gap，而不是增加优化点数量。

---

# 20. 一句话定义

> **Inference Foundry：先冻结一个真实推理合同，恢复它的真实执行，建立并通过实验收紧可实现性能边界，再按最大 Gap 重构 Runtime，通过 correctness 和正式 E2E 验证，最后把整个过程沉淀成下一模型可以重复使用的方法、工具和证据体系。**

