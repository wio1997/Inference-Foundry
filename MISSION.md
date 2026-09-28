# DeepSeek Extreme P0

## Mission

为 Inference Foundry 做出第一款极限性能产品，并用它完整验证第一条可复用的 Foundry 极限优化闭环：

**DeepSeek Extreme P0**

当前固定目标：

- DeepSeek V4 Flash
- W4A8
- 8×Ascend 910B3
- DP1×TP8
- DSpark7
- Prefill / Decode 混部

目标不是单纯优化 vLLM-Ascend。

目标是：

> 在模型语义正确的前提下，重新研究 DeepSeek 在这套固定硬件上的完整执行方式，建立 Current、Execution DAG、Resource Model、Optimistic/Achievable Bound，并通过真实实现与 E2E 实验持续缩小现实性能与可实现性能边界之间的差距。

---

## P0 Calibration Contract

当前 `48×32K→1024, c12` 是 DeepSeek Extreme P0 的：

**Calibration Workload / Primary Performance Anchor**

它的作用是：

- 建立稳定、可重复的 Current；
- 恢复真实 Execution DAG；
- 区分模型必要工作与框架制造的顺序/开销；
- 建立 Framework/Scheduling-only Bound；
- 用实验校准理论假设；
- 按最大 Gap 重构 Runtime；
- 用 correctness + formal E2E 验证；
- 重算 Bound；
- 完整跑通一次 Inference Foundry Method。

该冻结 workload 是第一个受控案例，不代表未来 workload、未来模型或最终生成 Runtime 必须采用相同的 32K、c12、固定 cohort 或执行 shape。

P0 可以为该 calibration workload 做激进专项化；但所有重要结论必须区分：

- 固定模型/硬件导致的专项化；
- 当前 workload 导致的专项化；
- 可以复用于下一 workload / 下一模型的方法论与工具。

---

## 产品裁决与最终裁判

DeepSeek Extreme P0 在以下固定模型/硬件/算法合同下工作：

- DeepSeek V4 Flash W4A8
- 8×Ascend 910B3
- DP1×TP8
- DSpark7

正确性是硬约束。

正式、同口径、可重复的 E2E 是最终性能裁判。

Stock 是公平基线。

vLLM/vLLM-Ascend 是：

- 启动支架；
- correctness / semantic oracle；
- 算子和已验证执行语义来源；
- 公平 baseline；
- 必要时的 serving scaffold。

它们不构成最终性能架构边界。

每一个候选，无论是 Runtime、通信、调度、Graph、Host/Device control 还是未来的算子改动，都必须先证明语义和 serving 合约，再用相同协议的正式 E2E 判断 KEEP / REJECT / INCONCLUSIVE。

局部阶段变快、某个百分比门槛、一次高 TPS 或超过 Stock 都只是里程碑，不能代替对剩余 Gap 的持续审查。

---

## 当前阶段：先完成 Framework / Scheduling 极限闭环

当前阶段暂不以“更快 primitive/operator”为主要变量。

优先固定当前正确 primitive/kernel 及其 shape-conditioned cost，研究：

- execution order；
- Host issue；
- async queue；
- Graph organization；
- buffering；
- prefetch；
- metadata/state preparation；
- Scheduler / control plane；
- Host↔Device synchronization；
- communication/compute overlap；
- pipeline；
- request scheduling；
- Device-side state；
- persistent execution。

目标不是猜一个 TPS，而是建立：

**Current → Optimistic Bound → Achievable Bound → Remaining Gap**

并通过实验逐步收紧 Bound。

理论模型和工程实现必须一起推进：

- 理论指出最值得验证的 Gap；
- 实验校准依赖、资源竞争和 overlap 假设；
- 实现验证新的执行结构是否真正可行；
- 正式 E2E 决定是否 KEEP；
- KEEP 后重新构建 Current / DAG / Bound。

---

## 第一阶段基础资产

### Baseline / Current

冻结并记录：

- 软件版本；
- Git commits；
- CANN / torch-npu；
- 模型和量化配置；
- 启动参数；
- benchmark workload；
- correctness protocol；
- noise floor。

得到稳定可复现的：

- TTFT；
- TPOT；
- Output TPS；
- Product wall；
- Runtime cycle count；
- useful tokens/cycle；
- acceptance；
- Prefill / Decode；
- Kernel / HCCL / Host profiling。

### Performance Map

建立当前性能地图：

- Prefill 时间分布；
- Decode 时间分布；
- Kernel 时间；
- HBM / 数据流；
- TP8 通信；
- Host / D2H / metadata；
- Kernel gaps；
- Framework overhead；
- idle / bubble；
- request completion；
- 当前最大的性能未知和性能机会。

### Execution DAG

恢复真实执行，而不是仅看源码模块：

- value dependency；
- state-version dependency；
- collective order；
- buffer lifetime；
- resource ordering；
- Host issue / enqueue / native start；
- current-order-only；
- unknown edges。

持续区分：

- `E_current`
- `E_must`

### Resource Model

测量：

- primitive cost by shape；
- resource occupancy；
- overlap legality；
- resource contention；
- uncertainty / confidence。

关键原则：

> Dependency freedom ≠ resource freedom.

### Achievable Bound

逐步估算和收紧：

- 必要计算成本；
- 必要权重/HBM读取；
- 必要 KV 读取；
- 必要通信；
- 必要依赖；
- 资源竞争；
- 可以重叠的部分；
- 当前无法证明的未知边/成本。

得到：

**Current Runtime vs Optimistic Bound vs Achievable Bound vs Remaining Gap**

---

## 优化方式

不预设第一刀。

当前允许的 Framework/Scheduling 方向包括：

- Graph / replay；
- persistent execution；
- Host issue / enqueue 重构；
- communication/compute overlap；
- metadata/state overlap；
- stable buffers；
- Device-side state；
- Scheduler / ModelRunner hot-path removal；
- D2H/H2D 减少；
- KV / state ownership 重构；
- output drain 与 hot path 分离；
- Standalone Decode Runtime；
- 跨模块 execution restructuring。

未来如果证据表明 primitive/operator 已成为主要剩余 Gap，再进入：

- Ascend C Kernel；
- SuperKernel；
- 数据流融合；
- W4A8 专项实现；
- 权重布局；
- Attention；
- MoE；
- HCCL/MC2 等。

由证据决定，不按模块习惯决定。

---

## Framework Exit

持续区分：

**Compute Gap**

和

**Framework / Scheduling Gap**

如果主要剩余 Gap 来自 vLLM-Ascend 的执行组织，而且无法在其边界内合理消除，则建立 Standalone Layer / Decode Runtime，与原框架做公平性能对比。

如果独立路径显著更接近 Bound，则逐渐把性能 Hot Path 从通用框架中拿出来。

Serving、HTTP 等外围能力如果不在性能关键路径，不需要为了“自研”而重写。

---

## 最终产品

DeepSeek Extreme P0 必须是完整可运行、语义正确的 DeepSeek，而不仅是 microbenchmark。

最终需要比较：

- vLLM-Ascend Baseline；
- DeepSeek Extreme P0；
- Optimistic Bound；
- Achievable Bound；
- Remaining Gap。

并能够解释：

1. Baseline / Current 的主要浪费在哪里；
2. 哪些顺序/等待是模型必须的，哪些是框架制造的；
3. Optimistic Bound 如何得到；
4. 哪些理论假设被实验否定或收紧；
5. Achievable Bound 如何得到；
6. P0 消除了什么；
7. 为什么 P0 更快；
8. 当前还剩什么 Gap；
9. Bound 在 Runtime 重构后如何变化；
10. 哪些过程、工具和证据格式可以直接用于下一 workload / 下一模型。

长期目标是从制造 DeepSeek Extreme 的过程里提炼规律，形成能够针对新模型/硬件/工作负载生成专项 Execution Runtime 的 Inference Foundry。

当前先把第一条完整闭环跑通。

---

## P0 专项 Runtime 边界

DeepSeek Extreme P0 的专项 Runtime 服务当前固定模型/硬件/算法配置：

DeepSeek V4 Flash W4A8、8×Ascend 910B3、DP1×TP8、DSpark7。

vLLM/vLLM-Ascend 只承担参考实现、正确性 oracle、权重/算子加载来源和公平 Baseline 的角色，不构成架构边界。

任何通用 Scheduler、ModelRunner、metadata builder、request object、dynamic dispatch、backend dispatch、兼容分支、通用 KV 管理、Python orchestration、CPU bookkeeping 和 Host↔Device 状态同步，只要没有被模型语义、硬件约束、当前 calibration contract 或必要 serving 合约证明为必须，就应通过：

- 删除；
- 静态化；
- 预计算；
- 合并；
- Device 常驻；
- Graph；
- persistent execution；
- 专项实现；

来减少或消除。

但必须明确：

> 因当前 32K/c12 calibration workload 而成立的固定 shape、固定 cohort 或固定调度假设，仅属于 P0 的 calibration specialization，不自动成为 Inference Foundry 的通用方法或下一产品的强制架构。

当前阶段的 KEEP 仍由冻结 workload 的 correctness 与可重复 formal E2E 性能裁决；Foundry Method 的跨 workload / 跨模型可复用性将在后续 calibration case 中独立验证。
