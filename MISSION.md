# DeepSeek Extreme P0

## Mission

为 Inference Foundry 做出第一款极限性能产品：

**DeepSeek Extreme P0**

当前固定目标：

* DeepSeek V4
* W4A8
* 8 × Ascend 910B
* DP1 × TP8
* Prefill / Decode 混部

目标不是单纯优化 vLLM-Ascend。

目标是：

> 在模型语义正确的前提下，重新研究 DeepSeek 在这套固定硬件上的完整执行方式，并持续缩小现实性能与硬件可实现性能上限之间的差距。

## 第一阶段

首先建立：

### Baseline

冻结当前：

* 软件版本
* Git commits
* CANN / torch-npu
* 模型和量化配置
* 启动参数
* benchmark workload

得到稳定可复现的：

* TTFT
* TPOT
* Output TPS
* Prefill
* Decode
* Kernel / HCCL / Host profiling

### Performance Map

建立当前性能地图：

* Prefill 时间分布
* Decode 时间分布
* Kernel 时间
* HBM / 数据流
* TP8 通信
* Host / D2H / metadata
* Kernel gaps
* Framework overhead
* 当前最大的性能未知和性能机会

### Achievable Bound

逐渐估算：

* 必要计算成本
* 必要权重/HBM读取
* 必要 KV 读取
* 必要通信
* 可以重叠的部分

得到粗略：

Current Runtime
vs
Achievable Bound
vs
Remaining Gap

## 优化方式

不预设第一刀。

可能包括：

* Ascend C Kernel
* SuperKernel
* 数据流融合
* W4A8 专项实现
* 权重布局
* Attention
* MoE
* HCCL/MC2
* 通信计算重叠
* ACL Graph
* D2H/H2D
* Model Runner
* KV
* Sampling
* Device-side state
* Standalone Decode Runner
* Persistent execution

由证据决定。

## Framework Exit

持续区分：

**Compute Gap**

和

**Framework Gap**

如果主要剩余 Gap 已经来自 vLLM-Ascend 的执行组织，而且无法合理消除，则建立 Standalone Layer/Decode Runner，与原框架做公平性能对比。

如果独立路径显著更接近硬件下界，则逐渐把性能 Hot Path 从框架中拿出来。

Serving、HTTP 等外围能力没有性能理由时不需要重新实现。

## 最终产品

DeepSeek Extreme P0 必须是完整可运行的 DeepSeek，而不仅是 microbenchmark。

最终需要比较：

* vLLM-Ascend Baseline
* DeepSeek Extreme P0
* Estimated Achievable Bound

并能够解释：

1. Baseline 的主要浪费在哪里；
2. P0 消除了什么；
3. 为什么 P0 更快；
4. 剩余最大的 Gap 是什么；
5. 下一步是否应该继续现有路线或进入更专项的 Runtime。

长期目标是从制造 DeepSeek Extreme 的过程里提炼规律，最终形成能够自动生成专项 Execution Image 的 Inference Supercompiler。

但当前先把第一款产品做出来。


## 专项 Runtime 产品边界（2026-09-21 冻结）

DeepSeek Extreme P0 的交付物是只服务以下固定产品配置的专项推理 Runtime：DeepSeek V4 Flash W4A8、8×Ascend 910B3、DP1×TP8、DSpark7。vLLM/vLLM-Ascend 只承担参考实现、正确性 oracle、权重/算子加载来源和公平 Baseline 的角色，不构成架构边界。

任何通用 Scheduler、ModelRunner、metadata builder、request object、动态 batch/shape、backend dispatch、兼容分支、通用 KV 管理、Python orchestration、CPU bookkeeping 和 Host↔Device 状态同步，只要没有被真实模型语义、硬件约束或 serving 合约证明为必要，就应通过删除、静态化、预计算、合并、Device 常驻、Graph、persistent execution 或专项实现消除。最终 KEEP 仍由冻结 workload 的正确性和可重复 E2E 性能决定。
