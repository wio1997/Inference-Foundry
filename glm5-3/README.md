# GLM Execution Research

从 **GLM-5.3 的语义、现有算子与 Ascend 910C 资源**研究执行计划。SGLang / vLLM 是外部对照和可选集成环境，候选不以它们的类层次和执行顺序为边界。

2026-10-12 重置：[旧研究审计](docs/RESEARCH_RESET.md)。旧补丁和实验原件保留；当前从 [AGENTS](AGENTS.md)、[HANDOFF](HANDOFF.md)、[MISSION](MISSION.md) 恢复。

## 当前可以运行什么

新增独立、纯 Python 标准库的 `execution_lab/`：

- **Problem**：必须保留的操作和依赖、资源容量、带来源的条件成本。
- **Plan**：额外串行边与 ready 节点优先级；不能删操作或必要依赖。
- **Estimate**：在资源容量约束下生成时间线，未知成本拒绝估计，不自动当成零。
- **Reference replay**：实际调用显式注册的 CPU callbacks，比较不同合法执行顺序的中间/最终输出。

它不导入框架，不要求权重或设备，不执行 shell/plugin 字符串。当前只验证固定操作集合的顺序/并发问题；切分、布局、placement 和状态生命周期仍需显式建模与等价性审查。

```sh
# 在仓库根执行。纯本地 synthetic 示例，无网络和设备调用。
python3 glm5-3/execution_lab/cli.py demo

# 替换为自己的、经过校准的条件成本与计划；此示例仍是 toy。
python3 glm5-3/execution_lab/cli.py estimate \
  glm5-3/examples/execution_lab/toy_problem.json \
  glm5-3/examples/execution_lab/toy_concurrent.json

# 依赖/资源/成本校验、竞争排程与 callback 输出一致性测试。
python3 -m unittest discover -s glm5-3/tests -p 'test_execution_lab.py' -v
```

toy 的模型估计为串行 9ms / 双槽 6ms；这只是检验排程逻辑的数据，**不是 GLM/910C 加速比**。CPU replay 是顺序执行的 reference，不声称验证真实设备并发。完整 schema/API/假设见 [示例说明](examples/execution_lab/README.md)。

## 下一块真实工作

[GLM-PLAN-0001](studies/GLM-PLAN-0001.md) 定义真实片段和校准要求：先从已有源码/trace 提取必要语义与实际输入条件，再比较机制不同的执行方案。保留层宽、路由、工作集及拓扑，不默认以小 Dense 模型替代 GLM。

**尚未实现/证明**：真实 GLM tensor fixture、Ascend 算子 adapter、设备并发正确性、真实成本采集、新的模型性能收益。独立执行实验内核不等于完整推理框架。设备测试仍按此前用户指令暂停；本次本地重构没有恢复设备执行。

| 阶段 | 结论边界 |
|---|---|
| 工具测试和 synthetic 估计 | 代码/成本模型内成立 |
| 经原模型校准的真实片段 | 声明条件内的执行机制与收益 |
| 原模型与动态服务回归 | 合同内真实 E2E / SLO / 产品 Current |

## 资料入口

- [MISSION](MISSION.md)：模型/硬件身份、语义合同、动态服务及正式 PD 验收。
- [PLAN](PLAN.md)：问题、替身校准、计划、实测、集成的推进方式。
- [RECORDING](RECORDING.md)：每轮最小证据和不能越级的结论。
- [REUSE](REUSE.md)：旧资产怎样复用，哪些默认值不继承。
- [20261007 patch 包](patches/20261007/README.md)：H6/H5 等已有证据，未改写。
- [legacy](legacy/README.md)：发布基线入口原件；旧恢复指令不再生效。

本地重构不创建新性能 Run、不提升产品 Current。正式 80K/600/93% 条件作为历史产品回归保留，不能把它写死到执行器或视为所有客户的请求分布。
