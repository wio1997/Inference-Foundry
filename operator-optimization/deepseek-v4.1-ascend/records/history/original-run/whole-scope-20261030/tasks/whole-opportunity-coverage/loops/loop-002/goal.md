# Loop 目标：166同调用MoE专家计数与有效路由证据

- Loop ID：`loop-002`
- 模式：`correctness`
- 所属 Task：`whole-opportunity-coverage`
- 执行 Skill：`none`
- 冻结时间：`2026-10-30T15:31:28Z`

## 目标/假设

已验证的TensorList graph observer可在隔离诊断入口中读取实际FusedExpertsResult计数、mask、topk和调用序号，避免capture常量及跨层共享buffer覆盖误读；不改变模型返回值。

## 允许修改的路径

- `/Users/wio/work/166-vllm-analysis-continuation-20261030`
- `/Users/wio/work/169-vllm-ascend-reproduce-172/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/whole-scope-20261030`

## 目标用例

- c8-input32k-output1k-prefix90-166

## 接受条件

固定C8用例校验通过；实际target/draft FUSED_MC2按rank/layer/group/call绑定，有效路由与专家计数可做守恒检验、容量身份明确，原服务恢复健康。

## 否定条件

取证仅得到warmup/capture常量或身份无法核实；返回计数与路由守恒冲突且不能解释；原服务未恢复。

## 固定条件与禁止变更

执行前在此记录固定接口、测量口径和实现边界。
