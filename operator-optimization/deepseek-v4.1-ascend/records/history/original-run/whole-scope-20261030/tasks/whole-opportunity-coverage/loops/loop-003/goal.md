# Loop 目标：扩展启动与请求形状域的166同调用MoE取证

- Loop ID：`loop-003`
- 模式：`correctness`
- 所属 Task：`whole-opportunity-coverage`
- 执行 Skill：`none`
- 冻结时间：`2026-10-30T16:03:15Z`

## 目标/假设

取证范围按实际配置和显式资源预算定义，512容量常数不作调用硬上限；同一返回点TensorList可保留真实计数/mask/topk/ordinal并支持跨rank路由守恒复核。

## 允许修改的路径

- `/Users/wio/work/166-vllm-analysis-continuation-20261030`
- `/Users/wio/work/169-vllm-ascend-reproduce-172/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/whole-scope-20261030`

## 目标用例

- c8-input32k-output1k-prefix90-166

## 接受条件

完整固定C8验证通过，所有实际FUSED层/rank身份与计数/掩码/容量可复核，覆盖缺口显式，原服务恢复健康；值与调用身份经独立CPU复核。

## 否定条件

仅有capture或warmup常量、身份不明、采集数据不守恒且不能解释、观察预算不足未覆盖关键范围或原服务未恢复。

## 固定条件与禁止变更

执行前在此记录固定接口、测量口径和实现边界。
