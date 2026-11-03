# Loop 目标：Profile主导的全模块融合候选证据分级

- Loop ID：`loop-004`
- 模式：`spec`
- 所属 Task：`whole-opportunity-coverage`
- 执行 Skill：`none`
- 冻结时间：`2026-10-30T16:37:20Z`

## 目标/假设

已有profile的模块/框架算子/kernel关联、时序、形状与调用栈可先对实际运行路径的融合拆分重组候选分级，明确不能定案的最小问题后再定点查代码。

## 允许修改的路径

- `/Users/wio/work/166-vllm-analysis-continuation-20261030`
- `/Users/wio/work/169-vllm-ascend-reproduce-172/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/whole-scope-20261030`

## 目标用例

- c8-input32k-output1k-prefix90
- c8-input32k-output1k-prefix90-166

## 接受条件

对全模块/阶段覆盖表和已有候选逐项给出profile证据、支持/排除/不足判定、关键路径或资源意义；每个源码补查有明确profile缺口和定点目标，未覆盖范围保留。

## 否定条件

以整库文件审阅或AST数量代替profile分析；只有相邻kernel就判融合合法；只看TopN后宣称全模块/分支完成。

## 固定条件与禁止变更

执行前在此记录固定接口、测量口径和实现边界。
