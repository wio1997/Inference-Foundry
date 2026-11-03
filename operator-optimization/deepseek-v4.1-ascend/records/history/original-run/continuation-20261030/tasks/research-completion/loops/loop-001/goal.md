# Loop 目标：主模型执行与相邻间隔的同次证据归因

- Loop ID：`loop-001`
- 模式：`baseline`
- 所属 Task：`research-completion`
- 执行 Skill：`none`
- 冻结时间：`2026-10-30T10:47:03Z`

## 目标/假设

后续小行数窗口存在与主模型跨度相近的迭代间隔；同次导出可区分其中设备工作与主机包络

## 允许修改的路径

- `/Users/wio/work/169-vllm-ascend-reproduce-172/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/continuation-20261030`

## 目标用例

- c8-32k-1k-prefix90

## 接受条件

32份CSV哈希一致；重复签名跨全部rank成立；16rank间隔区间账和rank0/8同次host轨道完成

## 否定条件

签名不稳定、边界不唯一或哈希不同则拒绝归属，不把间隔视为空闲

## 固定条件与禁止变更

执行前在此记录固定接口、测量口径和实现边界。
