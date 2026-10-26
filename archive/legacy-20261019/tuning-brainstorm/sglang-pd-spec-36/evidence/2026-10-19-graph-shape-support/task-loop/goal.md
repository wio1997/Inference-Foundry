# Loop 目标：M021 NPU图形状桶与MTP输入刷新边界

- Loop ID：`L008-npu-graph-shape-support`
- 模式：`spec`
- 所属 Task：`sglang-pd-spec-36`
- 执行 Skill：`NONE`
- 冻结时间：`2026-10-18T18:11:06Z`

## 目标/假设

安装版NPU图继承通用decode runner，实际形状键和准入条件决定2/3请求是否缺桶；MTP正确长度/页刷新不可被旧graph状态替代

## 允许修改的路径

- `/Users/wio/work/sglang-learning-lab/mechanism-task-20261018`
- `/Users/wio/work/sglang-learning-lab/repo-export/tuning-brainstorm`

## 目标用例

- source-contract

## 接受条件

固定源码追踪target/draft和父类桶选择、宽度、fallback及NPU更新入口；纯CPU精确谓词/桶逻辑探针，实例未观察与NPU graph未执行明确保留

## 否定条件

仅凭图开关或类名认定已replay；未定位父类/消费者就增加桶；CPU结果外推graph性能

## 固定条件与禁止变更

执行前在此记录固定接口、测量口径和实现边界。
