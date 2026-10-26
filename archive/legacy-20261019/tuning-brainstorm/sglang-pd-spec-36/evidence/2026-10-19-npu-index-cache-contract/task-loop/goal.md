# Loop 目标：M023/M024/M025 NPU index cache与operator边界

- Loop ID：`L018-npu-index-cache-native-boundary`
- 模式：`spec`
- 所属 Task：`sglang-pd-spec-36`
- 执行 Skill：`NONE`
- 冻结时间：`2026-10-18T21:02:23Z`

## 目标/假设

NPU actualpool and native wrapper determine index cache dtype/layout, quant branch and shortpath support. Generic DSATokenToKVPool assumptions and checkpoint filenames are insufficient.

## 允许修改的路径

- `/Users/wio/work/sglang-learning-lab/mechanism-task-20261018`
- `/Users/wio/work/sglang-learning-lab/repo-export/tuning-brainstorm`

## 目标用例

- source-contract
- c2-short
- c3-long

## 接受条件

Trace fixed installed NPU pool selection, cache layouts and operator registrations without native activation; record known/unknown and a frozen minimal CPU or native experiment where justified.

## 否定条件

Infer actual runtime dtype, shortpath or performance from filename/comment/sys.path candidate alone; allocate devices or run operators without task resource ownership.

## 固定条件与禁止变更

执行前在此记录固定接口、测量口径和实现边界。
