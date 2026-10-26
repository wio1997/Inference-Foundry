# Loop 目标：M033 默认增量反分词与结束回收契约

- Loop ID：`L030-incremental-detokenize-finish`
- 模式：`correctness`
- 所属 Task：`sglang-pd-spec-36`
- 执行 Skill：`NONE`
- 冻结时间：`2026-10-19T03:36:02Z`

## 目标/假设

固定源码的默认请求增量IDs与反分词状态机应在多token回传、交错请求、Unicode和finish条件下保持输出顺序并回收状态；先验证现有能力，不预设性能补丁。

## 允许修改的路径

- `/Users/wio/work/sglang-learning-lab/mechanism-task-20261018`
- `/Users/wio/work/sglang-learning-lab/repo-export/tuning-brainstorm`

## 目标用例

- source-contract
- tokenizer-cpu-c2
- tokenizer-cpu-c3
- lifecycle

## 接受条件

固定源码来源与选中完整方法可追溯，目标tokenizer能恢复则校验精确hash后实测，否则保留不可用边界；所有结论限定CPU组件。

## 否定条件

流式拼接与相同token序列全量decode或stop裁剪不一致、请求错配、finish后状态残留，或真实tokenizer/完整方法不可执行。

## 固定条件与禁止变更

执行前在此记录固定接口、测量口径和实现边界。
