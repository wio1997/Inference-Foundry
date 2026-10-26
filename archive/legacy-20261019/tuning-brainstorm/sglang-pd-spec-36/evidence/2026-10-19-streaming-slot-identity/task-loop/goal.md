# Loop 目标：M036 真实流式composer身份与生命周期

- Loop ID：`L036-streaming-composer-identity`
- 模式：`correctness`
- 所属 Task：`sglang-pd-spec-36`
- 执行 Skill：`NONE`
- 冻结时间：`2026-10-19T06:33:16Z`

## 目标/假设

原normalizer、registry、Scheduler→Session.create_req与完整StreamingSession及facade match组合应在身份变化时阻止旧slot索引恢复，并保护同身份/重试/关闭释放。

## 允许修改的路径

- `/Users/wio/work/sglang-learning-lab/mechanism-task-20261018`
- `/Users/wio/work/sglang-learning-lab/repo-export/tuning-brainstorm`

## 目标用例

- source-contract
- lifecycle
- c2-short
- c3-long

## 接受条件

原组合路径可执行且严格记录identity与页/释放前提；变化身份拒绝或安全重算，同身份与取消重试生命周期保持。

## 否定条件

变化adapter/key/salt可通过实际Session/Scheduler并在facade之前命中旧slot CPU索引；或所谓修补破坏同身份/释放/关闭边界。

## 固定条件与禁止变更

执行前在此记录固定接口、测量口径和实现边界。
