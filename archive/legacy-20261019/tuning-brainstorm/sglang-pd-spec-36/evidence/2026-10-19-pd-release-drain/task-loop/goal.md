# Loop 目标：M005/M031 Ascend PD 传输完成、abort 与释放所有权

- Loop ID：`L027-pd-transfer-release-ownership`
- 模式：`spec`
- 所属 Task：`sglang-pd-spec-36`
- 执行 Skill：`NONE`
- 冻结时间：`2026-10-19T02:05:08Z`

## 目标/假设

Trace actual Ascend sender/receiver inheritance and completion/drain/timeout semantics before claiming shared KV pages are safe to reuse after explicit count item bypass.

## 允许修改的路径

- `/Users/wio/work/sglang-learning-lab/mechanism-task-20261018`

## 目标用例

- source-contract
- c2-short
- c3-long

## 接受条件

Canonical source closure and actual caller/manager/transport ownership matrix with validated state transitions; native correctness acceptance remains separate.

## 否定条件

Generic backend assumptions do not match Ascend inheritance, timeout/poll cannot establish last writer completion, or metadata/free ownership remains unresolved.

## 固定条件与禁止变更

执行前在此记录固定接口、测量口径和实现边界。
