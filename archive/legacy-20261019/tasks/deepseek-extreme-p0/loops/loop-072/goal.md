# Loop 目标：Parked-tail full MoE active-row architecture screen

- Loop ID：`loop-072`
- 模式：`design`
- 所属 Task：`deepseek-extreme-p0`
- 执行 Skill：`NONE`
- 冻结时间：`2026-09-26T15:58:09Z`

## 目标/假设

Inactive parked request rows in one non-hash MoE block may be removable without changing active outputs, and the saved resource work may shorten the all-rank complete MoE endpoint after compaction overhead

## 允许修改的路径

- `evidence/20260926_loop072_parked_moe`
- `scripts/loop072`
- `tasks/deepseek-extreme-p0`

## 目标用例

- mixed_32k_1024_c12

## 接受条件

All8 same-prestate private Graph A/A/B/A active-output correctness and routing validity, plus complete block common-entry-to-final-communication improvement above A/A drift

## 否定条件

Active output/router divergence, unsupported collective shape/rank participation, compaction cost erases block gain, or no latest-rank endpoint saving

## 固定条件与禁止变更

执行前在此记录固定接口、测量口径和实现边界。
