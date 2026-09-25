# Loop 目标：Formal-trajectory serving boundary attribution

- Loop ID：`loop-059`
- 模式：`design`
- 所属 Task：`deepseek-extreme-p0`
- 执行 Skill：`NONE`
- 冻结时间：`2026-09-25T16:20:51Z`

## 目标/假设

The 1.3-1.9s per-cohort Run99 client-minus-decode swing is concentrated in prefill/admission or output-publication boundaries rather than TP8 decode wall

## 允许修改的路径

- `scripts/run_loop059_boundary_capture.sh`
- `evidence/20260926_loop059_boundary`
- `PERFORMANCE_MAP.md`
- `ACHIEVABLE_BOUND.md`
- `HANDOFF.md`

## 目标用例

- mixed_32k_1024_c12

## 接受条件

Legal warmed 48-request diagnostic captures common-clock all-rank boundaries for four cohorts and separates exposed phase variation without profiler or correctness loss

## 否定条件

Boundary records fail to align with client waves, materially perturb serving, or reveal variation dominated by an uninstrumented segment

## 固定条件与禁止变更

执行前在此记录固定接口、测量口径和实现边界。
