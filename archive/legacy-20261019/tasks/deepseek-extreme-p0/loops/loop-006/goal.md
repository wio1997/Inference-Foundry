# Loop 目标：Benchmark valid coupled path with stable functional gate

- Loop ID：`loop-006`
- 模式：`optimization`
- 所属 Task：`deepseek-extreme-p0`
- 执行 Skill：`NONE`
- 冻结时间：`2026-09-20T16:56:35Z`

## 目标/假设

After replacing invalid exact reasoning-text equality with stable API functional checks, the runnable SP/DSA-CP-off path may reduce exposed TP8 communication enough to improve warmed full E2E throughput

## 允许修改的路径

- `/data/wio/Inference_Foundry/scripts/run_flashcomm_candidate.sh`
- `/data/wio/Inference_Foundry/scripts/check_functional.py`

## 目标用例

- mixed_32k_1024_c12
- correctness_short

## 接受条件

Functional suite passes, 48/48 success on each of three warmed full passes, and median output TPS improves >5 percent over 543.65 without material TTFT/TPOT regression; performance result remains provisional pending numerical equivalence

## 否定条件

Functional suite fails, 48-request run fails, or robust E2E performance does not improve beyond baseline spread

## 固定条件与禁止变更

执行前在此记录固定接口、测量口径和实现边界。
