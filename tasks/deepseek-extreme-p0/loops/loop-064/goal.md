# Loop 目标：Frozen c4 DSA CP fan-out scheduling and resource-constrained E2E calibration

- Loop ID：`loop-064`
- 模式：`optimization`
- 所属 Task：`deepseek-extreme-p0`
- 执行 Skill：`operator-task-loop`
- 冻结时间：`2026-09-26T04:22:23Z`

## 目标/假设

After indexer cache update, main compressor/scatter and indexer query/QLI are semantically independent in actual CP Target; legal Graph fork/join may reduce latest-rank full Target/cycle critical path if shared AIC/HBM contention and event overhead do not erase overlap

## 允许修改的路径

- `scripts/loop063_cp_fork_patch.py`
- `scripts/run_loop063_cp_fork_diag.sh`
- `/data/wio/vllm_ascend_26/framework/vllm-ascend/vllm_ascend/attention/context_parallel/dsa_cp.py`
- `runtime`
- `evidence/20260926_loop064_cp`

## 目标用例

- mixed_32k_1024_c12

## 接受条件

One-layer candidate actually selected on all 8 ranks, visible cache alias and same-state/continuous correctness pass, Graph device timeline confirms legal branch overlap; all21 implementation shows stable latest-rank complete cycle reduction and repeated frozen formal E2E KEEP without acceptance or serving-boundary confounding

## 否定条件

Graph cannot safely capture/replay fork, state/KV/logit acceptance diverges, or full Target/cycle/E2E does not improve despite legal branch overlap; falsifies this schedule only, not overall architecture bound

## 固定条件与禁止变更

执行前在此记录固定接口、测量口径和实现边界。
