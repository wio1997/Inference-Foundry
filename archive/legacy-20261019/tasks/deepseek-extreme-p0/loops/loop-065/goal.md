# Loop 目标：Target decode hidden AllGather and local Q scheduling

- Loop ID：`loop-065`
- 模式：`optimization`
- 所属 Task：`deepseek-extreme-p0`
- 执行 Skill：`operator-task-loop`
- 冻结时间：`2026-09-26T07:33:47Z`

## 目标/假设

The decode hidden AllGather completion to local Q start is an implementation edge; delaying wait until first gathered-hidden consumer can advance latest-rank Target path without changing model work.

## 允许修改的路径

- `/data/wio/vllm_ascend_26/framework/vllm-ascend/vllm_ascend/attention/context_parallel/dsa_cp.py`
- `scripts`
- `evidence/20260926_loop065_gather`

## 目标用例

- mixed_32k_1024_c12

## 接受条件

Fixed-layer numerical parity and all-rank FULL Graph replay/dependency evidence; repeated formal E2E gain required for KEEP.

## 否定条件

Async gather cannot capture/progress safely, numerical gate fails, or exposed latest-rank path and formal E2E do not improve.

## 固定条件与禁止变更

执行前在此记录固定接口、测量口径和实现边界。
