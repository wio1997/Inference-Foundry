# Loop 目标：M005/M017 NPU CPU镜像页计数候选设计

- Loop ID：`L024-npu-cpu-page-count-design`
- 模式：`design`
- 所属 Task：`sglang-pd-spec-36`
- 执行 Skill：`NONE`
- 冻结时间：`2026-10-18T22:49:51Z`

## 目标/假设

Under fresh authoritative CPU mirror inputs, NPU alloc_extend can derive default new-page count on CPU while preserving explicit count and original fallback. Removing default item may affect hidden stream/WAR/lifetime ordering; native validation is mandatory.

## 允许修改的路径

- `/Users/wio/work/sglang-learning-lab/mechanism-task-20261018`

## 目标用例

- source-contract
- c2-short
- c3-long

## 接受条件

Produce a reviewable isolated single-function patch plus fallback/mirror/native gates, then freeze CPU differential correctness without claiming native success or speedup.

## 否定条件

Narrow explicit-count/fallback behavior; alter page units, loc/dtype, capacity semantics or kernel; infer device ordering safety from CPU tests; apply production change before native PD+MTP validation.

## 固定条件与禁止变更

执行前在此记录固定接口、测量口径和实现边界。
