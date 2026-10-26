# Loop 目标：M005/M006 PD prefill准入与chunk预算合同

- Loop ID：`L020-pd-prefill-admission-budget`
- 模式：`spec`
- 所属 Task：`sglang-pd-spec-36`
- 执行 Skill：`NONE`
- 冻结时间：`2026-10-18T21:36:12Z`

## 目标/假设

PD prefill actualtokenrows emerge from admission/chunk decisions, separately frommemory/outputreserve andrequestconcurrency. NPU cannotinheritAMD-only exact-fill/tile gates withoutsource evidence.

## 允许修改的路径

- `/Users/wio/work/sglang-learning-lab/mechanism-task-20261018`
- `/Users/wio/work/sglang-learning-lab/repo-export/tuning-brainstorm`

## 目标用例

- source-contract
- c2-short
- c3-long

## 接受条件

Trace fixed installed PD PrefillAdder caller, shape-selection/commit andallocator budget; establish safe minimal CPU/state andtokenwork contracts preserving PD/MTP andNPU alignment.

## 否定条件

Treatbatch2/3 orconfiguredchunk size asactualtokenrows; confuseadmissionmemory withcompute; borrowHIP-only optimization onAscend; dropinput/output/MTPwork orclaimperformancewithoutnativeevidence.

## 固定条件与禁止变更

执行前在此记录固定接口、测量口径和实现边界。
