# MC2 preparation mechanism closed by same-call boundaries

**Run258 closes the missing native boundary attribution.** The repeated V4 availability predicate is currently the largest specific native preparation cost identified and validated by an intervention. It is not77% of the model step or a claim that the whole D265ms/token is explained by one helper. Accept the independent Challenger's mechanism closure; stop prioritizing descriptor/executor caching or V1→V2 migration from the old unexported residual.

The [frozen Reset](RESET_NATIVE_BOUNDARY.md) permits exactly one original-stock golden2334/8 PD request after an independent CPU fixture. The source-stack Run257 is not repeated. CPU fixture A's tracefs seek failure and B's registration/cleanup/selfcheck are preserved. B has20 complete triplets/120 events, zero lost; median task-clock reference error4.730us, cold max177.840us. This supports coarse phase attribution, not overhead-free fine CPU accounting.

[Run258 raw](../../runs/GLM-RUN-0258/artifact_identity.json) is4,156,190bytes, SHA256 `cde09fd699de83da084286a183df63e9c7bfa6400a8eae31788e37c0acdc4408`. Original server path ends `runs/GLM-RUN-0258/native_boundary_raw.json`. Every event reads the **same target TID's task-clock group leader** with CLOCK_MONOTONIC wall and equal enabled/running times. Native entry/return uses actual installed prologues and source/BL evidence, not nearest-symbol names. Existing Run254 exact stacks identify the actual libopapi_transformer V4 workspace wrappers. Source/ELF SHA and file-offset translation are in probe_identity/resident_native_entries.

The [reducer](reduce_native_boundaries.py) asserts9120 events, zero lost,380 dispatch+380 combine per rank, exact alternating native order, exactly one contained predicate and matching workspace per native, no outside-native calls, monotonic counters, raw commonPID identity, source order and partition conservation. Original golden token IDs/chunks1/2/2/2/1, prompt2334/external KV2334/finishlength, original native and both-role all-worker guards are unchanged. Probe group removed, other groups unchanged; phase exits0. Diagnostic TPOT259.430706ms and observer CPU76.678140ms are not performance results or quantities to subtract.

| Actual stock main thread | Native wall | Native task CPU | Predicate task CPU | Predicate/native CPU |
|---|---:|---:|---:|---:|
|D13|337.578752ms|337.569740ms|260.094670ms|77.0492%|
|D15|331.271064ms|331.196320ms|257.133130ms|77.6377%|

| Same-call D15 phase | Dispatch median wall/CPU | Combine median wall/CPU |
|---|---:|---:|
|Native entry→predicate entry: allocations/checks|39.687/39.540us|12.821/12.440us|
|Availability predicate|345.599/345.540us|323.338/323.385us|
|Predicate exit→workspace entry: descriptor/cache/determinism preparation|23.826/23.880us|21.726/21.945us|
|V4 workspace wrapper|24.821/24.820us|24.786/24.850us|
|Workspace exit→native return: workspace/submission/cleanup mixture|23.436/23.405us|22.501/22.415us|

Signed wall-minus-task-clock is retained: D15 phase extrema−0.990/+2.402us, D13−20.839/+10.033us. These intervals predominantly execute on CPU; **off-CPU is not declared exactly zero**. Task CPU includes probes/kernel overhead. The remaining D15 native CPU74.063190ms includes necessary output allocation, dynamic descriptor/executor/query/submission and unknown details; it is not a removable budget.

Five groups of76 MoE calls correspond by source counts to75 target MoE layers plus one MTP MoE. D15 availability task CPU by group53.347/47.992/46.732/54.249/54.814ms; remaining native14.703/14.567/14.429/15.403/14.961ms. These are grouped CPU contributions, not complete ModelRunner step durations or per-emitted-token costs.

## Causal code and independent intervention

Actual `MoeDistributeDispatchV2KernelOpApi.cpp:190` and `MoeDistributeCombineKernelV2OpApi.cpp:91` call `check_aclnn_kernel_available` each time. Actual `op_api_common.cpp:798` checks both API and workspace symbols with GetOpApiFuncAddr. Its custom/default library path traverses realpath, handlers/symbol lookup and ownership checks before fallback. EXEC's separate static function pointers do not cache the preceding bool. This predicate has no tensor/request-dependent input; the ready library set is fixed during the worker lifetime.

The [minimum two-site patch](mc2_capability.patch) uses function-local `static const bool`. It preserves V3/V2 fallback, allocations, tensor format, descriptor/workspace/executor construction, CANN arguments, EP communication and output. Library replacement requires a worker restart; negative capability caching does not permit hot installation.

Run254 identifies the caller, Run258 measures the same-call phase, and [Run256](../../runs/GLM-RUN-0256/summary.md) supplies independent correctness and matched intervention. Both23-token natural-EOS complete PD pairs preserve output/KV/cumulative MTP signature. D TPOT225.712→194.973 and223.947→192.152ms/token, decreases13.62/14.20%; D wall−0.772341/−0.748999s exceeds frozen drift0.064642s. Complete PD wall decreases11.51/14.41%, separate P contributions−0.110304/+0.084961s are not D gain. Timed modes share workers/common diagnostic binary; this is not a timed stock-library deployment comparison.

**Close this mechanism/diagnostic chain; retain scoped POSITIVE / formal PERF_KEEP PARKED.** No further availability sampling or bracket retry, no speculative V2/descriptor patch. Native production deployment, complete dynamic API/SLA capacity and a full exact event-release/control-layer DAG remain separate unaccepted work. Current=None/stack empty; 本阶段没有新增代码级性能 KEEP。

[Full first-dispatch/last-combine task-clock envelope and post-combine source attribution](INTERLAYER_SOURCE_DECISION.md) explicitly retains the larger1636.228ms inter-layer CPU region. The257.133ms predicate is about12.13% of that2120.453ms measured envelope, not77% whole-step. No cross-epoch scaling or globally maximal deletable-wall claim.
