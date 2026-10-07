# Current Decode source dependency map — corrected actual async path

2026-10-07 checkpoint172 supersedes the earlier H10 preparation map. All16 actual D267 workers report async_scheduling=true. TargetFULL bucket2/CompilationNONE, eagerK1 MTP, retained H6/H5, H9/H10off. This map records dependencies, not measured wall budgets. Old eager/profileON Run249 timing cannot budget current FULL/async execution.

Actual installed config (async_sources/identity.json) returns max_concurrent_batches=2 for PP1/async, for either V1 or V2. Actual EngineCore source hash matches the protected snapshot (current_core_source_identity.json); therefore queue branch selection is source-supported. Queue occupancy/timing in the actual requests remains unmeasured. The synchronous three-RPC diagram does not describe this current path.

| Boundary | Actual dependency | Unmeasured exposure |
| --- | --- | --- |
| P→KV→D | D may reuse local prefix then import remaining P blocks. Native all-rank transfer and canonical failed-block/error handling preserved. | P local reuse, D local reuse, imported tokens/bytes and compute must be separately accounted; external hits are not P HBM hits. |
| Scheduler/queue→Executor | AsyncScheduler reserves output placeholders. Core submits execute and, absent pending grammar, sample nonblocking before waiting for prior execute. FIFO queue capacity2; KV response aggregation can involve all16 ranks. | Queue occupancy, MQ enqueue/dequeue/response/consume and exposed host gaps. |
| ModelRunner preparation | input-prep fence protects CPU staging reuse; state/DCP metadata correction uses prior GPU sampled/count state. | staging/count readiness versus host preparation. |
| TargetFULL | Captured math/TP/DCP/EP, retained H6/H5. Original target pre-replay host sync retained. H9 had no repeated measured gain. | Necessary device compute/comm, peer service/wait and host gaps in current capture. |
| Sampling/bookkeeping | `_sample` repairs CPU history only when requested; `_bookkeeping_sync` async branch stores placeholders, not synchronous target-token D2H. | Histories/penalties versus current greedy route; source function name is not runtime evidence. |
| Eager MTP | Device next-ID/count, separate hidden buffers, DCP preparation, one MTP model step, own count/draft events. Ordinary async path skips draft CPU RPC; deferred grammar retains distinct draft visibility. | Current host submission versus device math/communication and count correction exposure. |
| Async sampled copy | Output constructor is after MTP/finalize; copy stream waits current/default stream, copies sampled/logprob and optional routed/EP-fault data, records event. Worker returns owned object before copy completion. | Copy wait may include MTP. Its duration is not all removable; old ready event also covers iteration completion. |
| Output thread→Future→Scheduler | WorkerAsyncOutputCopy calls get_output before SUCCESS; CPU parse/response then FIFO Future consumption, abort processing, Scheduler update and possible KV release. | Which readiness/consumer actually stalls critical path; early publication needs KV/error/lifetime proof. |

Root personally read ASYNC_OUTPUT_REVIEW.md and accepts its PIVOT, safety boundary, ranking-as-question not measured gain, and no repetition of H8/H9/H10 absent new evidence. Keep a late iteration/error gate if exploring earlier sampled copy. Preserve staging/count/draft/DCP/current-stream/state/routed/KV contracts. Largest current removable gap remains unknown. Only decisive bounded completion/queue evidence may justify a new diagnostic; no performance candidate or new NPU Run is activated by this map.

User's later cache warning is active: audit prior salts and missing local-cache measurements; future comparisons isolate cache namespaces, preserve controlled within-round warm prefix and measure P/D local/external deltas separately. 93% shared content/declared KV condition is not a measured guarantee.
