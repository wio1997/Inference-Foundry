# checkpoint163 — real Decode attribution and first minimal code comparison

2026-10-06; source parent e36a52b4f93cb0d4884d985b4b6a54ea8156155b; branch glm5-3-autonomous-20261001; GLM-RESEARCH-RULES-v2. User requested continuing source/raw research without new large NPU workload/profile or parameter scans. No new kernel/math or parallel performance intervention.

**本阶段没有新增代码级性能 KEEP。** H4 has positive scoped code evidence but formal repeatable complete Product Gain is INCONCLUSIVE/PARKED for promotion. Current=None, active stack empty; full API and workload/SLA not accepted. Research has progressed beyond checkpoint162's unknown specific code path; the project is not declared complete.

## What the evidence now establishes

Existing Run249 D16 SQLite/CANN connection and all-thread enqueue/dequeue relations establish repeated late host supply inside the target-layer chain. Latest CANN host-launch rank equals latest collective device-entry rank in1896/1901 matched operations. D15 has902.406ms of exposed host-launch-late intervals, partitioned747.341ms before producer Enqueue,94.418ms Enqueue-start→Dequeue,60.647ms Dequeue→CANN. These are profile-ON bounds with profiler/scheduling/necessary host work; not a removable OFF-profile budget.

Five target+MTP rounds, not eight device steps. Each target78 attention/75 MoE plus draft1/1. Actual backend is SFA/DCP, not legacy dsa_cp or DSv4 dsa_v1.395 DCP gathers (79 first-step compact KV,316 later packed Q),316 output/LSE all-to-alls;800 TP reductions,380 MoE gathers,10 logit gathers;380 EP dispatch/combine pairs inside MC2. Standard1901 HCOM records do not cover EP kernel communication.

Representative round3: D15 extent603.685ms, named math union36.355ms, HCOM3.832ms, EP4.144ms, event wait3.470ms, all non-event union47.175ms. Early D0 sees HCOM413.963ms, EP156.179ms, event union609.603ms within609.603ms extent. Overlap/peer waiting explains why these sums cannot be deleted as barriers. Scheduler/Executor/ModelRunner exact separate durations are not in this acquisition. Most repeated late supply is within model layers, not the few-ms device boundary intervals.

## Minimal H4 and actual execution

[Patch](moe_gather.patch) replaces the equal-positive-shard,512B-aligned MoE list gather with direct `all_gather_into_tensor`, preserving private output, logical rank order, group, dtype, unpad, stream lifetime and fallback. Exact torch_npu source shows list gather has16 post-collective copies, base gather writes directly.380 calls ×16 =6080 redundant output-copy submissions. D15 contained CPU copy scopes112.574ms plus only finalize's380 splits14.629ms;127.203ms disjoint ON-profile host scope union, not predicted E2E saving.

Final guarded4-rank Gloo240 cases passed. Run250 failed an ambiguous old diagnostic gate; raw failure and unequal complete74A/8B trajectory retained, not credited. [Run251](../../runs/GLM-RUN-0251/summary.md) recovered owned D, passed320 native HCCL full-byte cases, then two all16-rank real-model B byte gates. Actual padding NaNs preserved. Same resident A1/B1/A2/B2, P unchanged, profileOFF: short matched TPOT263.843→246.900 and260.413→233.540ms/token, reductions6.42/10.32%. Complete same3-token natural EOS PD wall1.693434→1.660282 and1.745766→1.566286s. First pair near noise; second includes31.16ms faster unchanged P. No formal PERF_KEEP.

## Remaining highest-value attribution

Largest localized region: eager within-model producer preparation/submission on slow ranks. Strongest safely removable specific path: list-gather output materialization, now tested. Exact globally largest removable gap and the remaining profiling-OFF host cost are unknown; no speculative runner rewrite, early-output/KV barrier removal, MTP disabling or config scan is enabled. Additional attribution should separate op/output allocation/preparation and CPU scheduling/profiler overhead using the existing raw scopes/source before another device diagnosis. H4 is not promoted by a short fixture or by complete3-token arithmetic.

## Actual retained service and evidence

P Run249 root1916718 unchanged; D Run251 fresh root2532689. Both health200, idle,16 owned NPU workers each; controller251 completed/exited; original on-disk source SHA54e8ac… restored, selector0. D retains diagnostic selector in memory at stock mode. Recheck identity before operations. No active NPU Run/profile.

Primary report [CRITICAL_PATH](CRITICAL_PATH.md), reducers/compact evidence beside it, [Run251 execution summary](../../runs/GLM-RUN-0251/execution_summary.json), [final site](../../runs/GLM-RUN-0251/final_site.json), source SHA map and evidence index.66 relevant installed source files were independently rehashed on D after restoration with no change. Full source snapshots remain in the Mac workspace; raw trace/native logs remain on servers. Independent Astra final review accepted scoped correctness/signal and formal INCONCLUSIVE, with no blocker or additional Run request.
