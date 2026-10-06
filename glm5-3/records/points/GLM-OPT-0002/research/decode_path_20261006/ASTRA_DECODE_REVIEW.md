# Independent Run249 decode Challenger — 2026-10-06

**PIVOT the attribution work to cross-rank collective readiness and its producers. Do not select an async-output/KV-aggregation removal or an eager-runner rewrite from the current totals.** The largest proven removable Gap and its milliseconds remain unknown. This review reads existing source/Run249 evidence only; no model request, service action or NPU Run. It is not PERF_KEEP; Current=None.

## Evidence personally read

Installed source snapshots: Ascend model_runner_v1, llm_base_proposer, deepseek_mtp, dsa_v1, context_parallel/dsa_cp, fused_moe runner/shared_experts; vLLM multiproc_executor, gpu_model_runner (AsyncGPUModelRunnerOutput), deepseek_v2; prior EngineCore and KVOutputAggregator consumers. Provenance: installed_sources.json and supplied later source files. Read CPU reduction code analyse_existing.py, local_only/rank0.json and raw-index-preserving selected_rank0.json.gz. Other ranks' reduction was still pending at this writing; aggregate/rank-tail causality is not filled from a summary.

Reported profOFF TTFT1.309s/TPOT265.29ms (2334 input, 8 output) and profON~336ms are different timing regimes. The raw profile does not numerically decompose the unprofiled 265ms. Profiling effects, output token timestamps/count semantics and exact benchmark TPOT formula are required before mapping device cycles into public TPOT. Eight output tokens are not eight target-model invocations under MTP.

## Facts versus inference

- D0 raw has ten CPU embeddings. Independent raw-event counting between each adjacent pair gives alternating **78 MLA/75 MoE scopes** and **1 MLA/1 MoE scope**, repeated five times. Counts are consistent with five backbone+MTP rounds, not 395 independent decode steps. Target model forward embeds then loops its decoder layers; llm_base_proposer invokes a draft model separately. The full inherited DeepSeekMTP forward/model registration and runtime layer configuration remain needed to make the model-count identity entirely source-complete.
- D0 hardware extent is 2.859885s; main CPU-op union is 2.566724s. Stable target CPU embedding segments span about533–553ms; first is475ms. Four complete draft-embedding-to-next-target CPU intervals are14.35/15.77/16.98/16.84ms. These are **host submission intervals**, not device model latencies or removable Python time. The last CPU-defined period includes device drain and must not be labelled a steady-state cycle.
- Rank0 reducer reports only29.66ms of >20us direct task gaps on main device stream47. Much larger EVENT_WAIT/communication unions overlap compute and each other. A nearly continuously occupied stream can still be waiting for a peer's late host enqueue; conversely, busy CPU scopes can enqueue ahead of a device bottleneck. Neither observation proves host-bound or network-bound execution.
- Raw non-main-thread (tid307382) event synchronizations associated with successive host periods grow approximately2.55→29.36→85.35→118.34→132.13ms. Their exact output-copy association needs the thread/connection mapping; the trend is compatible with accumulating device/dependency backlog. It argues against treating all host submission occupancy as exposed host starvation. These waits are not additive lost wall time.

## Source-level dependencies worth testing

1. **Per-layer DSA-CP and MoE collective chain.** dsa_v1 selects context_parallel/dsa_cp for the enabled CP path. In that implementation, decode/specdecode does not take the prefill full-weight shortcut; `_restore_tp_head_layout` performs `all_to_all_single` on tp_group when tp_size>1 (1835–1863). This restores the head layout needed by projection, not an arbitrary barrier. Target model also has conditional TP gather/reduce-scatter around sequence-parallel MoE. Ascend MoE has routed/shared work, TP/EP/MC2 domains and event-controlled overlap. `vllm::moe_forward_shared` wraps more than a shared-expert kernel. HCCL names alone do not identify logical TP versus EP versus DCP ownership. Match communicator membership and producing call site. Potential code value is eliminating a proven late producer, redundant layout/collective or unnecessary serialization while retaining math, not deleting waits on required data.

2. **Output publication after draft submission.** Ascend sample_tokens performs target sampling/bookkeeping, invokes the padded drafter, finalizes connector, then constructs AsyncGPUModelRunnerOutput (2655–2832). That object's copy stream waits on default_stream before D2H, and get_output synchronizes the copy-ready event. WorkerAsyncOutputCopy resolves get_output before queueing a response. Thus sampled-token visibility can depend on default-stream work submitted after sampling, including draft work. The sampling_done_event exists earlier. A narrower event boundary is a plausible question, but tokens/logprobs/state lifetime, MTP acceptance, faults and buffer reuse are real dependencies. Earlier public visibility need not accelerate the next round, which still needs draft/state. Current raw does not establish net TPOT/capacity Gain.

3. **All-rank RPC/KV aggregation.** Enabling KV aggregation causes collective_rpc to collect every worker reply, while the aggregator counts completion contributors across calls. Model-output delivery can therefore wait for a slow rank even when no new KV transfer completes. Receiving successful KV on all required ranks is necessary; bundling unchanged connector state with every token-output reply is a design choice worth testing only if an actual post-compute rank tail blocks a scheduler opportunity. The asynchronous worker output thread can wait concurrently with the main worker; its synchronize duration is not automatically serialized worker downtime. EngineCore queue ordering and next-batch readiness must be included.

## Gap ranking / Top3 questions

Research priority, not measured savings:

1. Cross-rank late readiness on repeated layer collectives: **for the same collective ordinal/group, is the critical rank late because its host submitted late, prior device work completed late, or transfer progressed slowly after all ranks were ready?** This can explain both communication and event wait without double-counting.
2. Repeated host/framework submission and layout work: **is there a causal device starvation edge attributable to an avoidable host/layout cost, locally or on the peer gating the collective?** CPU scope totals alone cannot choose graph/eager rewrite.
3. Post-sampling output/commit gate: **after required device results are ready, does draft-tail/copy/all-rank aggregation delay public output or miss a legal next scheduler boundary?** Distinguish latency-only early delivery from throughput improvement.

MTP's one-layer host interval is small relative to the backbone submission interval in this trace. This weakens “MTP alone causes265ms” but does not measure MTP's device contribution or acceptance efficiency. Disabling it is not the first causal diagnostic and would change useful tokens per round.

## Smallest distinguishing evidence: reuse existing Run249

From D16 existing profiles, choose one long representative repeated collective in a stable backbone phase and one normal neighbor, not a parameter sweep. Match same communicator/operation/sequence across ranks; preserve raw indexes and clock uncertainty. For each rank extract host enqueue/CANN connection, preceding producing device task completion, collective start/end, downstream main-stream wait end and next compute. Reconstruct the causal predecessor of the last-finishing rank. Then repeat only on enough occurrences to establish that the selected edge represents a substantial recurring part of the critical path.

Decision table:

- Last rank's required operation was submitted late, device otherwise ready: investigate the specific host producer; no generic Python-total attribution.
- All ranks submitted, but one input device result is late: follow that producer backward; communication interval includes peer readiness.
- All inputs ready with stable clocks, collective itself remains long: investigate actual group/layout/communication organization within existing kernels.
- Collective completes promptly, but output/scheduler stalls afterward: inspect narrower sampled-token event and rank-output publication path.
- Missing connection/group/clock data prevents these distinctions: report unknown and specify that missing edge; do not infer savings from unions.

For output-copy attribution separately, existing sample/D2H/event/response flow IDs may settle whether copy waits for the whole draft tail and whether another batch is already in flight. Absence of explicit scheduler/commit markers limits counterfactual admission claims. No new NPU Run is authorized by this review.

## Stop List / likely misreading

Stop adding compute+COMM+EVENT_WAIT or rank totals; converting profON fractions into profOFF265ms; dividing the complete2.86s device window by8 and calling it a step; treating CPU embedding windows as device steps; treating get_output event synchronize as removable synchronous worker stall; removing all-rank KV safety; disabling MTP based on counts; choosing graph/eager rewrite from CPU inclusive scopes; configuration scans or new kernels.

Most likely misreading: **COMM/EVENT_WAIT is measured idle time that can all be deleted.** It is largely overlapping dependency time and can hide either essential communication or a late peer producer. Next most likely:2.57s busy CPU proves host starvation. The local direct-gap result and growing non-main synchronization waits do not support that shortcut. The decisive missing evidence is a matched cross-rank dependency edge, not another utilization total.

## D16 evidence update — independently checked

The pending reduction arrived under `records/points/GLM-OPT-0002/jobs/GLM53-DECODE-TRACE-20261006/reduced/`. Personally read cross_rank_collective_start.json and all16 raw-index-preserving collectives*.json.gz, independently recomputing the common-name matches:1901 operations; median start skew1083.25us; median last observed start→latest end15.25us; latest-start rank15 in1271 operations, rank13 in534. These name/group/sequence correspondences plus rank_size/count/type agreement support matching; clock alignment semantics still constrain microsecond claims. This replaces the earlier pending-rank limitation.

Decisive raw examples:

- `hcom_allReduce__503_476_1`, BFP16/count12288/rank_size16. D0 raw index450255 starts1791285355672589.689us, lasts6565.529us; D13 index450181 starts1791285355672584.242us, lasts6573.329us; D15 index449463 starts1791285355679152.238us, lasts9.94us. D15's connection_id is184606; D0 is191753. All-rank final completion follows the latest observed entry closely. This is strong evidence that the long early-rank collective interval is mostly waiting for late participation, not 6.5ms of transfer after all peers arrive.
- `hcom_allGather__503_387_1`, BFP16/count6144/rank_size16. D13 index450387 starts1791285355451585.297us, lasts7.84us while D0 index450093 starts1791285355450500.506us and lasts1091.479us. D13 is the late participant here, so the effect is not exclusively rank15.
- Normal near-aligned neighbor class `hcom_allGather__097_375_1` has D0/D13/D15 durations15.56/14.619/13.42us and close entries, illustrating that the raw representation can show short collectives when ranks arrive together.

**Updated leading Gap:** repeated late-rank readiness is now the strongest localized explanation of the profiled communication waits. Its *removable component* is still unknown. Next source/trace question is specifically the producer of rank15/13's late entry: host submission, previous device/MoE work, communication-stream scheduling, or another dependency. SQLite COMMUNICATION_OP.connectionId→CANN_API/host enqueue plus prior device dependencies is the correct next existing-evidence step. Do not subtract the sum of start skews from E2E: operations overlap, clock/entry semantics matter, and late readiness may be necessary work. Do not call this “network bandwidth bottleneck” or already-proven “Python bottleneck.” No change to Current or PERF_KEEP.

## Final SQLite/flattened-gather review — supersedes the initial candidate deferral

**Route KEEP: freeze one candidate replacing list-output all-gather with equal-shard all_gather_into_tensor in PrepareAndFinalizeWithAll2All.finalize, inherited by MC2.** This is approval of the bounded code question for implementation/correctness review, not deployment or PERF_KEEP. This is now the strongest concrete removable path identified in the supplied evidence; it is not proven to be the globally largest possible optimization.

Personally read `GLM53-DECODE-TRACE-20261006-B/reduced/db_rank15.json.gz`, task/CPU/communication columns and launch_analysis.json, and prepare_finalize.py prepare/finalize/MC2 inheritance. Independently recomputed latest-host-rank==latest-device-rank for1896/1901 matches and median last-rank launch-end→device-start24.825us. The same-host SQLite domain removes the cross-host ambiguity; small negative intervals in some rows still prohibit overprecise sub-100us ordering claims.

The previous peer-readiness inference is now localized to **host submission lateness on the slow ranks in this profiled run**, rather than pure collective transfer duration. Example hcom_allReduce__503_476_1 on r15 has host launch1791285355679048450–5679120210ns and device start5679152238ns. Its stream47 predecessors include small multiply/logical-not/embedding/masked-fill work, with masked-fill finished5679006201ns, then record-event submitted5679020860ns. This is a real late-feed sequence; it is not evidence that this particular longest collective was itself caused by the MoE final gather. Host scheduling, profiler overhead and runtime API cost remain contributors whose profOFF magnitude is unknown.

Concrete redundant work: r15 contains380 `c10d::allgather_` CPU intervals, total209.72587ms and median546.905us. In a personally inspected middle occurrence, CPU1791285356069989590–6070582230ns contains one stream40 AivKernel followed by16 separate `aclnnInplaceCopy` submissions. Copy task IDs31094–31109, connections212647–212734, each execute for roughly0.6us while host issues them separately. The gather CPU interval includes collective scheduling plus unpack/copy work, so209.7ms is an inclusive cost, **not predicted saving**. The supplied trace supports removing repeated list-output materialization, while retaining one real all-gather. Existing `_allgather_base_` calls are different work and cannot provide a matched speedup ratio.

### Correctness conditions for the minimal patch

- Preserve the existing fresh output allocation. The source explicitly forbids reusing prepare's split views because shared experts may still read their backing input. A permanent/reused output buffer is a separate lifecycle change and should not enter this candidate.
- Use the same `moe_config.tp_group.device_group`, process-group rank order, dtype/device, collective order and stream semantics. The intended result is row concatenation in group-rank order, followed by exactly the existing `self.num_tokens` slice.
- Fast-path eligibility must establish `output.shape == (tp_size * input.shape[0], *input.shape[1:])`, positive/legal dimensions, dense contiguous compatible tensors and equal shards. Match output numel and group size, not just a divisible first dimension. NPU physical-format/backend support must be verified; PyTorch logical contiguity alone is not a proof of backend-format compatibility.
- Eligibility must be consistent across participating ranks. Do not silently choose different collective APIs based on an unconstrained rank-local shape/contiguity predicate. Prove prepare/finalize invariants for the selected equal-shard path or use a configuration/metadata condition known consistently across the group. Do not add a per-layer guard collective just to choose this path.
- Preserve TP1, replace_allreduce, MC2 sequence-parallel handling and the original unequal/special-shape branch. All2All.prepare only pads when token_count<TP; larger nonmultiples can produce unequal tensor_split chunks, so unconditional replacement is incorrect despite the method comment suggesting uniform slicing.
- Do not catch a failed collective and retry the old API in-place: participants may already have entered different collective states. Validate the branch before collective entry.

### Updated Top3 questions and minimal evidence

1. Does the equal-shard tensor-output API preserve exact rank-concatenated contents and shared-expert alias safety while removing16 output-copy submissions per layer?
2. Do those removed submissions shorten slow-rank host readiness and corresponding peer waits, rather than merely shifting time into another backend path?
3. Does the local reduction survive profiling-off, same standard-PD request/workload/MTP conditions and complete E2E measurement?

First implement/review this one path off-service. CPU/fake-collective tests can verify branch/layout/unpadding and fallback invariants but cannot certify HCCL ordering or backend copy elimination. When separately authorized, the smallest device correctness fixture uses rank-distinct sentinel rows under the actual group, equal positive shard sizes, original-token unpadding, and the source's fallback/alias-sensitive cases. A short matched trace should show the380 list gathers'16-copy pattern removed/reduced, same collective semantics and correct outputs. Then profiling-off matched PD E2E decides Gain. Until then there is no numeric savings estimate and no performance KEEP.

Stop wider runner rewrites, early-output decoupling, KV-barrier changes, new kernels and MTP disabling while this bounded question is active. Continue to preserve the earlier Stop List against summing waits or mapping profON cost to265ms profOFF. The strongest remaining misreading would be to relabel all902ms reported slow-rank host gaps or209.7ms inclusive list-gather cost as guaranteed recoverable product time.
