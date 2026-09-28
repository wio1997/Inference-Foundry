# Run592 preflight — native slice timing and arrival acquisition

2026-09-28. Independent read-only Bound review. No NPU/service execution or borrowed-source edits. Recommendation: **reduce existing Run246 all8 traces first; defer a new live window until its incremental uncertainty reduction is explicit.** Run246 contains actual native copy timestamps, so timing-field absence is not a reason to rerun.

## 1. What existing evidence can answer now

Run589 final admission, its astra_final_review.md and Run591 findings.md certify the same-run source-labeled production partial→RS→copy static cut, plus a conditional source-typed HC-post consumer edge. They do not supply device time. Run246 Level1 MemoryAccess profiles supply real device intervals under their own instrumented workload. A cross-run numeric Model45/stream/task match is a hypothesis prior, not a same-generation semantic certificate.

I directly inspected:
- evidence/20260926_loop060_resource/run246/profile/rank0_160893_20260925171406705_ascend_pt/ASCEND_PROFILER_OUTPUT/{kernel_details.csv,task_time.csv,trace_view.json}
- runtime/extreme_decode.py profile begin/end and step hooks.
- scripts/loop081_slice_capture_run589.py and evidence/20260928_loop081_bound/run591/findings.md.
- The established Run580 Level0 collector/offline-export/reducer method and Run247 resource findings from the ongoing review.

The first inspected Run246 rank0 occurrence has:
| Native role / observed signature | Model/stream/task | Start µs | End µs |
|---|---|---:|---:|
| MatMulV2 BF16 [96,1024;4096,1024]→[96,4096] | 45/1/40 |1790356446887733.633|1790356446887748.5135|
| AivKernel hcom_reduceScatter |45/0/13|1790356446887749.773|1790356446887763.813|
| MEMCPY_ASYNC SDMA_SQE |45/1/43|1790356446887765.374|1790356446887766.114|
| HcPost [1,12,4096;1,12,4,4096;1,12,4;1,12,4,4] |45/1/44|1790356446887766.134|1790356446887775.074|

These are a spot check, not an all8 admission or a bound. Exact Decimal trace duration gives producer end; kernel_details rounds producer duration to14.880µs whereas trace has14.8805. In this occurrence the observed P_end→copy_end span is17.6005µs, decomposed into pre-RS gap1.2595, RS task14.040, post-RS gap1.561 and copy0.740. This is an instrumented native interval, not a claimed unmarked Current cost, unavoidable latency or E2E saving.

The copy trace has args Model Id45, Task Type SDMA_SQE, Physic Stream Id1, Task Id43, Batch Id0, Subtask Id4294967295 and connection_id38457. task_time.csv has identical start/stop and MEMCPY_ASYNC role. Thus same-profile trace supplies the Model identity missing from task_time.csv. kernel_details.csv omits this copy; it also contains stream99/task43 QuantBatchMatmul. Task ID alone would produce a false join. connection_id38457 repeats across Graph tasks/occurrences and is not a unique replay key.

### Minimal offline reducer before any live run

1. Inventory every Run246 profile directory independently by rank, worker PID, profiler start metadata and raw/export manifest. Do not concatenate repeated sessions or assume rank directory suffixes are simultaneous windows.
2. Parse trace decimals without binary-float recomputation. Select Model45/stream1/task40 with exact MatMul shape/type; separately Model45/stream0/task13 communication, Model45/stream1/task43 SDMA and task44 HcPost. Verify exported physical stream mapping; record all signatures.
3. For each profile window, partition full model occurrences using ordered complete repeated native task signatures and host target/profile window evidence. TaskId, BatchId0, or connection_id alone cannot partition replay. Require one producer/RS/copy and consistent event-neighbor chain per occurrence; boundary-truncated or ambiguous occurrences fail, not nearest-neighbor fill.
4. Join trace↔task_time by rank/profile, physical stream, task, task type and exact start; check stop/duration within documented export quantization. Cross-check kernel_details shapes and interval precision. Inspect event record/wait rows corresponding to the static chain where exported; event IDs must be certified in this generation before claiming semantic event identity.
5. Report per-rank occurrence counts and paired distributions, all8 min/median/max, missingness and windows. Preserve per-rank vectors; no unsupported cross-device timestamp subtraction.
6. Admission tier H: “Run246 observed instrumented native MatMul→communication→SDMA neighborhood.” Tier S requires independent Run246 source/Graph/cohort evidence binding this neighborhood to the actual layer0 ordinary wo_b partial/RS/copy. Run589 may inform the hypothesis, not close that historical gap.
7. Even if tier S fails, keep tier H timing observations and the exact rejection reason. Do not discard real measurements; do not label them Run589 or cohort5 fixed-W0 cost.

Run246 profiling used stronger Level1/counter overhead and target synchronization in relevant prior windows. Its historical intervals cannot determine Level0 overhead, unprofiled Current or current rank arrival skew. Full contract/acceptance/output parity must be established independently if comparing to current frozen W0. Shape equality is insufficient.

## 2. Minimal future live acquisition, only if needed

### Frozen identity and placement

Reuse Run589 native tags at capture time only:
- partial: SequenceRowParallelOp ordinary quant_method output immediately before tensor_model_parallel_reduce_scatter.
- copy: DSACP layer0 attention output assignment completed at dispatch, tagged by the queue1 sidecar.
- Preserve original HCCL group label; do not overwrite HCCL ExtendInfo. No new Python HC-post hook/graph break.
- Capture strong Graph entry reference, source/script/binary hashes and branch/typed storage descriptors. This records object identity, not replay last-writer.

At runner's existing bind_cohort boundary explicitly gate cohort5 AND exact12 request IDs/run generation. Runtime cycle_index resets per cohort, so the existing start-cycle test alone would profile multiple cohorts. Record rank/local device/worker PID/capture serial/entry descriptor/FULL96/model ID/replay submission ordinal/cycle index in a tiny host ledger.

Place profiler lifecycle outside the selected slice at ordinary cycle boundaries in ExtremeDecodeRuntime.step. Prefer an interior declared window: begin at cohort5 cycle8, one warmup cycle, active cycles9–10, then stop after the next ordinary step boundary. These indices are prospective design choices, not a statement that existing Run246 windows have these identities. Exclude start/stop transition cycles from timing. Use all8 identical logical cohort/cycle gates without adding a rank barrier.

Use CANN/PyTorch NPU CPU+NPU Level0 explicitly, record_shapes=False, profile_memory=False, with_stack=False, analyse_flag=False. Do not silently inherit Level1 MemoryAccess. Reuse Run580 raw collection and post-stop copied offline export; original raw stays immutable with hashes. Export only after service stop, avoiding CPU/storage interference in measured execution.

Do not copy Run580 terminal fixture's barrier/synchronize into production. Do not enable EXTREME_RUNTIME_PROFILE_SYNC_TARGET. Disable optional DAG timing events, diagnostic tensor clones, self-replay and extra counters. The source has optional sync before/after target and optional timing events after stages; these materially alter queueing and are unnecessary when native times are available.

Host record_function scopes around ordinary target execute and one replay-submission ledger entry are sufficient. They certify submission association; return is not device completion. No synchronous writes or tensor-to-host reads in the selected hot slice. Flush ledger after normal cohort completion. Profiler start/stop can itself perturb, hence discarded edge cycles and matched controls.

### Same-generation task and replay join

Each active rank must have its own captured graph, sidecar labels, static dependency/event mapping and raw profile from the same process generation. Discover Model ID and streams from these records; do not hard-code Model45 or task40/13/43 in a new run.

Key static nodes by process/generation/rank/model/graph-stream/task/type, with tagged roles and typed storage. Explicitly map graph logical streams to profile physical streams. Key dynamic instances by profile session plus unique reconstructed replay occurrence/native launch correlation, not Model ID alone. A connection_id repeating across replays is insufficient.

Join host cohort5 cycle9/10 target/replay submission records to device model occurrences using supported profiler launch correlations and complete ordered native signatures. Require replay-count arithmetic, exact occurrence counts, no missing/duplicate target replay, all8 request/cohort/cycle agreement and rank-local monotonicity. If correlation cannot distinguish two queued identical replays, retain an unordered/window observation only; do not invent exact cycle assignment.

For each joined instance acquire P_start/end, RS_start/end and C_start/end from raw native trace/TASK/task_time cross-check. Existing event record/wait nodes are passive observables if Level0 exports them. Preserve event IDs from same-generation Graph; profile event starts may precede producer completion because a wait task can already be queued.

## 3. Arrival semantics and honest interval outputs

Let P be partial MatMul, H the certified RS task, C the output copy. Report:
- task durations P_e-P_s, H_e-H_s, C_e-C_s;
- producer-ready→copy-done cut C_e-P_e;
- observed gaps H_s-P_e and C_s-H_e, plus event wait/record intervals if valid;
- producer-start→copy-done C_e-P_s.

P_e is the partial producer completion. It does not certify all HCCL resources/metadata or peers are ready. Under the certified stream/event prerequisite path and with all other local prerequisites checked, local eligibility to execute H lies in [P_e,H_s]; otherwise call this only a producer-end→RS-start envelope. Event wait release can tighten the local bracket if exporter end semantics are verified. H_s is task launch, not necessarily first network read/protocol arrival. H_e is task completion, not pure link-transfer latency.

Never subtract per-rank absolute P_e or H_s without a documented device-clock calibration/error bound in the same collection. Common wall-like timestamps and simultaneous host starts are not calibration. If clock alignment with maximum errors e_r is available, widen each readiness bracket accordingly and propagate interval arithmetic into rank-arrival spread; otherwise report rank-local brackets only. No inserted barrier to manufacture aligned arrivals.

HCCL duration contains waiting/progress/resource competition. Differences of intervals cannot uniquely identify peer wait, transport latency or removable synchronization. A rank's largest duration is not automatically global critical-path cost. Observed serial constraints prove Current order, not semantic inevitability under another runtime architecture.

## 4. Matched controls and perturbation budget

Use controls with identical capture-time tags, host ledger, source, exact Graph structure and serving contract:
- A0: Level0 off, no extra device markers.
- B: narrow Level0 window above.
- A1: Level0 off again.
All runs need normal ownership/idle/restore checks and fixed output/acceptance/cycle ledgers. If existing full-state replay provides verified identical KV/state/RNG/draft/input/sequence positions, use that same W0; otherwise these are protocol-matched runs, not exact-state matched. A0 and A1 adjacent cycles within one cohort are also not the same workload merely because FULL96 shape matches.

Compare full target/cycle/cohort time with pre-existing ordinary measurements, full algorithm trajectory, thermal/clock/background state and unprofiled A0↔A1 drift. CPU scopes measure submission unless ordinary completion is already part of the method. Do not add boundary synchronizations solely to generate comparable completion timing.

Pre-register an uncertainty budget relative to the intended claim: a proposed microsecond removable gap cannot be admitted if profiler/control uncertainty is of the same size. No universal percentage proves neutrality. If B differs beyond A0↔A1 variability or controls lack power, accept B as explicitly instrumented Current only, reject “negligible overhead” and unmarked savings. Do not correct kernel times by subtracting one scalar whole-cycle overhead.

For this first timing acquisition, a full A0/B/A1 service sequence may have less information value than offline Run246 reduction. It becomes mandatory only before promoting the instrumented cost to unmarked Current or using it to support a gain claim. A two-cycle all8 window is sufficient to test identity/coverage; it does not establish stable latency tails. Expand sampling only for a stated unresolved confidence question, not by default.

## 5. Admission and rejection gates

Admit native Current timing only when:
- all8 raw/export coverage; immutable source/env/runtime/Graph generation; selected branch FULL96 DP1TP8; exact same-run native label/task join;
- complete per-occurrence partial/RS/copy, no task collision, durations finite/nonnegative, timestamp resolution recorded;
- compatible trace/task_time/kernel_details, no silent dropped events or parser errors;
- cycle/request/replay ledger consistent with advertised scope; algorithm output/acceptance/count unchanged;
- service exits, all8 idle and borrowed-source restoration verified before final admission.

Missing one rank rejects all8 arrival/product claims, but may retain clearly scoped rank-local observation. Missing copy rejects copy timing, not producer/RS timing. Missing clock alignment rejects cross-rank skew, not local durations. Missing exact-state controls rejects fixed-W0 causal/overhead conclusions, not instrumentation diagnostic data. Suspicious overlap violating certified dependencies triggers join/export investigation rather than an architectural discovery.

## 6. Which Bound uncertainty shrinks

Offline Run246 reduces uncertainty about the historical instrumented native cost and whether a purported task gap is observable at all. Same-generation Level0 reduces the Current Scheduling DAG node/edge cost uncertainty for one fixed-algorithm cut, including its rank spread and input-ready bracket. It can falsify an assumption that communication dominates this cut, or that a large host/stream gap is present.

Neither acquisition proves any cost removable, unavoidable, or independently composable. No sum over43 layers, profiler-byte sum, HCCL FLOPS/BW shortcut, speculative acceptance change, or perfect-hit cycle floor is admitted. Resource compulsory traffic and exact-board attainable compute/HBM/HCCL remain separate evidence obligations. Whole-cycle dependency/overlap/resource constraints and fixed workload useful-token accounting are still needed before finite Scheduling-aware/Product endpoints or a numeric Current→Bound gap.

Current Formal E2E remains571.681tok/s. Run592 should record narrowed uncertainty and residual questions, not a new finite TPS ceiling. The next live acquisition is justified only if the offline reducer cannot close the specific same-generation/cycle/low-perturbation uncertainty needed by the model.
