# Run401 design — count-copy completion, source reuse, and Host consumers

Design only. No source edits, service launch, NPU work, or timing results are authorized by this document. Execution must use the frozen DeepSeek V4 W4A8 / 8×910B3 / DP1TP8 / DSpark7 / c12 contract and the clean Run394 lifecycle. This gate targets the normal cycle64→65 path; it does not prove safety for arbitrary rescheduling or all future executions.

## Hypothesis and useful outcome

The current cycle's asynchronous D2H read of state.num_sampled finishes before the next cycle overwrites that tensor. Its host destination is consumed only after the existing synchronization, then contributes to progress and Draft metadata.

Measure the existing schedule. Add no wait_event, wait_stream, synchronize, host polling loop, barrier or blocking tensor read to the hot path beyond operations already present. Do not replace the D2H source with a snapshot: that would remove the hazard under test.

The minimal successful observation is a positive, uncertainty-resolved same-device completion-to-overwrite margin on each of eight ranks, plus a correctly linked host-copy generation and actual CPU consumption. It establishes observed ordering under the instrumented original schedule. It does not create a source-level happens-before edge or a universal correctness proof.

## Frozen scope

Capture one producer generation c=64 and its normal consumers/overwrite in c+1=65 in every eligible cohort. Freeze the selection at Runtime step entry; never test a cycle counter after it increments. The existing 48-request warmup plus 12-request diagnostic should produce five 12-slot cohorts; tag warmup versus diagnostic separately. Do not promote the 12-request pass to a formal 48-request E2E result.

At these cycles, with zero initial output and at most8 accepted tokens/cycle, no request can reach1024; parking should be absent. Record and assert that condition. A separate future selected parking-generation experiment is needed for the conditional path. Continue logging branch identity so unexpected parking becomes a classification failure rather than a mislabeled next-cycle wait.

## Exact event sites

Preallocate unique timing-enabled event objects and fixed-size records outside the selected hot path; never recycle a recorded diagnostic event across generations. Account for lazy event initialization and validate API behavior in an isolated preflight outside workload timing. Keep the existing production _host_copy_event object, options, placement and reuse unchanged.

| Label | Site and stream | Meaning |
| --- | --- | --- |
| R_BEGIN_64 | In _launch_host_count_copy, inside the existing _host_copy_stream context, after its existing wait_stream(current), immediately before _host_count_copy.copy_(counts, non_blocking=True) | Earliest instrumented D2H execution boundary; optional for the strict safety test, useful for service-envelope diagnosis |
| R_DONE_64 | Same context, immediately **after the existing _host_copy_event.record()** | Conservative timestamp after D2H and its original completion event; does not postpone the original event record to insert a timing marker |
| W_PRE_65 | fixed_decode.advance_state, on its actual current stream, after state.accepted_tokens.copy_(sampled) and immediately before the unchanged state.num_sampled.copy_(torch.where(...).to(...)) expression | Conservative boundary before RHS kernels and the actual overwrite; leave the compound expression unchanged |
| W_POST_65 | Immediately after that unchanged copy_ expression, same current stream | End of the overwrite expression; diagnostic bracket, not precise store-start time |

R_BEGIN is not required to infer safety, so if overhead proves material, retain only R_DONE and W_PRE for the minimal repeat. W_POST helps reject false “race” claims when W_PRE and R_DONE overlap; it does not identify the actual first write.

Store device index, actual stream handle/ID, event identity, cohort/request identity, entry cycle and producer generation with every record. The R_DONE stream must equal the copy stream; W_PRE/W_POST must equal the stream submitting the overwrite. No cross-stream waits are added between these markers.

The W_PRE marker can delay the overwrite slightly and thus enlarge the observed margin. R_BEGIN can delay the copy; R_DONE is a conservative completion upper marker. Quantify instrumentation effects and never treat marginal positive ordering as proof of the uninstrumented case.

## Actual Host consumption sites

Use time.monotonic_ns (or perf_counter_ns consistently) and preallocated host records. Do not add device queries or tensor serialization during the selected path.

In _commit_host_mirrors, capture:
1. function entry, pending flag, copy generation and caller reason (normal_proposer, park_completed_slots, validation/drain);
2. immediately before and immediately after the **existing** _host_copy_event.synchronize();
3. immediately before _committed_emitted_count[:batch].add_(counts): this is the first numeric CPU use of the copied counts, not merely the preceding slice construction;
4. after that add_, before/after the first unique sequence-length mirror add_, and after all sequence/computed-length mirror updates. Record mirror pointer/alias identities without extra tensor cloning in the timed section.

Track generation through launch→pending→sync→consumption; normal commit in cycle65 consumes generation64 even though cycle65's device advance has already been submitted. The launch in65 must not overwrite the pinned host destination until generation64 consumption finishes. Assert this host ordering explicitly.

Record the serving _committed_progress read as a separate progress consumer. It reads the lagged accumulated host count, not the current D2H tensor directly.

Do not label DSpark entry, a device marker, or prepare_inputs_padded entry as “true Host mirror consumption.” In the inspected proposer, prepare_inputs_padded mostly passes seq_lens_cpu/_seq_lens_cpu references; its query_start_loc CPU operations concern a different field. For the first downstream Draft numeric read, resolve the actual Draft backend/builder at preflight and pin its source. Insert host timestamps immediately around its first value-consuming operation on the updated mirror or a derived copy:
- the active _prepare_parallel_draft_seq_lens_cpu branch if it performs arithmetic first;
- otherwise the active metadata builder's max/item/tolist/CPU arithmetic or clone read. For the inspected DSA path, candidates are build_prefill_metadata's _seq_lens_cpu[reqs_start:].max().item() and build_decode_metadata's max().item()/tolist(); these are conditional sites, not an assumption that this backend/branch is used.
Record backend, method, field and pointer/lineage. If no such site fires, report Host-mirror publication/first direct count use only; the downstream-consumer part is INCONCLUSIVE. Preserve reference propagation separately from numeric reads.

## Clock semantics and calculations

Same-device cross-stream timing is sufficient **in principle** for the read-done versus overwrite ordering test; a CPU clock or cross-rank alignment is unnecessary. It is not sufficient merely because both objects are called Event.

Before execution, verify the installed torch_npu/CANN timing contract permits timestamps from these two streams on the same device and that an event after non_blocking D2H denotes completion of the read and host-visible destination. Pin implementation/documented semantics and perform an isolated known-order two-stream timing/CPU-data check outside the workload. If cross-stream comparison is unsupported or uncertain, this gate cannot settle the hazard; use a verified device timeline mechanism in a new design, not a fabricated global clock.

Prefer direct nearby cross-stream elapsed_time(R_DONE_64, W_PRE_65), with both events completed before querying. If the API does not support negative differences, export all selected event-to-common-terminal-anchor elapsed values after the cohort and subtract them. Ensure the terminal anchor is after every selected event by synchronizing those diagnostic events **after the measured loop**, then recording/synchronizing the anchor. Do not reuse the old long-lag anchor without carrying its float/timestamp quantization error. Export direct and anchor-based calculations when supported and compare them.

Define margin M = timestamp(W_PRE_65) − timestamp(R_DONE_64).
- M greater than the documented/measured timing uncertainty: observed D2H completion precedes even the conservative overwrite boundary.
- M overlapping zero or negative: INCONCLUSIVE, not proof of a data race, because W_PRE precedes RHS work and R_DONE follows actual D2H completion.
- Corrupted mirror values or generation mismatch: correctness failure, irrespective of margin.

Report R_BEGIN→R_DONE and W_PRE→W_POST as envelopes. R_DONE is after the existing production event, so the existing CPU synchronize may return before R_DONE executes; do not assert the opposite or infer a broken wait.

Host sync duration and consumption order use only the same monotonic host clock. Relating a host consumer to device completion relies on the existing synchronize's semantics, not subtracting NPU milliseconds from host nanoseconds. Optional near-time bracket calibration may aid visualization but must retain uncertainty and must not add hot-path barriers. Evaluate every rank separately; all8 passing means eight local-order checks, not an all8 makespan.

## Controls and overhead

Use A0–B–A1: two uninstrumented clean controls bracket the sparse-event condition. Each uses the same launch, warmup48/diagnostic12 protocol, request ordering, configuration and source baseline. The A0/A1 difference measures repeat noise/drift; compare B against that envelope. If existing formal protocol requires additional repetitions, obey it; broaden only when results are ambiguous.

Record workload output/acceptance trajectory, cohort/cycle counts, runtime wall, client E2E and host mirror correctness in all conditions. Differences in acceptance or cohort eligibility make a raw overhead ratio uninterpretable. Outside workload timing, measure event/timestamp overhead and resolution on the actual installed stack. Aggregate E2E equivalence alone cannot certify that a particular inserted event did not shift one tiny local margin.

An observed margin must exceed device timing uncertainty and a conservative local instrumentation allowance to support extrapolation to the original schedule. Otherwise repeat with the two-event minimum or retain instrumented-only scope. Even a large margin is empirical slack, not a synchronization guarantee for an accelerated future schedule.

## Metadata and hard gates

- Exact model/dtype/DP/TP/DSpark/batch/context/output contract; hardware IDs and stack/source hashes.
- Explicit actual schedule mode, Target FULL replay versus capture, enable_enpu, is_draft_model/use_eagle predicate, Draft graph mode/backend, profiler/diagnose flags. No competing profiler or altered replay barrier.
- State.num_sampled device pointer, storage range, shape/dtype; pinned host-copy destination; producer and overwrite alias identity; unique host mirror aliases. Production event and diagnostic event identities remain distinct.
- Per-cohort request IDs, client IDs preserved from SSE, source run ID, per-cycle acceptance/useful output, counts generation, selected branch and parking state.
- Exact 60 HTTP POSTs per 48+12 run, maximum client concurrency12, matching request ledger, five Runtime cohorts × eight ranks, all outputs1024, no error, FULL Target, zero post-handoff oracle/ModelRunner calls, host-mirror exact.
- Verify no orphan benchmark/health-poll subprocess before launch; bind service/client run identity and clean up the entire process tree. Source install/restore SHA must match. Record closure after each run.
- Complete expected event/host records on all ranks; selection and pair identity exact. One missing rank or unmatched generation prevents all8 acceptance.

## Disqualifiers and reporting

Disqualify frozen-workload calibration for extra/missing requests, mixed run IDs, external load, wrong branch/mode, changed copy source/lifetime, added synchronization, forced Graph serialization, runtime failures, unmatched pointers/generations, or a changed correctness/acceptance trajectory that invalidates the comparison. Invalid cross-stream timer semantics invalidates ordering, even if event values look plausible.

Report ACCEPTED only for the narrowly observed local ordering and linked consumers, REJECTED for demonstrated correctness failure, otherwise INCONCLUSIVE. Preserve per-rank raw records, minimum margin with uncertainty, branch flags and controls. Neither a safe measured margin nor zero exposed host wait proves all count-copy cost removable or the complete Scheduling DAG closed. Algorithm/Hardware floors and finite Product ceilings remain null.

Source locations inspected for this design: runtime/fixed_decode.py; bootstrap/vllm_dspark_handoff.py; runtime/fixed_serving.py; framework llm_base_proposer.py (prepare_inputs_padded, _propose, build_draft_attn_metadata); DSA metadata builder. Only this design document was written.
