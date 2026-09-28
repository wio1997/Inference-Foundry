# Run602 Astra next-measurement review — prospective, before live admission

Status: **design review only; Run602 was loading/in progress at inspection. No posterior conclusion, service action or live-source modification.** Frozen DSpark7 behavior, acceptance, cycle trajectory and token semantics remain constraints. All strict numerical Bound endpoints remain null.

## Evidence identity and limits

- Run601 source_gate.json: cb000237994c8183575ecbf1fe5d0f757c04b12c50a91f154b6d027dd80b350f
- Run601 astra_source_review.md: a11ce4be6da9309203b899065f5c3b05701cfd551eb6d27af62d5a5409101560
- Final Run602 preflight review: 1c9d8e4e9dc15887ec7a24f89f388ed56b67efdeb8d66a14160a7f26d382b8eb
- Run602 patch_check: 83d0698f7c97d756446c3f2f121accfa56e6fdd2b82d6dac2a89186f0c81c39a
- Live patch_state/manifest.json: 0d9c6d18385f5b0014e130e8e6f7a466441997673bb54976e60ef9d43484aad8
- dflash_proposer.py: 6be46559f9f869861c676efb4531b0c01eef8a0445521ecadf5c1fec1c4412f8
- llm_base_proposer.py: e3e1ff579e67f845f155c983294ae5546e6ee76ba330098d7867e870c3f9f6e4

Run601 binds nine original sources and five evidence files to anchor locations. Its PASS is not branch coverage or proof that an asynchronous consumer executed a wait. Original runner inspection in this review used Run602's pinned backup, not a mixture of patched and unpatched line numbers.

## Identifiability matrix

| Edge / quantity | Source + Run602 can establish after live gates | Still unidentifiable |
|---|---|---|
| Ordinary Target → proposal | Host call intervals, current-stream elapsed, same execute ordinal | Full resource demand, side-stream task completion, whether a stream span includes waits/idle/contending work |
| Proposal → initial Runtime seed/Draft KV | Observed producer calls by request and 96-token handoff IDs; compact initial basis | Final producer generation for every state object, actual stream IDs and writer/consumer joins, device ready time |
| build_extreme_runtime return → cycle0 | Host construction boundary and Runtime ledger | Host return is not device completion; Python calls enqueue work asynchronously |
| Runtime output → scheduler | Exact retained basis token list joins incoming bulk; actual pre/post IDs and accepted prefix | Whether current bulk/Host ordering is necessary in a redesigned execution architecture |
| Scheduler placeholders → handoff | Placeholder counts separately from actual output IDs | Cached output count cannot identify actual emitted/received prefix; unresolved placeholders are not automatically pending useful outputs |
| API engine IDs → SSE → client | DELTA IDs including suppressed parser events; emitted payload hash joins actual client payload receipt | Token-text first-delivery mapping, socket send completion, timing for suppressed IDs |
| All8 / Host ↔ device timeline | Server time namespace and per-rank Host spans | Cross-device timestamp alignment; current-stream event elapsed is not automatically a Host-correlated absolute timestamp |

This acquisition materially closes Product ownership/accounting uncertainty. It does not identify an Engineering/Scheduling lower time endpoint by itself: a measured Current path is not a lower bound on all legal schedules, and observed node costs are not proven minimum costs.

## Single highest-value next measurement

**A bounded all8 producer/consumer frontier capture at the ordinary-proposal → first Runtime cycle cut, joined to the same acquisition's compact W₀ ledger and Product Host spine.** First finish Run602's offline audit. Run this next only if Run602 confirms the expected chain and leaves this readiness frontier as a material uncertainty; do not reflexively acquire another trace if its actual results point to a larger missing Product edge.

The falsifiable question is: for each handoff slot, are all actual seed/context-KV/state writers already ordered before its first consumer by the current stream or an existing wait; which existing wait/queue segment is exposed before the first Runtime Target/proposer, and which Host publication delays occur after device completion? Answering it separates legal data dependencies from inherited submission order. It does not yet test moving the work.

### Concrete event packet and hook sites

Use semantic keys (run, phase, rank, cohort, execute ordinal, request ID, slot, producer generation, state-object role). Request ID alone is insufficient when buffers are reused or slots reordered. Record opaque stream identity, event generation, Host enqueue start/end, graph/capture/replay mode and branch-taken bits; do not dereference tensor data for these Host metadata records.

1. **Ordinary proposal production.** In the ordinary sample_tokens closure retain Target-return/proposal-start/proposal-end boundaries. Identify the actual _draft_token_ids generation. Record context projection/store enqueue completion around the existing dflash build_model_inputs_first_pass → precompute_and_store_context_kv call, and query/Draft completion at the outer model/proposal boundary. Join affected requests and existing context slot ranges. These are command-enqueue/end-of-stream markers, not immediate Host-ready assertions. If the context hook is inside compiled/captured execution and does not execute per replay, reject that proposed hook; use the existing replay boundary or a compatible profiler marker, never silently treat a capture-time callback as a live event.
2. **Existing side-stream writes and waits.** Record actual branch execution for valid-sampled-count D2H, rejection-count D2H, optional draft-token D2H, sampling_done_event and global-stream state update. At _copy_draft_token_ids_to_cpu explicitly distinguish early return, zeros_only and actual copy. Log existing record/wait/synchronize object+generation associations without adding waits. Record an end marker on an existing writer stream after its final relevant command, and around an existing consumer wait where a real wait exists. If no consumer join is found, retain UNKNOWN; do not repair the algorithm merely to make the diagnostic pass.
3. **Handoff consumption.** Mark the existing input-prep entry/exit and its existing preparation event, 96-token prepared-input binding, build return, first cycle0 Target enqueue, and first Runtime Draft/context consumer. Join producer generations through previous/current slot mapping. A last global proposal call is not sufficient provenance for every request; earlier mixed prefill calls and existing KV blocks must remain in the object lifetime ledger.
4. **Publication spine.** Add Host timestamps immediately after the actual ModelRunnerOutput object is constructed and at its sample_tokens return branch; scheduler update entry/bulk append end/return; API engine-output arrival and existing yields; client payload receipt and DONE. Preserve real pre_ids/placeholders/accepted_bulk_ids. Where a framework transport introduces another queue, add its actual producer/consumer IDs before naming that queue's delay. Run602's JSON flush-before-construction and pre-DONE writes must remain separately labeled observer work.
5. **Clock domain.** Archive boot/time-namespace identity for client as well as server. Device event differences are valid locally only under the tested timing API conditions. Never subtract an NPU event timestamp from Host monotonic_ns or compare raw clocks across ranks. If absolute Host↔device alignment is required, use the installed CANN version's supported correlated trace method and validate its time mapping; otherwise report per-rank local intervals/partial order and Host enclosing brackets.

### Minimal execution and controls

- Preallocate a bounded event/Host-record pool outside the measured window. Record all necessary ordinary producer generations for the four measured handoffs; choose final producers offline. Pool overflow is a failed diagnostic, not silent sampling loss. Use metadata references/counters and fixed-capacity Host records, no new tensor clones, value copies or device reads solely for this frontier packet.
- Read event results only after the existing cohort terminal synchronize; flush records outside the Product timing boundary where practical. Do not insert event.synchronize, stream.wait_event, extra barriers, event-query polling, new collectives or a stream switch that routes model work differently. Recording on a side stream must not migrate subsequent model calls.
- This preserves the program's computation and dependency semantics; **zero timing perturbation cannot be guaranteed**. Events, CPU callbacks and storage all cost time. Capturing an earlier return is not equivalent to executing future work early.
- For ownership/partial-order admission, one successful new same-W₀ diagnostic can suffice. Before using timing magnitudes to narrow a performance interval, require an observer OFF/ON control with identical basis instrumentation, service configuration and warmup. Compare only runs whose initial state, output ledger, count/park trajectory and relevant dispatch/shape ledger match; otherwise present each W₀ separately. Run602 cannot automatically serve as this control, and no historical time may be subtracted as if it were the same state.
- The control calibrates observer overhead; it is not a performance optimization E2E claim. If a subsequent intervention is proposed, retain the existing correctness and repeated formal E2E gates.

## Admission, failure and interpretation

Required: all8 complete event-generation/stream maps and handoff request order; no missing producer or stale reused-event identity; same-acquisition Runtime/basis/scheduler/API/client token joins; final source/script restoration and all cleanup exits zero. CPU negatives should cover stale generation, missing side-stream writer, wrong slot remap, copy-early-return mislabeled complete, graph-capture-only hook mislabeled replay and mismatched Host/device clock domains.

Reject quantitative readiness conclusions if a needed writer/consumer is untraced, timing APIs cannot compare the relevant events, observer controls change trajectory or overhead is unresolved. Retain useful scoped ownership evidence rather than inventing a ready time.

An end event on a writer stream proves its preceding covered commands have executed only when that event has actually completed/read after the existing terminal sync. Its recorded occurrence does not prove the consumer waited. A same-stream or existing wait edge plus a traced consumer provides ordering; an absent wait is neither proof of an error nor proof of overlap until all alias/state dependencies are checked.

A measured exposed wait is Current behavior, not necessarily removable. If a needed producer lies on it, removing a Host wait can merely move the stall to the device. If work is legally movable, attainable overlap still needs resource-contention evidence. Resource work/traffic and capacity uncertainties remain distinct; this experiment cannot promote current traffic to compulsory traffic or attained throughput to C⁺.

## Posterior update slot

Pending Run602 live admission. Update with the actual basis/output identity, cleanup, ordinary-call coverage, real scheduler prefix/retained suffix and received yield-associated IDs. Then select whether the next packet above remains highest value. No new TPS interval or claim of proximity to the limit is justified in this prospective review.
