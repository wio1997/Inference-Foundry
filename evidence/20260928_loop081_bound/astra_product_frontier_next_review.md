# Astra High: same-W0 Product frontier acquisition review

Scope: independent read-only review of Run239/241, Run341, Run597 and Run599. No service/NPU action, source mutation or new benchmark. Different runs remain different workloads and clock domains; no historical durations are combined into one W0.

## Decision

Prioritize a narrow **same-new-W0 Product preparation -> Runtime -> delivered output ledger** over an isolated Target-to-Draft overlap timing run. Reuse existing boundary/basis collectors after exact-source checks. Do not repeat Run239's coarse phase experiment or claim the Run99 client-minus-Runtime remainder is removable. The missing result is an identity- and dependency-joined timeline, not another phase sum.

Run239/241 already establish a large preparation region in one instrumented pass; Run341 provides its naturally admitted shapes and current-stream costs in a different pass; Run597 provides a full Runtime basis in another pass. Run599 identifies a possible internal fork but provides no timing. No single existing run supplies all four.

## Evidence coverage and missing-field matrix

| Product edge / fields | Run239/241 | Run341 | Run597 | Run599 | Minimum missing field in one new W0 |
|---|---|---|---|---|---|
| Client request -> engine request -> rank/cohort/slot | client timing; cohort mapping largely by order | response/request IDs joined to Runtime; worker entry IDs | admitted 96 response IDs, dataset and all8 cohort/slot joins | none | preserve exact ID normalization; add client send/first SSE/usage/DONE/end and server admission keyed by ID |
| Arrival -> prefix-conditioned residual work | first worker execute and scheduled-token count | new/cached req IDs, prompt_len, computed prefix, scheduled/spec counts and finished IDs | frozen request bodies; initial Runtime state only | none | per-request actual prompt length, prefix-hit/computed prefix, each residual position interval, cache/block-table generation |
| Residual prefill execution | execute-entry timestamps, coarse first-execute->handoff | original _model_forward Host start/end/thread CPU + current-stream event | absent | none | bind each call to execute ordinal/request positions; record return plus actual output writer-stream completion |
| Last prefill -> initial DSpark/seed/state/KV-ready | coarse handoff/build region | runtime_built is Host only | cycle0 last/draft/positions and sparse input witnesses | internal steady-cycle source fork only | initial seed producer call, token-position ledger and ready events, required metadata/KV writer streams, existing consumer waits |
| Handoff output ownership / remaining Product tokens | Runtime count matrices; pre-handoff published p_i missing | cached num_output_tokens is scheduler state, not delivered-output proof | external 1024 and Runtime staging joined by request; no full prefix->suffix ownership proof | active certificate count explicitly not Product work | per-request pre-handoff generated, worker-published, server-emitted and client-observed token prefixes; sequence/token IDs, truncation and ownership |
| Runtime entry -> first Target -> final Draft/serve completion | Host serve start/end; counts and parks | full Runtime rows, no complete time frontier | full accepted/count/next-draft/park basis; sparse actual entry witnesses; perturbing | conservative source cut after binding.forward including output gather | Host serve interval, rank-local first Target input-ready/replay, final consumer completion; keep compact full Runtime basis |
| Worker output -> engine consumption -> server SSE -> client completion | publication_ns is output construction, BEFORE file dump/return; not SSE publication | worker preparation-focused | raw client SSE/admission and response IDs; no worker-to-server timing joins | absent | worker result ready/return, engine bulk-output consumption, server emission of final token/usage/DONE, client corresponding receive/end |
| Shared clock and complete asynchronous dependency | same-host Host marks; no full device frontier | current-stream scope may miss side streams | no complete timing DAG | explicitly no ready/overlap claim | Host boot/time-namespace/monotonic clock identity; rank-local device event relations and existing cross-stream waits; never subtract raw timestamps across cards |

Run239/241 are one source trajectory and a derived calibration, not independent trials. Run239 'publication' is a naming limitation. Run341's 353-384 ms is the range PER prompt-bearing call, not the total of26 calls; do not reuse the abbreviated Performance Map wording as a sum.

## Smallest live acquisition

One frozen warm48 + measured48 c12 diagnostic pass, with all four measured cohorts covered, is enough for this identity/frontier question. It is a NEW W0; do not demand or imply equality with Run99 or Run597. Three formal repeats are for a subsequently justified product intervention, not required to establish this diagnostic ledger.

1. **Identity/Host spine:** use preallocated in-memory records at client, API/engine and all8 worker boundaries. Record run ID, dataset index/hash, response ID, engine ID, rank/PID/cohort/slot, execute/call ordinal and monotonic_ns. Include actual send/admit, all execute entries/returns, runtime construction begin/end, serving begin/end, worker output ready/return, engine consume, SSE final token/usage/DONE and client receipt. Preserve asynchronous semantics; no admission holds or wave barriers.
2. **Preparation work:** reuse Run341 new/cached/scheduled fields. Add explicit per-request prompt versus generated-token position intervals and output ownership at handoff. Record prefix-cache and block-table/storage generations; neither prompt length nor num_computed alone establishes cache contents.
3. **Minimal all8 device frontier:** preallocate events outside the measured path. Mark original residual-prefill call boundaries, initial Draft/seed producer boundaries, first Target input-ready/submission and final Runtime completion on the actual participating streams. For metadata/KV/seed side streams, record completion after their last writer and the existing consumer wait identity. Record events only; do not insert new joins, synchronize, or query readiness mid-window. A current-stream end event is a complete-ready witness only after its existing dependencies are audited. Missing stream ownership remains explicitly unknown.
4. **Frozen Runtime basis:** reuse compact Run597 first state + count/token + next-draft + explicit Host park ledger, with cycle0/1 and first-postpark checkpoints. Reuse existing Host-readable outputs where possible; keep copies preallocated and defer decoding/JSON writes. Full per-cycle aux/logit dumps and all-operator profiling are unnecessary for this question.
5. **End-of-window flush:** read events only after the already-existing cohort/final synchronization and keep serialization outside client timed wall where feasible. If flushing before subsequent cohort/publication is unavoidable, timestamp it as observer work and reject it as an unmarked Product timing estimate.

For the first implementation, missing direct API admission instrumentation may be bracketed by client-send and worker-entry, but must stay a combined interval, not be mislabeled scheduler or transport cost. Missing side-stream ready events similarly permit a Host-spine result but not a complete device-ready DAG.

## Algorithm and correctness invariants

DSpark7, draft/Target algorithm, acceptance logic, c12 admission policy, max1024, output ordering, prefix-cache policy, Host park behavior and graph/source configuration remain unchanged. No oracle counts, forced seed, acceptance replay, early publication, refill, branch reordering or cache copying that changes ownership.

A1 records its actual full algorithm trajectory rather than substituting another run's. Same-source A0/A1 controls cannot be assumed same W0: compare actual cycles, active masks, prefix hits, preparation shapes and output ledgers, and report differing trajectories separately. Use same-state differential checks at sensitive new hooks. The observer may alter Host park timing; a changed ledger is a perturbation finding, not algorithm improvement.

At least a bounded A0/A1 observer gate is necessary before interpreting timings: measure marker/copy/flush overhead and changes to queueing, cohort wall and Host park counts. If overhead is comparable to the interval being interpreted or controls differ materially in work, retain identity evidence and decline uninstrumented-time claims. No arbitrary universal percentage is justified by current evidence.

## Required admission / negatives

- All96 responses, warm/measured distinction, exact request-body/dataset join, all8 identical cohort/slot identity,64 Runtime rows and32 measured basis records; legal48x1024 client output ledger.
- Each scheduled prefill/seed position has an owner and subsequent consumer; detect missing/duplicate call ordinal, wrong rank/request, wrong cache generation and shifted cycle.
- Worker-generated, worker-published, server-emitted, client-received and Runtime-staged counts are separate fields. A deliberately incorrect pre-handoff p_i or duplicate token prefix must fail reconciliation. Preserve overshoot explicitly.
- Host timestamp order must satisfy causal edges within its verified clock domain. Client waves are allowed to overlap; do not add wave durations or per-call rank maxima as a critical path.
- Device event recorded on the wrong stream, missing last-writer event or absent existing wait must fail complete-ready admission. Do not manufacture a wait merely to pass it.
- No .item(), new global synchronize/barrier, file I/O or dynamic allocation inside compiled FULL Graph hooks. Exact-source patch anchors, CPU negative tests and same-state semantics precede service.
- Hash all raw artifacts/source/collector, record observer work, exit codes, owned-process stop, all8 idle and exact restore with mandatory cleanup keys.

## Bound interpretation

The product is a same-W0 **measured Current causal spine with unresolved branches explicitly marked**. It can distinguish preparation work, exposed Host submission, initial seed/KV join and actual output delivery, and locate which uncertainties dominate the Product interval. It cannot by itself certify compulsory work/traffic, maximum cumulative C+/B, ideal overlap or a numeric TPS ceiling. Formal Current remains Run99 571.681 tok/s; strict Resource/Scheduling/Product finite endpoints remain null.

## Reviewed primary paths

- scripts/loop045_boundary_patch.py; run239/boundary/rank0_cohort5.json; run241/calibration.json; prior all8 Run239/347 reviews.
- scripts/loop074_prefill_ready_patch.py; scripts/loop074_run341_phase_patch.py; run341/marks/rank0.jsonl; run341/findings.md.
- run597/live/b/{basis_admission.json,client_admission.json}; Run598 independent all8/readmission review.
- run599/source_gate.json v2 and run599/astra_review.md.

All paths above are under /data/wio/Inference_Foundry, with evidence under their existing dated Loop directories. Raw source/evidence SHA bindings for this review are saved in the adjacent provenance JSON.
