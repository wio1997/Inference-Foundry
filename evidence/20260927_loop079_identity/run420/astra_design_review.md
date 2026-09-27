# Run420 — independent Run419 handoff and row-identity design review

2026-09-27. Requested role: Astra High. Session exposes GPT-6, not an independently verifiable backend variant/effort; preserve parent dispatch metadata separately. Only this file written. No source edits, service/NPU work or process control. Docker inspect and docker exec were used only to read mount metadata/source hashes; no framework import or device initialization.

## Verdict

ACCEPT Run419's narrow conclusion: configured/internal zero is established; external pre-handoff published p_i=0 is not established for Run403. Existing source contains no zero-output handoff predicate, and the saved clean/length gates cannot supply it. A cheap Host-only authoritative Scheduler bulk ledger should be the first additional hook. A full API/SSE hook is conditional: it is unnecessary to prove zero if a stronger append-only zero-generated-output certificate succeeds.

Run407's source/runtime branch certificate remains the minimum sensible row-identity approach. Bind it to the compiled graph and selected replay; use shadow labels only at unresolved transformations. Do not instrument43 redundant label collectives.

## 1. Mounted-source findings and the false zero inference

Current container dsv4ab bind mounts the host framework trees into /vllm-workspace/vllm and /vllm-workspace/vllm-ascend. Container-read hashes match Run419 for:
- ModelRunner:004dbd0d5b1a5c3f9533fa1d12327bd0ae421fe24bb2674544b2c4a25d74aaba.
- KVDelivery patch:5ba894b9ec003502c008e12281c260787ed64a419327c451c8e67374bf525e8d.
- Request:92124fbad28cda49bd06fa12c2c4fd5f53fc9381ddb4dc35f275c5ccfbd27378.

These prove current mounted file identity, not which old native/Python objects were loaded in every Run403 process. Run403 install manifest lists its six instrumentation patches; ModelRunner/Scheduler are not among them. Keep historical provenance distinct.

ModelRunner model_runner_v1.py:
-2170–2195 forms req_ids and gates a new cohort on12 requests,96 scheduled tokens, speculative metadata and not-already-served. It does NOT require CachedRequestState.output_token_ids to be empty.
-2280–2315 bootstraps last_sampled_tokens from column0, draft_tokens from columns1..7 and positions from the existing input batch. Existing draft inputs do not establish prior publication or its absence.
-3389–3394 hardcodes initial_output_counts=[0]*12.
-3411 handoff_before_model_forward=True means before this intercepted forward; it does not mean before every prior prefill/sample/output for those requests.
-3457–3473 returns the1024-token Runtime frame through extreme_bulk_output.

KVDelivery patch:
-966 snapshots len(request._output_token_ids).
-970–978 routes bulk output to KVDeliveryScheduler._update_request_with_output.
-1194–1202 appends tokens individually, calls check_stop, and trims the rest of the incoming list when stopped.
-Upstream sched/utils.py:114 stops once request.num_output_tokens reaches max_tokens.
-1068–1074 constructs EngineCoreOutput from the resulting new_token_ids.

Therefore the following counterexample is source-consistent: g>0 tokens already appended before handoff, Runtime produces1024 more, Scheduler admits only1024-g, client finishes at1024. Exact client length, Runtime generated_output_counts=1024 and zero configured initial count all still pass. This is a logical possibility, not an observation that Run403 actually had g>0. Run419 must not promote p_i=0 or infer a discovered semantic bug.

OutputProcessor processes EngineCoreOutput.new_token_ids at628, detokenizes at653, and enqueues RequestOutput at678. It maps internal request_id to external_req_id through RequestState. Chat serving emits SSE chunks with its own request_id; role/empty/prompt-echo chunks are not generated output tokens. Current mounted hashes:
- output_processor.py:ee10351275d90796c8b901a5f4b23d5a046ef6ee72fd2921aff2ae78ca58bd9b.
- chat_completion/serving.py:860e27d548ec7bb0497a46e0356a5acc5500998944cffa5e67e336041693c067.

## 2. Smallest request-correlated ledger: staged design

### Stage A — authoritative zero test, no device instrumentation

At the existing Scheduler bulk branch immediately BEFORE update_request_with_output, append one buffered Host record per req_id:
- run/process and concrete Scheduler class/method identity, internal request_id and request-generation identifier;
- scheduler step/model-output transaction identifier, extreme_bulk_output;
- g_before=len(request._output_token_ids), incoming_bulk_len BEFORE mutation;
- request.max_tokens, num_prompt_tokens, resumable/streaming-input state, stale/drop/preemption/restart status;
- Host monotonic timestamp;
- after update: g_after, admitted_delta, outgoing new_token_ids length, stopped/status and emitted EngineCoreOutput length.

Check g_after-g_before == admitted_delta == outgoing length for this ordinary generation path. Capture the incoming length before the callee trims the list; retaining a reference and measuring it later loses the original bulk length. A checksum/prefix identity is useful to detect truncation, but counts suffice for the zero test.

Join each record to existing rank0 Runtime req_ids and cohort, requiring exactly one bulk transaction per request generation and all8 slot order agreement. Add a cheap rank0 handoff record at the actual start containing req_ids, each CachedRequestState.output_token_ids length, prompt length, CPU known position/count metadata, schedule identity and timestamp. The authoritative bulk hook alone can prove no earlier generated outputs under the conditions below; the handoff record disambiguates generations/timing and diagnoses positive results. Neither requires hot-path D2H or per-cycle logging. Buffer records and flush after the cohort.

Zero-promotion conditions:
1. Every relevant bulk g_before==0; unique request generations and complete record coverage.
2. Normal generation is append-only from request admission through this bulk record; no prior-output reset or ID reuse.
3. All generated API outputs originate from the covered Scheduler output path; no connector/echo/custom bypass contributes generated tokens.
4. Actual classes/overrides, source hashes and run flags establish those premises; bulk records join the measured handoff/request trajectory.
5. Frozen success/output gates remain satisfied.

Under these conditions p_i at handoff must be0, since publication of a generated token requires an earlier append. A later authoritative zero is stronger than a handoff-local cache zero. No API hook is required for this zero proof.

Important reset exception: upstream Scheduler._update_request_as_session clears session._output_token_ids at1314 during streaming-input updates. The Request list is NOT universally monotone. Record non-resumable/no-session-update operation, or explicitly ledger generation/reset events. Current Run403 gates do not independently record this premise. CachedRequestState may lag async Scheduler state; its zero alone is insufficient.

### Stage B — only if Stage A is positive/inconclusive

g_before>0 measures previously generated/accepted tokens, not necessarily tokens already published at handoff. Distinguish:
- generated before handoff;
- queued in EngineCore/OutputProcessor;
- yielded by API;
- actually received by client.

For exact API-published p_i, add the smallest mapping/egress ledger:
- OutputProcessor records internal request_id↔external_req_id and generation plus token-delta/cumulative ordinals;
- chat stream records actual response/SSE id, choice index, contributing output token ordinals and timestamp at generated-output yield, excluding role/usage/prompt echo;
- rank0 records the handoff boundary with the same Host monotonic clock domain and uncertainty.

If “published” means ASGI send rather than generator yield, correlate chunk sequence to send completion; do not silently equate yield with receipt. Text detokenization/reasoning parsing can delay, suppress or regroup bytes; count output token provenance, not text retokenization, number of SSE chunks or usage at completion. Borderline timestamp intervals must remain unknown. To test zero, a proof that NO generated output reached an earlier API queue is sufficient and may avoid precise wire timing.

A timestamp alone without internal→external→SSE identity is not a request ledger. Existing client Run403 JSON lacks this direct ID join; do not invent it from client index.

### Run419 validator corrections before promotion

The script checks string occurrence, not control-flow/path applicability. gate_available=bool(gate) only checks nonempty JSON and is not validation of the gate verdict. Require explicit gate status/schema, source/provenance/coverage and per-request uniqueness checks. Its capture/runtime assertions are useful narrow structural checks; source anchor presence cannot prove the output lifecycle.

Five-cohort640 cycles is a conditional ordinary-DSpark7 capacity relaxation, not a Product floor. Recompute per-request external remaining output once p_i/other already-generated reusable outputs are known; do not transfer diagnostic five-cohort arithmetic to formal four-cohort Run99. Closing p_i=0 does not certify every Runtime-retained token is externally delivered under every other stop/format path.

## 3. Minimum-perturbation all43 router row certificate

Reuse Run403 route buffers and Run407's composed maps. First add only Host metadata at capture/handoff and selected-boundary snapshots exported after the cohort:

1. Seed actual labels once: (request generation, slot, candidate offset, position) from the real input/query layout, with padding=-1. Attach Runtime req_ids directly to the capture metadata; cohort-local slot text is an avoidable limitation. Token values/positions alone are not unique identity.
2. At graph capture, enumerate all43 layers' concrete embedding/attention/linear/MoE prepare/finalize/dispatcher branches, native op/build identities, shape/stride/live/padded sizes and ordered TP/EP/DP/PCP ranks. Record FlashComm1 and separate sequence-parallel-MoE flags, DSA CP attention state/skip-A2A/full-weight output path, mmrs and prepare class.
3. Bind descriptor hash to compiled graph/cache key, capture generation and actual route-buffer outputs. Capture-time hooks certify a replay only through this association; a Python hook that does not execute inside replay is not a per-cycle branch observation.
4. At selected replay export actual input IDs/positions/query boundaries/active mask, actual target_logits_indices and route-buffer generation for cycles64/65, joined to the same req_ids. Snapshot on the producer stream or after the existing join; retain buffer lifetime until export.
5. Compose maps offline through ordered gather, pad/unpad, chunk/reduce-scatter, DSA query/head redistribution, residual merges and expert inverse permutation. Corresponding residual branches must carry the same query labels before addition.

For verified N=96,TP=8,L=12, the normal DSA head A2A can reconstruct global row12*r+u by concatenating source rank blocks while redistributing head chunks. This is a conditional algebraic proof; capture the active branch/group order and native query-output contract. Full-weight/skip-A2A has a different map. Do not infer it from the normal branch.

No per-layer label collective is needed when the same compiled operator's row-order contract and actual indices/group order prove the map. At an unresolved native transform, perform one targeted semantic check or propagate labels using its ACTUAL indices/permutation metadata. A parallel guessed label map or independent arange at each router merely restates the assumption. Do not inject labels as floating model features: they change computation/routes and are not a neutral identity test.

Promotion requires all43 exact pre-dispatch router rows joined to the real candidate labels and selected topk_ids, source/binary/replay provenance, all rank consistency and a held-out selected replay/branch check. If one native contract remains unproved, retain a conditional result rather than force global success. Sampling late/parked behavior is needed only for a claim covering that branch; an early64/65 certificate must stay early-normal-path scoped.

This promotes exact measured retained expert unions for the certified trajectory and checked formats. It does NOT make them compulsory HBM reads, prove causal row pruning or close Draft's noncausal/body/head/state dependencies. The observed acceptance mask remains hindsight information.

## Evidence limits

Reviewed Run419 script/JSON, Run403 install/gates and prior Run405/407 source audits, actual current mounted ModelRunner/KVDelivery/Request/output publication sources, and the streaming-reset exception. Did not independently re-evaluate all40 raw capture matrices or certify historical loaded binaries. No existing p_i value was inferred beyond what those records establish. Next Run can combine Host-only ledger/capture descriptors with already-authorized identity acquisition; no standalone service run is needed solely for source-anchor counting.
