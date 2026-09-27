# Run426 — minimal handoff-to-publication ledger design

2026-09-27. Requested independent Astra High review. Session exposes GPT-6, with backend variant/effort not independently attested; preserve parent dispatch configuration separately. DESIGN ONLY. No service, NPU, benchmark or borrowed-source operation performed. Only this file is written.

## Decision and scope

Use four instrumented source areas for the core ledger: **ModelRunner boundary; Scheduler ordinary+bulk committed output; OutputProcessor ID/raw-output/queue boundary; Chat parser-consumption+actual yield boundary.** All first-pass records can use existing Host data and buffered per-process logs. Reuse Run421 terminal bulk accounting. Add ASGI/client observation only if the question explicitly requires send/receipt, not as a prerequisite for generation-work accounting.

The minimal useful result is not necessarily a single p_i number. Return a request-correlated record distinguishing committed output, raw-token delivery/consumption, yielded bytes and any publication watermark. A raw model token can be a control token, be suppressed, or contribute bytes only in a later parser delta. Under the unchanged API, raw-token “publication count” has no automatic one-to-one interpretation. Do not force a false scalar by summing len(output.token_ids) only on yielded chunks.

Run421 proves584 Scheduler-committed tokens before terminal bulk append and584 truncated Runtime suffix tokens. It does not locate those commits/publications before handoff. Nor does it measure device-computed reusable output at handoff.

## 1. Define endpoints before acquisition

Choose the primary Host cutoff H_i explicitly: the pinned ModelRunner site immediately before invoking FixedCohortServing.run for the cohort, after fixed Runtime state/buffers have been assembled. Call it **H_run, Runtime-suffix entry**. Record the source line/descriptor, not just the old handoff_before_model_forward boolean. This is not automatically the earlier bootstrap ownership transition at build_extreme_runtime nor a device-ready instant. If that earlier boundary matters, record H_bootstrap as a second cheap marker in the same source area and report both; do not retroactively equate them.

All eight ranks record local H brackets without a barrier. Preserve rank0 cutoff and [min H_rank, max H_rank] global transition envelope. Events before its beginning precede every rank; events after its end follow every rank. Events inside it are rank/cutoff-specific. A maximum of unrelated durations is not a common-clock timestamp.

For each request generation define:

| Symbol / event | Exact meaning | What it does NOT establish |
|---|---|---|
| D_i(H) | Valid, reusable output tokens whose required device computation has completed by H | Not directly measured by a Host cache/count hook |
| G_i(H) | Output ordinals already appended/committed to Scheduler Request by H | May lag device computation; may precede external publication |
| R_i(H) | Raw output ordinals received by OutputProcessor by H | Not necessarily queued or sent |
| Q_i(H) | Raw ordinals represented in RequestOutputs submitted to output collector by H | Collector may merge outputs; not API consumption |
| A_i(H) | Raw ordinals actually consumed by Chat generator/parser by H | Parser may emit nothing |
| Y_i(H) | Complete SSE frames/bytes yielded by Chat before H, with exact channel/choice identity | Not ASGI send completion or client receipt |
| S_i(H) | Response-body bytes accepted by ASGI send await before H | Not TCP acknowledgement or peer delivery |
| C_i(H) | Bytes/complete SSE frames observed by benchmark client before H | Not when server computed/committed them |

Use per-request ordinal sets/contiguous prefix endpoints, not just totals, to detect duplication or omission. p_i must name its chosen boundary and unit. For raw-token publication, distinguish (a) explicitly exposed raw token IDs, (b) raw ordinals whose visible contribution is released by a parser with proven provenance, and (c) raw-consumption watermark associated with a chunk. Only (a)/(b) support literal token-publication counts; (c) is a watermark, not proof every consumed token was disclosed.

For the work numerator, D_i(H) or a certified already-available continuation prefix is relevant. G_i(H) is a useful committed lower observation under valid normal execution. Do not count1024-p_i as remaining generation when already computed/queued tokens exist, and do not substitute terminal G_i(bulk) for D_i(H).

## 2. Source findings that determine the design

- Ascend model_runner_v1.py:4436–4461 appends sampled_ids=[-1] in async mode. Upstream gpu_model_runner.py:1356–1365 can optimistically append prev_num_draft_len additional placeholders;1406–1410 trims inflated cache lengths. CachedRequestState.output_token_ids length is neither a monotone committed count nor proof of device-generated tokens.
- Ascend input preparation:1054–1061 keeps optimistic CPU seq lengths;1132–1155 corrects authoritative device values using prior accepted counts. CPU positions/computed counts must retain their optimistic/stale provenance.
- CachedRequestData contains per-request num_output_tokens, num_computed_tokens, resumed_req_ids and optional all_token_ids; it is a Scheduler snapshot taken for that schedule, not current Scheduler state at H. new_token_ids is normally empty without PP. Capture its actual values, not an assumed returned-token list.
- Scheduler's append/stop path precedes EngineCoreOutput.new_token_ids publication. Ordinary AsyncScheduler override adds placeholder/cache bookkeeping; bulk explicitly calls KVDeliveryScheduler base. Capture both paths at their common update_from_output append/result boundary.
- Request._output_token_ids is append-only for ordinary non-resumable requests, but Scheduler._update_request_as_session:1314 clears it for streaming-input updates. Preserve generation/reset/stale/abort semantics.
- InputProcessor.assign_request_id:232–240 records external_req_id then normally appends a random suffix to internal ID. Never recover identity by stripping a guessed suffix.
- OutputProcessor.add_request:530–551 has the authoritative internal/external mapping. make_request_output:283–310 can suppress FINAL_ONLY/stream-interval outputs. RequestOutputCollector.put:62–72 merges DELTA RequestOutputs while the producer outruns its consumer; queue puts and Chat frames are not1:1.
- Chat serving:303–305 maps one API request to one or multiple engine subrequests. At624 previous_num_tokens increases BEFORE parser-suppression continue640–647. A None delta may consume control tokens or buffered text without yielding; a later delta may expose earlier material. Role/echo/usage frames are separate from generated content/reasoning.
- The current stream can omit raw token IDs. Turning return_token_ids on changes the protocol and is not part of this diagnostic design.

## 3. Minimum core hooks

### H — one source area, cohort-entry Host certificate

At H_run record a bracket around copying existing Host metadata into an in-memory record:
- run, process/rank, cohort, ordered internal request IDs and per-request generation identity;
- H_begin_ns/H_end_ns, canonical boundary name and source SHA;
- concrete ModelRunner class, async flags, actual schedule object fields for those req_ids;
- CachedRequestState output list length and placeholder count, plus short value/hash evidence if useful; explicitly label cache-derived;
- prompt length, CPU num_computed/num_tokens_no_spec and previous draft lengths with optimistic flags;
- schedule snapshot num_output_tokens, num_computed_tokens, resumed IDs and scheduled speculative lengths;
- Runtime's configured initial counts/max length and source/replay identity.

Do not call .item(), .cpu(), synchronize, or a fresh collective. Request IDs are Python strings; do not copy them to NPU.

This hook locates H and captures discrepancy evidence. It does NOT certify D_i(H). Where existing bootstrap inputs contain immutable saved device state, export them later under their established lifetime/order contract, not by reading mutable buffers after the cohort. If eventual D_i(H) proof needs a new device snapshot/marker, treat that as a separate targeted gate after these Host results expose the remaining ambiguity.

### G — Scheduler ordinary AND bulk output commit/emit ledger

Extend Run421 beyond extreme_bulk_output: at the common pre-update and post-update sites for EVERY nonempty generated-token result of the measured requests, record:
- internal req_id/generation, monotonically increasing Scheduler-local record sequence;
- before/after time brackets, g_before/g_after, incoming/admitted delta and actual emitted EngineCoreOutput token length;
- admitted raw token IDs or immutable ordinal-range plus digest; copy before later list mutation;
- normal/bulk flag, stale/drop/reset/resumable/status, source class, input/output transaction descriptors;
- current observed scheduler-step value labeled as observation, not assumed origin step.

Include a record for reset/abort/streaming-input-generation change or assert with evidence that none occurs. A zero-output result can be summarized unless needed to resolve an outstanding transaction.

This suffices to reconstruct committed G_i(H) from clock-bounded append events without sampling Scheduler state synchronously at H. A commit bracket straddling H gives an interval; do not pick its midpoint. Do not label self.current_step the producing execution's ID in async mode. If origin order is unresolved, add a serialization-safe explicit diagnostic sequence to the existing request/output transaction or a verified immutable descriptor; an arbitrary dynamic Python attribute may not survive RPC serialization. Request ID, ordinal continuity and actual token hashes often avoid adding such a wire field.

Capture after actual EngineCoreOutput creation in the same source area if later stale/error/structured-output branches can change emitted tokens after append. The existing bulk-only after-hook is before downstream emit handling; it is adequate for append arithmetic but not a full emit ledger.

### R/Q — OutputProcessor receipt, ID map and queue certificate

At receipt of each EngineCoreOutput for a measured req_id, use req_state to log:
- internal ID, external ID, parent ID/choice index if present, request generation and output_kind/stream_interval;
- immutable new_token_ids ordinal interval/digest and receipt bracket;
- prior/after detokenizer raw output count;
- whether make_request_output returns None;
- actual RequestOutput raw token range/length submitted to queue and queue-put bracket.

The per-output record can carry the ID map, so a separate mapping hook is unnecessary for successful requests that emit output; add one admission/reset record only if complete lifecycle or empty-output/error mapping is required. Capture RequestOutput values BEFORE collector mutation/merging. Logging a mutable object reference and serializing it later is invalid.

For frozen n=1,DELTA, source-proven lossless aggregation allows joining successive Chat-consumed token ranges to the queued raw-token prefix without a dedicated collector hook. Validate this by ordinals/counts/hashes. Add a collector put/get hook only if that join fails, modes differ, or exact queue occupancy is needed; no need to instrument every queue preemptively.

### A/Y — Chat consumption and actual generator yield

At each res.outputs entry:
- API request_id, res.request_id/subrequest ID, choice, parser class/options and output_kind;
- consumption bracket and raw token IDs/range BEFORE parse_delta;
- delta text/hash, parser result kind (None/content/reasoning/tool/control/terminal) and previous_num_tokens before/after;
- whether this iteration suppresses/continues or yields.

At EACH actual yield site used by the frozen path record frame sequence, exact encoded byte length/digest, frame class, raw consumption watermark and yield-before time. Do not mark a would-be chunk as yielded before a continue branch. Include role/usage/echo/error categories separately; never credit those as generated raw tokens.

If the async generator is resumed, an after-yield timestamp is a resumption marker, not a direct send completion timestamp. A wrapper around the output generator can observe delivered bytes without modifying wire content; correlate it to inner parser records by exact frame bytes/hash and sequence, retaining duplicate-frame ordinals.

This closes exact Y_i bytes and the API-consumption watermark at H. It closes literal raw-token publication only when the parser's contribution mapping is separately established. Current source does not provide that mapping automatically: skipped control tokens may never be visible, buffered text may emerge later, and one output token may contribute zero/multiple bytes. Report UNKNOWN or a conservative provenance interval instead of renaming A_i as p_i.

## 4. Optional send/receive endpoints

ASGI S: an opt-in wrapper around send records entry/return of http.response.body, body byte length/hash, more_body and connection/request identity. Preserve every byte and await semantics. A successful await means the ASGI server accepted/processed the send according to its flow control; it does not certify delivery/ACK. Match yields to the concatenated body byte stream because middleware can split/coalesce frames. Record exceptions/disconnects. No global middleware instrumentation is needed if a request-scoped response wrapper suffices.

Client C: extend bench logging at SSE parsing to retain returned SSE id, choice/frame sequence, raw byte/digest and receive monotonic timestamp; retain request index↔response id. Complete-frame receive is an observation after bytes reach the client parser, not the first packet arrival. Do not infer every frame timestamp from TTFT, end time or chunks count. No changes to prompts, token-return flags or c12 workload.

S/C are optional for the next bound calibration. If the product question is explicitly client-received p_i, these become required and parser-token provenance remains a separate issue.

## 5. Clock and causal binding

Use time.monotonic_ns consistently for new Host records. At process start and finish record:
- host boot_id;
- Python clock implementation/resolution/monotonic/adjustable metadata;
- PID, container/process time-namespace identity and accessible timens_offsets;
- run generation and source/helper hashes.

Same Linux host is not alone proof of identical clock origin: time namespaces can offset CLOCK_MONOTONIC. Compare actual namespace/offset metadata for worker, EngineCore, API and benchmark. Existing client perf_counter values must not be mixed with new monotonic_ns merely because both increase; verify implementation/domain or bracket both in the same process.

If all relevant processes demonstrably share the same monotonic domain, no wall-clock synchronization or network protocol is needed. If not, use bounded ping-pong clock-offset intervals through an existing control/Host channel outside measured hot paths. No symmetry assumption; retain RTT/asymmetry as uncertainty and repeat to bound drift. Wall-clock date/NTP is not a substitute.

Represent H and each commit/yield/send event as [lo,hi]. An event ending before H begins is definitely before; one starting after H ends is definitely after; overlap is unresolved. Sum ordinal deltas definitely before for a lower count and all possibly before for an upper count. Exact G_i(H) or watermark requires collapsed equal endpoints. Do not use midpoint ordering for ties. Respect producer→consumer causal edges even when timestamps fall within error; missing records or impossible order must fail the gate.

Host hooks identify Host publication/commit cutoffs. They do not align NPU device completion clocks. D_i(H) requires an existing dependency proof of prior device completion or a separate calibrated same-device event/lifetime certificate; no new global synchronize in this acquisition.

## 6. G_i versus D_i: stop at the resolved level

The first-pass Host ledger can yield exact or interval-valued committed G_i(H), API-consumed A_i(H), Y/S/C byte counts and queue-delay relationships. It may show all pre-bulk commits precede H; only then can terminal g_i^bulk equal G_i(H) for that trajectory after checking no intervening changes.

To certify D_i(H)/reusable state, reconcile:
- actual accepted output transaction validity and prefix identity;
- current input last-token/positions and already-present KV/continuation state;
- outstanding async result completion/commit and possible rejected/stale predictions;
- GPU-authoritative accepted counts versus optimistic CPU placeholders.

If only committed output is proven, name it G_i(H), not already-generated g_i^H. A legal same-trajectory continuation may permit using G_i(H) as a conservative already-available subset once ownership/state consistency is proved; it cannot become a universal algorithm floor over unrelated possible outputs/routes. Newly required generation and already executed computation must be accounted separately.

No decision to add a GPU snapshot should occur until the Host ledger shows which exact transaction/state value remains unproved. This preserves the minimum-measurement objective.

## 7. Acceptance gates and implementation constraints

- Freeze exact source/extension versions, inserted hook sites and actual concrete classes; no assert-by-string-count proof.
- Exactly60 request generations for the48+12 diagnostic, correct n/choice map, no request-ID suffix guessing, no generation reset or untracked error, expected single service and max concurrency12.
- Per-request Scheduler append ordinal continuity, raw token identity to OutputProcessor and parser input after allowed aggregation, final bulk accounting and1024 final total.
- H rank/cohort/request membership matches Runtime and logs; distinguish H_bootstrap/H_run and rank transition envelope.
- Clock provenance sufficient for requested ordering; all event brackets/lost records/ambiguous cutoffs explicit.
- Completed cleanup/source restoration; no formal Current TPS replacement.
- Buffered append-only records per process, immutable Host primitives, flush after cohort/process completion. Avoid per-event disk open/fsync and print in hot paths; log failures/overflow explicitly. Do not add request-wide barriers, sleeps, device copies or serialized API sends to make the diagram simpler.
- Compare overhead only as a diagnostic control; logging itself can alter prefill/admission and timing. Source proof/accounting can pass while performance calibration remains inadmissible.

Required output schema separates G_at_H_interval, D_at_H_status, API_received_raw_interval, API_queued_raw_interval, Chat_consumed_raw_interval, yielded_bytes_before_H, yield_raw_watermark_interval, sent_bytes_before_H, client_received_bytes_before_H, parser_provenance_status, publication_definition and exact_p_i_if_proven. Null unsupported fields; never substitute one level for another.

The smallest initial implementation is H + G + R/Q + A/Y. It resolves the practical accounting question without pretending all notions of publication coincide. Optional ASGI/client and native-device completion work should be selected only for the remaining explicit claim.

## Evidence inspected

Current host-mounted source (read-only): Ascend ModelRunner _update_states/_prepare_inputs/sampling-cache/handoff; upstream CachedRequestState, GPUModelRunner optimistic corrections, CachedRequestData/SchedulerOutput; OutputProcessor RequestState/RequestOutputCollector/receipt/queue; InputProcessor ID assignment; Chat request creation, parser suppression and yield. Prior Run421/422 ledger certificate and Run420/407 designs supply lifecycle and graph-identity context. This document proposes no optimization or source mutation.
