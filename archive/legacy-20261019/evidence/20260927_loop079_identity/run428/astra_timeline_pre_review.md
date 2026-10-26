# Run428 — independent Run427 timeline pre-review

2026-09-27. PRE-RESULT SOURCE REVIEW ONLY. Run427 was running; no live service/NPU/process/patch operation was performed. Only this file is written. Raw completed results will be reviewed after parent notification.

Requested role Astra High. Session exposes GPT-6 without independent backend variant/effort attestation; preserve parent dispatch metadata separately.

## Verdict

The acquisition can provide valuable source-bound Host observations, especially Scheduler committed G near the handoff probe. **The current reducer overstates the timing semantics of API consumption and generator yield.** Correct those offline after completion; do not mutate running instrumentation or discard the entire run automatically.

Acceptable candidate scope: append brackets and token-digest delivery to OutputProcessor; request mapping; post-consumption markers; pre-yield markers. Not yet certified: literal yield-before-H counts, exact API-consumed upper bounds at H, Q→A token identity under aggregation, device-completed output, raw-token publication, ASGI/client delivery or formal timing.

## Priority findings and offline dispositions

### 1. Y is a pre-yield marker, not a completed-yield observation

Patch calls api_yield(...) immediately BEFORE the Python yield. The helper emits api_generator_yield, then, for finished=True, performs synchronous flush BEFORE returning to execute that yield. Even nonterminal cases have an unbounded preemption possibility between marker and yield.

Therefore reducer fields api_generator_yields_before_H and api_yield_raw_consumption_watermark_before_H do not prove actual generator yields before H. A marker before H is necessary/possible evidence, not sufficient completed-yield evidence. Exact counterexample: marker occurs before H, task/process is delayed or terminal flush runs across H, actual yield executes after H.

Offline salvage:
- Rename raw event to api_pre_yield_marker and raw counts/watermarks accordingly.
- A subsequent event from the SAME request/choice generator, whose execution is source-proven to follow its resumption, gives an upper bound on the earlier actual yield. The bracket is [pre-yield marker, next such successor marker]. Another request's event in the same PID is not a valid successor.
- If that successor timestamp is before H_lo, the earlier yield definitely happened before the handoff envelope. If the current pre-yield marker is after H_hi, it definitely did not.
- Last pre-H marker without a qualifying before-H successor remains uncertain. Terminal marker has no guaranteed successor in the current logging path. A later client completion proves eventual success, not exact before-H timing.
- Do not promote last pre-H watermark as actual published raw tokens. Parser provenance still excludes that inference.

This is a concrete reducer semantic error, not a need to stop the running diagnostic.

### 2. A is a post-consumption marker; its possible upper cutoff is too small

api_consume is called after parser.parse_delta and previous_num_tokens increment. Its token_digest argument is evaluated before emit takes the timestamp. First-event clock-origin reads can add further delay. Thus api_consume.monotonic_ns is an after-marker, not consumption start or an instantaneous consumption timestamp.

Reducer consumed_before (marker before H_lo) is a sound witnessed lower observation, assuming contiguous ordinals and complete logging. consumed_possible (last marker<=H_hi) is NOT a conservative upper bound: the first marker after H may correspond to consumption that occurred before H.

Offline salvage:
- Name observed time post_api_consume_marker.
- Bound each consumption operation between a preceding same-generator progress event and its own post marker, when that causal predecessor exists. For the first event, a matching queue/receipt event can supply an earlier lower-time bound only after its identity/order join is established.
- At minimum include the first post-H consume event as potentially pre-H when its start cannot be located; subsequent consumptions follow that marker and cannot also precede H.
- Using the full1024 output as a conservative upper bound is preferable to a falsely tight interval when no valid bracket is recoverable.
- Enforce cumulative_i = cumulative_(i-1) + new_token_len_i, not merely nondecrease and final1024.

R and Q are also post-site markers (hashing/emit follows receipt or queue.put); the same directional timestamp caution applies if later reducers calculate R/Q cutoff intervals. Do not invent a symmetric timing error around an after-only marker.

### 3. G hook is authoritative append accounting, not EngineCoreOutput emission

scheduler_before is before both ordinary AsyncScheduler update and the direct bulk base call; scheduler_after has after_ns captured after the call. These genuinely bracket the append operation. G(H) lower/upper arithmetic using after/before is structurally appropriate.

But scheduler_after runs BEFORE structured-output handling, stop/free lifecycle handling and EngineCoreOutput construction. The source can change status and, in streaming sessions, reset state downstream. It is false to call after_ns an EngineCoreOutput emit timestamp.

For the ordinary frozen no-reset generation, reducer's per-request ordered equality of (admitted_len, SHA256) to OutputProcessor (new_token_len, SHA256) is a strong actual delivery check. It validates that the recorded payloads eventually arrived with exact boundaries/content, without claiming an emit timestamp. Add causal gate scheduler after_ns <= matching OutputProcessor post marker; the latter must be later for valid clocks.

Keep nonempty normal and bulk results, including any zero admitted result semantics. Empty/stopped-only EngineCoreOutputs are currently not logged by scheduler_before; extra zero-length R events may cause an exact-list mismatch. Such a mismatch needs investigation, not deletion of inconvenient rows. Conversely, success of digest equality is a scoped dataflow gate, not proof that every runtime output path in all configurations was instrumented.

Run427 helper does not log resumable/drop flags or full generation lifecycle as Run421 did. Input contract, actual class/status, successful no-reset run and G ordinal continuity must supply that premise in the final review. Do not infer universal append-only behavior just from a final cumulative1024.

### 4. Digest join stops at R; Q→A is not validated by the reducer

Scheduler→R equality is checked. The reducer collects Q hashes and A hashes but never compares them; for A it checks total length1024 and cumulative monotonicity. Those conditions do not prove token identity or sequence order.

The actual RequestOutputCollector can merge successive DELTA outputs. An API chunk may concatenate several queue-submitted token arrays. SHA256(chunk1) and SHA256(chunk2) cannot in general be combined into SHA256(chunk1 || chunk2), especially with the current JSON-array encoding. If coalescence occurs, existing per-chunk hashes alone cannot independently recompute that concatenation.

Offline disposition:
- Where queue/API segmentation matches exactly, compare ordered lengths AND hashes directly, including choice index.
- If aggregation occurs, current source may explain the transformation, but label Q→A as source-supported aggregation/count consistency unless raw arrays or a boundary-independent digest already exist elsewhere. Do not claim a cryptographic join that the artifacts cannot establish.
- For a future capture only if needed, record raw Host token arrays or prefix hashes over a fixed-width stream encoding at each queue/consume boundary. Compare matching cumulative ordinal endpoints. No change to Run427 helper while running.
- Explicitly validate internal→external mapping and choice0/n1 assumptions; reject duplicate/mixed output modes. res.request_id in the API is already external, despite helper parameter name internal_res_id. The reducer's assertion request_id==external is appropriate for this frozen single-input map but should be named accurately.

### 5. Q-after-put is usable here, with a narrow alias condition

The hook runs immediately after req_state.queue.put(request_output). In the inspected single-event-loop code, put is synchronous; when an older output is queued, RequestOutput.add mutates the older receiver and extends from the new input token list. There is no await between put and the hook. Hashing the submitted input immediately therefore can record the submitted delta, not the full queue contents.

Do not call it queue occupancy or the merged output. Values are converted into immutable scalar/hash fields before buffering, which avoids later JSON serialization of mutable model objects. Verify actual single-loop/no-concurrent-thread usage. A hypothetical generic “all after-put data is corrupted” rejection would overstate this source risk.

## Clock, boundary and data-durability gates

### H is the logging probe envelope, not a device-ready instant

handoff_entry_ns precedes gathering cached Host fields; event.monotonic_ns follows those reads. The code records all8 rank/slot IDs and the reducer constructs [earliest entry, latest post marker]. This is a coherent all-rank **H_probe transition envelope**.

It is still before function return, FixedCohortServing construction and .run invocation. Do not claim the last marker brackets the exact later first Runtime step: preemption/constructor work can follow it. Either explicitly define H as this source-pinned probe boundary or preserve that uncertainty when comparing to H_run. The bootstrap ownership transition and NPU state-ready boundary are distinct again.

Cached output length includes async placeholders; helper correctly records placeholder count. scheduled_output_including_placeholders is a label, not a proof of producer semantics; retain actual cached.num_output_tokens source definition. Neither this field nor an optimistic computed count is D(H).

### Equality of missing clock metadata is not clock proof

Reducer accepts one unique tuple even if boot_id, namespace or offsets are all None. Require nonempty boot_id and time-namespace evidence plus known CLOCK_MONOTONIC implementation, or another independently justified domain certificate. Validate an origin for EVERY event-producing PID, consistent file PID, repeated origins and namespace/offset stability; equal tuples among the subset of origins are insufficient.

The helper emits a fresh clock_origin whenever ROWS becomes empty after a flush. This is not inherently wrong; it provides repeated domain observations but adds /proc I/O and must not be treated as a new clock epoch. Cross-process common-host claims require the actual non-null values. Different namespace IDs with explicitly equal offsets could still be comparable, but current reducer conservatively rejects them; do not weaken it without a documented proof.

No NPU clock or actual client clock is captured. Server Host cross-process ordering does not become device completion, ASGI or client receipt evidence.

### Flush and completeness are conditional, not guaranteed by >=300 rows

Helper is buffered between terminal boundaries, with explicit flush:
- worker after Runtime.run and synchronize;
- Scheduler after EACH bulk request;
- OutputProcessor on terminal receive and terminal queue submission;
- API terminal PRE-yield.

These hooks should persist required records on a fully successful ordinary request lifecycle. They also add synchronous disk I/O within terminal handling, potentially delaying publication and next admission; no unperturbed timing/TPS inference.

Concrete residual risks:
- atexit is not guaranteed on SIGTERM, forced kill or multiprocessing exit; rely on explicit successful terminal flush evidence, not atexit promise.
- runner counts timeline rows BEFORE cleanup. Late explicit/atexit writes can make the count stale; a later mismatch does not automatically invalidate already persisted rows, but final count/provenance must be reconciled offline.
- no per-process sequence number, flush generation, end-of-run record or checksum covers missing middle records. Strong expected event cardinalities, G/R digest continuity, terminal coverage and source path are required; >=300 is only a smoke check.
- file write failure propagates, but a crash may leave a partial JSON line. Never silently skip it.
- shared ROWS/clear is not protected against concurrent emit/flush threads. Current intended paths appear synchronous within their process/event loop; actual threading must remain compatible. Do not assume generic thread safety.
- process identity is PID-only in filenames; verify no PID reuse/restart or old files mixed in the fresh run directory.

Final raw review must check40 H records/all rank memberships, exactly60 bulk records, per-request complete normal→bulk G/R histories, all60 R/Q/A/Y terminal identities, source restoration and cleanup, plus combined48+12 client concurrency and service60 POSTs. Current reducer omits combined concurrency and several lifecycle/source binding checks; independent review must supply them.

## Y scope omissions

Only the main generated-output yield branch is instrumented. Role, prompt echo, usage, error and DONE handling are not included. No frame bytes/hash/sequence is saved, only has_content/has_reasoning/has_tool_calls and raw-consumption watermark. Consequently:
- do not call the count all SSE frames or publish a byte total;
- a marker is “about to yield a generated-output branch,” not literal token exposure;
- booleans do not prove exact text/token provenance;
- no ASGI send or client receipt conclusion is possible.

These are intentional/narrow capture limits if stated. They become invalid claims only if promoted beyond their evidence.

## Source binding at pre-review

All four current host-mounted source files match Run427 install patched hashes:
- scheduler8768f0e318e2db3537b3daa9cd3e192ebf4e4219c8f711826747f65e0de1a1cf;
- runner9f7e3eb383adce2f0cc6a381608ba22ebb5c0534c824105f1cdacc51b9f1c274;
- outputad6b014582f882edee92a286f66652548c18795523a39b91675eec6b2be19465;
- api65e1aefa8614983f731bb1f382988674e5e0e661cda350b6052f6a63202b8ce2.

Helper20b0fff1b8d9fbb2b40ebda41a14e1730743d992f5586a9c3f6bb31b05c39527 matches install.
Patch script e2a37e31f605daf85b7f1b70ad2eed3d75d8cec7c0bb353b5b82b15766884989.
Reducer adaa5b16a9f1dd2e92d5786e46fb6a832ba772c37b8ab2860c99ab1e9bc1bfb9.
Runner8cd35e2d2717535015460c3b23a9aaebe5ce942be866f0c95c5a68937ff1455d.

Read scripts, install manifest, current relevant source and RequestOutput.add/collector get semantics. Did not execute patch/runner/reducer, inspect current raw result rows or interact with the running service. One read-only extraction command had a Python syntax error (exit1), then was corrected successfully; it made no mutations.

## Required final-report wording

If raw gates pass, accept “G committed within/around the source-pinned H_probe envelope; exact append→OutputProcessor payload delivery; observed API post-consumption and pre-yield markers.” Publish actual-yield/consumption intervals only with correctly directed or successor-derived brackets. Keep literal raw-token p(H), device-completed D(H), ASGI/client receipt and finite performance bounds unresolved unless additional evidence independently closes them.

The running diagnostic does not need a live patch to make its scoped accounting useful. Offline reducer corrections and an honest narrowed verdict should precede any decision to repeat acquisition.


## Final independent adjudication — Run427 completed (2026-09-27)

This section supersedes provisional invalidity concerns above where the completed raw ledger closes them. It does not change the historical pre-review. Scope: read-only raw/source review and this review-file append; no service, NPU, source, patch, or reducer mutation. Requested review role is Astra High; the session exposes GPT-6 but does not independently attest the backend variant/effort beyond parent dispatch metadata.

### Independent result

**Accept the Run427 source-bound Host accounting diagnostic, with interval-valued H_probe semantics. Do not promote it to a device-completion, raw-token publication, finite achievable-throughput, or formal E2E certificate.**

I independently parsed the ten original timeline JSONL files, reconstructed each request's append and downstream event sequence, and recomputed all 60 per-request intervals without executing the production reducer. The revised results agree:

| Cohort | Runtime cycles | H envelope width ms | Terminal prebulk G | G(H_probe) | A(H_probe), causal-capped | Generated-output yields before H |
| --- | ---: | ---: | ---: | --- | --- | --- |
| 1 | 291 | 16.296762 | 122 | [122,122] | [122,122] | [37,49] |
| 2 | 291 | 10.159740 | 176 | [176,176] | [144,176] | [48,59] |
| 3 | 314 | 15.257828 | 191 | [191,191] | [157,191] | [50,61] |
| 4 | 304 | 16.323590 | 193 | [193,193] | [162,193] | [56,67] |
| 5 | 288 | 11.125300 | 114 | [113,114] | [89,114] | [22,32] |
| Sum | — | — | 796 | **[795,796]** | **[674,796]** | **[213,268]** |

Here each cohort's H envelope runs from the earliest of its eight rank probe-entry timestamps to the latest rank probe-emission timestamp. The total sums five separate cohort cutoffs; it is not a single simultaneous cutoff for all 60 requests. G is Scheduler committed/admitted output count, A is cumulative raw token IDs consumed by the Chat generator/parser path, and Y counts instrumented generated-output yield executions, not tokens. H remains the pre-run Host probe, not constructor completion, Runtime .run entry, or device-ready time.

All 60 Runtime bulk inputs have length 1024; total incoming 61,440, terminal prior Scheduler output 796, admitted bulk 60,644, final Scheduler output 61,440. Do not substitute terminal 796 for G(H_probe) without its [795,796] qualification. One request has G lower bound zero; aggregate positivity is not evidence of positive pre-H G for every request. API consumption has a positive certified lower bound for 55 requests.

### Raw, lifecycle, clock, and cleanup gates independently checked

- Exactly 40 Runtime rank/cohort reports, covering rank 0..7 x cohort 1..5. Within each cohort, all ranks have identical ordered request IDs and cycles. Reports pass, host_mirror_exact=true, target graph FULL, generated counts [1024]x12, and zero ModelRunner cycles / oracle target calls after handoff.
- Client files contain 48+12 successful requests, each output_tokens=1024, error=null. Combined interval-derived peak concurrency is 12; diagnostic starts 0.698290967 seconds after the last warmup request ends. Saved server POST count is 60. Client JSON does not retain the API request ID, so this is population-level client agreement, not a client-receipt request-ID/time ledger.
- 2,200 raw rows: 280 clock_origin, 40 runtime_handoff, and 376 each of Scheduler append, OutputProcessor receive, queue submission, API consume and generated-output pre-yield. Exactly 316 normal and 60 bulk appends. Each request starts with g_before=0, preserves object identity within that lifecycle, has continuous nondecreasing append counts, and terminates with one bulk append. All append rows stale=false; class AsyncScheduler. 316 RUNNING→RUNNING and 60 RUNNING→FINISHED_LENGTH_CAPPED transitions.
- R/Q/Y each have 60 terminal events; every terminal identity is present. All observed API choice indices are zero; internal↔external request mapping is one-to-one for these 60 lifecycles.
- All event PIDs have clock origins; file PID and row PID agree; per-file timestamps are nondecreasing; run ID is LOOP079-RUN427. All origins share boot ID 6e5ca76b-a856-4531-8ab3-8d6c3ea9c8d4, namespace time:[4026531834], zero monotonic/boottime offsets, clock_gettime(CLOCK_MONOTONIC), and reported resolution 1e-9 seconds. This supports cross-process ordering for this host/domain; resolution is not a measured synchronization error bound.
- Every raw timeline SHA matches analysis.json. Final raw row count remains the saved 2,200 count. Terminal coverage, append continuity and complete downstream sequence agreement provide the substantive flush-completeness evidence; the count alone would not.
- Four current framework sources and four saved originals match their original manifest SHA values. Applying the patch function in memory to each original reconstructs its recorded installed SHA. The helper SHA matches install/restore records. source_before.sha256 and source_after.sha256 are identical. This is file/install/source-path evidence, not an independent process-memory import attestation.
- All seven cleanup status fields are zero. Saved stop-process probe is []; saved eight HBM readings are [3443,3440,3442,3435,3441,3435,3446,3435] MiB. These are saved stop evidence, not a new device query.
- This helper did not log request.resumable. The frozen bench sends an ordinary complete Chat request with response stream=true; this is distinct from streaming input. The source's Request default is resumable=false, while AsyncLLM streaming-input requests explicitly set it true. Append-only lifecycle inference therefore remains source/workload-bound, reinforced by unique IDs and uninterrupted normal→bulk histories. Do not generalize it to resumable input, whose continuation path can clear output IDs.

### Why the corrected A upper bound is valid here

The original marker-only post-consumption interval is [674,12962]. Its loose upper is mainly cohort 1: the next post-H consume marker is a terminal bulk marker for each request, so timestamps alone permit 12x1024. The corrected cap at G's upper bound gives [674,796].

I checked the additional facts required for this cap, beyond the reducer's original Scheduler→R check:

1. For all 60 requests, ordered (length,SHA256) lists agree exactly across **Scheduler admitted delta → R receive → Q submitted delta → A consumed delta**. All 376 segments have matching boundaries as well as digests; there was no observed queue chunk aggregation. Thus the pre-review aggregation/hash-join concern is closed for this acquisition.
2. Every matched segment satisfies S.after_ns ≤ R.monotonic_ns ≤ Q.monotonic_ns ≤ A.monotonic_ns. These are causal sanity checks; post-markers alone are not operation-start times.
3. Source order makes Scheduler append precede EngineCoreOutput construction, OutputProcessor receive precede RequestOutput creation/queue put, and AsyncLLM's queue get precede Chat consumption. The collector clears its fetched slot and yields each consumed delta once. Observed recurrence is exact: each A cumulative count equals its previous cumulative count plus that delta length and ends at 1024.

Consequently at any candidate H inside the same cohort envelope, API raw consumption cannot exceed the corresponding cumulative committed Scheduler tokens. Taking G's conservative upper bound is valid. The 674 lower bound remains based on consumption already completed by a marker strictly before H_lo. This does not prove that those raw IDs were literally published or that all their semantic text was exposed.

The reviewed reducer correctly reverses the problematic A/Y timestamp interpretations and applies this cap. The parent subsequently encoded these independently checked Q/A exact segment equality, choice==0 and per-segment causal ordering gates in the reducer; I reread that revision and independently rechecked its latest analysis. Future merged chunks would require a compatible prefix/raw-ID join and must not silently inherit this run's exact-segment guarantee.

### Why Y's successor lower bound is valid; nonempty reasoning finding

For each same-request single-choice generator, a later A or Y marker proves execution resumed past the prior yield. If that later marker is before H_lo, the prior yield was completed before H_lo. A pre-yield marker at/before H_hi only makes its yield possible before the cutoff; it does not prove completion. The corrected [213,268] count uses these proper directions.

Independently, all 213 successor-certified lower-bound Y records are has_reasoning=true, has_content=false, has_tool_calls=false, finished=false, and involve **50 unique requests**. Source constructs the response choice from this same delta_message, serializes the chunk, then executes the instrumented yield. maybe_filter_parallel_tool_calls modifies only tool_calls, not reasoning. Therefore the defensible statement is:

> In this Run427 acquisition, at least 50 requests had yielded, in total, at least 213 generated-output chunks containing nonempty reasoning deltas from the Chat generator before their respective cohort's earliest H_probe entry.

“At least” matters: the 50 is a certified subset, not proof that the other ten yielded nothing. This rejects an all-zero pre-H **generator-level semantic-output** premise for this run. It does not count literal published raw token IDs, bytes, all SSE frames, or tokens represented by each delta. It does not establish ASGI send, network write, client receive, or when a reader saw the reasoning. Role/echo/usage/error/DONE branches remain outside Y instrumentation.

### Conclusions still barred from promotion

- Neither p_i^H as a literal raw-token publication count nor device-completed/generated D_i^H is determined. Do not relabel G or A as either quantity.
- [795,796] is not a device generation work-numerator certificate. Tokens sampled/accepted on device but not yet committed by Scheduler may exist. A raw-output count remaining to be admitted is not necessarily fresh model computation still required.
- The per-rank ModelRunner cache length, placeholder count and scheduled counts do not become authoritative generation counters merely because the Host ledger closes.
- No ASGI/client publication latency, token receipt identity, or zero pre-H external-delivery claim follows. A client/API/SSE boundary ledger is still required if that precise endpoint is the hypothesis; avoid a new acquisition solely to re-prove the Host facts now closed.
- No formal TPS promotion: hooks/digests/flushes perturb Host timing; this is 48 warmup+12 diagnostic, not a matched clean formal run.
- Do not transplant Run427 counts or cutoffs into Run403's retained routing numerator. Run403 still needs its own temporal/identity certificate; active FlashComm1/DSA-CP candidate-row↔all43-router identity is not closed by H/G/R/Q/A/Y.
- No new finite Algorithm/Resource, Hardware, Scheduling/Execution or product-throughput endpoint is established. Capacity ceilings and necessary same-run work remain separate obligations.

### Review provenance

Reviewed corrected reducer SHA256: 6a53847273cb91c010661f64ec7224acfdc59b967b0ac04e198375da0a2eb3ff.
Reviewed analysis.json SHA256: f6539903508382bb863dae04d431347b849640d6585e4ad5058a07c7ea0fc12f.
Other helper/patch/source hashes are recorded above and in install/restore manifests. Ten raw timeline hashes were independently matched to analysis.json, not recomputed by trusting its results.

One preliminary independent check intentionally compared against the uncapped A formula and stopped with an assertion after the parent had corrected analysis to the causal cap. It made no writes. The subsequent independent calculation checked the capped values, all 60 per-request results, and the full 376-segment downstream digest chain successfully.

### Latest reducer revision rechecked

After the parent integrated the downstream digest/causality and reasoning-yield gates, I reread the updated reducer and reran the independent raw computation against the new analysis. All 60 per-request intervals and the 376-segment chain still agree. The latest source also asserts the lower-bound Y flags; latest analysis records 50 requests with confirmed reasoning-delta yields.

- Latest reducer SHA256: 17cd5c0297cc6882727aba0f09ecebc670ecd189228244c0e4f73abcb3391649.
- Latest analysis.json SHA256: d2c77c6852731e8dcf29ade947953b0f12c1008cca7e6121caa9ee66d0f0ae55.

The two earlier hashes above are retained as review chronology, superseded by these latest hashes. **H_probe is not H_run and does not measure D(H). Y's certified lower bound is nonempty reasoning chunks at the server Chat-generator boundary; it is not raw token publication or network/client receipt.**
