# Run422 — independent Scheduler bulk-ledger certificate review

Date: 2026-09-27. Status: COMPLETED RAW-RESULT REVIEW. Final verdict below supersedes the initial hypothetical-zero design review.

Requested role Astra High; session exposes GPT-6 but does not independently attest backend variant/effort. Keep parent dispatch metadata as configuration provenance. Only this review file is written. No running-service/NPU/patch operation or benchmark was performed.

## Completed raw-result review — supersedes the hypothetical zero branch

**ACCEPT the Run421 terminal Scheduler ledger and its request-level truncation arithmetic. REJECT promotion of p_i=0. Exact pre-handoff published p_i remains UNKNOWN; the result does not prove p_i>0 either.** The observed variable is g_i^bulk, the Scheduler's committed output-list length immediately before terminal bulk append. It is not a timestamped handoff-time generated count.

Independent read/recomputation used the60 raw rows in ledger/pid541.jsonl, forty Runtime files, both client JSON files, the actual server log, install/source-check/restore manifests, source-before/after hashes and saved cleanup probes. It did not treat reducer status as the primary evidence and did not operate the service, NPU or patches.

### Checked result and clean-run gates

| Cohort | Runtime cycles | Sum g_i^bulk | Incoming Runtime tokens | Bulk tokens admitted |
|---|---:|---:|---:|---:|
|1|294|121|12,288|12,167|
|2|314|120|12,288|12,168|
|3|321|110|12,288|12,178|
|4|342|134|12,288|12,154|
|5|297|99|12,288|12,189|
|Total|1,568|584|61,440|60,856|

- All60 g_i^bulk are positive, range1–33. For every request incoming=1024, g_after=1024, admitted_len=append_delta=1024-g_before, with ordered timestamps and stopped=true.
- All60 rows use AsyncScheduler in EngineCore PID541, RUNNING→FINISHED_LENGTH_CAPPED; resumable=false, stale count0, output_is_stale=false and drop_stale_output=false.
- Placeholders16 and in-flight tokens8 are recorded on every row. They are async scheduling state, not an additional published-output count or proof of a handoff-time value.
- Exactly60 unique ledger request IDs equal the60 Runtime IDs; five disjoint12-request cohorts, eight rank reports each, consistent slot order/cycles. All reports pass FULL Target, exact mirrors, zero post-handoff oracle/ModelRunner calls and1024 internally generated tokens per slot.
- There are only52 distinct (PID, Python object ID) pairs across60 rows. Request objects are reused after lifecycle completion; unique request IDs and cohort membership carry identity. Object address alone is not a generation certificate.
- Both48+12 clients succeed at exact1024, with combined timestamp-sweep concurrency12. Warmup ends4139068.010851 and diagnostic starts4139068.684319 in recorded client time; phases do not overlap.
- Actual service log contains60 POSTs, not merely a saved count claim. Startup confirms TP8, DP1, PP1, async scheduling and DSpark7. EngineCore PID matches ledger PID.
- All seven cleanup statuses are0; saved stop process probe is[]; saved eight-card HBM is3435–3445 MB with no VLLM process row. Source-before/after identical; current Scheduler has restored original SHA and helper SHA still matches install.
- Ledger SHA256:7aaf2537777f1da6fd038c93d55b71b7bde0152265dfa399b520b1a968c974b3.

The actual base-class bulk call bypasses the ordinary AsyncScheduler placeholder-decrement override as designed. The preceding list was already nonempty when this terminal call ran. The append/stop code directly explains why client exact1024 and Runtime exact1024 coexist:584 tokens of the Runtime's terminal lists are removed by request-level maximum-length trimming. This is an observed internal-versus-external accounting distinction, not by itself a semantic correctness failure or584 tokens' worth of independently removable critical-path work.

### Three distinct counts and the precise inference

Define:
- g_i^bulk: recorded committed Scheduler count at terminal bulk-before time;
- g_i^H: already-generated, semantically valid reusable output at the actual Runtime handoff boundary;
- p_i^H: generated output already published at the chosen external API boundary at that handoff.

Run421 directly measures only g_i^bulk. It confirms that584 output tokens had been accepted into Scheduler histories before the Runtime terminal append. It does NOT timestamp those append/sample/publication events relative to the earlier handoff.

The Runtime contains no post-handoff ModelRunner re-entry, but async EngineCore may process an earlier outstanding model result after Runtime begins. Thus “before terminal bulk” must not be silently relabeled “before handoff.” Device-computed results, Scheduler-committed results and queued API publication can also differ at that boundary. g_i^H needs a source/transaction/time certificate of its own before being used as already available work.

For the inspected ordinary non-resetting lifecycle, pre-handoff publication requires prior committed output, so the conservative enclosure is0 <= p_i^H <= g_i^bulk (aggregate0..584). No exact p_i^H, no strictly positive p_i^H and no handoff-time g_i^H follows from this enclosure. A general g_i^H ordering with g_i^bulk additionally depends on outstanding-result validity/commit semantics; do not equate them by definition.

The source-default non-resumable input contract, successful normal lifecycle, unique IDs and actual flags support use of the ordinary append-only path here. Positive counts make the earlier zero proof inapplicable regardless. Reasoning/content buffering or async queues can delay publication even after generation. Role/usage framing is not a generated token.

### Is API/SSE instrumentation necessary?

For **the exact claimed p_i^H**, yes: existing records lack a request-correlated handoff/publication boundary. A source proof of a specific all-output-drained-before-handoff ordering could substitute, but no such ordering was found or measured. Client TTFT is not sufficient: client indices lack a direct internal-ID join, it reports only first content arrival, and it does not reconstruct all prior generated-token publications.

Minimum exact-publication extension:
1. A Host monotonic timestamp/transaction certificate at actual Runtime handoff, with cohort/slot and internal request IDs.
2. Internal→external→SSE/choice identity mapping in OutputProcessor/chat serving.
3. Cumulative generated-token ordinals at the selected publication boundary, excluding role/usage/echo; align with handoff using the same verified clock domain and state uncertainty. Distinguish generator yield, ASGI send and client receipt. If text parsing buffers or suppresses token text, retain token provenance rather than retokenizing chunks.

For **a resource numerator or necessary remaining generation**, a full SSE timing ledger is usually unnecessary and less direct. First close g_i^H, including valid pending/generated outputs and KV/continuation state:
- record request-correlated ModelRunner cached output count and actual input/position/seed at handoff;
- bind that boundary to scheduler execution/output transaction sequence and previous committed counts;
- establish which pending prior results are already computed, still valid and reusable, without inferring their count from placeholders;
- preserve later bulk accounting as an independent reconciliation check.

Only then can remaining newly required generation use1024-g_i^H under its defined continuation contract. Using1024-p_i^H would wrongly charge queued but already-generated tokens again when g_i^H>p_i^H. A later bulk g_i^bulk may replace g_i^H only after proving their equality for this path, not simply because both are counts of outputs.

No next diagnostic should add a hot-path device synchronize merely to force this equality. Source/execution sequence and Host-only metadata may close it with minimal perturbation.

### Impact on Run403 retained numerator and Current Bound

1. Run403's current96-row routes, histograms, ownership and standard arithmetic remain valid within their existing capture scope. Run421 is another trajectory and supplies no missing row identity or prior count for Run403.
2. Run403 “retained” means Runtime-internally clipped/staged rows under initial_output_counts=0. It must not be called the exact externally admitted suffix without its own lifecycle/clipping ledger. Run421 establishes that this distinction actually occurs in the current product path, rather than being merely hypothetical.
3. The Run403 early cycle64/65 union is NOT automatically disproved. External truncation mainly affects a terminal suffix; early retained predictions could all belong to the eventual admitted prefix. But that requires same-run prior counts, cumulative output and row mapping. Run421's1–33 counts cannot be pasted into Run403. The existing all43 FlashComm/DSA row-identity and clairvoyant-mask caveats remain unchanged.
4. Within Run421, first four cohorts have485 earlier Scheduler tokens and48,667 admitted Runtime tokens, totaling49,152 for that48-request phase. Fifth has99+12,189=12,288. Never replace formal throughput Q=49,152 with Runtime-only admitted counts: earlier outputs are still part of the client's workload and elapsed serving time.
5. No finite Algorithm/Hardware/Scheduling/Product endpoint is promoted. The cardinality relaxation must be recomputed using certified already-available/required-output semantics. Its old numerical512/640 terms might coincidentally remain unchanged for fixed-cohort request chains; neither a mandatory decrease nor retention can be asserted from terminal g_i^bulk alone.
6. Current formal median571.681 and best single repeat612.962 remain unchanged. Host ledger I/O and48+12 diagnostic protocol prohibit a new formal TPS claim.584 discarded terminal-list tokens do not directly translate to saved cycles/time, because fixed-slot execution, acceptance, KV/state consumers and cohort stopping still need causal analysis.

Final confidence: high for Run421 raw arithmetic, request-set join, truncation and clean/restored run; insufficient for exact pre-handoff generation/publication values or any cross-run numerator correction. The completed result supersedes the initial hypothetical zero branch below; those sections are retained as design history, not active pending instructions.


## Pre-result verdict (historical design review)

The patch selects the correct authoritative Scheduler point: immediately before and after the explicit KVDeliveryScheduler._update_request_with_output call in the extreme_bulk_output branch. It records incoming list length before the callee can trim it and records the returned admitted length. This is a valid design for a zero-prior-generated-output certificate.

IF every measured request has g_before=0, the additional lifecycle/provenance conditions below can prove that no generated model output for that same request generation was externally published before its Runtime handoff. This is a causal source argument; it does not require an API timestamp or an SSE-to-client-index join when the stronger zero-generated claim holds. Exact client length alone cannot prove this.

No conclusion about actual Run421 p_i is made until raw results and lifecycle gates are reviewed. No conclusion may transfer automatically to Run403, Run239, Run99, future requests or other scheduler variants.

## Inspected artifacts and source binding

Read:
- scripts/loop079_bulk_ledger.py;
- scripts/loop079_bulk_ledger_patch.py;
- scripts/loop079_bulk_ledger_reduce.py;
- scripts/run_loop079_bulk_ledger_run421.sh;
- Run421 install/source_check/selftest/backup manifests;
- actual current host Scheduler source in the known framework bind mount;
- Request constructor/append/count, ordinary AsyncLLM input processing, Scheduler session reset/admission/stop, Ascend async override and PP wrapper source;
- prior Run420 source/API pathway audit.

At review:
- helper SHA2566530154480e36d63731deda202a0d910c5ad0c2fc63ccf62017e5f7392fe347f;
- patch script e5860353d4158d5fb34bb3d3450acc01291ec80ccd4055564c1e37cc1b893a7e;
- reducer4a389ce3f425f9b55789f81b2d47a938a4b812e308f4a4a5730c79ddf5d3c036;
- runner4606bdaf32ba6066123f91f39e574441771f11cf34e0e62feb086c3b1c5c7c9b;
- original Scheduler5ba894b9ec003502c008e12281c260787ed64a419327c451c8e67374bf525e8d;
- actual patched Scheduler9e93848744fb9fd6e878d2afe87f5ef3e8cb491a96bbd408936ff7c03dd0ecf4, matching install.

The patch is SHA-pinned and checks unique replacement anchors; it directly surrounds the base-class call, so an AsyncScheduler subclass does not redirect this particular bulk append through its normal placeholder-decrement override. The logged concrete scheduler class still matters for surrounding update_from_output behavior.

Compile/selftest verifies syntax and synthetic helper accounting. It does not independently test loaded Scheduler inheritance, historical Request lifecycle or API publication. A host on-disk hash is not a loaded-code attestation; fresh-process launch after install, correct bind mount/PYTHONPATH, actual helper-produced rows, logged class and no intervening mutation supply the operational binding. Verify these from the completed run artifacts without touching the running process.

## Necessary zero-certificate conditions

1. **Complete same-generation join.** Exactly60 unique bulk request IDs equal the60 Runtime IDs in five cohorts; all8 ranks agree on each12-slot order/cycles. No duplicate ID, previous completed generation reused under the same ID, hidden second service/client cohort or ledger from another run. PID/object ID is supporting identity, not a persistent generation key by itself.

2. **Authoritative zero and append accounting.** For every row:
   g_before=0;
   incoming_bulk_len=1024;
   g_after=1024;
   admitted_len=append_delta=1024;
   stopped=true;
   timestamps ordered.
   Concrete status must be compatible with ordinary running→finished-length handling. No stale/drop/reset flags. Preserve positive counterexamples if present; never normalize them to zero.

3. **No earlier reset of output history.** Request.__init__ initializes _output_token_ids empty; append_output_token_ids only appends/extends. However Scheduler._update_request_as_session at upstream line1314 clears the list. Its admission/stop callers handle streaming-input sessions. The ordinary Chat→AsyncLLM process_inputs path uses resumable=False by default; streaming input separately requests resumable=True. Need the executed request contract/class and ledger resumable=False together, not the final flag alone, to rule out an earlier resumable generation/reset. A late false flag is not universal proof: structured-output error code can set resumable=False. Ordinary successful frozen n=1 input with no streaming-input/reset/error path discharges this particular concern.

4. **Append precedes every generated publication.** In inspected KVDelivery update_from_output, tokens enter Request through the append/stop call before EngineCoreOutput is constructed; OutputProcessor consumes its new_token_ids, and the API derives generated chunks from RequestOutput. Confirm the actual scheduler/overrides and frozen API path match that chain. Prompt echo, role/usage/error framing are not generated model tokens. No custom connector/API bypass may fabricate or publish model tokens independently of this Request ledger.

5. **Handoff precedes the observed bulk return.** Join to Runtime record for that request generation, with handoff_before_model_forward and no post-handoff ModelRunner/oracle re-entry; ModelRunner constructs the terminal bulk frame only after FixedCohortServing.run. There can be async earlier work, but any earlier externally published generated output would already have passed the append step. Thus an eventual zero at this bulk boundary excludes it when history is append-only.

6. **Clean lifecycle and provenance.** Fresh service launch under recorded source/helper,60 POSTs for sequential48+12 requests, all outputs successful/exact1024, coherent unique Runtime membership, no reset/abort/retry/generation reuse, valid cleanup/restoration. API RequestID mapping need not identify each client index to prove universal zero for the covered set; set coverage must be complete. Do not claim an unrecorded direct client-index mapping.

Proof: suppose a generated token for a covered request was published before handoff. Covered API dataflow requires an earlier Request append for that generation. Non-resetting output history then has length>=1 at the later bulk-before point, contradicting g_before=0. This proves p_i=0 in the defined generated-output sense for those requests. It does not assert that zero SSE bytes/role frames/headers were sent.

If g_before>0, no equivalent proof of exact p_i follows: generated, queued, yielded, sent and received are distinct. Then use Run420's optional internal→external→SSE ledger and a defined publication boundary. Missing proof premises should yield scoped INCONCLUSIVE, not infer a semantic defect.

## Reducer and runner review

Correct:
- reducer keeps external_pre_handoff_p_i_verified=false and labels zero only as candidate for source review;
- checks60 ledger IDs against Runtime, forty rank/cohort reports, rank agreement, FULL Graph, mirrors, exact output counts, cleanup and source restoration;
- computes append_delta consistently and rejects stale/resumable cases from zero_ready;
- runner executes48 warmup then12 diagnostic sequentially and requires60 POSTs/ledger rows.

Independent result review must strengthen/check:
- incoming_bulk_len=1024 is presently reported but not asserted; zero_ready alone can pass an oversized input trimmed to1024.
- scheduler class/path values are collected, not enforced; inspect actual class/override chain.
- g_before/g_after/admitted field types, nonnegative values, status transitions, PID↔file consistency and timestamp grouping should be checked.
- per-client peak concurrency is checked separately. Compute the combined timestamp sweep and confirm nonoverlapping client phases; shell sequencing is supporting evidence, not a substitute if artifacts are mixed.
- actual runtime/report request order and IDs must form disjoint cohorts, all60 ledger IDs unique; no implicit client-index inference.
- verify install/helper SHA, source-before/after/restore and relevant mounted lifecycle sources. The reducer does not itself validate helper binding, loaded module provenance or all Request-reset paths.
- cleanup_status text is not by itself evidence of idle hardware/processes; saved stop probes should be reviewed after parent announces completion.
- no API post-append emit hook is present. That is acceptable for the zero exclusion proof; positive output/publication accounting would require more.
- placeholders/in-flight counters may legitimately be nonzero in async scheduling and are not published tokens. Interpret them through source; do not reject merely because they are positive or equate them to g_before.
- the helper creates a directory and opens/writes JSONL immediately for each bulk request. It is Host-only and after Runtime execution, but it is not buffered and can delay terminal publication/next admission. No Run421 TPS or output timing promotion is justified.
- absence of a Runtime-entry Host count snapshot does not defeat the stronger later-zero argument under monotonicity. Do not add a new hook merely to timestamp a fact already implied causally.

## Originally planned actual-result review (completed above)

Historical plan, now completed: independently read60 raw ledger rows, forty Runtime reports, both client artifacts, server count/log, install/restore/helper manifests, cleanup probes, reducer output and relevant class/runtime flags. Recompute counts/joins rather than treating reducer status as proof. Append a final scoped verdict here.

If all zero-certificate gates pass: promote Run421-only p_i=0 for the covered60 ordinary request generations; the diagnostic cardinality ledger may use their verified remaining lengths. Do not retroactively rewrite earlier Run403/Run99 p_i or promote a finite Hardware/Scheduling/Product ceiling. If any row is positive/reset-ambiguous: preserve the measured g_before and post-trim accounting and request the targeted API ledger only for the unresolved statement.
