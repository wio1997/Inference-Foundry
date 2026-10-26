# Run602 independent Astra posterior review

Verdict: **SCOPED PASS for this new diagnostic W₀'s Product ownership and Host accounting.** No formal E2E result, removable-time conclusion or strict Bound endpoint is promoted. This review ran CPU validators and read evidence/restored sources only; no service or NPU workload was launched.

## Independent admission and provenance

Verified all 281 product_admission raw digests against current files. Both source_before/source_after and scripts_before/scripts_after are byte-identical; independently hashed each listed current restored source/script against the recorded digest. Checked exact nine cleanup keys, all zero, and the saved stop verification: no remaining VLLM processes, all8 idle HBM 3435–3446 MiB. These checks corroborate the reported controller success; product_admission alone did not certify cleanup.

Re-ran Product validator on the host: exit 0 and exact JSON equality. Initial host attempt to repeat the phase validator failed because host Python lacks aiohttp (exit 1, no source/evidence mutation); reran both phase and basis validators using container Python, CPU only: both exit 0 and exact JSON equality with saved admissions. This includes the 96-client raw SSE/base64/payload checks and 32-basis/64-Runtime all8 source/ledger gates. The failed host-environment attempt is not a failed live acquisition.

Admission identities:
- product_admission.json: a257c14a4a39bbeb152acd78ada54e8194f347497689bd430e38ad778a7b4fa6
- basis_admission.json: 964825f4b07082f5e65c9aaa39fae4457c8b364de4335f24cc3c593568521bde
- client_admission.json: a29ab49eab3c886a2eb2fcfeb42f0a1f46b934a81567a8fc4859ccf66e572c4a
- cleanup_status.txt: cd06ecab09b860d341c260eb9fe671532b7543ef42559f84d61672f69147157a

The new compact basis has 1189 cycles, 100912 active Target8 rows and 114144 issued physical Target rows per rank. It is not Run597's 1200-cycle W₀ or a reconstruction of Run99's three formal trajectories.

## Clock conclusion

Actual client admission records CLOCK_MONOTONIC/perf_counter using the same clock implementation, kernel boot ID 6e5ca76b-a856-4531-8ab3-8d6c3ea9c8d4, time namespace time:[4026531834], and zero monotonic/boottime offsets. Worker, scheduler and API records all have this same namespace; the guarded controller ran them and clients in the same container. Thus this run's Host comparisons are supported by archived clock identity rather than merely assuming the client matches the server. This does not align NPU clocks with Host clocks or clocks across devices.

## Token ownership result and precise meaning

| Measured cohort | Actual scheduler pre IDs | Placeholders | Retained incoming Runtime bulk | API IDs | Yield-associated IDs | Received yield-associated IDs before max all8 build |
|---|---:|---:|---:|---:|---:|---:|
| 5 | 277 | 192 | 12011 | 12288 | 12288 | 251 |
| 6 | 372 | 192 | 11916 | 12288 | 12287 | 341 |
| 7 | 193 | 192 | 12095 | 12288 | 12288 | 164 |
| 8 | 175 | 192 | 12113 | 12288 | 12288 | 142 |
| Total | 1017 | 768 | 48135 | 49152 | 49151 | 898 |

For every request: actual pre IDs + accepted incoming prefix == actual scheduler post IDs; post has 1024 IDs; API engine DELTA IDs concatenate to that post. Each full 1024-ID incoming Runtime list independently joins its reconstructed basis output. Consequently, 49152 Runtime bulk IDs were supplied, 48135 retained, and 1017 tail IDs not appended because the scheduler already owned that many actual prefix IDs. This is a same-W₀ identity/accounting observation. It does not authorize skipping Runtime cycles/tail work under the frozen trajectory, and it does not imply all pre IDs had reached the client at handoff.

Placeholders are 16 per request at the recorded bulk boundary (192 per cohort). They are neither generated-token counts nor additional output on top of pre IDs. Cached output-position counters include placeholders and cannot replace this actual-token ledger.

Exactly one engine DELTA ID is parser-suppressed: cohort6, response chatcmpl-8c82d90205e4480e, token ID4, zero-based output position27 (cum_output_tokens28), finish_reason None, no yield hash. That request has 36 scheduler pre IDs, 988 retained Runtime IDs, API1024 and yield-associated1023. The suppressed ID lies in its ordinary prefix. It is **not a missing output token, failed SSE delivery or a 1023-token model result**; usage and API token ledger remain1024. Do not assign it a client receipt time or assume its text appears in a later chunk without parser-specific proof.

The 898 figures count IDs associated with API payloads whose exact hash was received before the latest all8 runtime_built_host timestamp. They do not count every text token already visible, do not prove a simultaneous all8 ready barrier and do not prove first-delivery time for every ID. The 119 difference from scheduler pre IDs comprises one suppressed ID plus 118 other prefix IDs not classified as received before that cut; those can be delayed API yields/receipt relative to the cut. All yielded payloads are eventually hash-joined. The latest-versus-earliest rank build spread is 10.3–11.5 ms, so this is explicitly a max-build Host convention, not a hardware-ready event.

## Current Host accounting, not an attainable Bound

Independently checked all worker mark sequences are chronological and ordinary call Host intervals end before build. Rank0's four first-execute-entry → serve_synced_host intervals are pairwise nonoverlapping and contained within the actual measured client interval in the verified Host clock domain.

- Client measured duration: 91.483702 s.
- Sum/union of those four rank0 intervals: 90.909688 s.
- Client duration outside that union: 0.574014 s.

The subtraction is valid **accounting for this diagnostic**. The remainder includes boundaries/transport/observer publication and inter-cohort work; it is not proven idle, removable or an irreducible Product term. The union itself includes async submission, execution, waits, basis capture and observer costs. No formal Run99 time is subtracted from it.

Rank0 preparation (first execute → build) by cohort is 4.943204, 6.756057, 5.859611, 5.447533 s; serve_start → serve_synced is 16.006716, 16.325938, 17.067861, 18.502292 s. The last ordinary proposal Host end → build is only 23.569, 21.643, 73.561, 84.872 ms on rank0. Those small final gaps do not establish early device readiness: outstanding work can continue after proposal Host return. They do show that measuring only this final Host gap would miss most preparation.

Current-stream Target elapsed spans total approximately18.736282 s across the four rank0 preparations; proposal spans total2.083750 s. These are descriptive per-stream event totals, not compulsory compute time, independent resource time, or an E2E critical-path lower bound. Do not subtract their sum from a Host interval as "framework waste"; asynchronous queueing, stream waits, contention and overlap remain unresolved.

## Call denominator and preparation work conditions

The four measured cohorts contain respectively16,26,15,14 ordinary Target calls **per rank**, and the same16,26,15,14 proposal calls. Therefore "71 ordinary calls" means71 Target call instances plus71 corresponding proposal call instances per rank, before the four Runtime handoffs. It excludes warmup and the1189 internal Runtime cycles. Across all8 rank replicas this is568 Target records and568 proposal records, not568 independent full-model evaluations. All8 execute semantic sequences and call kind/ordinal/padded-shape/request-ID sequences were independently equal within each cohort.

All48 new requests already report num_computed_tokens32768 at admission. Actual prompt lengths/scheduled suffixes are46×(32851,83), one(32853,85), and one(32852,84), yielding3987 newly scheduled prompt-suffix tokens in the observed preparation path. These are current scheduler coordinates under the warm-cache initial state. They are not proof that the measured workload freshly computed48×32768 prompt positions, nor proof that cached KV is physically unique or universally available at zero cost. The frozen product's initial cache/weights/state boundary must be explicit in Resource work accounting.

## Priority revision and exact offline next step

**Next action should first be an offline all8 preparation-domain reducer using this already admitted evidence. Do not spend another live run merely measuring the last handoff gap.** The producer→consumer frontier remains a useful possible next acquisition, but it should span the ordinary preparation domain and its initial KV/state conditions, then target its largest unresolved dependency. Its status as highest-value live action is conditional on that reducer, not already proven.

Reducer fields, without adding a new experiment:

1. For each rank/cohort/execute ordinal: new/cached request identity and order, prompt length, num_computed_tokens, scheduled tokens, speculative-token count, cached output-position counter explicitly labeled "including placeholders", and final96 handoff membership. Preserve monotonic request generation/slot mapping and explicit unknowns.
2. Join each Target/proposal call to that exact execute ordinal: actual scheduled-token sum, padded Target shape, call kind, current-stream elapsed, Host interval, observed producer req IDs/copy_event_present. Validate all8 semantic equality and report rank min/max; do not sum ranks into model work.
3. Per-request distinguish first admitted prompt suffix from ordinary cached/speculative scheduled rows and later internal Runtime rows. Preserve overlaps/repeated positions/unknown corrected positions. A scheduler count is an observed issuance coordinate, not a fresh-logical-evaluation certificate; raw ordinary acceptance/actual device positions are not fully captured here.
4. Tabulate physical padding versus scheduled rows, ordinal/request occupancy, prefix accumulation and accepted bulk truncation, while keeping Target and Draft roles separate. Missing actual Draft query/context shape or positions remain unknown rather than borrowing Runtime Q7 geometry for every ordinary prefill call.
5. Build a same-run Host interval union and call coverage matrix for the whole preparation → Runtime → scheduler → API/client chain. Mark synchronous observer writes and uncovered worker-return/side-stream-ready edges. Report current costs only; do not treat call boundaries or their order as immutable algorithm constraints.
6. Explicitly list the initial resident KV/weight/state assumptions, aliases and data lifetimes needed for any future compulsory-work claim. This can reveal whether another Resource witness or a scheduling-frontier trace would shrink more uncertainty.

If the reducer leaves unresolved execution readiness/resource contention as the largest useful uncertainty, apply the previously proposed all8 stream/event-generation frontier capture to the relevant ordinary calls, using existing waits and terminal synchronization, fixed algorithm semantics and an observer perturbation control before numerical timing claims. A future architecture may change boundaries, but it must preserve the frozen acceptance/output/work ledger; today's issued calls are not a proof of compulsory work.

## Bound consequence

Run602 closes a major Product ownership ambiguity and localizes the measured Host wall to preparation plus Runtime with a small unaccounted Host remainder in this diagnostic. It does not close exact compulsory traffic/compute, attainable cumulative capacity, cross-stream readiness or a legal optimal schedule. Current formal571.681tok/s remains unchanged; strict Resource/Scheduling/Product endpoints remain null. No evidence here supports "near the limit."

## Run603 follow-up — completed census and updated next-measurement choice

The initial offline census recommended above is now completed and independently reviewed. Final script SHA: 7f6a7d7577107e7598e9d63a7513ceb78266df936972f724260f6ee55e42651a. census.json SHA: 54e5d198bb266bdb35a712f112414f88e76b8e5136b740e9c55a2fcead9eda3b. Independently regenerated the final census into a temporary file, exit0 with exact JSON equality. All32 worker digests join the already verified Product admission.

The initial reducer could overwrite duplicate call ordinals in dictionaries, used zip without an explicit all8 ordinary-length check, and recorded rather than checked Product source hashes. All three fail-closed gates are now repaired and independently inspected. The actual raw cohort already had unique paired ordinals/all8 matching structure, so this changes admission robustness, not the measured totals.

Run603 confirms, per rank across measured cohorts5–8:

- 71 ordinary Target/proposal pairs, three zero-token execute entries, four final96-token handoffs.
- 7283 ordinary scheduled rows and7496 Target padded rows; the213 difference is observed padding, not a proof that213 full evaluations could be removed independently.
- The3987 first-admission prompt-suffix rows above leave3296 cached-request scheduled rows by arithmetic. Their fresh logical identity/repeated positions still need source/semantic proof.
- Source-specific execute-count assertions in this reducer are acceptable for regression against this exact recorded W₀. They must not become a universal product invariant for a future cohort schedule.

**Updated next acquisition:** after a bounded source-only preflight, prefer an all8 ordinary-preparation dispatch/producer–consumer packet, rather than a final-handoff-only timer. Its first question should be which of the71 ordinary Target calls actually takes prefill, decode or mixed attention and which Graph/captured/replay branch, and which seed/context/state dependency precedes each next consumer. Run603 gives counts and padded rows; it does not contain those actual dispatch states, actual ordinary Draft query/context geometry, or the necessary writer-stream joins.

Minimum extra metadata: exact forward-context aclgraph mode and batch descriptor at the existing set_ascend_forward_context/_model_forward boundary; existing attention-state prefill/decode counts and shape/sequence metadata (Host mirrors only where valid); current stream identity; actual Draft query/context shape at its real dispatch boundary; producer event generation and existing consumer wait relation. Preserve request/execute/slot joins to the same acquisition's basis/Product ledger. Do not infer Graph mode from padded shape or label every ordinary call "prefill."

Reuse source and current marks to classify anything already provable before adding probes. Unknown values need a targeted witness, not guessed CPU recomputation of corrected device positions. Prefer the preallocated, deferred-read event packet and existing terminal sync described in the prospective review; add only necessary fields. If this packet requires a new tensor copy, explicitly record its resource cost and apply the perturbation gate before using timing. Ordinary-prefix acceptance remains frozen; collecting it or its input identity does not authorize changing it.

This acquisition can distinguish obligatory per-request dependencies from Current batching/submission/Graph decisions and expose the relevant Resource work classes. It cannot by itself certify a new optimal schedule or a lower bound from event sums. The total Runtime still occupies most observed wall, so preparation should not be declared the largest globally removable gap; its present priority is to reduce the newly localized Product/initial-state uncertainty with a compact measurement. Sol should compare that information gain against remaining exact-board capacity/compulsory-work uncertainty before dispatching the live run.
