# Astra High — Run606 independent posterior / Bound review

## Verdict

**SCOPED PASS** for this diagnostic W0's Basis, Product accounting and ordinary dispatch observations. No unresolved identity/admission blocker found after independent raw rehash and replay. Strict Resource, Scheduling and Product Bound endpoints remain **null**. This is not a new formal performance result or the same W0 as Run99, Run597, Run602 or failed Run605.

## Independent verification

- Rehashed **97 Basis**, **281 Product**, **64 dispatch** manifest entries: **312 unique referenced files**, including upstream admission files, all match.
- Re-executed client phase, Basis, Product and dispatch validators using CPU-only container Python. All exit **0**, and regenerated JSON equals original JSON exactly.
- First review-only attempts used the wrong client module invocation and a relative Basis arm path; these failed before validation. Correcting to the controller's direct client script and absolute arm paths produced the results above. These were audit-command errors, not original acquisition failures.
- Verified exact **9 cleanup keys**, all **0**, including run_exit/final_exit. Source before/after and script before/after snapshots are identical; independently rehashed their **9 source** and **18 script/dataset** entries against current files.
- Final stop verification records no VLLM processes and idle all8 HBM. It is the recorded stop evidence, not a fresh NPU workload.
- Exact raw counts: 32 Basis, 64 Runtime, 32 worker Product records, 32 returns, warmup/measured each 48 requests plus summary. Product directory has 85 entries: 32 workers, 48 API, 4 scheduler and arm.
- Corrected dispatch reducer SHA **680c959b10afe96a3ad64df252db96609a4fa934562ecf0779bf4daeeb194d3a** now checks exact cleanup names, all three run_ts values, worker identity and complete all8 semantic_signature. Its root calculation arm.parents[4] resolves correctly for the frozen directory.
- Independently reran that reducer and reused Product reducer; both exit0 and output JSON identical. Product reducer SHA **30b13cc270627b933da35ff9fac3bbda2c9a5706154347511c5ccf657e9327a3**. The reused Product reducer only checks cleanup length itself; the exact-name gate and original file were independently verified here. Its measured.log equals the hashed client-admission measured_summary exactly.

## Recomputed same-W0 observations

Counts below are **per rank**, not summed across TP8.

| Cohort | Ordinary pairs | NONE / fallthrough | FULL / replay / existing sync | DSpark query rows | Context rows | Target padded rows |
|---|---:|---:|---:|---:|---:|---:|
| 5 | 9 | 7 | 2 | 413 | 1374 | 1400 |
| 6 | 12 | 6 | 6 | 504 | 1476 | 1496 |
| 7 | 8 | 5 | 3 | 343 | 1292 | 1312 |
| 8 | 11 | 6 | 5 | 553 | 1533 | 1552 |
| Total | **40** | **24** | **16** | **1813** | **5675** | **5760** |

Every ordinary DSpark record has Q=7, sample_from_anchor=true, use_cuda_graph=false, group_count=2 and batch_size equal to current ordinary request count. query_rows=sample_rows=7×batch_size. These are issued geometry/count observations; context rows are not certified fresh mathematical work, KV writes, or compulsory bytes. group_count=2 does not mean two model layers.

Rank0 existing replay stream.synchronize Host spans sum **2.928161 ms** (cohorts 0.320343/1.087243/0.550106/0.970469). This measures Host duration of existing calls. It is not communication time, device idle time, or removable E2E latency; removing a synchronization may merely move a wait or change correctness.

All8 finite context-event records are present. Independently checked every context event's stream ID equals that rank's first Target/Draft marker stream; Draft-input and context Host timestamps lie inside their own ordinary proposal span. All8 metadata/dispatch signatures agree. Target metadata has eight identical captured group summaries; an attn_state named ChunkedPrefill is not by itself a claim that all rows took a prefill kernel: num_prefills/num_decodes and actual Graph branch must be retained.

Basis totals: **1214 cycles**, **101848 active Target8 rows**, **116544 current physical Target rows**. These remain the captured diagnostic workload and do not replace formal Run99 workload.

## Product accounting

Recomputed **564 pre-bulk actual IDs + 48588 accepted bulk IDs = 49152** API/output usage IDs. Scheduler placeholders total **768**, distinct from actual IDs. Incoming bulk IDs total49152 before clipping. API yield-associated ID count49152; **461** are associated with client SSE receipt before the maximum allrank Runtime-built Host marker.

The 461 count is not “461 tokens of text first displayed,” a precise publication frontier, or all pre-bulk work already consumed. API parser/yield association and client fragment receipt are the captured semantics. This run has no missing association implied by the total. Namespace time:[4026531834], boot ID 6e5ca76b-a856-4531-8ab3-8d6c3ea9c8d4 and client clock description/zero offsets are consistent. Cohort Host envelopes formed with min/max across ranks are observation envelopes, not a single rank's dependency path; they must not be summed into a hardware lower bound.

## What the event cut proves — and does not prove

Let C_i be the marker after ordinary precompute_and_store_context_kv returns, T0 the first Runtime Target **entry marker**, and D0 the first Runtime Draft **entry marker**. This packet supports the current rank-local same-stream observed order **C_i before T0 before D0**. Terminal synchronization makes the event intervals readable. All C_i references produce consistent T0→D0 differences within 0.002ms rounding, and their age decreases with ordinary call order.

Across 32 rank/cohort endpoints, last-C→T0 is **56.165–64.626ms**, last-C→D0 **103.397–115.637ms**. The derived T0→D0 elapsed is **45.073–57.189ms**. These are **instrumented Current marker intervals**, not Bound endpoints or slack.

Three distinctions prevent promotion:

1. **Observed serialization is not necessary dependence.** Runtime Target does not consume Draft context KV. C_i→T0 is therefore not established as an intrinsic data edge. T0→D0 includes Target/logits/acceptance/state/Host execution; its current elapsed time does not fix the best legal schedule.
2. **Runtime D0 is not the first context consumer.** Installed llm_base_proposer._run_merged_draft calls build_model_inputs_first_pass (context precompute/store) and then the ordinary draft model in the same invocation. That ordinary query produces the next drafts before handoff. Arbitrarily postponing C_i until after T0 can violate this earlier consumer. Runtime then updates context from new Target aux before its own query; the packet does not identify which old slot generations survive and are read.
3. **Marker after a call is not a full KV-ready certificate.** slot_list_present proves only the list is non-None. Installed _store_standard_swa_kv can return for a per-layer None/empty mapping or absent cache. We did not record per-layer valid slots, actual scatter invocation/completion or exact consumer read. Same current stream constrains submitted work on that stream, not hidden side-stream work without a proven join.

Thus the packet narrows uncertainty about actual preparation dispatch/ownership and supplies a current execution skeleton. It does **not** establish a new unavoidable serialized duration, minimum context work, or the amount safely overlappable. For a future necessary Scheduling edge, bind retained logical position/prefix and per-layer KV slot generation to the earliest actual consuming query, prove stream/event joins, and maintain fixed acceptance/output semantics. Current marker age alone cannot supply that edge.

## Remaining scope, not hidden PASS claims

- No perturbation-controlled timing comparison. The repaired observer actually allocates ordinary context events; failed Run605 did not, so their times are not a valid A/B pair.
- Current stream IDs are rank-local; equal numeric IDs across processes do not synchronize rank clocks. Do not use device-event ages to infer a global TP critical path.
- Nine restored source identities plus admitted raw files are not a complete immutable hardware/software environment certificate.
- No active-row-to-compulsory-work promotion, KV-byte lower bound, attainable-capacity upper certificate, overlap gain, formal TPS replacement or ceiling conclusion.

## Artifact identities

- Basis admission: 5bbbbb3492d75c7c37679d911d1986174ddf8d3ebb599e982a90b894847defe7
- Product admission: 1d0a9bbfe941080a9b49b1b3f8b566460d94fa84664c6691ff67ae46bacd834f
- Dispatch admission: 721ab3206d1edaa437b8e630fed9e700c637f3e2ae778e50d408862bf2378a6d
- Client admission: f2b754ba1302915dad87a5f9ca171184ca6cc4c65d4f2eed3e3f254da210d7e4
- summary.json: c6e37798dc686d7bd7d6279813ac943398a282959e40e5e7aa4b102fb0d3de11
- product_summary.json: 0de47dc0821cfd86b4ff71ed8513ef8b74af06eeb7ae8730b504ddcd820c8029
- cleanup_status.txt: cd06ecab09b860d341c260eb9fe671532b7543ef42559f84d61672f69147157a
- Restored DSpark model source (source interpretation): b459fa373e89da1722085cffa63a9959502dd29fb1a0cf4b8f073e431d811867
- Restored llm_base_proposer source (source interpretation): e3e1ff579e67f845f155c983294ae5546e6ee76ba330098d7867e870c3f9f6e4

No service or NPU workload was launched by this review; only CPU artifact validation and read-only source inspection were performed.
