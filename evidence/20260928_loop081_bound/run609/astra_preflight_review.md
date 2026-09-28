# Run609 — Astra independent two-cycle packet design review

**Verdict: SCOPED DESIGN PASS.** This admits an offline prior-W0 selection and source-backed measurement checklist. It is **not LIVE-READY**, a completed native dependency packet, or a numerical Bound certificate.

## Identity and independent checks

- Script `loop081_two_cycle_gate_run609.py`: **7d19bc22ada7f15e467674d166598456110d98e5532a75b14de76c9e1b49f729**
- Output `two_cycle_packet_gate.json`: **5727e9e6738108e515e46c3ef72b42e33b03b1131d075ac03ed9181ac3416097**

Independently called build() without invoking main's write path. All nine installed-source SHA/predicate gates and eight selected raw Basis SHA joins pass; regenerated JSON equals the existing output byte-for-byte. Source and raw-data reads only; no service/NPU execution.

The pinned Basis admission was independently replayed in Run606 review. This small gate itself rehashes the selected eight cohort5 Basis files, not all97 upstream inputs, and does not re-run the full Basis correctness checker. Its scope is appropriately narrower.

## Selected prior and arithmetic

Run606 cohort5, **zero-based cycle indices64/65**, has12 positive masked counts in each selected cycle and branch_history scheduled_target=false, schedule_mode=off. All8 count vectors and request order agree:

- cycle64 counts4/6/5/2/8/1/3/5/2/1/2/8: **sum47**.
- cycle65 counts2/8/4/1/8/3/7/4/6/3/5/8: **sum59**.

The cycle64 actual Draft witness has96 context rows and84 query rows, Q7 and sample_from_anchor=true, with group labels2/3. Independently verified the full raw cycle64 witness is identical across all8. Each group has96 unique context labels and84 unique query labels; set intersection is **49**. Per request, the numeric overlap equals8 minus that request's cycle64 count, summing96−47=49. This consistency is descriptive, not proof of a memory hazard.

There is **no cycle65 actual sparse slot witness**. Do not infer an observed cycle65 overlap of37 from96−59. Counts47/59 and overlap49 are Run606 prior values; a new diagnostic W0 need not reproduce them. New-run output/acceptance gates must reconcile that run internally or an explicitly restored same-state control, rather than demand equality to this old run.

## Source typing and proof boundary

The nine pinned sources support the proposed roles: Target forward/logits, greedy acceptance, state advance, host-count mirror, input preparation, hidden combine, Draft context projection/scatter, query and next-draft commit. Substring predicates prove the reviewed source contains these anchors; they do not prove executed branches, native task identity, completion, consumer binding or legality of a new schedule.

Positive counts and off/unscheduled branch_history establish the prior active/normal stratum. They do not independently certify native FULL96 replay at these cycles. The proposed live gate must obtain actual Graph mode/generation/replay identity on all8; ordinary Run606 Graph records cannot be reused as Runtime-cycle identity.

Group2/3 are attention-group labels, not two layer counts. Equal slot numbers identify neither a common per-layer cache allocation nor its generation. Context writes may be skipped for None/empty layer maps or absent cache, and query stores can overwrite a generation before a later read. Retain cache identity, offset/format, allocation lifetime, actual write task, validity and first read before asserting WAW/RAW hazards, dependencies or compulsory traffic.

## Required live preflight / admission details

The stated gates are sound in direction, but the live implementation still needs source-exact hooks, canonical ordinary/Runtime ownership, native-label support and a tested controller/validator before authorization:

1. Capture actual per-layer producer and consumer descriptors plus existing cross-stream joins. A Host method return/current-stream event is not ready unless all writers are joined.
2. Include state/metadata and **lagged Host-count copy generations**. Cycle64 may consume the pending copy from63; cycle65 consumes64. Carry-in generation63 must be identified even if its work lies outside the selected trace.
3. Define open boundaries: two cycles close Draft64→Target65. If claiming the subsequent consumer of Draft65 too, add only the needed Target66 entry/first-use witness or explicitly leave that edge open. Do not classify terminal packet outputs as dead work.
4. Preserve actual HCCL message identity/dtype/participating bytes and peer ordering. Native task count, elapsed duration or matching addresses alone do not certify the mathematical collective dependency.
5. Specify exact offline/ON instrumentation costs, buffer ownership/capacity, raw-trace save-before-parse and exception/restore gates. “No added model work” permits diagnostic copies/events only when accounted as observer work; it is not a zero-perturbation claim.
6. Structural admission precedes timing use. A0/probe/A1 requires matching restored entry state or a clearly weaker uncertainty statement; adjacent natural cycles and the old Run606 trajectory are not same-state timing controls.

No acceptance/cycle-reduction objective, marker-age slack, compulsory-work promotion or finite endpoint is introduced. Formal Current571.681tok/s and strict Resource/Scheduling/Product null endpoints are preserved. No arithmetic or provenance blocker was found for this **offline design artifact**.

## Final delta recheck

The final script now asserts the per-request overlap relation on all8 and explicitly states that new-W0 actual Graph/shape must be measured,47/59/49 are not new-run acceptance gates, Draft65 successor closure needs Target66 or an open boundary, and alternative storage/layout remains admissible. Independently rebuilt this revised output byte-for-byte. The identities above supersede original script adcc2b5e… / output3232ede3…. Verdict remains SCOPED DESIGN PASS; no executable live collector/controller was reviewed in Run609.
