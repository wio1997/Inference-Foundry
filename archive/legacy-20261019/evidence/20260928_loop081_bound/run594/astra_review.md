# Run594 independent Astra review

## Verdict

**SCOPED PASS for the corrected offline Run566 A0 basis/state-transition regression.** This is not an independent Host-event measurement, a new live compact collector validation, a Draft-model evaluation ledger, a same-state Run99 join, or a necessary-work/capacity certificate. Resource, Scheduling and Product numeric endpoints correctly remain null; Current Formal remains 571.681 tok/s.

No service, NPU workload, or device query was run. The review imported the CPU-only reducer, verified hashes, recomputed all 32 rank/cohort results, and exercised mutation fixtures. Only these new independent review artifacts were written.

## Identity and correction history

- Initially reviewed script SHA256: `7933f785bfe16f1c926bc633a6722c5fbd29b096c45d556d8dedf3a15e84c23f`.
- Corrected script SHA256: `ffc6832018fa7097037e4f615ca63bd59eeb522541b474041794036aec419fe6`.
- Current summary SHA256: `2044b0ecf53f6cc8a25c7491e6fb34f2d4518e852490b7ee07d7a735b6e22de0`.
- Parent preserved the former result as `summary_pre_review_invalid.json`; its negative-test validation is **superseded/invalid**. Numeric census was unaffected.
- The summary bytes are unchanged after the fix: the original summary recorded labels rather than actual assertion outcomes. Therefore its hash alone cannot distinguish valid from invalid execution. Cite the corrected script identity and rerun exit record together. Independent review checks are in `astra_review_checks.json`.

The original negative fixture copied only three cycles but supplied the full-cohort staged counts. Even its unmutated baseline failed `active sampled/staged mismatch`. Four injected mutations happened to fail earlier at their intended checks, but the old harness accepted any ValueError and could not establish test validity; the wrong-staged test was vacuous.

The corrected version first reconstructs the full unmodified cohort successfully, then applies one mutation per full copy and requires the intended error substring. Independent execution confirms all five cases reject correctly. This closes the identified fixture defect. No further live experiment is needed for this repair.

## What was independently verified

All 64 trace/runtime source hashes match the summary. Every one of the 32 complete rank0..7/cohort5..8 reconstructions reproduces its summary row. In addition to the reducer’s aggregate consensus, raw trace bytes are identical across all eight ranks separately for each cohort; this is stronger evidence for these actual inputs, though not a general guard implemented by the reducer.

| Cohort | Cycles | Physical Target rows | Active Target rows | Parked rows |
|---|---:|---:|---:|---:|
| 5 | 318 | 30,528 | 24,368 | 6,160 |
| 6 | 283 | 27,168 | 23,824 | 3,344 |
| 7 | 299 | 28,704 | 26,064 | 2,640 |
| 8 | 306 | 29,376 | 24,240 | 5,136 |
| Total | 1,206 | 115,776 | **98,496** | 17,280 |

These are Target-8 row counts per replicated input ledger; do not multiply the logical model workload by eight ranks. Active samples match each runtime staged count, and generated lengths are 1024 for all slots. This reproduces Run570’s admitted current evaluation-class census, not an independently derived mathematical compulsory-work floor.

Given the initial position/last/draft basis, every cycle checks Target IDs = [last, draft7], Target positions = position + [0..7], accepted nonnegative prefix/count, last-token update while active, and next_draft -> next cycle draft. Parked slots freeze position/last while draft still advances. The raw snapshots in `runtime/extreme_decode.py` are tensor clones taken before/after preparation, not JSON values constructed offline from the formulas under test. Thus checking those snapshots against the reconstructed basis provides a real state-transition regression.

## Circularity and scope limits

1. **Parking is extracted from the expected state.** The reducer reads the following full-trace position and accepts a discrepancy only when it matches origin + 960, a rewind, and >=1024 staged tokens. This is an explicit extra input to the reconstruction. Once the park event is extracted, its following position match is partly tautological. The script and summary disclose this correctly; there is no independent proof of Host-mirror timing or that counts alone determine the park cycle. A future compact collector must record actual park events and anchor values. Split event extraction from pure replay and pass an explicit event list before claiming basis sufficiency.
2. **Initial state is assumed from cycle zero.** This does not certify prefill/seed/KV provenance or replay equivalence. The final next_draft has no following cycle against which lineage can be checked; only its shape/nonnegative values are checked. This is appropriate as captured basis, not evidence of necessary final proposer work.
3. **Acceptance is an input here.** `target_argmax` is not used by this reducer. Accepted-prefix structure and count consistency are checked, not target logits -> greedy acceptance semantics. The existing `check_loop035_trace.py`/Run566 admission can provide that distinct evidence; pin the admission/checker/source identities when promoting the evidence chain. No oracle or KV-value parity follows from this regression.
4. **Draft lineage is not Draft execution reconstruction.** next_draft -> draft_before is checked, but Draft query/context positions, context projection, KV ownership and Markov dependencies remain future ledger work.
5. **All8 consensus implementation is aggregate-only.** It compares cycle/row/park summaries, not request identities or complete semantic basis. Actual raw traces match in this review; future runs must explicitly compare semantic hashes and request joins, rather than rely on aggregate agreement.
6. **Freshness/necessity remains unproved.** 98,496 is current active evaluation-class work, including possible identical semantic states, terminal overrun and reusable results. Rejected verification rows cannot simply be removed either. Necessary-work classification needs consumer dependencies, actual prefix/KV state, admitted initial cache and justified dedup. No compulsory HBM/HCCL claim is supported here.

## Minimum follow-through before live compact acquisition

Keep the corrected negative harness. Pin reducer/source/admission identities. Feed an explicit extracted Host-event list into a separate pure basis reconstruction, then mutate missing, shifted, duplicate and wrong-anchor events; the unmodified fixture must pass first and each mutation must fail for its intended dependency. Add shape/missing-cycle/request/rank semantic-join guards where they are required by the future collector. These are bounded CPU checks on existing evidence, not reasons to rerun Run566.

The live collector still needs explicit Host events, initial provenance, its measured-window identity, Draft query/context reconstruction with sparse observed checkpoints, and a perturbation gate. Graph-external copies still add execution work/submission latency. This offline pass neither measures nor waives that gate.

One new diagnostic W0 cannot identify any of Run99’s three historical W0 trajectories. Keep historical aggregate-compatible envelopes separate from the new ledger. The matching total 1206 cycles with Run99 repeat3 is not a trajectory identity. No numeric Bound endpoint or formal TPS update follows from Run594.
