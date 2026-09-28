# Run630 independent Astra High Product hurdle review

**SCOPED PASS for paired formal accounting and hypothetical net E2E saving hurdles. No Bound or attainable performance interval is established.** Review is read-only except this report; no live service/NPU work.

Final reviewed generator `scripts/loop081_product_hurdle_run630.py` SHA256: `624ccbc96bd987acb68d6d5a2a81431b9f6b7246930cd116ebfb3d9f9b1ce3eb`.

Final `product_hurdle.json` SHA256: `63c8e7690136c3a74f4471b2cadd2d33f5d5cd4fb47e8ffafcb159cdb2ec7620`.

I independently executed main with output writes intercepted in memory and reproduced the saved JSON byte-for-byte. Run629 independently checked the underlying three Run99 clients and all8 measured cohort identities/counts; Run630's paired arithmetic matches that check and checks Run153/238 against V3.44. The recorded source hashes match the reviewed artifacts. This is reproducibility and accounting validation, not a new physical measurement or universal schema-security test.

## Corrections resolved

The first reviewed version labeled its uniform saving `all_1212_cycles` although it divided by the paired median repeat's1206 cycles. It also used wall implied by rounded571.681 TPS as numerator while dividing by exact repeat3 phase sums. The final version explicitly selects repeat3/cycles1206, uses its exact85.97796410089359s client wall for all hurdles, and separately records the rounded-TPS implied85.97801921001397s and55.109120µs difference. Both issues are resolved.

Because every repeat reports the same49,152 outputs, sorting client wall and selecting the middle repeat selects the same sample as median exact TPS. Its exact TPS is571.6813664292052, reported rounded571.681. Median of durations would not generally imply median throughput with unequal token counts; that case is outside this fixed artifact.

## Independently checked formulas

For landmark p, desired Product wall is49152/p; required net saving is exact repeat3 wall minus that desired wall. Fractions divide by the **same repeat3** Runtime scope69.03965463861823s and arithmetic remainder16.938309462275356s. Uniform per-cycle equivalents divide by1206; they require a hypothetical equal **net E2E-exposed** saving on all these cycles, with other work and trajectory unchanged.

| Landmark tok/s | Net E2E saving required s | Uniform equivalent ms/cycle |
|---:|---:|---:|
| 581 | 1.378996803 | 1.143446769 |
| 607 | 5.002675798 | 4.148155720 |
| 616 | 6.185756309 | 5.129151168 |
| 682 | 13.907582869 | 11.531992429 |

These calculations are correct. They neither identify where that saving exists nor demonstrate that any runtime can attain it. In particular, fractions of the arithmetic remainder are comparison units, not a claim that the remainder is an independently removable phase. Uniform equivalents do not imply that reducing one kernel by that duration gives the required net saving. Overlap, resource interference, critical-path exposure and boundary effects still apply.

## Wave accounting and scope limits

The final wording correctly says summed client wave envelopes exceed client wall by roughly7ms and do not partition it. Run629 checked adjacent nominal waves overlap2.589–2.687ms. The sum excess is **not exactly the sum of overlaps**, because the single client timer also contains small leading/trailing boundaries. Run630 does not make that erroneous equality or treat the excess as timer error.

Run99 lacks absolute server scope endpoints and a client/server request-ID token join. Pairing is by the accepted repeat/cohort assignment, not a certified interval-containment relation. T−sum(rank0 Runtime) is therefore an arithmetic residual. The JSON explicitly forbids naming it removable Host time or using approximately69s Runtime as a compulsory floor. Distinct Run602/610 W0 timing is explicitly excluded from filling these formal costs. Sample ranges are descriptively correct and not promoted to noise distribution or causal attribution; their extrema need not occur in the same sample, so range values must not be summed as a variance decomposition.

All four strict endpoint fields remain null. The old581–607/616–682 rates are clearly landmarks, not ceilings, stopping conditions or current confidence intervals. No remaining correction is needed for this narrow scope.

The next-measurement sentence is a design direction, not live-readiness approval. Follow Run629's offline-first recommendation: reuse admitted same-W0 Run602/606 and local Run610/611 coverage before acquiring a new ledger. OFF/ON/OFF performance control alone does not prove identical acceptance trajectory; retain exact W0/Basis/Product identity checks and treat unpaired trajectories separately. Whole-Product assessment must keep the dominant Runtime uncertainty visible alongside the residual.
