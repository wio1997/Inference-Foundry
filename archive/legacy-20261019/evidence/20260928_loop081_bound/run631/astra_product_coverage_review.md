# Run631 independent Astra High Product coverage review

**SCOPED PASS as two separate diagnostic Host-marker coverage ledgers.** This corrects attribution of record boundaries; it establishes no removable time, device occupancy, attainable interval or cross-W0 timed DAG. No live/NPU experiment was performed; only this report was written.

Final generator SHA256: `986654afa7bfb0280d3ed36cd06b5265834e1556bd95d352be8e0bb7f0d04151`.

Final `product_coverage.json` SHA256: `ceafd884c8df096bd07e80d9f37a13a378879ac93b9c95b29c87981c261abee2`.

## Reproduction and admission challenge

I independently executed the final main with output writes intercepted in memory and reproduced the saved JSON byte-for-byte. I also directly rehashed all32 consumed measured product rank/cohort files and client-admission manifest references per W0 against the prior Product admission; all match. Same file appearing in relative/absolute manifest entries is not independent evidence.

Initial reducer revisions did not authenticate the consumed raw files and accepted wrong run tag, wrong marker rank, duplicate built request ID and altered timing. Sol corrected these gates. Nine independent negative challenge classes now reject: raw product SHA mismatch; raw client-admission SHA mismatch; wrong run tag; wrong marker rank; duplicate built request ID; duplicate response join; duplicate sync marker; rank-local built/serve order reversal; and mismatched time namespace. Structural mutations were injected after parsing in memory to exercise checks independently of file hashing; hash mismatches were separately injected. No source evidence was modified. This verifies the named gates, not every possible malformed-input case.

The final reducer checks each rank/cohort identity, run tag and namespace, exactly one built/serve-start/sync marker per rank, per-rank marker ordering,12 unique built request IDs matching the joined cohort,48 unique server request IDs, and a bijection to48 measured client response IDs. Prior admitted token/body/output joins are inherited rather than reinvented. The output records the admissions' hashes, with raw identity transitively authenticated by their manifest.

## First-recorded versus first-own attribution

I inspected every measured rank/cohort execute marker independently. In both W0s, cohort5 has no prior-cohort or no-request execute. In each of cohorts6–8, **each rank** has exactly one previous-cohort-only execute, one execute without request operands, then own-cohort-only executes. There are no mixed previous/current request-ID sets in these admitted records.

Per-rank own execute counts for cohorts5–8 are17/26/15/14 in Run602 and10/12/8/11 in Run606, including the handoff execute. Thus the all8 aggregates of eight prior and eight no-request markers per later cohort are supported, not inferred merely from aggregate counts. A no-request operand marker does not prove no device activity or no residual side-stream work.

First-recorded→first-own ranges **0.420616970–0.476266056s** across the later cohorts. This is a correction to which cohort owns the initial recorded calls. It does not remove time from the actual client path or prove that any of the interval is free/idle. First-own is the earliest Host execute with that cohort's operand on any rank, not simultaneous all8 start, admission completion or first physical device work. Last-allrank built is the maximum Host built marker, not initial KV/seed readiness. The output states these scope limitations correctly.

## Clock and endpoint accounting

The source admissions bind one server/client Host time namespace per W0; recorded client boot identity is retained. This is current Host-clock evidence, not a certified synchronized device clock or causal readiness certificate. The two W0s remain separate.

The final reducer distinguishes outer client clock anchors, measured wall monotonic anchors and the original reported perf-counter duration:

| Run | Outer clock span s | Measured monotonic wall s | Reported perf-counter s | Outer−measured s |
|---:|---:|---:|---:|---:|
| 602 | 91.485045846 | 91.483705141 | 91.4837018204853 | 0.001340705 |
| 606 | 83.397178606 | 83.395760301 | 83.3957535708323 | 0.001418305 |

The3.321µs/6.730µs differences between separately sampled inner monotonic and perf-counter durations are preserved, not forced to equality. Request end is later than the prior client DONE marker by approximately0.147–0.217ms. Consequently this reducer's end must not silently replace the earlier DONE endpoint. The old summaries' broader first-execute→built scopes remain historical observations; their labels should carry the new previous-cohort contamination qualification.

## Independent union calculation

I independently swept only the full cohort first-recorded→last-client-end envelopes, using timestamp endpoint counts rather than the reducer's phase-signature sweep. Union and multiple-envelope time exactly match; sums of mutually exclusive active signatures equal the union:

| Run | Covered Host-envelope union s | Uncovered within outer clock s | Multiple envelopes active s |
|---:|---:|---:|---:|
| 602 | 91.217689051 | 0.267356795 | 0.843905122 |
| 606 | 83.138430547 | 0.258748059 | 0.773280844 |

Covered plus uncovered equals the corresponding outer client span exactly in integer ns. Multiple-envelope time is a subset of covered time; it must not be added to the union. It counts intervals with multiplicity greater than one once, not an arbitrary summed overcount. The phase-signature sweep partitions **Host envelopes**, not physical work. A large interval between two Host markers remains covered even if it contains idle time, NPU work, queued work, waits, serialization or activity from another phase. Conversely, uncovered time is outside these selected envelopes, not a proved Host overhead or elimination budget.

The model correctly prohibits transferring these durations into Run99, subtracting Run606 from Run602 as an intervention gain, or treating a broad envelope as compulsory work. All strict Resource/Scheduling/Product endpoints and numeric Current-to-limit distance remain null; Current Formal571.681tok/s is unchanged. No further correction is required for the declared current Host-marker accounting scope.
