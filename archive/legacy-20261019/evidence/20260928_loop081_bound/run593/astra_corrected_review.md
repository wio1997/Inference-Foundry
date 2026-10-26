# Run593 corrected independent review

2026-09-28. **PASS for the corrected historical native/CPU temporal coverage census. Both issues raised in astra_final_review.md are resolved in the reviewed fields. No phase ownership, complete Scheduling DAG or numerical Bound is promoted.** CPU-only parsing/hashing; no service/NPU and no source edits.

## Pinned versions

- Corrected summary.json SHA256: 2d658c3babd338cefcb63b27c44f99d94f64f086eb3b0fc903bfb32310a2a596.
- Corrected scripts/loop081_phase_coverage_run593.py SHA256: 0e831cc5ec05b9cd84c42c65bf020a5de3613cad70d8971656fb0e1c6659ca29.
- Preserved summary_pre_review_invalid.json SHA256: 336610a9aa7b192cb3dc67622591492aac6ead9077a6419305e5611c5c07abcc; exact match to the originally reviewed summary.
- All24 pinned original trace/profiler-info/window source hashes reverified. The source-hash map is unchanged between old and corrected summaries.

## Independent recomputation

Parsed all8 raw trace files directly using Decimal. Rebuilt each Host phase interval, native midpoint/model/type census and fully-contained count without using the corrected reducer functions. All phase rows match. All native ph=X records carrying Model Id have one of the two admitted IDs in this dataset; no unlisted-model native record is silently omitted here.

Recomputed every native task against **all28 phase start/end boundary instances**. A boundary is counted only when it lies strictly inside the task interval (a < boundary < b), regardless of midpoint position. Counted each intersecting task once in the distinct-task statistic. Results exactly match corrected summary:

| Rank | Phase-boundary point intersections | Distinct tasks crossing a phase boundary | Native midpoint outside selected scopes |
|---|---:|---:|---:|
|0|3|2|13|
|1|0|0|4|
|2|6|2|86|
|3|1|1|4|
|4|1|1|6|
|5|0|0|4|
|6|18|4|102|
|7|18|8|125|

This corrects the old omission of tasks with midpoints in a gap. Important definition: these are **boundary-point instances**, not task–phase noncontainment pairs reported in the first review. A long task spanning a complete phase crosses its start and end, so corrected point counts can be larger than the first review's pair counts (e.g. rank2 6 points versus4 pairs). This is a legitimate, explicit metric distinction, not a mismatch. Exact endpoint touches are excluded by strict inequalities and should not be described as included intersections.

All unchanged phase midpoint/containment rows, outside counts and rollups equal the preserved old summary. Thus this correction did not perturb the underlying census or reassign task ownership.

## DSpark scope correction

The summary now uses dspark_layer_scope_counts_are_overlapping_repeated_scopes_not_replay_count=true. The previous nested assertion is absent.

Independent all8 check sorted layer43/44/45 scopes into two chronological five-scope groups per layer. Each rank has24 adjacent pairs; all satisfy strict crossing overlap a < c < b < d, not strict nesting. The corrected wording is supported. Multiplicity does not certify model replay count or identify why the repeated scopes occur.

The remaining nested_scope_counts field names the separate DSpark Host-mirror/context/prepare/pack/model subphase inventory. It is not the rejected assertion that each of the ten layer scopes is geometrically nested. Do not conflate these two inventories.

## Residual scope boundaries

The corrected fields preserve:
- phase_ownership_from_midpoint=false;
- assignment_rule explicitly temporal overlap only;
- no fixed-W0 or direct request join in the contract;
- resource/scheduling/product bounds null.

No new residual ownership overclaim was found in the correction. Prior limitations remain unchanged:
- Model45 cardinality/containment is an observed historical inventory, not a typed source DAG or arbitrary-node uniqueness proof.
- Sentinel-model task midpoints within proposer/Target scopes are not automatically owned by that stage.
- Outside counts and boundary-point counts quantify timing geometry, not the number of semantically unjoined tasks or device idle.
- A complete outer Host scope appears once; this does not certify every asynchronous producer/consumer has completed there, nor prove the second cycle failed.
- Host scope length can include CPU work, queueing, synchronization/wait and profiler overhead. It is not a pure submission-cost measurement.
- Latest-session/window association remains the historical ordinal/temporal association reviewed in Run592; no direct cohort/request/cycle linkage is added.
- Native/event timing and task counts do not identify mandatory dependencies, compulsory work, legal overlap, service minima or removable E2E time.
- Two basic negative tests are not a general admission test suite; no additional tests are necessary merely to retain this descriptive result.

## Bound impact and decision

Admit the corrected census as historical **coverage and attribution-gap evidence**. It confirms useful native Runtime/DSpark/Host material is available outside Model45 and now honestly records cross-phase task spans. It reduces uncertainty about what can be studied offline and where midpoint classification breaks down.

It does not independently narrow a numerical fixed-W0 Scheduling interval. The next useful step remains causal Host/API/native/event and cross-cycle storage/KV correlation on existing evidence. Only a concrete missing producer/consumer certificate should trigger the minimal same-generation cycle-sandwich acquisition. No new local kernel scan or optimization follows from these counts.

The old summary is preserved for traceability; use corrected summary plus this review for future temporal-coverage references. Keep astra_final_review.md as the historical explanation of the two resolved findings. Current Formal E2E remains571.681tok/s, and numerical Resource/Scheduling/Product endpoints remain unpromoted.
