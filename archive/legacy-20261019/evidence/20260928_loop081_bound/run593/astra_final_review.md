# Run593 independent final review

2026-09-28. **PASS for historical native/CPU temporal census; two field-definition corrections required before treating the summary as a reusable coverage certificate. No Host→native ownership, complete cycle DAG, critical-path cost or numerical Scheduling Bound is certified.** CPU-only source/trace review; no NPU/service execution and no source edits.

## Reviewed version and independent checks

- scripts/loop081_phase_coverage_run593.py SHA256: b56d24cdbf3b4531641f98ce2b79b438a0d9d2494014d8367813f5c791d97b2f.
- run593/summary.json SHA256: 336610a9aa7b192cb3dc67622591492aac6ead9077a6419305e5611c5c07abcc.
- All24 source hashes match existing bytes.
- Imported pure reducer functions without main; independently re-ran reduction from all8 original trace files. Every stored rank row matches exactly. Both existing negative tests reject on all8.
- Independently enumerated all ph=X records with Model Id, including IDs outside the reducer's whitelist. No additional model IDs exist in these selected traces; the current whitelist loses no such native records here.
- All8 have10824 Model45 records, plus1782–1794 sentinel-model4294967295 records. Seven top-level phase scope types each occur twice and are nonoverlapping within each reported occurrence; one enclosing complete outer cycle scope is present.
- Session selection uses the same final profiles already independently verified by Run592: same PID per rank across five sessions, chronological metadata ends, and only final session end inside final window. Run593 newly pins profiler_info/window. It still has no direct profile→request/cohort join. Run592 review documents the exact limitation.

Recomputation confirms the published midpoint counts. Agreement establishes reproducibility of the statistic, not correctness of a semantic task-to-phase join.

## Two substantive definition corrections

### 1. DSpark repeated scopes overlap; they are not strictly nested

The boolean dspark_layer_scope_counts_are_nested_not_replays=true overstates the geometric relationship. The reducer checks only10 names per layer, not nesting.

I sorted layer43/44/45's ten scopes by start and examined adjacent pairs within each five-scope cluster. All24 pairs per rank, across all8 ranks, satisfy crossing overlap: startA < startB < endA < endB. None of these tested pairs is strictly nested.

Concrete rank0 layer43 first pair, µs:
- A=[1790356535662120.780,1790356535673053.660]
- B=[1790356535662159.470,1790356535673085.310]

The valid conclusion is **repeated overlapping CPU scopes are not a certified model-replay count**. Suggested field: dspark_layer_scope_counts_are_overlapping_not_replay_count. Do not claim their cause is duplicate instrumentation, distinct model calls or nesting without API/call-stack/source-generation evidence. They must not be summed into work or latency.

### 2. native_boundary_crossings is not all boundary crossings

The reducer increments only when a task's midpoint lies inside a phase and one endpoint lies outside it. A task whose midpoint falls in an inter-phase gap can cross phase boundaries yet contribute zero. Thus this count is a subset statistic and cannot quantify all carry-in/carry-out.

Independent counts of task–phase noncontained interval intersections, compared with stored counts:
| Rank | Stored midpoint-assigned crossing count | All noncontained task–phase intersections |
|---|---:|---:|
|0|1|3|
|1|0|0|
|2|2|4|
|3|1|1|
|4|1|1|
|5|0|0|
|6|2|12|
|7|6|12|

These added counts are **intersections**, not distinct-task counts; a task can cross two phase boundaries.

Concrete rank0 MatMul interval:
[1790356535699033.121,1790356535699061.0215]µs.
It intersects the end of proposer0 (ends1790356535699036.590) and beginning of draft_commit0 (starts1790356535699061.000), but its midpoint is in the gap, so it is classified outside and absent from native_boundary_crossings. Another Slice task crosses draft_commit's end and is counted. This is direct evidence that midpoint partitioning cannot assign completion/ownership.

Either rename the existing field to native_midpoint_assigned_boundary_crossings or additionally report complete interval intersections plus unique task counts, carry-in/out and outside-midpoint crossings. Do not rewrite existing counts without preserving their definition/provenance.

These issues do not invalidate the published midpoint census. They block interpreting the current booleans/count labels as a complete overlap/lifetime certificate. Parent was notified with exact examples; this review pins the version above.

## Why temporal classification is not ownership

The script correctly labels each row as temporal overlap only and sets phase_ownership_from_midpoint=false. Preserve those qualifiers wherever numbers are quoted.

The following would be false joins:
- “591–711 DSpark native tasks”: currently these are non-Model45 native midpoints during proposer Host scopes, not proven DSpark-owned launches. Pending earlier work, Host copies, runtime events and asynchronously completing work may share the interval.
- “41 Target tasks outside Graph”: these are41 sentinel-model task midpoints during each Target scope. Ownership or semantic purpose still needs launch/source correlation.
- “0 prepare tasks means preparation absent”: zero midpoint count can mean work executes later, including during metadata/proposer scopes.
- “outside4–125 tasks means only4–125 unattributed operations”: most temporally classified tasks also lack ownership. Outside is the complement of selected CPU time windows, not a global attribution-error metric.
- “two5412-task Target DAGs certified”: counts and temporal containment support two historical Model45 task inventories, but cardinality alone does not establish unique static node identity, event IDs, typed dependencies or exact source generation.
- “non-Graph means unnecessary framework overhead”: sentinel Model Id is an export association, not a work-necessity classification. It includes real device kernels, DMA, CPU tasks, events and model execution control.
- “Host scope length is CPU-only submission time”: these instrumented scopes can include synchronization/device wait, Python work and profiler overhead. The existing field usefully rejects device-service equivalence, but ‘submission plus profiler’ is not an exhaustive decomposition.
- “one outer scope means second cycle never completed”: profiler lifecycle can clip scope export. Two phase sets are observed; a complete second-cycle closure is not certified.

No task time sum, phase duration sum or midpoint count ratio measures useful compute, capacity, critical-path contribution or removable wall time. The historical Level1 synchronization/overhead also prevents direct transfer to passive Current.

## Validator scope and remaining gaps

Current negative tests cover missing phase and duplicate outer scope only. They are appropriate basic shape checks, not a comprehensive admission suite.

Before a complete DAG claim, validate:
- Native unique keys/types/durations and duplicate export rows; model/replay boundaries via correlation.
- Host PID/TID and call identity, not name alone; native physical stream mapping and clock provenance.
- Full interval overlap including midpoint-outside cases and profile boundary clipping.
- Nested/overlapping DSpark scope topology, not count alone.
- Actual launch→native correlation; producer→consumer events; tensor/KV/page/version lifetime and off-stream completions.
- Previous/current/next cycle buffer lineage, Host-copy consumption, acceptance/draft state, and complete output ledger.
- Direct same-generation cohort/request/cycle identity. Current final-session chronology remains conditional.
- For new datasets, fail on unexpected Model IDs rather than silently whitelist them; this did not lose rows in the reviewed input.

No large test expansion is needed solely to retain the existing descriptive census. Strengthen only the gates needed for the next promoted claim.

## What uncertainty really narrowed

Run593 demonstrates that the retained historical profile has substantial native evidence beyond Model45 and preserves recognizable full Runtime/DSpark/Host phase markers. A Target-only model omits1782–1794 observed native records in this two-window sample, although their count is not a cost. Thus the next causal frontier investigation can reuse existing data instead of assuming a fresh full service run is necessary.

It also shows the limits of temporal attribution concretely: variable non-Graph phase counts, pending work crossing the proposer/draft boundary, incomplete outer scope coverage, repeated overlapping DSpark scopes. This reduces **measurement coverage/attribution uncertainty** and identifies which edges still need correlation. It does not shrink a numerical optimal-time interval by itself because task ownership, necessary dependencies, service floors and feasible resource overlap remain unclosed.

The next high-value action is bounded offline correlation of the whole-cycle frontier: use existing Host-to-device flow/API correlation, native event wait/record and source-defined state/Host-copy lifetimes to determine whether previous proposer work or current preparation is executing across these boundaries. A successful result must join identities and completion, not merely rebin timestamps. If those correlation fields cannot distinguish the required producer/consumer, retain an opaque fence and use the Run593 preflight's minimal same-generation cycle-sandwich acquisition.

Do not optimize a phase merely because it has many midpoints. Use the resulting causal DAG and resource-contention uncertainty to decide whether removing one current boundary could shorten the complete fixed-algorithm path. Run579/580 remain a counterexample to ideal whole-chain overlap, not a prohibition on all runtime overlap.

Current Formal571.681tok/s is unchanged. Resource, Scheduling and Product numerical bounds remain null; no new Current→Bound distance is admitted.
