# Run434 V3.14 proof-obligation review — Astra

2026-09-27. Read-only inspection of generator and generated JSON; stdlib JSON checks only. No service/NPU/framework import or source modification.

## Verdict

**Safe as a conservative evidence ledger; not yet a machine-enforced proof-obligation DAG.** No erroneous promotion of Run432/433 to all43 or numerical Product bound occurs in the current artifact. All three latency floors and Product finite ceiling remain null, and all43 is explicitly `open_conditional_native`. Fix the contradictory inherited decision rule before treating the document as an authoritative gate specification.

## Findings

### P1 — inherited Product decision rule contradicts the strict outer-ceiling gate

`next_measurement.decision_rule` still says every numeric Product bound requires a legal resource-constrained DAG and corrected E2E calibration. The new `strict_outer_ceiling` correctly requires only a matched necessary-work lower bound and true capacity upper bound. A positive matched `W_minus/C_plus` can establish a strict outer ceiling without constructing an attainable schedule or running a new intervention.

Separate the rule explicitly:

- Strict outer ceiling: prove a positive in-window unavoidable W_minus for the declared legal execution class, with a matching true upper capacity C_plus, both scoped to the same timed/output contract.
- Attainable engineering performance: exhibit a legal implementation/schedule and repeated frozen E2E evidence; add practical mixed-resource evidence for a predictive model.
- Scheduling relaxation: classify whether its endpoint is a lower latency bound or an attainable latency estimate; these require different evidence.

This is an over-restrictive/confusing rule rather than an unsafe numerical promotion today. It also means all43 identity is not a universal prerequisite for every possible W_minus: another independently justified necessary subset could suffice. The suggested all43 measurement is relevant to the current retained-route numerator, not the sole route to a strict bound.

### P2 — Scheduling node lacks explicit numerical-duration and endpoint semantics

`typed_all8_critical_path` lists stream/all-rank completion, state reuse and overlap feasibility, which are necessary to interpret the observed schedule. But typed edges alone cannot generate a numerical bound. The DAG must state whether task durations are observed Current intervals, proven lower service durations, or attainable estimates; also their common time domain/error envelope and timed-window boundary ownership.

For a lower latency bound, a proven necessary dependency path plus valid task-duration lower bounds can suffice as a relaxation; full overlap feasibility is not required. For an achievable schedule, full resource/storage feasibility is required. Calling both a `scheduling_aware_interval` without endpoint types risks later confusing a longest-path relaxation with an attainable schedule. Missing arrival/publication/preexisting-work boundary obligations should be referenced explicitly, not merely left in the inherited descriptive ledger.

### P2 — certificate_graph is a descriptive hierarchy, not a validated dependency DAG

Only Product entries have `requires`; several are free-text phrases with no addressable node. No unique node IDs, dependency resolution, acyclicity, conjunction/alternative semantics, status propagation, scope compatibility, or numeric promotion checks are implemented. The script pins sample counts and searches review keywords, but does not bind source/result hashes or validate all per-case predicates. It would accept a mutated result retaining the pass label/case counts while changing geometry, range or diagnostic contents. Assertions also disappear under Python optimization.

Current null assertions prevent a finite-bound output in ordinary execution, so this is not evidence of an already published false bound. Label schema1 as a human-reviewed obligation ledger or add an actual fail-closed graph validator before future automated promotion. Specifically bind Run432 mutation source_sha256 to the reviewed sentinel script and snapshot evidence hashes. Immutable records matter because result.json and review files have already advanced through versions.

## Sampled ABI scope: correctly retained, with one useful explicit addition

The eager node remains sampled_pass and lists absent tiling provenance, EP ranges/branches and scale/GMM pairing. The all43 node stays open, with actual FULL entry/kwargs, branch/group/EP maps, native boundaries and logits/retention joins missing. Its effect correctly limits Run433 to isolated synchronized fixed-storage generation refresh. `run430_433_native_scope.not_proved` additionally names exact graph all-slot sensitivity. No all43 proof or full-model W_minus is inferred.

For clarity, put `exact graph all-slot sensitivity` directly into the all43 node's missing list too. Run433's position-based payload and0.015 tolerance do not inherit Run432's1440 k-pair mutation sensitivity. Those CPU checks support Run432's revised oracle, not the graph oracle. Current JSON contains the limitation elsewhere, so this is visibility/edge completeness, not a factual error.

## W_minus / C_plus sufficiency

The inherited formula `T_star >= max_i(W_i_minus/C_i_plus)` is sound under its stated matching-work assumptions. One positive necessary subset is enough; a complete model work census is not required. Avoid adding resource times unless seriality/nonoverlap is proved. Hardware OEM identity alone is correctly not promoted to C_plus; measured practical throughput is correctly separate from a true upper rate cap.

Before any numeric promotion, make these compatibility checks explicit fields/edges:

1. Same admissible algorithm class, output/accuracy contract, timed window, and successful output count N; no cross-run handoff/publication or acceptance transfer.
2. W_minus excludes already-completed/reusable pre-window work and remains unavoidable under the allowed placement/recompute/reuse alternatives.
3. Matching units, precision/sparsity convention, aggregation across devices/resources, and complete set of engines allowed to perform that work. C_plus must bound aggregate rate even under boosts; an attainable mixed rate is not an upper-cap certificate.
4. Positive finite W_minus and C_plus, conservative uncertainty treatment, and conversion `TPS <= N / lower_latency` only for that fixed output-count/time definition.

Several of these exist in prose across the inherited model, but the two-node requires list does not enforce them. The current no-finite-ceiling conclusion remains correct.

## Product engineering gate

`complete legal execution variant + practical mixed capacity + repeated frozen formal E2E` is a reasonable evidence plan. A statistical/achievable interval additionally needs endpoint semantics, repeated-run variability policy and unchanged correctness/output contract; a best sample is not an upper achievable endpoint guarantee. Conversely, measured repeatable E2E establishes a witnessed performance point even without a complete resource model. V3.14 correctly keeps Run99 median571.681 and best single612.962 separate and adds no hypothetical gap.

## Recommended disposition

Accept V3.14 as the current conservative ledger. Correct the inherited decision_rule, declare the graph non-executable, and add typed scheduling endpoint/duration requirements before using it to authorize numerical promotion. All43 metadata remains a reasonable targeted diagnostic for the retained-route line of inquiry, with native opaque boundaries as explicit stops; sampled MoE success alone does not warrant advancing W_minus or Product status.

## Checks and identity

Stdlib inspection exit0 confirmed algorithm_resource/hardware_resource/scheduling_execution latency_floor_s=null and product finite_tps_upper_bound=null. No regeneration or source edits performed.

- Generator SHA256: `4e2e524431cf4d4ca9e327b7e3e24437715fe8bbb44a18eb1ac953cb5872fbe0`.
- JSON SHA256: `1a6af9fb0891be6d59b4b7208b28fc6e37fd5d16176889095485facfdcb0a726`.
