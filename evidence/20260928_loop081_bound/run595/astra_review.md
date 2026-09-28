# Run595 independent Astra review

## Verdict

**PASS as a conditional integer outer envelope for active rows in the declared fixed Target-8/current serving class.** No constraint error was found. The model retains Host-lag trajectories because it does not force parking on the first crossing of 1024. It is not an exact characterization of feasible serving trajectories, necessary fresh work, or a Resource/Scheduling/Product time bound. Null numeric endpoints and 571.681 Current Formal are appropriate.

Review used only CPU Python/SciPy, reading existing files. No service, NPU workload, or hardware query was run. Original script and summary were not changed.

- Reviewed script SHA256: `d9fb880e1b7592c1429c6d2fc2e85c24949313c0b8bfd8f8389384ca8764ea25`.
- Reviewed summary SHA256: `3af0ceaf4a56fed3b222ec968d1bc7f00a40a2ba18ae05bf6f19be37a7c99df9`.
- Independent executable audit: `astra_cpu_audit.py`; result: `astra_cpu_audit.json`; exit 0; SciPy 1.13.1 in the existing container. No torch import is required.

## Input and integer recovery

All 128 all8 source hashes in the revised summary match. Independently compared rank0..7 for all 12 Run99 and four Run566 A0 cohorts: window lengths/totals, per-slot staged counts and request IDs agree, with valid rank/cohort identities and runtime pass flags. The revised reducer itself requires all8 equality of request IDs, cycles, generated counts, staged counts and acceptance-window means, and pins all 128 file hashes. The original rank0-only artifact is retained as `summary_rank0_provenance_only.json`. All numeric rows/endpoints remain unchanged after the provenance revision; the independent CPU audit was rerun successfully against the revised artifacts.

Window names partition [0, cycles) contiguously. Multiplying each saved mean by 12 times its window length recovers the integer count sum with maximum residual **4.547473508864641e-13**, far below 1e-6. Window sums equal per-slot staged totals. `fixed_serving.py` currently computes these means from the staged masked `state.num_sampled` history, not raw acceptance for parked slots. The historical-source equivalence remains a declared condition until the Run99 collector/source identity is explicitly joined; arithmetic agreement alone is not a source-version attestation.

The 1024..1039 staged admission range fits these existing inputs and a one-cycle Host-mirror lag with up-to-eight counts. It should not be exported as a universal DSpark law or silently applied to a collector with different lag. Here staged totals are observed fixed values, so this admission check does not truncate the per-input MILP feasible set.

## Constraint proof and Host lag

For slot s and window w, A counts active cycles, Y counts staged samples, and binary Z means active at window start.

- Z <= A <= L*Z is exactly A=0 iff Z=0, otherwise 1..L.
- A <= Y <= 8*A is the exact integer projection of A positive counts in [1,8]: every integer Y in that interval can be distributed among A counts.
- Z[w+1] <= Z[w] prevents reactivation.
- A[w] >= L[w]*Z[w+1] makes every window before the terminal active window full. The terminal window can contain an active prefix of any length A, followed by zeros.
- Z[0]=1 represents initially active slots; per-slot Y totals and per-window aggregate Y totals are exact saved equalities.

Thus every positive-prefix count trajectory with the declared aggregates maps into the model, including a slot that continues executing after its output has reached 1024 due to Host mirror lag. Conversely, an integer solution can be expanded into a positive-prefix count trajectory; there is no missing intra-window integrality obstruction. This establishes exactness **for this relaxed count class**, not the actual scheduler.

The model omits the threshold-crossing/Host-mirror relation, exact park-call order, semantic token/state dependencies, and the requirement that a cohort ends at the actual completion boundary. These omissions admit extra trajectories; they can loosen minimum downward or maximum upward, but do not make the outer envelope unsafe. No observed Run566 park timetable is imposed on Run99.

A future tightening may require at least one slot active through the final recorded cycle, and encode source-proven mirror-lag/crossing constraints. First pin the historical serving semantics; do not import an assumed immediate-stop rule or a new run’s parking events. Such tightened trajectories would still not prove necessary fresh work.

## Solver evidence and limits of optimality wording

All **24 Run99 solves** (12 cohorts, min and max) and **eight A0 solves** report HiGHS Optimal with zero relative MIP gap. `solve` rejects nonzero SciPy status or absent solutions. Both directions agree with the rounded active-cycle objective. There is no timeout-as-optimum admission in the implementation.

This is solver-certified numerical MILP optimality for the stated integer model, not an independently checked exact-arithmetic proof. The original evidence does not save primal vectors, dual bounds, constraint residuals, HiGHS build identity or a proof log. I verified the recorded statuses and independently tested the formulation; I did not rerun all 32 production optimizations. For stronger auditability, future solver runs should retain A/Y/Z assignments, rounded integer feasibility/residual checks, objective/dual bound and solver version. Do not call the current summary a formal mathematical proof certificate.

## Independent CPU tests

1. Enumerated **all 584** positive-prefix sequences with counts 1..8 over three cycles. Pairing two free slots yields **341,056** trajectories and **5,952** distinct aggregate keys; ten other slots are fixed to one count in cycle zero. This embeds the test directly in the unmodified 12-slot production solver, with windows [1,2].
2. For **48 deterministic randomly selected aggregate keys**, both MILP endpoints equal the minimum and maximum of complete enumeration. This is exhaustive trajectory enumeration followed by 48 solver comparisons, not a claim that all 5,952 groups were solved.
3. Built actual A/Y assignments from Run566 raw traces plus the already disclosed Run594 park-event extraction. For each of the four cohorts, every window total and per-slot staged sum matches; all prefix and A<=Y<=8A constraints hold. Recovered active rows are 24,368 / 23,824 / 26,064 / 24,240, total **98,496**. This is stronger than merely checking that the total lies between two endpoints. Park events remain derived from the old trace, not independently measured Host timing.
4. Negative cases reject noninteger recovered window sums, inconsistent staged totals, wrong cycle coverage, and a zero-output initial window despite all slots initially active. Each mutation has a valid baseline or explicitly infeasible input; there is no truncated-fixture false pass as in the superseded Run594 harness.

## Interpretation of the tighter historical envelope

| Run99 repeat | Cycles | Prior lower | Prefix-aware lower | Upper |
|---|---:|---:|---:|---:|
| 1 | 1,217 | 49,672 | **79,968** | 116,608 |
| 2 | 1,212 | 49,672 | **78,360** | 114,672 |
| 3 | 1,206 | 49,672 | **80,856** | 115,144 |

The earlier 49,672 bound was a weaker aggregate/per-slot necessary inequality. It did not enforce that a slot supplying output in a later window must have been active throughout earlier windows. Adding that prefix condition forces additional active cycles in low-acceptance earlier windows, raising the lower endpoint by 30,296 / 28,688 / 31,184 rows. The earlier bound is not disproven; it is superseded by a stronger conditional relaxation. Upper endpoints happen to remain unchanged.

These are active **Target-8 row occurrences**, not unique semantic evaluations and not compulsory FLOPs or bytes. They do not imply proportional runtime, HBM, HCCL or E2E savings. They do not incorporate Draft context/query/Markov work, prefill/seed/KV dependencies, exact-board capacity or serving critical path. New diagnostic W0 trajectories must remain separate from Run99. Resource, Scheduling and Product endpoints must not be promoted from this result.
