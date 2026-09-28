# Run607 — independent Astra High fixed-algorithm Bound review

Scope: read-only review through Run606. No service, NPU load, source mutation or algorithm experiment. Fixed DSpark7 acceptance, cycle occurrences, output ledger and model semantics are held constant. Formal Current remains **571.681 tok/s**. Review inputs and SHA values are in astra_bound_review_inputs.json.

## Decision

There is still **no certified positive Resource/Hardware time floor, dependency-aware Scheduling time floor, or finite Product TPS ceiling** for the unrestricted architecture search under the frozen product contract. No trustworthy numerical distance from571.681 to that ceiling can yet be stated. Confidence in this negative certification judgment: **high**.

This does not mean that Engineering calibration must remain empty. Existing feasible service observations can populate explicitly conditional node/service envelopes; they cannot supply missing no-faster-than certificates. Separate (a) observed Current, (b) conditional Engineering feasible schedules and sensitivity bands, and (c) strict physical/algorithmic relaxations. Do not wait for a perfect universal ceiling before constructing (b), and do not relabel (b) as (c).

A finite observed execution time is a feasibility witness for its own admitted trajectory and observer configuration. It is not the lower time endpoint sought here, a deterministic repeated formal guarantee, or a universal TPS ceiling. The formal median and Run606 diagnostic wall must retain their separate meanings.

## Certificate matrix

| Quantity | Admitted evidence | Missing decisive certificate | Consequence / confidence |
|---|---|---|---|
| Fixed logical workload | Run597/606 all-cycle compact basis and sparse actual-input witnesses; Run598 role counts; Run595 formal aggregate outer relaxation | Required fresh semantic evaluations after exact reuse/entry-state credit; parked/rejected/terminal consumer requirements; complete ordinary prefill/seed/state provenance | High occurrence/issued-count confidence; low compulsory-count confidence. Active rows are not W-minus. |
| Conventional BF16 / W4A8 work | Run569 wo_a and MoE formulas; Run576 real operand/route geometry; Run598 conditional counts | Declared evaluation class, necessary fresh keys, dtype-specific mapping and ownership; Draft/attention/KV/seed uncovered terms | Conditional arithmetic is usable now. It cannot become an unrestricted mathematical minimum just because the source evaluates it. |
| Compulsory HBM | Run247–249 current main-memory counters; Run578 current GMM traffic; checkpoint footprints | Memory hierarchy and initial residency, live-value/representation/recomputation class, reuse across calls, write/read consumer graph, counter-to-physical-payload units | Current18.965GB read /2.380GB write is not W-minus. Neither is GMM9.244GB or1.092× packed estimate. |
| Exact-board compute C+ / burst B | Installed910B3/bin/cores/configuration; Run571/577 attained service | Authoritative maximum supported clock/issue envelope, precision/engine coverage, boundary convention/B, installed applicability | No strict denominator. Additional throughput samples cannot prove a universal maximum. |
| Exact-board HBM C+ / B | Core counters, sampled/configured clocks, memory capacity and isolated attainments | Active channels/bus/transfers/maximum clock including tolerance; read/write sharing, encoding/ECC and counted-unit relationship; finite boundary allowance | No strict physical bandwidth cap paired with compulsory bytes. |
| Communication work and C+ / B | Current265-collective chain; Run253/254 official HCCL Test; Run590 all8 link inventory | Necessary information across actual cuts under allowed repartition/schedule; physical peer/direction/alternate-route graph and maximum lane rate; fresh timestamped statistics | Library bytes/current collectives and “standard224Gb/s” are not the needed pair of certificates. |
| Mixed service | Run579/580 serial~14.4–14.6ms versus concurrent~16.2ms for independent-ready real-weight GMM + current HCCL chain | Workload/state/residency-matched service region for legal production branches, with allocation and stream/engine contention | High confidence that this particular whole-chain overlap loses. No general overlap impossibility. |
| Scheduling lower relaxation | Run589 native identity; Run591 typed source edge; Run592 historical local timing; Run599 source split; Run606 ordinary branch/marker identity | All8 same-W0 actual last writers, consumers, aliases/generations, joins and resource contention across a complete Runtime cycle; compulsory node-duration floors | Current topology improved; shortest legal execution remains unbounded numerically by this evidence. |
| Product completion | Run602/606 same-W0 Scheduler/API/client joins, ordinary preparation and output accounting | Low-perturbation per-W0 preparation/Runtime/publication DAG and legal c12 arrivals; causal service transfer to formal repeats | Counts/Host envelopes high confidence; removable/required duration low. Cannot mix Run99/602/606. |

For a resource class q, the required proof has the form **service_q(I) <= C_q+ × duration(I) + B_q**, and workload proof **required_q(W0) >= W_q-minus**. Only then does **T >= max_q max(0, W_q-minus - B_q)/C_q+** follow for that declared interval/class. Shared resources and a legal dependency graph can tighten this relaxation; adding independent floors double-counts overlap unless justified. Attained rates establish feasibility in the opposite direction and must not substitute for C+.

For Scheduling, optimize over legal schedules of the same required work with initial-state, lifetime, resource, stream and serving constraints. Current layer/operator/collective/cycle boundaries are observations, not mandatory edges. A critical-path calculation using current kernel durations is a conditional schedule model, not a hardware floor. Keep conventional BF16 and W4A8 GEMM-equivalent resource units separate.

## What has changed since the earlier projections

Run577–580 materially improve isolated and mixed Engineering priors, but still do not close exact-board C+/B. Run590 closes supported static inventory and installed-package identity, not dynamic device implementation, maximum serializer rate or statistic freshness. The archived500ms cache timer is not a bounded sampling delay; another short pre/post npu-smi delta remains inadmissible.

Run593–598 close broad occurrence/role cardinality and reduce historical Run99 row uncertainty without inventing same-state identity. Run595's79,968–116,608 /78,360–114,672 /80,856–115,144 rows are conditional active Target-8 geometry, not fresh model work. Run599's49,523 prefix-decision incidences exclude consumers needed for raw parked acceptance/context; they do not justify pruning verification work. No512-cycle/perfect-hit or acceptance improvement enters this review.

Run606 closes actual ordinary Graph/DSpark ownership:40 pairs/rank,24 NONE and16 FULL replays,5675 context/1813 query rows. Its context marker age does not measure slack. Ordinary Draft query consumes context before handoff; Target does not consume Draft context, and Runtime context updates can replace old generations. Non-null slot lists do not certify every layer's scatter or later read. Do not turn56–65ms or103–116ms into a necessary edge, overlap budget or savings.

The old581–607/616–682tok/s projections therefore remain historical scenarios, not newly verified intervals. A future conditional range must explicitly sweep workload, service, readiness and contention uncertainties, and expose which assumptions move the endpoint most.

## Most plausible architectural omissions

1. **Runtime Host/device dispatch coupling and whole-cycle structure.** Target is graphed while Draft has substantial eager orchestration, metadata, state and serial head work. A layer-local kernel census can miss submission serialization and opportunities to retain state/device-side scheduling. Current evidence does not quantify a removable total.
2. **Finer legal branch structure.** Target auxiliary output may feed Draft context while logits/acceptance feed query preparation; both join before the real query. Run599 identifies the source split, but aliases, actual readiness and resource interference determine whether the split has execution value. It must not be assumed beneficial.
3. **Cross-call lifetimes / materialization / information placement.** Existing layouts and TP boundaries can force repeated stores, reads and collectives. A storage-generation/consumer graph, including cached prefix and rejected/parked state, may expose reusable or non-required materializations while retaining every frozen algorithmic evaluation. We have not measured the size of this opportunity.
4. **Request preparation and Runtime can share resources.** Preparation is not just an external Host remainder. Legal pipeline choices require true c12 admission and fixed-work ownership, not an artificial smaller cycle count.

These are ranked uncertainties, not claims that any implementation will win. The16 ordinary replay sync Host calls (roughly3ms over this pass) are a low-priority standalone target on present evidence.

## Highest-information next minimal experiment

**Recommendation: one new guarded same-W0 acquisition of a complete all8 Runtime two-cycle producer/consumer packet**, rather than another preparation marker, narrow layer0 RS trace, peak test or full-chain overlap benchmark.

Predetermine cohort5 cycles64/65 (normal FULL96, both admitted active; reject this selected stratum if the actual state differs rather than cherry-picking a faster pair). Retain the existing whole-pass basis/Product joins. Two adjacent cycles are the minimum useful unit here because next-draft/state/KV consumers cross the cycle boundary. The aim is identification and conditional service calibration, not changing acceptance or reducing cycles.

Packet:

- Bind run/request/slot/cycle, actual source/Graph generation and native task identity. Reuse Run588/589 supported native labeling, all8 source/schema gates and existing profiling methodology. No Python logging/import in compiled decoder.
- Observe full Target replay, actual auxiliary-output last writer/readiness relation, logits, greedy acceptance/state advance, Draft context combine and per-layer KV projection/scatter, query input readiness, actual first consuming query, serial head/next-draft commit and the next Target input. Cover existing collectives/stream joins throughout this cycle, not only one convenient kernel.
- For context/KV, record cache presence, each layer's slot-map valid count/range and storage generation plus producer-to-consumer lineage. Method return, raw pointer equality or non-null outer list is insufficient. Use canonical buffer identity with lifetime/version and actual task labels; selected device metadata snapshots must have explicit observer costs.
- Record source/Host submission separately from native execution/completion. A current-stream event only serves as readiness after all producers' joins are proved. If an output lives on another stream, retain the existing wait/event edge; do not add synchronization to manufacture an apparently clean DAG.
- Reuse exact all-cycle acceptance/output and stage ledger, no model recomputation or new collective on the serving hot path. Save raw traces before parser invocation (Run578 lesson), keep exact source restoration/owned stop and phase/client gates.

Falsifiable result: either the presumed aux/context versus acceptance/query split has independent actual producers and a real common consumer, or a previously hidden alias/writer/stream dependency serializes it. The same packet distinguishes eager Host submission gaps from busy device time and locates mixed-resource overlap. This is new evidence relative to Run246/592 historical traces and Run606 entry markers because it binds **state/role/native identity and adjacent consumers in the same W0**.

Admit structural findings first. For service durations to enter even a conditional Engineering endpoint, require an A0/probe/A1 perturbation gate with controlled entry state or an explicitly weaker uncertainty bracket. Adjacent natural cycles are not same-state A/B. If checkpoint/private-buffer replay cannot reproduce touched state and outputs, do not claim isolated branch costs or causal probe overhead. A sparse packet cannot be multiplied by1214 cycles or43 layers; expand only the state/shape stratum whose uncertainty actually drives the model.

Failure gates: missing producer/consumer generation, truncated native tasks, forced collective equivalence without message identity, different branch/shape, unexplained output/count drift, new synchronization, mutable snapshot aliases, parser-only data loss or source/cleanup mismatch. A timing failure can still leave scoped topology evidence but supplies no bound endpoint.

Expected model effect: close one major **Scheduling identifiability** gap and assign same-W0 Current costs plus conditional service priors to the complete cycle. It does not certify global optimum or C+/B. If the split is already serialized by a necessary producer or contention dominates, pivot to whole-cycle dispatch/lifetime architecture; do not interpret that local failure as a ceiling.

## Parallel strict Resource work — proof, not another benchmark

The nearest useful strict capacity step is the existing exact-board certificate request/package proof: maximum clock/issue (per precision/engine), physical HBM interface, and HCCS directed-edge envelope with a defensible finite B. Preserve the source-to-booted-image and API freshness gaps identified in Run590. Repeat the previously inaccessible OEM fetch or device query only if a new authoritative source/permission exists.

For the numerator, the next offline work should classify the existing full ledger by semantic input/prefix/context key and declared initial availability, keeping unknown reuse as uncertainty; then choose one broad declared online dense class. This can strengthen a conditional W-minus once required freshness/consumer facts are proved. Do not relabel all active rows or all observed context stores to force a positive number. HBM demands a separate hierarchy/residency/recomputation proof. A partial strict numerator plus an honest capacity certificate would be valid but may be numerically weak; missing terms need not block that partial floor.

This parallel obligation must remain visible while the live experiment reduces Scheduling uncertainty. Neither direction is a reason to resume arbitrary kernel scanning.

## Historical priors checked on demand

Searches: “overlap context DSpark” and “R09 R21” via the project's knowledge search; source commit db3beef223e0b5acd81ccccb444600c1b22aac8a.

- R09: DP2×TP4/EP8 eager DSpark; fixed counts rejected a narrow duplicate-call hypothesis. Host span could not distinguish submission versus completion coupling. “No duplicate calls” did not prove a performance ceiling.
- R11: same older family, short profiled workload; exposed large wrapper/Host cost around modest device work, with profiler inflation. Useful prior for the Runtime whole-cycle packet, not a numerical transfer to current DP1×TP8 or an instruction to reproduce the old micro-kernel scan.
- R21: actual prefill Compressor plus shape-faithful W4A8 overlap was bit-exact but slower; resource contention defeated apparent available time. Current Run579/580 corroborate the risk in another fixture. Different shape, precision placement, workload ownership or narrower independent branches are reasons to re-test; “two streams” or marker age alone are not.

No historical KEEP/REVERT is applied to current Extreme. No new reusable global conclusion is promoted without Sol review.
