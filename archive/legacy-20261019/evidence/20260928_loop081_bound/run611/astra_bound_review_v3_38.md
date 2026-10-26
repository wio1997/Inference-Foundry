# Run611 / V3.38 independent Astra High Bound review

Verdict: **SCOPED PASS for V3.38's instrumented-Current update; no strict finite Resource, Scheduling or Product endpoint is newly justified.** Confidence: high. Formal Current remains Run99 **571.681 tok/s**; no numerical Current-to-credible-limit distance is certified. Fixed DSpark7 behavior, acceptance trajectory, cycle occurrences, outputs and model work remain the objective's constraints.

This review is read-only except this report. No service, NPU workload, raw reparse or historical experiment was launched. I inspected Run607–611 design/model/recovery evidence and installed-repository rules, independently rehashed the eight parsed task_time CSVs and eight trace_view files, and executed V3.38 with its sole output write replaced by an assertion against the existing JSON. All seven input pins, status/identity gates, derived scope ranges and JSON equality passed.

## What V3.38 proves

The preserved recovery admission and parsed manifest establish a usable same-run evidence source after Run610's INVALID controller outcome. Run610 must remain INVALID. The recovery review's independently checked producer markers and hashes support byte preservation, not universal native task coverage or semantic correctness of a reconstructed DAG.

V3.38 records 24 rank-local Host scope rows: three occurrences for each of eight ranks. The derived Target Host scope range is 2,469.10–25,934.22 us and proposer Host scope range 50,775.95–59,494.98 us. These include submission, possible waiting and instrumentation. They are not device service durations, necessary work times, available overlap or removable Product time. Summing Target and proposer scope durations would not produce a Resource or Scheduling floor.

The corrected CLOCK_MONOTONIC_RAW versus Python CLOCK_MONOTONIC distinction is soundly bounded by the recovery review. Present-time API-domain witnesses do not retroactively calibrate historical clocks. The assumed 1 ms wall conversion margin remains a conditional containment check. Neither the client endpoint offset difference nor same time namespace supplies a strict error bound.

The model properly preserves null strict endpoints and keeps 555.1125 diagnostic tok/s distinct from formal Current. It adds evidence scope, not a new performance formula. The pinned V3.37 base makes the inherited null claims auditable; current V3.38 hardcoded-null checks pass, but are not a substitute for validating any future dynamically computed endpoint.

## Formula and certificate review

For each resource q, a strict interval floor needs both required_q(W0) >= W_q_minus and service_q(I) <= C_q_plus * |I| + B_q, with matching counted units, initial availability and resource boundary. Then T >= max_q max(0, W_q_minus - B_q) / C_q_plus is valid. The denominator is an upper service envelope, not an attained benchmark rate. A positive partial numerator can yield a valid weak floor without completing every model term; absent terms alone do not prohibit all partial Bounds.

V3.38's text “attainable cumulative ... C-plus/B” should be read or revised as **certified upper cumulative capacity envelope**. Attained compute/HBM/HCCL is instead appropriate for explicitly conditional Engineering schedules. Run611 supplies neither missing certificate: there are no HBM counters, compulsory physical bytes, exact-board maximum capacity, or new proven-fresh logical-work terms.

Scheduling must optimize over legal schedules with fixed work and outputs, storage generations/lifetimes, resource sharing and serving dependencies. Existing stream ordering and layer/cycle boundaries cannot all be declared mandatory. Conversely, merely deleting current waits from a trace cannot prove a legal alternative. A critical path weighted by observed profiler durations is a Current or conditional scheduling model, not a hardware floor.

The strict/global track must not stall an actionable Engineering track. Build conditional schedules now from admitted node service intervals and clearly identified unknown edges, then sweep those uncertainties. They can guide the next experiment before a universal hardware certificate exists. Do not assign a false precise TPS endpoint to an unclosed same-W0 Product DAG.

## New independent native-correlation observations

On **every rank**, the exported native Model45 records contain **16,236 events**, exactly **5,412 distinct static keys**, each appearing **three times**. My key was (physical stream ID, task ID, task type, event name). This is strong internal evidence for three repeated static Graph task inventories. It does not yet bind their occurrences to cycles64/65/66, prove required event ordering, or establish the generation identity of every tensor.

The exported connection_id is **not a replay identifier**. On rank0 a single Model45 connection_id covers12,339 events across replays; other connection IDs also recur. Grouping only by connection_id would merge occurrences and generate false durations/dependencies. Use the static-task occurrence sequence and actual Graph-launch/flow semantics, with independent boundary checks.

I also found rank7 has5,400 async_npu flow events versus5,410 on ranks0–6, and14,069 HostToDevice events versus14,074 on ranks0–6. No data-loss conclusion follows: boundary clipping, rank-specific calls or exporter correlation may explain it. The exact Model45 inventory is nevertheless equal. The difference must be reconciled for any eager-task completeness claim. Parser success and per-file .done sizes do not settle that question.

## Highest-information next step

**First finish an offline all8 native occurrence/ownership reducer using these existing copies.** This is cheaper and more discriminating than another live profile with the same scope. For each repeated static Model45 key, retain three ordered occurrences and validate the Graph boundary/order using actual launch/flow records, not nearest CPU scope or connection_id alone. Cross-check task_time CSV against native events. Explicitly account for cycle63 carry-in, cycle66 stop-fence effects, the rank7 eager-flow discrepancy, unmatched flows and exporter duplicates. Reject unique cycle/owner assignment when evidence is ambiguous; retain an unassigned interval.

Join admitted native identities to this run's Basis/Product/dispatch. Produce a conditional Current DAG cut spanning Target aux/logits, acceptance/state, Draft context/query/commit and next Target, with each edge marked source-only, actual stream/event, or typed producer/consumer. Keep Host submission and device-ready completion separate. Report coverage and unknowns before any latency sum.

**Then perform only the missing exact live experiment:** a same-state/private-buffer two-cycle replay or guarded sparse acquisition that records per-layer Draft context KV writer, storage generation and first query consumer, plus the acceptance/query preparation branch and next-Target commit dependency. Capture actual existing joins without adding fences. Pair observer OFF/ON/OFF under controlled touched state and ledger if assigning transferable service time; otherwise admit topology alone. Preserve the actual acceptance/output trajectory. This discriminates whether the source-level aux/context versus logits/acceptance fork is legal and whether shared resources erase its scheduling value.

Historical R11 Host-versus-device decomposition and R21 resource-contention failure remain priors; Run579/580 already reject ideal whole-chain overlap in their tested fixture. The knowledge search “DSpark overlap Host” was rerun against pinned history commit db3beef223e0b5acd81ccccb444600c1b22aac8a. No historical absolute timing or KEEP/REVERT is transferred. The next proposal is evidence completion, not a new ungrounded overlap implementation.

Resource work should proceed alongside this reducer: classify fresh semantic keys/entry-state credit in the existing W0 ledger under a declared evaluation class; maintain exact-board capacity applicability/freshness uncertainties. Further generic peak samples cannot by themselves close C-plus. The next scheduling result should reduce one named uncertainty and update conditional sensitivity; a failed local fork does not imply proximity to the overall limit.

## Reviewed identities

- scripts/extreme_bound_calibration_v3_38.py: 39b69215059535f5213970d1572392d80480f446b8477c15d4bd8faedae52774
- bound_calibration_v3_38.json: 8206db47c4a36ee9c61090d8c3415f1cb48a1397d7f0bdbe2ef2eebaedcb546b
- scripts/loop081_profile_scope_ledger_run611.py: 4a3c82370f62559dc4cff99339d269ca87e9546101e1654a5d143000cf28ea68
- scope_ledger.json: 3c9bb23a28674fae048daad9c32a2803324d5509aa418c41d66c129fe27fd2ff
- raw_recovery_admission.json: 9903001902801471e005601bf331028a4d02a8b04cb8689f11a5d81c645bd802
- offline_parse/manifest.json: b51eac01dcdc9f183544d51d20040f371f5ffb23a686ded7372b2cb366c84e83
- astra_recovery_review.md: d9ef7aae0db0cd5eaa4585755ad5b20472dfbf32efe09f141807a26d4455356b
- astra_host_clock_witness.json: fabf5ca8968316529864dc28cff7eef9e7c39851db3c675bd06083772a14b42d
- Run607 astra_bound_review.md: 65b1215d5b9ebf394cc02881521daafe094b76ae0bb95eb44c836c33a461c53e

The parsed-file SHA identities and all8 path mapping are in the reviewed offline manifest. This review leaves every strict numerical endpoint and numerical gap null; it narrows the next correlation method and prevents false replay grouping.
