# Run570 — same-arm A0 active versus parked Target-input rows

Run569's source/admission gate pins all32 Run566 A0 measured trace files to their original all8 Runtime reports and request IDs. Run570 uses that same diagnostic trajectory and the frozen `fixed_serving.py`, `extreme_decode.py` and `fixed_decode.py` source hashes to locate each slot's first parking transition. For every slot, active `num_computed_before` advances by the previous raw acceptance count (1–8); a single rewind lands at initial position + 960, then remains fixed. The cumulative active count reaches at least1024 and equals that slot's Runtime `staged_output_counts`. The last active slot may terminate the cohort without a following parked cycle. The complete slot transition/count pattern matches across all eight ranks in each cohort.

| Measured A0 cohort | Cycles | Current 96-row Target inputs/rank | Active rows in current Target-8 class/rank | Parked input rows/rank |
|---:|---:|---:|---:|---:|
| 5 | 318 | 30,528 | 24,368 | 6,160 |
| 6 | 283 | 27,168 | 23,824 | 3,344 |
| 7 | 299 | 28,704 | 26,064 | 2,640 |
| 8 | 306 | 29,376 | 24,240 | 5,136 |
| **Total** | **1,206** | **115,776** | **98,496** | **17,280 (14.92537%)** |

This identifies an **observed current physical-row versus active-row distinction** under A0's own acceptance trajectory. `FixedCohortServing._park_completed` explicitly keeps fixed-shape parked slots executing while freezing logical progress. The 17,280 rows are neither a mathematical compulsory-work subtraction nor a proven removable wall-time opportunity: they are Target input geometry, not a per-kernel FLOP/HBM census; fixed Graph, collective and resource sharing may prevent saving proportional time. A0 differs from other diagnostic arms and from the historical Run99 formal W₀. No acceptance/cycle reduction or DSpark algorithm change is proposed.

The generator self-test exited0, rejecting zero/wrong active position increments, wrong park anchor, resumed or repeated rewinds and staged-count mismatch. Its actual-data gate passed all384 rank/cohort/slot records and exact per-cohort arithmetic. `slot_rows.csv` and `summary.json` carry the admitted hashes, per-slot parking cycle and interpretation. This closes one useful Resource census variable for the **A0 diagnostic fixed Target-8 execution class**; full model necessary arithmetic/traffic, formal W₀, exact-board C⁺/B, mixed all8 Scheduling and numeric Product Bound remain unknown. Formal Current remains Run99 median **571.681 tok/s**.

Next use this all8 active-row ledger to avoid counting parked input rows as required work in a same-trajectory conditional model, then measure whether any architecture that removes those rows remains shape/layout compatible and actually shortens the critical path. Do not prioritize a parked-row optimization before formal W₀ and capacity/mixed-service uncertainties are compared.
