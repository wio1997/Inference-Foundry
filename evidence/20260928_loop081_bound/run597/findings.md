# Run597 — new fixed-algorithm diagnostic W₀ lineage

Run597 used the frozen 48 warmup + 48 measured, c12, 1024-output, DP1×TP8 DeepSeek V4 W4A8 + DSpark7 contract. It is one **new, instrumented diagnostic trajectory**, not the formal Run99 state or a formal TPS run. The guarded five-source observer patch recorded every measured cycle's first state, Target accepted/masked counts, next Draft tokens, explicit Host park events and scheduling branch, with sparse device snapshots of actual Target, DSpark handoff and Draft model input tensors. The sparse checkpoints include cycle0/1, fixed later cycles and the first postpark cycle in every measured cohort.

The 96 client responses, 64 all-rank Runtime files, 32 measured basis files, request/cohort joins, rank consensus, full basis replay and sparse actual-input assertions passed. The collector's 13-mutant CPU preflight passed; the live patch check covered all five sources. The wrapper reported `run_exit=0`, tagged stop0, all8 idle0, source restoration0, source/script hash comparisons0. The first transient NPU release probe failed and a subsequent bounded probe passed; the saved stop log preserves both. Diagnostic observer work and output serialization perturb timing, so its elapsed time is not used as a performance result.

| Cohort | Cycles | Current physical Target-8 rows | Active-slot Target-8 rows | First postpark checkpoint |
|---:|---:|---:|---:|---:|
| 5 | 296 | 28,416 | 23,808 | 172 |
| 6 | 282 | 27,072 | 24,768 | 181 |
| 7 | 317 | 30,432 | 24,888 | 155 |
| 8 | 305 | 29,280 | 26,512 | 221 |
| **Total** | **1,200** | **115,200** | **99,976** | — |

All sampled Draft model checkpoints report source-static `num_query_per_req=7`, `sample_from_anchor=True`, 96 context rows and actual query IDs/positions consistent with the compact basis. The all-cycle count in Run598 is conditional on this fixed source configuration. The 48 Host park events and all-cycle Target branch ledger reconstruct the observed state transitions; all measured Target cycles took the existing Graph branch. This narrows current Runtime workload cardinality and dependency identity for this new W₀. It does not identify required fresh semantic evaluations, initial prefill/seed/KV work, compulsory HBM or HCCL traffic, exact-board cumulative capacity or a legal all8 Scheduling lower bound.

Run99 Current Formal remains **571.681 tok/s**. Resource/Hardware, Scheduling/Execution, Product E2E finite endpoints and numerical Current→credible-limit distance stay **null**. Run597's diagnostic W₀ is not a matched intervention on Run99, and the formal acceptance trajectory is unchanged by this study.
