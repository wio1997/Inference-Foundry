# Run579: independent-ready GMM × full TP8 HCCL mixed service

## Scope and admission

The frozen DeepSeek V4 Flash W4A8, 8×910B3, DP1×TP8, DSpark7 service completed warm48 and measured48 at c12/1024. After the final cohort, a terminal diagnostic used loaded production W4A8 Target weights, the SHA-pinned Run576 cohort5/cycle170 route groups, private nonzero activation, and the Run378 current-order 265-collective ledger (AG135/RS87/A2A43). Each collective had distinct input and output buffers. The diagnostic does **not** change or measure DSpark7 acceptance, cycle count, or formal E2E TPS.

Eight ranks passed three fresh HCCL generations with all 265 outputs checked, plus a fourth common-start concurrent generation with both branch outputs poisoned before replay, GMM valid rows checked against a preexisting independent reference, and all 265 collective outputs checked after the two branch events joined. CPU admission passed eight negative cases. The post-stop final gate checked 96 POSTs, 64 all-rank Runtime records, all8 idle, exact source restoration and script hashes. `final_admission.json` is the machine-readable admission and provenance record.

## Same-run device-event observation

Four arms were measured in balanced forward/reverse order, 20 samples per arm per rank. Times are **rank-local joint completion medians** in ms. They are not a synchronized cross-rank makespan.

| Rank | GMM only | HCCL only | Serial GMM→HCCL | Concurrent | Concurrent − serial |
|---:|---:|---:|---:|---:|---:|
| 0 | 10.047 | 4.167 | 14.402 | 16.202 | 1.801 |
| 1 | 10.106 | 4.136 | 14.377 | 16.142 | 1.766 |
| 2 | 9.909 | 4.180 | 14.396 | 16.192 | 1.797 |
| 3 | 10.032 | 4.151 | 14.401 | 16.205 | 1.804 |
| 4 | 9.985 | 4.064 | 14.334 | 16.129 | 1.795 |
| 5 | 10.266 | 4.052 | 14.312 | 16.076 | 1.764 |
| 6 | 9.946 | 4.097 | 14.342 | 16.118 | 1.776 |
| 7 | 9.807 | 4.074 | 14.335 | 16.076 | 1.741 |

The slowest rank median is 10.266 ms GMM only, 4.180 ms HCCL only, 14.402 ms serial and 16.205 ms concurrent. On every rank, the minimum concurrent sample exceeds the maximum serial sample by at least 1.11 ms. The concurrent HCCL branch completion median is about 16.1–16.2 ms and GMM branch completion about 12.8–13.3 ms. This observation is consistent with substantial concurrent resource/stream interference in this specific dual Graph protocol; it does not isolate the exact mechanism or establish that no finer overlap schedule can win.

## Bound effect

Run579 falsifies the **specific** optimistic assumption that independently ready, whole-chain GMM and current-order 265 HCCL Graphs can simply be launched concurrently and attain `max(GMM-only, HCCL-only)` service. It improves the Engineering mixed-service calibration by giving same-run serial and concurrent measurements with full correctness. It does not certify compulsory traffic, exact-board attainable C⁺/B, production producer/consumer legality, Target/DSpark/KV/Host overlap, a finite Resource/Hardware or Scheduling/Execution endpoint, or a numeric Current→limit distance. Formal Current remains Run99 median 571.681 tok/s.

The next high-value diagnostic is a short-window all8 task timeline of the **same serial and concurrent protocol**, interpreted against Run579's unprofiled baseline. It should separate late HCCL launch, interleaved but inflated GMM service, and collective rank-arrival wait. Profiler timing must be treated as mechanism evidence, not service time. The next product-facing Bound measurement is the **actual production dependency DAG**: timestamp collective input-ready, submit, completion and consumer-ready edges together with Target, DSpark, KV and Host edges for an unchanged W₀ trajectory. Where an edge is legally movable, test smaller dependency-preserving segments rather than assume whole-chain concurrent replay is the best schedule.
