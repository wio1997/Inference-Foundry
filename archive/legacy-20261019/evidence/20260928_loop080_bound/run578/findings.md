# Run578: scoped real-weight GMM MemoryAccess counters after online parser failure

## Execution and recovery

Run578 used the guarded warm48 + measured48, c12/1024 client contract. At terminal cohort8, all eight workers borrowed the resident production W4A8 Target operands and the SHA-pinned Run576 cohort5/cycle170 group fixture for a private 43-layer GMM1→GMM2 Graph with deterministic nonzero synthetic activation. The original plan was unprofiled A0 → Level1 MemoryAccess → unprofiled A1. The online profiler daemon failed to parse on all eight workers before the helper could persist A0 or start A1. The controller returned **exit1**. Its stop, idle, restore, source SHA and script SHA gates all returned **0**; the service remains stopped, eight NPUs idle, borrowed source files restored.

The raw profiler packages survived. The installed torch_npu offline `analyse(..., export_type="text")` completed for all eight ranks. The salvage reducer pins its own code, collector/controller code, raw PROF/FRAMEWORK files, offline parse logs, metadata, exported CSVs, the 96-request client admission, all64 Runtime cohort records joined to phase/dataset, Run576 census/fixtures, service/stop logs, and source restoration. Its CPU gate rejects wrong rank/metric/schedule metadata, duplicate task IDs, NaN, all-zero counters, and a duplicated second replay. Astra independently reviewed the repaired admission. This is a **partial counter diagnostic**; the original paired-service experiment failed.

## Scoped observation

Each rank exports exactly **172 native tasks**: two time-distinct Graph replays of task IDs `0..85`, alternating 43 `GroupedMatmulSwigluQuantV2` and 43 `GroupedMatmul`. Each CSV has one device/model/stream identity; CANN 9.1.0 Level1 `ACL_AICORE_MEMORY_ACCESS` metadata matches rank and schedule. The two replay intervals do not overlap. Per-rank median of the two AIC+AIV **counter-reported main-memory** totals:

| Rank | Read GB/replay | Write GB/replay |
|---:|---:|---:|
| 0 | 9.226467 | 0.223727 |
| 1 | 9.211232 | 0.216798 |
| 2 | 9.072913 | 0.207924 |
| 3 | 9.155434 | 0.213006 |
| 4 | 9.073470 | 0.214713 |
| 5 | 9.504296 | 0.226891 |
| 6 | 9.156949 | 0.211179 |
| 7 | 8.878161 | 0.197254 |

Across ranks, median read is **9.156191 GB** and median write **0.213860 GB** per replay. The two replay read totals differ by <0.001 GB on each rank. Rank0's first replay comprises about 5.957 GB AIC GMM1 reads, 3.017 GB AIC GMM2 reads, plus 0.253 GB AIV reads. GM→L1/UB fields are recorded separately and are not added again to main-memory totals. Installed CANN9.1 `ai_core_config.py` computes read/write KB from hardware register sums in 128-byte units; exported KB is multiplied by 1024 to decimal GB. Official 9.1 documentation describes these as core-side MemoryAccess counters, without certifying a one-to-one mapping to physical HBM-controller payload. See the [official 9.1 fields](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/910/devaids/Profiling/atlasprofiling_16_0069.html) and [AIC/AIV prefixes](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910/devaids/Profiling/atlasprofiling_16_0066.html).

This current-implementation counter range is comparable in order of magnitude to Run247's different-trajectory FULL Graph median GMM read 9.244 GB/rank-cycle. The numerator includes non-weight reads and could include repeat reads, so the ratio to Run576's selected-expert-slice arithmetic does **not** measure weight amplification or compulsory bytes. Run577's isolated 10.1311698 ms slowest-rank service is from another trial. Because A0 was not persisted and A1 was not executed, **no same-run attainable bandwidth** follows from Run578.

## Bound effect and next discrimination

Run578 narrows the **Current implementation traffic** uncertainty for the Run576-matched, real-weight, cross-layer isolated GMM fixture. It does not close physical HBM traffic, mathematical compulsory work/bytes, formal Run99 W₀ equivalence, mixed Target+DSpark+KV+HCCL resource contention, legal dependency overlap or exact-board cumulative C⁺/B. V3.34 keeps all strict finite Resource/Hardware, Scheduling/Execution and Product endpoints and numeric Current→credible-limit distance **null**. Current Formal E2E remains **571.681 tok/s**.

Next measure one production-sized GMM × HCCL independent-work four-condition service (each alone, serial, dual-stream event-joined) under unchanged work and all8 correctness, then bind the production collective producer/arrival/consumer edges before transferring any overlap to Scheduling Bound. Historical R20's partial E2E realization of timeline overlap and R27's exposed EP collective dependency are mechanism priors from different A2/EP8 conditions, not current verdicts. In parallel, formal necessary-work/traffic W-minus and exact-board cumulative C⁺/B certificates remain required for a finite Hardware/Resource bound.

Evidence: `counter_salvage.json`, `salvage_gate.json`, `live/b/bench/profile/rank*/`, `live/cleanup_status.txt`, V3.34 model and Astra review.
