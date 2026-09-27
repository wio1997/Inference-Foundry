# Run571 — empirical capacity matrix, with no strict Bound promotion

This offline reducer reopens Run354–357, Run253/254, Run394/395 and Run487 raw evidence. It ran no service, model or NPU benchmark. Its 23 source-pinned rows comprise 18 **attained isolated-service observations** and five **instrumented current-path intervals**. The generator checks same ordinal64 route counts, packed format and 30 Graph replays per rank for the GMM bank A/B/A2 group; exact official HCCL Test payload/result, exit code and invocation; the full 5×8×{64,65} Run394 event grid and raw event joins; and all40 Run487 capture hashes. Each row records shape, timing class, source SHA and the transfer it does not support. `capacity_matrix.json` keeps every strict capacity and Product eligibility field null/false.

| Evidence | Recomputed result | Valid use |
|---|---:|---|
| W4A8 GMM1 bank8 B | 153.025 µs per call, median of eight rank medians; 0.212% above per-rank A/A2 midpoint median, 4/8 ranks strictly slower | Synthetic-zero, isolated Graph working-set sensitivity only |
| W4A8 GMM2 bank8 B | 81.919 µs per call; 8.037% above per-rank A/A2 midpoint median, 8/8 strictly slower | Same scope; shows bank residency can matter |
| HCCL Test 98,304B BF16, normal Run253 | AllGather 159.32 µs, ReduceScatter 114.36 µs, AllToAll 211.07 µs | Isolated tool average, exact board and payload, not Graph collective latency or wire bytes |
| HCCL Test 98,304B BF16, Run254 t1 | AllGather 40.03 µs, ReduceScatter 39.74 µs, AllToAll 62.17 µs | Device-only tool mode with 256MB HCCL buffer; not a necessary on-path latency |
| Run395 selected original-path events | Target 49.465 ms, proposer 6.397 ms, same-rank adjacent begin 56.548 ms | Instrumented Current, not separable service capacity |
| Run487 Target FULL selected frontier | R0→R1 45.986 ms; T→U 46.904 ms | Output-completion-conditional replay-stream interval; not passive or all8 makespan |

The GMM fixture has real route-count vectors but synthetic zero weights/inputs; independent rank Host windows overlap, but per-replay all8 device overlap and original FULL Graph residency have not been proved. HCCL `data_size` is participating bytes per NPU, not actual link bytes. Run395 and Run487 are different marked acquisitions; their medians cannot be spliced, added to other runs, or treated as removable critical-path time. These rows are **empirical Engineering priors**, not certified cumulative `C⁺/B`, strict Hardware/Resource lower time, Scheduling lower time, or Product TPS ceiling.

The matrix therefore changes no numeric Bound endpoint: formal Current remains Run99 median **571.681 tok/s**; finite strict Resource/Hardware, Scheduling/Execution and Product endpoints and Current→credible-limit distance remain null. It narrows the next measurement design: quantify real-data, residency-matched, concurrently active all8 GMM service together with Target/DSpark/HCCL resource contention on a source- and workload-admitted trajectory. Separately establish formal `W₀` fresh work/traffic and exact-board cumulative capacity. A new mixed-service result can fill an Engineering/Scheduling column only after rank overlap, working set, timing and output semantics are certified; it is not automatically a Hardware ceiling.

Independent Astra High review passed the arithmetic and scope and suggested stronger raw-event and invocation gates; both are implemented in the final reducer. Review files and hashes are recorded in the TaskCtl Run.
