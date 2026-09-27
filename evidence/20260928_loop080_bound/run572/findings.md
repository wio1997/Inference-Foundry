# Run572 — rank-local FULL Graph profiler interval audit

This offline audit reopens the latest Run246/247 Level1 FULL Graph profiler export (`capture=4`, two `extreme::target` windows × eight ranks). It checks each trace and kernel CSV against Run247 export SHA256, exact window counts (43 GMM1, 43 GMM2, 265 `hcom_*` pseudo envelopes and 265 HCCL `AivKernel` tasks), and the published target scope duration. All16 windows have 2,836 start-in-scope tasks, zero carry-in and zero cross-end tasks. Decimal-to-integer-nanosecond arithmetic avoids large absolute timestamp float rounding. It launched no model or NPU run.

| Rank-local export interval, 16 windows | Median | Range |
|---|---:|---:|
| Marked Target scope | 54.796 ms | 47.228–75.982 ms |
| AI Core task interval union | 27.749 ms | 26.994–28.626 ms |
| Vector task interval union, including HCCL `AivKernel` | 24.514 ms | 18.865–41.155 ms |
| `hcom_*` pseudo envelope interval union | 11.200 ms | 5.513–27.846 ms |
| Pseudo envelope with no exported AI task | 0 ms | 0–0 ms |
| Any exported task interval union | 49.746 ms | 44.398–67.557 ms |
| No exported task interval | 5.250 ms | 2.711–8.495 ms |

The profiler's `COMMUNICATION` classification contains **two different kinds of exported rows**: 265 `hcom_*` pseudo envelopes and 265 nonzero-AIV `AivKernel` tasks in each window. Counting both as physical communication would double represent intervals; classifying both as communication while excluding AIV would falsely create ~10 ms of apparent “communication without AI.” After splitting them, the pseudo envelopes have no interval unaccompanied by exported AI tasks in these windows. The 5.250 ms median with no exported task is only an export-coverage gap: it is not proved device idle or removable wall time. Interval unions across AI Core, Vector and pseudo envelope can overlap; their medians cannot be added.

The trace time axes have **not** been calibrated across rank/device clock domains. `cross_rank_physical_overlap` is null and no all8 makespan or concurrency is inferred. Target scopes and task export are instrumented/synchronized, not passive Run99 Current; pseudo envelope durations are not HCCL link transit or necessary collective latency. Without producer/consumer stream joins, none of these intervals certifies compulsory work, a service capacity, removable scheduling time or a strict critical-path lower bound. All finite Resource/Hardware, Scheduling/Execution and Product endpoints and numeric Current→credible-limit distance remain null; formal Current is Run99 median **571.681 tok/s**.

This audit sharpens the next Scheduling measurement: correlate each collective's actual message/producer readiness, `AivKernel` and pseudo-envelope semantics, all8 arrival and completion in a validated common clock domain, while measuring real mixed Target/DSpark/HCCL service under the same source-pinned workload. That can distinguish a rank-arrival/synchronization envelope from physical communication and test whether any overlap change shortens the complete critical path. Run571's isolated HCCL numbers are measurement priors only. The strict Resource track still needs formal `W₀` necessary work/traffic and exact-board cumulative `C⁺/B`.
