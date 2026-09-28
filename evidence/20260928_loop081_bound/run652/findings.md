# Run652 — rank-local Host versus current-stream handoff interval

Read-only SHA-pinned Run638 ON packet replay. For all8 ranks × four measured cohorts × two adjacent transitions (64 rows), compare the Host-submit interval `proposer_after(c) → target_before(c+1)` with the elapsed device Event interval between exactly those two markers on the **same current stream**. All event generations are1 and issue ordinals increase; no cross-rank device clock subtraction is used.

| Paired rank-local interval | Median | Range |
|---|---:|---:|
| Host marker submits | 6.265 ms | 5.790–7.001 ms |
| Same-current-stream Event elapsed | 0.873 ms | 0.798–4.054 ms |
| Host interval − Event interval | 5.195 ms | 2.325–5.598 ms |

Host interval exceeds Event interval in64/64 pairs. The intervals refer to different clocks and event execution points, though each duration is rank-local; the positive difference is **not** a measured idle duration, Host-only overhead, exposed Product time or schedulable saving. It is consistent with asynchronous submission/queued work keeping the stream behind the Host and compressing the device-marker separation, but the packet alone does not certify queue depth, producer-ready, all side streams, collective completion or exact overlap. Run638 cross-arm fixed-W₀ failed; no direct observer-neutral or formal E2E transfer.

Bound effect: Run651's6.265ms median Host-submit gap cannot be inserted as a disjoint edge cost. The fixed-primitive critical-path model should carry separate Host issue and same-stream progress clocks, and only credit a Host reduction when the selected consumer/collective/Product completion advances under real resource contention. Formal Current571.681tok/s and numerical Framework-only ceiling unchanged.
