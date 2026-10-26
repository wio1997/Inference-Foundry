# 执行日志

- `2026-09-22T09:27:52Z` Loop 已冻结。下一步：Run the implemented side-stream host-mirror handoff through the real-weight eight-rank eight-cycle correctness gate, then capture a scoped optimized trace if parity holds.

- `2026-09-22T09:28:14Z` 为用例 `mixed_32k_1024_c12` 创建 Run `real-weight-host-mirror-20260922`（test）。

- `2026-09-22T09:38:40Z` 为用例 `mixed_32k_1024_c12` 创建 Run `optimized-runtime-profile-20260922`（profile）。

- `2026-09-22T09:58:34Z` 为用例 `mixed_32k_1024_c12` 创建 Run `refined-host-mirror-64cycle-20260922`（benchmark）。

- `2026-09-22T09:58:58Z` Run `optimized-runtime-profile-20260922` 记录为 `pass`；正确性为 `pass`。Eight-rank optimized trace passed correctness. DSpark refresh median fell from 12.483 to 0.366 ms and proposer from 66.825 to 52.170 ms; profiled cycle median fell 7.495 ms (-1.40%). A p90 tail remained in end-of-cycle mirror commit, motivating one-cycle-delayed consumption before the next proposer.

- `2026-09-22T09:58:58Z` Run `real-weight-host-mirror-20260922` 记录为 `pass`；正确性为 `pass`。All eight ranks completed eight continuous real-weight cycles with identical rank state, exact state advance, 67 cache tensors, no retained ModelRunner, and no oracle target calls after handoff. Non-profile wall was 3.464-3.470 s per rank for the full eight-cycle window.

- `2026-09-22T10:11:44Z` Run `refined-host-mirror-64cycle-20260922` 记录为 `pass`；正确性为 `pass`。Refined one-cycle-delayed mirror handoff completed 64 continuous real-weight cycles on all eight ranks. Host mirrors exactly matched device query/next-sequence/computed state after the timed window; all rank state matched. Median wall 27.7495 s (433.587 ms/cycle), 1062 emitted tokens, internal decode-window rate 38.271 tok/s. This excludes E2E ingress/refill/egress and is not comparable to Stock 543.65 tok/s.

- `2026-09-22T10:12:13Z` 暂存知识变化 `extreme-target-replay-priority-20260922`：After host-mirror removal, the direct target remains the dominant structural gap: optimized trace median target host wall is 469.789 ms and target device union 465.056 ms, while the handoff currently forces CUDAGraphMode.NONE and skip_compiled=True. The next runtime migration should test a fixed target replay/graph boundary with owned buffers and metadata before lower-value acceptance/state micro-optimizations.

- `2026-09-22T10:12:13Z` 暂存知识变化 `runtime-owned-host-mirror-overlap-20260922`：Replacing three per-cycle device-to-host common-state pulls with a side-stream 12-count acceptance handoff reduces DSpark refresh median from 12.483 to 0.366 ms and proposer median from 66.825 to 52.170 ms. The refined runtime consumes the pending copy one target cycle later and preserves exact host/device mirror parity for 64 real-weight cycles on all eight ranks.

- `2026-09-22T10:13:31Z` 主控结论为 `PIVOTED`。The structural hypothesis is supported: real-weight correctness holds for 64 cycles, all runtime-owned host mirrors equal device state, and the matched profile shows the DSpark refresh barrier falling by 12.117 ms median with a 7.495 ms (-1.40%) full-cycle median reduction. TaskCtl cannot register an accepted optimization verdict because the already-recorded Loop031 profile Run omitted a metric field; the evidence comparison remains preserved explicitly and the implementation is retained.
