# 执行日志

- `2026-09-20T21:05:37Z` Loop 已冻结。下一步：Revisit retained cold TP0 trace and per-request TTFT alignment, inspect ScatterNdUpdateSk and Compressor call sites, then select one falsifiable prefill intervention.

- `2026-09-20T21:09:47Z` 为用例 `cold_32k_128_c1` 创建 Run `cold-kernel-shape-audit-20260920`（design-check）。

- `2026-09-20T21:09:47Z` Run `cold-kernel-shape-audit-20260920` 记录为 `pass`；正确性为 `not-applicable`。Original-source cold TP0 trace has ScatterNdUpdateSk shape cache[34091,32,1,512], indices[8096,2], updates[8096,1,512], 736 calls across 4 requests, median task1017us and median AIV67us; total761ms (all phases), plus Compressor ratio4 336 calls median1503us total506ms. Source ScatterNdUpdateSk enforces deterministic sort plus SyncAll because duplicates could corrupt rows. Need capture real indices and verify valid-row uniqueness before considering specialized path; current durations include decode and cannot be treated as TTFT savings.

- `2026-09-20T21:12:01Z` 为用例 `cold_32k_128_c1` 创建 Run `scatter-slot-uniqueness-20260920`（profile）。

- `2026-09-20T21:26:22Z` Run `scatter-slot-uniqueness-20260920` 记录为 `pass`；正确性为 `pass`。All eight ranks captured one representative 8096x2 long-prefill DSA compressor slot mapping; each had 8096 valid unique pairs, zero duplicates. Functional gate and cold request passed. Single-call observation does not establish a general invariant; probe latency is non-comparable due sync.

- `2026-09-20T21:27:51Z` 为用例 `cold_32k_128_c1` 创建 Run `scatter-v2-screen-20260920`（benchmark）。

- `2026-09-20T21:27:51Z` Run `scatter-v2-screen-20260920` 记录为 `pass`；正确性为 `pass`。Captured-rank0 8096x2 unique indices, 34091x32x1x512 float32 cache and 8096x1x512 updates: V2 output bit-equal to SK but 25-call device median 2.415ms vs SK 1.427ms (69% slower); reject V2 substitution. Isolated one-NPU diagnostic, not E2E.

- `2026-09-20T21:28:18Z` 主控结论为 `PIVOTED`。Captured one real 8096-row long-prefill scatter mapping per rank: all unique, but uniqueness not yet a general invariant. Existing V2 op gave bit-equal output and was 69% slower than SK in 25-call isolated NPU timing; no safe E2E patch or paired TTFT gain. Source SK deterministic sort+SyncAll suggests a direct unique-index path merits a separate build experiment.
