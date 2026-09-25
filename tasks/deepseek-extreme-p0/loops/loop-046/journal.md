# 执行日志

- `2026-09-25T06:02:29Z` Loop 已冻结。下一步：Run168 read-only source/write-set and metadata feasibility audit for first warmed measured genuine prefill forward.

- `2026-09-25T06:06:35Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run168`（review）。

- `2026-09-25T06:06:35Z` Run `run168` 记录为 `pass`；正确性为 `not-applicable`。Existing DSA CP graph capture rejects prefill; guarded exact-state capture would need full prefill metadata and SWA/compressor/c4 indexer state restoration. Async all-gather/connector compatibility untested. One-off graph product value must be bounded against natural shape reuse.

- `2026-09-25T06:07:59Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run169`（review）。

- `2026-09-25T06:07:59Z` Run `run169` 记录为 `pass`；正确性为 `not-applicable`。Run158/161 measured cohorts have respectively 10/12 distinct token-count forward calls on every rank; only 8 and83 recur across runs, and equal count does not prove equal metadata. Exact-state per-shape capture has no demonstrated amortizable reuse. Large calls sum median2.729/2.928s but are only opportunity ceilings. Pivot next to source-grounded admission/coalescing mechanism.
