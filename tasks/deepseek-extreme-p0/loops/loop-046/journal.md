# 执行日志

- `2026-09-25T06:02:29Z` Loop 已冻结。下一步：Run168 read-only source/write-set and metadata feasibility audit for first warmed measured genuine prefill forward.

- `2026-09-25T06:06:35Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run168`（review）。

- `2026-09-25T06:06:35Z` Run `run168` 记录为 `pass`；正确性为 `not-applicable`。Existing DSA CP graph capture rejects prefill; guarded exact-state capture would need full prefill metadata and SWA/compressor/c4 indexer state restoration. Async all-gather/connector compatibility untested. One-off graph product value must be bounded against natural shape reuse.

- `2026-09-25T06:07:59Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run169`（review）。

- `2026-09-25T06:07:59Z` Run `run169` 记录为 `pass`；正确性为 `not-applicable`。Run158/161 measured cohorts have respectively 10/12 distinct token-count forward calls on every rank; only 8 and83 recur across runs, and equal count does not prove equal metadata. Exact-state per-shape capture has no demonstrated amortizable reuse. Large calls sum median2.729/2.928s but are only opportunity ceilings. Pivot next to source-grounded admission/coalescing mechanism.

- `2026-09-25T06:12:45Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run170`（profile）。

- `2026-09-25T06:13:14Z` Run `run170` 记录为 `invalid`；正确性为 `invalid`。Shell redirection failed before runner start because run170 output directory did not exist; no source patch, service or NPU work. Retry as Run171 after creating output directory.

- `2026-09-25T06:13:26Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run171`（profile）。

- `2026-09-25T06:28:53Z` Run `run171` 记录为 `pass`；正确性为 `pass`。Legal warmed 48+12 requests pass 60/60 and 40 rank-cohort runtime records. Measured client starts span12ms; Core add_request processing spans1.696s, from0.226s to1.922s after client start, with running count advancing0->10. Schedule rows absent, so exact queue arrival and scheduler budget unmeasured. Borrowed scheduler restored and service stopped. Diagnostic TPS634.456 not formal.

- `2026-09-25T06:30:47Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run172`（benchmark）。

- `2026-09-25T06:45:31Z` Run `run172` 记录为 `pass`；正确性为 `pass`。Legal8-rank48+12 passes60/60 and40 rank-cohort records. Bounded250ms Core drain collected3-4 initial requests and measured prehandoff execute sequence249,107,449,321 on all ranks. Measured diagnostic envelope19.278s vs separate-service Run17119.368s; 0.090s net difference is noise-sized, TTFT p50 worsened. Borrowed Core/model sources restored and service stopped. No formal E2E claim; test one bounded500ms dose.
