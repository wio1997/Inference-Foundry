# 执行日志

- `2026-09-25T11:12:33Z` Loop 已冻结。下一步：Run211 read-only source closure: actual custom-op call, context, mutable state and graph API contract; choose a minimal one-layer capture implementation or reject

- `2026-09-25T11:12:33Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run211`（design-check）。

- `2026-09-25T11:13:35Z` Run `run211` 记录为 `pass`；正确性为 `not-applicable`。MoE custom-op source closure excludes direct DSA/KV but has forward-context index, DP/SP sizes, routing, multistream/HCCL and possible mutable counters; live ABI/state probe required before capture

- `2026-09-25T11:17:06Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run212`（profile）。

- `2026-09-25T11:20:16Z` Run `run212` 记录为 `invalid`；正确性为 `invalid`。Stopped before requests after detecting probe BF16-to-NumPy hash incompatibility; dry import passed but no ABI data, no benchmark; exact source restore and service stop verified

- `2026-09-25T11:21:20Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run213`（profile）。

- `2026-09-25T11:41:10Z` Run `run213` 记录为 `pass`；正确性为 `pass`。72/72 legal requests and8-rank Runtime pass; first88 MoE layer0 per-rank [11,4096] BF16 stable alias/address/context with changing inputs and outputs; no graph/performance claim

- `2026-09-25T11:44:17Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run214`（design-check）。

- `2026-09-25T11:44:44Z` Run `run214` 记录为 `pass`；正确性为 `not-applicable`。Bound one layer/shape shadow graph capture with two distinct real inputs, eager self-noise controls, retained serving eager output and all8-rank collective gates; no performance claim

- `2026-09-25T11:47:02Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run215`（test）。

- `2026-09-25T12:06:08Z` Run `run215` 记录为 `invalid`；正确性为 `invalid`。72/72 legal requests succeeded, but zero graph fingerprint files: wrapper lowercase run212 state name was not changed to run215, so MoE hook never armed; no capture/parity/performance result

- `2026-09-25T12:09:07Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run216`（test）。

- `2026-09-25T12:28:56Z` Run `run216` 记录为 `pass`；正确性为 `pass`。8/8 first88 MoE graph capture and second-cohort replay;72/72 legal requests and6x8 Runtime pass; shared bit-exact, routed graph maxabs0.0078125 within observed eager self order; A/B input distinctness and product saving unproven

- `2026-09-25T12:32:15Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run217`（test）。

- `2026-09-25T12:53:41Z` Run `run217` 记录为 `pass`；正确性为 `pass`。72/72 legal requests and48/48 eight-rank Runtime records pass; all8 first88 layer0 graphs capture A/replay B with distinct input/output hashes; shared exact, routed maxabs0.0078125 vs eager self max0.015625; one-layer diagnostic only, no stage/E2E claim

- `2026-09-25T12:56:31Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run218`（test）。

- `2026-09-25T12:59:17Z` Run `run218` 记录为 `invalid`；正确性为 `invalid`。Launcher referenced nonexistent patch file and exited2 before source install or requests; no graph substitution result; original source hashes and stopped service verified

- `2026-09-25T12:59:55Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run219`（test）。

- `2026-09-25T13:19:10Z` Run `run219` 记录为 `pass`；正确性为 `pass`。96/96 legal requests and64/64 eight-rank Runtime pass; first88 layer0 graph outputs served on B/C, B shared exact and routed maxabs0.0078125; one-call local timing plausible but no causal stage/E2E gain

- `2026-09-25T13:20:17Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run220`（design-check）。

- `2026-09-25T13:20:17Z` Run `run220` 记录为 `pass`；正确性为 `not-applicable`。Frozen eight-rank interleaved G/E/E/G/G/E/E/G one-layer synchronized completion screen with four paired slowest-rank comparisons, legal48+8x12 requests, Runtime and exact source restore gates; no E2E claim
