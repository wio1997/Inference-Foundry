# 执行日志

- `2026-09-25T03:39:52Z` Loop 已冻结。下一步：Run154: add temporary common-clock marks at vLLM admission/prefill/handoff/runtime-build/FixedCohort completion/publication, and persist per-slot count history after cohort; run one legal 8-rank 12x1024 diagnostic, restore borrowed sources and stop service.

- `2026-09-25T03:41:03Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run154`（profile）。

- `2026-09-25T04:02:59Z` Run `run154` 记录为 `pass`；正确性为 `pass`。Cold-service legal 8-rank 12x1024 diagnostic passed 12/12; eight boundary/runtime files, exact1024 outputs, source SHA restored and idle8. Same-host median client-to-first-execute0.383s, first-execute-to-handoff5.954s, handoff-to-build0.083s, build0.041s, serve16.636s, publication-to-client-end0.175s. Parked slot-cycles14.484%. Cold diagnostic527.999 tok/s is not formal E2E. EngineDeadError appeared after completed requests and is retained in teardown log. Warm Run155 needed before causal decision.

- `2026-09-25T04:03:23Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run155`（profile）。

- `2026-09-25T04:23:39Z` Run `run155` 记录为 `pass`；正确性为 `pass`。Legal 8-rank 48x1024 warmup + 12x1024 measured diagnostic, all 60 requests and 40 rank-cohort traces pass. Warm measured client envelope20.779s; client-to-first execute0.241s, pre-handoff model execution3.621s, runtime construction0.00264s, fixed decode16.728s, publication-to-client0.184s. Parked slot-cycles18.525%, exposure only; no formal E2E claim. Source restored and service stopped.

- `2026-09-25T04:24:48Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run156`（profile）。

- `2026-09-25T04:25:44Z` Run `run156` 记录为 `pass`；正确性为 `not-applicable`。Offline 8-rank Run155 cohort audit: warm cohorts3/4 each ~50.6k scheduled prefill tokens in 11 calls, first execute-to-handoff ~3.66s; measured cohort5 only 1.63k tokens in 12 calls, ~3.62s. Prefill latency is not explained by token volume alone; direct per-call admission/tokenization/device timing needed. Parked slots 9.7-19.5% of cycles across cohorts, exposure not speedup.

- `2026-09-25T04:27:48Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run157`（design-check）。

- `2026-09-25T04:27:48Z` Run `run157` 记录为 `pass`；正确性为 `not-applicable`。Source inspection found 12 client requests launch within milliseconds while Run155 measured cohort uses 12 pre-handoff execute calls. Existing boundary patch only marks entry; Run158 will record entry/exit across 8 ranks to split in-call wall from inter-call gap. No device-time claim without NPU events.

- `2026-09-25T04:28:55Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run158`（profile）。

- `2026-09-25T04:48:17Z` Run `run158` 记录为 `pass`；正确性为 `pass`。Legal 8-rank 48x1024 warmup + 12x1024 measured diagnostic: 60/60 success, all 40 rank-cohort records pass, 8 rank timing logs. Across ranks median measured pre-handoff span3.433s: 10 prefill execute_model calls total2.928s, inter-execute gaps0.478s of which sample_tokens0.447s, final handoff preamble0.026s. Dominant worker in-call; no device-only claim. Restored sources, stopped service, 8 NPU idle. Diagnostic TPS622.761 not formal E2E.

- `2026-09-25T04:49:28Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run159`（design-check）。

- `2026-09-25T04:49:28Z` Run `run159` 记录为 `pass`；正确性为 `not-applicable`。Inspected NPUModelRunner prefill source boundaries. Run160 will timestamp method entry, prepare end, model forward entry/exit and method exit without device synchronization, excluding final Extreme handoff call. Attribution limited to Python wall; device time still requires NPU events/profile.
