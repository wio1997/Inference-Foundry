# 执行日志

- `2026-09-25T03:39:52Z` Loop 已冻结。下一步：Run154: add temporary common-clock marks at vLLM admission/prefill/handoff/runtime-build/FixedCohort completion/publication, and persist per-slot count history after cohort; run one legal 8-rank 12x1024 diagnostic, restore borrowed sources and stop service.

- `2026-09-25T03:41:03Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run154`（profile）。

- `2026-09-25T04:02:59Z` Run `run154` 记录为 `pass`；正确性为 `pass`。Cold-service legal 8-rank 12x1024 diagnostic passed 12/12; eight boundary/runtime files, exact1024 outputs, source SHA restored and idle8. Same-host median client-to-first-execute0.383s, first-execute-to-handoff5.954s, handoff-to-build0.083s, build0.041s, serve16.636s, publication-to-client-end0.175s. Parked slot-cycles14.484%. Cold diagnostic527.999 tok/s is not formal E2E. EngineDeadError appeared after completed requests and is retained in teardown log. Warm Run155 needed before causal decision.
