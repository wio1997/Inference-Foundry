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

- `2026-09-25T04:50:17Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run160`（profile）。

- `2026-09-25T04:50:54Z` Run `run160` 记录为 `invalid`；正确性为 `invalid`。No service or NPU work. Temporary stage patch installed, then timing patch install failed because both used the same /tmp backup path. Trap restored stage and boundary sources to original SHA and stopped service. No throughput/stage data; fix backup collision in Run161.

- `2026-09-25T04:51:08Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run161`（profile）。

- `2026-09-25T05:11:07Z` Run `run161` 记录为 `pass`；正确性为 `pass`。Legal 8-rank warmed-service diagnostic 60/60 success and 40 rank-cohort records pass. In measured cohort, median prefill execute_model wall3.397s, of which _model_forward Python wall3.122s (~91.9%) and preparation0.258s. Device/communication split unresolved; diagnostic TPS580.055 not formal E2E. Temporary sources restored, service stopped, 8 NPU idle.

- `2026-09-25T05:12:45Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run162`（profile）。

- `2026-09-25T05:27:14Z` Run `run162` 记录为 `invalid`；正确性为 `invalid`。Service startup failed before any benchmark or profiler capture: Worker TP6 KV block table CpuGpuBuffer pin_memory allocation raised torch.OutOfMemoryError aclrtMallocHostWithCfg 207001. Host memory at failure ~16Gi free and swap full; root cause/ownership not established. Stopped service, restored temporary sources, 8 NPU idle. No performance conclusion.

- `2026-09-25T05:27:41Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run163`（profile）。

- `2026-09-25T05:30:53Z` Run `run163` 记录为 `pass`；正确性为 `not-applicable`。Read-only resource audit after Run162: dedicated vllm container cgroup has 3646 live process IDs, including 3270 multiprocessing.forkserver processes (~716GiB aggregate RSS), 327 spawn processes (~194GiB) and 8 python stdin processes (~59GiB). Host swap 71GiB full; no container memory cap and 8 NPU idle. Strong resource pressure explanation for pinned Host allocation failure, not direct proof. Next controlled container restart after stopped-service check.

- `2026-09-25T05:31:17Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run164`（test）。

- `2026-09-25T05:33:18Z` Run `run164` 记录为 `pass`；正确性为 `not-applicable`。Controlled cleanup of stopped dedicated container: docker restart first exited1 (daemon did not receive exit event), but released most orphan memory; explicit docker stop exited0 and docker start exited0. Cgroup process count3646->1, Host used memory841GiB->20GiB, swap71GiB->749MiB. Mounts and source SHA verified, service stopped, 8 NPU idle. Initial restart error preserved.

- `2026-09-25T05:34:10Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run165`（profile）。

- `2026-09-25T05:52:42Z` Run `run165` 记录为 `pass`；正确性为 `pass`。Legal 8-rank warmed-service 48x1024+12x1024 diagnostic, 60/60 success and all 40 rank-cohort records pass. One rank0 measured prefill _model_forward profile captured after four cohorts. CANN step trace: stage444.180ms, compute39.597ms, exposed communication2.862ms, Free/no recorded kernel401.721ms (90.44%). 264 paired HCCL/Aiv logical duplicates removed in independent kernel-union check. Single rank/call profiler with synchronization, no formal E2E or achievable speedup claim. Source restored, service stopped, 8 NPU idle.
