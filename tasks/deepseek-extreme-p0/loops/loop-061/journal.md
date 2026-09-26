# 执行日志

- `2026-09-26T01:02:12Z` Loop 已冻结。下一步：Audit Run246 HCCL payload for all 80 target windows, then map sizes to API semantics before testing transport capacity

- `2026-09-26T01:02:18Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run250`（profile）。

- `2026-09-26T01:02:43Z` Run `run250` 记录为 `pass`；正确性为 `not-applicable`。Run246 offline HCCL trace census passed 80/80 target rank-cycle windows, latest16 across8 ranks; 265 collective events and 25,651,200 reported size bytes per rank-cycle with one common signature; INVALID_TYPE link fields and zero transit sizes prevent physical-byte inference

- `2026-09-26T01:04:35Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run251`（build）。

- `2026-09-26T01:08:00Z` Run `run251` 记录为 `pass`；正确性为 `not-applicable`。CANN 9.1.0 bundled HCCL Test built in /tmp/extreme_hccl_test_cann910 with container OpenMPI 4.1.2; stock Makefile initially failed linking MPI C++ bindings, temporary copy amended with -lmpi_cxx; final build exit0, 11 executables; no NPU benchmark yet

- `2026-09-26T01:08:12Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run252`（benchmark）。

- `2026-09-26T01:09:37Z` Run `run252` 记录为 `invalid`；正确性为 `invalid`。CANN HCCL Test -i 0 repeated fixed 96K indefinitely rather than terminating at -e 96K; manually interrupted exact mpirun PID, rank processes now defunct, 8/8 NPUs idle. Per-iteration success lines are diagnostic only and excluded from capacity estimate; rerun finite sweep with -i positive

- `2026-09-26T01:09:43Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run253`（benchmark）。

- `2026-09-26T01:12:38Z` Run `run253` 记录为 `pass`；正确性为 `pass`。Official CANN9.1 HCCL Test six fixed payload cases on eight 910B3 ranks returned exit0, data_size exact, check_result success; 96KiB normal-path average AllGather159.32us, ReduceScatter114.36us, AllToAll211.07us. One preliminary -i 1K invocation failed parser, corrected to integer1024; graph product path remains incommensurate and no wire bytes or E2E savings inferred

- `2026-09-26T01:12:45Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run254`（benchmark）。

- `2026-09-26T01:15:00Z` Run `run254` 记录为 `pass`；正确性为 `pass`。Paired CANN9.1 HCCL Test at exact 98304B on8x910B3 with HCCL_BUFFSIZE=256MB and -t0/-t1, 50 iterations20 warmups, all exit0/check success. AG167.40/40.03us, RS102.23/39.74us, A2A205.65/62.17us; standalone device-only times still do not identify graph-path collective service time or physical link bytes. 8 NPUs idle

- `2026-09-26T01:15:05Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run255`（review）。

- `2026-09-26T01:15:51Z` Run `run255` 记录为 `pass`；正确性为 `not-applicable`。Independent Astra High read-only review rejects promoting V0 scenario savings to a hardware/engineering achievable bound; flags cross-cycle GMM 1.092 ratio, missing mixed-resource dependency DAG and HCCL wire bytes; identifies Compressor unique BF16 weight footprint 0.6082GB versus 1.5309GB gross read without labeling difference waste. Sol incorporated caveats in findings and next mechanism experiment

- `2026-09-26T01:17:13Z` 主控结论为 `PIVOTED`。Eight-rank graph HCCL operation payload and same-host official HCCL Test capability are measured, but trace Size(Byte) is not physical link bytes, standalone test is not product Graph path, and critical-path overlap remains unknown; Astra High confirms V0 future ranges are sensitivities rather than attainable bounds. No E2E candidate or true hardware ceiling established.
