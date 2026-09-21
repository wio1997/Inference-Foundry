# 执行日志

- `2026-09-21T01:13:57Z` Loop 已冻结。下一步：Inspect Loop014 no-profiler builder traces and active DSA-CP builder source; quantify repeated allocations/work by shape before editing.

- `2026-09-21T01:21:39Z` 为用例 `mixed_32k_1024_c12` 创建 Run `active-dsacp-builder-stage-20260921`（profile）。

- `2026-09-21T01:57:19Z` Run `active-dsacp-builder-stage-20260921` 记录为 `pass`；正确性为 `pass`。DP1TP8 diagnostic service functional checks passed, 12/12 c12x512 sample passed; 174 pure-decode builder steps/rank. First builder median total3.61-6.50ms/rank, request-metadata2.74-5.74ms, shared setup0.57-0.66ms. Inclusive host timing, not an E2E saving.

- `2026-09-21T02:00:18Z` 为用例 `mixed_32k_1024_c12` 创建 Run `active-dsacp-request-subphases-20260921`（profile）。

- `2026-09-21T02:49:27Z` Run `active-dsacp-request-subphases-20260921` 记录为 `pass`；正确性为 `pass`。DP1TP8 second diagnostic: short functional passed, 12/12 c12x512; 169-170 pure-decode first-builder steps/rank. Within first build_req_metadata, QLI stage median1.735-4.306ms/rank dominates; device-local0.477-0.548ms, CPU-local0.255-0.292ms, SAS0.422-0.485ms. Inclusive host scopes, no E2E saving claim. API stopped, framework restored.

- `2026-09-21T02:59:05Z` 为用例 `mixed_32k_1024_c12` 创建 Run `qli-scalar-versus-op-trace-20260921`（profile）。

- `2026-09-21T03:14:26Z` Run `qli-scalar-versus-op-trace-20260921` 记录为 `pass`；正确性为 `pass`。No-profiler DP1TP8 diagnostic passed short functional and12/12 c12x512. QLI uncached decode q.max().item median0.402-3.716ms/rank, k.max().item0.110-0.135ms, metadata op+clones0.356-0.411ms, ~155-156 calls/rank. Earlier Loop011 CPU-max removal yielded only +0.52% mixed TPS within noise and worse TTFT, so this host wait is not proven E2E-removable.

- `2026-09-21T03:14:26Z` 主控结论为 `PIVOTED`。QLI first NPU scalar read accounts for most measured first-builder host time, but the already-tested all-step CPU-max removal passed parity and gave no robust mixed TPS gain (+0.52% within noise, TTFT worse). Metadata op itself is only ~0.4ms/call. No new semantics-safe, high-value decode patch justified; host span is not a removable E2E bound.
