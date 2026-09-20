# 执行日志

- `2026-09-20T18:18:35Z` Loop 已冻结。下一步：Apply reversible Tensor.item wrapper only during prepare input, restart without profiler, capture warm short workload and source line timing

- `2026-09-20T18:20:34Z` 为用例 `mixed_32k_1024_c12` 创建 Run `itemtrace-20260920`（profile）。

- `2026-09-20T18:34:55Z` Run `itemtrace-20260920` 记录为 `pass`；正确性为 `pass`。Reversible no-profiler tracer identified per-step NPU Tensor.item at dsa_cp.py:1008; CPU equivalent max_local_query_len/max_local_seq_lens already computed from mirrored local metadata. TP0 warm approx 210ms/54 calls; cold approx 1268ms/70 calls. Eight ranks show skew; no performance gain claimed.

- `2026-09-20T18:35:39Z` 主控结论为 `ACCEPTED`。Targeted wrapper found exact DSA CP QLI NPU item callsite and matching CPU local maxima already available; warm/cold rank timing supports a falsifiable candidate, without claiming E2E gain
