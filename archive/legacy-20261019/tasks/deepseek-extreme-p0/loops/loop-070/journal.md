# 执行日志

- `2026-09-26T14:58:20Z` Loop 已冻结。下一步：Run313: bash scripts/run_loop070_dynamic_metadata.sh run313

- `2026-09-26T14:58:25Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run313`（test）。

- `2026-09-26T15:17:06Z` Run `run313` 记录为 `fail`；正确性为 `not-applicable`。Private same-prestate dynamic Graph gate: all8 delta0 A/B/A2 pass. At synthetic delta-1 or -4 five ranks stop in original Graph A because only Sparse differs versus eager full-A despite exact Graph metadata, owner typed writes and QLI; three ranks complete all scenarios. No B observation on stopped arms, so owner conclusion unavailable. This is a synthetic metadata sensitivity control, not Product correctness. Borrowed sources restored SHA and cards idle. Run314 will use Graph A as same-mode control while retaining eager comparisons.

- `2026-09-26T15:17:16Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run314`（test）。

- `2026-09-26T15:40:22Z` Run `run314` 记录为 `fail`；正确性为 `not-applicable`。Real-entry delta0 private Graph metadata and producer A/B/A2 exact on 8/8; five ranks stop in synthetic full A with NaN Sparse even eager A; carrier incomplete, no Product verdict; source restored and cards idle

- `2026-09-26T15:40:22Z` 主控结论为 `PIVOTED`。Real-entry private Graph parity 8/8; synthetic shifts create nonfinite Sparse in original A, so no further synthetic repeats or Product claim
