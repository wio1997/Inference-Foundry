# 执行日志

- `2026-09-20T19:20:35Z` Loop 已冻结。下一步：Stop current baseline, apply prefill-only source patch, restart DP1TP8, validate function and run warmed full benchmark plus fresh paired cold prompts

- `2026-09-20T19:24:14Z` 为用例 `mixed_32k_1024_c12` 创建 Run `qli-prefill-candidate-20260920`（benchmark）。

- `2026-09-20T19:44:32Z` Run `qli-prefill-candidate-20260920` 记录为 `pass`；正确性为 `pass`。Prefill-only QLI CPU maxima: functional gate passed; three 48/48 warmed mixed passes 547.55/538.53/553.35 TPS, median547.55 vs paired original537.60 (+1.85% within noise); median mean TTFT1143.58 vs1262.74ms, no regression; sixteen same-offset cold prompts all improved, mean TTFT2362.89 vs2626.03ms (-10.02%).

- `2026-09-20T19:44:33Z` 为用例 `cold_32k_128_c1` 创建 Run `qli-prefill-cold-20260920`（benchmark）。

- `2026-09-20T19:44:33Z` Run `qli-prefill-cold-20260920` 记录为 `pass`；正确性为 `pass`。Same-prompt 16/16 cold requests improved; candidate mean TTFT2362.89 vs original2626.03ms, -10.02%, delta range -355.75 to -202.61ms, zero prefix-cache hits over 525616 queried tokens.

- `2026-09-20T19:44:40Z` 已记录对比（`comparable=yes`）：16 exact same cold prompts (offsets24-31,40-47), all improved: original mean TTFT2626.03ms, candidate2362.89ms (-263.14ms, -10.02%); both cache-hit counters zero; same DP1TP8 image, workload and full warmed mixed protocol. Baseline run metric2633.90ms covers first 8; comparison.json is all16 source.

- `2026-09-20T19:47:27Z` 为用例 `cold_32k_128_c1` 创建 Run `original-cold16-20260920`（benchmark）。

- `2026-09-20T19:47:36Z` Run `original-cold16-20260920` 记录为 `pass`；正确性为 `pass`。Original-source 16 exact cold prompts mean TTFT 2626.0327 ms, all 16 successful and zero prefix hits across two measured groups

- `2026-09-20T19:47:36Z` 已记录对比（`comparable=yes`）：Same 16 cold prompts: mean TTFT 2626.03 to 2362.89 ms (-263.14 ms, -10.02%); all 16 faster, zero prefix hits. Mixed median output TPS 537.60 to 547.55 (+1.85%, within noise), median TTFT 1262.74 to 1143.58 ms. Keep prefill-only QLI.

- `2026-09-20T19:47:44Z` 主控结论为 `ACCEPTED`。Runtime CPU/NPU QLI maxima parity passed on all eight ranks in Loop011. Prefill-only patch passed functional gate. Same 16 cold prompts all improved: 2626.03 to 2362.89 ms mean TTFT (-10.02%) with no prefix cache hits. Full mixed median output TPS 547.55 versus paired original 537.60 is within baseline noise; median TTFT 1143.58 versus 1262.74 ms shows no observed regression.
