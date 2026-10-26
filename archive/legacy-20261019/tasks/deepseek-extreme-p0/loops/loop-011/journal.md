# 执行日志

- `2026-09-20T18:35:51Z` Loop 已冻结。下一步：Stop tracer service, restore framework source, implement CPU-max QLI candidate with controlled parity verification, restart DP1TP8 and measure

- `2026-09-20T18:38:42Z` 为用例 `mixed_32k_1024_c12` 创建 Run `qli-cpu-candidate-20260920`（benchmark）。

- `2026-09-20T18:58:14Z` Run `qli-cpu-candidate-20260920` 记录为 `pass`；正确性为 `pass`。QLI CPU maxima candidate parity >=192 per rank on warm/cold, functional pass, 3x48/48 mixed passes median540.40 TPS (-0.60% vs frozen baseline); two cold candidate groups mean TTFT2338/2360ms require same-offset baseline

- `2026-09-20T18:59:31Z` 为用例 `cold_32k_128_c1` 创建 Run `paired-baseline-20260920`（benchmark）。

- `2026-09-20T19:19:10Z` Run `paired-baseline-20260920` 记录为 `pass`；正确性为 `pass`。Original-source same-offset baseline after matched full warmup+3 mixed passes: 8/8 cold prompts paired candidate faster 205-343ms, mean TTFT candidate2349.06 vs baseline2633.90ms (-10.81%); baseline cold prefix hits 0/262808. Paired mixed median candidate540.40 vs baseline537.60 TPS (+0.52% noise), candidate mixed TTFT median1413 vs1263ms.

- `2026-09-20T19:20:20Z` 主控结论为 `PIVOTED`。All-step QLI CPU-max candidate passed >=192 per-rank parity checks and improved exact same cold prompts 8/8 by mean284.83ms (-10.81%), but paired mixed TPS +0.52% is noise and median mean TTFT worsened11.9%; refine to prefill-only rather than KEEP
