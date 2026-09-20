# 执行日志

- `2026-09-20T18:35:51Z` Loop 已冻结。下一步：Stop tracer service, restore framework source, implement CPU-max QLI candidate with controlled parity verification, restart DP1TP8 and measure

- `2026-09-20T18:38:42Z` 为用例 `mixed_32k_1024_c12` 创建 Run `qli-cpu-candidate-20260920`（benchmark）。

- `2026-09-20T18:58:14Z` Run `qli-cpu-candidate-20260920` 记录为 `pass`；正确性为 `pass`。QLI CPU maxima candidate parity >=192 per rank on warm/cold, functional pass, 3x48/48 mixed passes median540.40 TPS (-0.60% vs frozen baseline); two cold candidate groups mean TTFT2338/2360ms require same-offset baseline
