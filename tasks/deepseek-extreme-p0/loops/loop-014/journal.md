# 执行日志

- `2026-09-20T20:28:19Z` Loop 已冻结。下一步：Inspect model_runner_v1.py prepare-input call chain and trace operator self-times; locate one repeated metadata operation with a measurable upper bound before editing.

- `2026-09-20T20:29:36Z` 为用例 `mixed_32k_1024_c12` 创建 Run `decode-host-source-audit-20260920`（design-check）。

- `2026-09-20T20:29:36Z` Run `decode-host-source-audit-20260920` 记录为 `pass`；正确性为 `not-applicable`。Per 170 TP0 decode steps, prepare input median21.42ms and draft median52.89ms. Typical prepare nested item only0.70ms; scope self median9.57ms. Source path includes _update_states, _prepare_inputs, Mamba preprocessing, _build_attention_metadata, _preprocess. No single call attribution yet; next use env-gated perf_counter spans on these boundaries with unprofiled c12 sample before proposing a code optimization.
