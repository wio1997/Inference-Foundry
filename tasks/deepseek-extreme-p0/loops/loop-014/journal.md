# 执行日志

- `2026-09-20T20:28:19Z` Loop 已冻结。下一步：Inspect model_runner_v1.py prepare-input call chain and trace operator self-times; locate one repeated metadata operation with a measurable upper bound before editing.

- `2026-09-20T20:29:36Z` 为用例 `mixed_32k_1024_c12` 创建 Run `decode-host-source-audit-20260920`（design-check）。

- `2026-09-20T20:29:36Z` Run `decode-host-source-audit-20260920` 记录为 `pass`；正确性为 `not-applicable`。Per 170 TP0 decode steps, prepare input median21.42ms and draft median52.89ms. Typical prepare nested item only0.70ms; scope self median9.57ms. Source path includes _update_states, _prepare_inputs, Mamba preprocessing, _build_attention_metadata, _preprocess. No single call attribution yet; next use env-gated perf_counter spans on these boundaries with unprofiled c12 sample before proposing a code optimization.

- `2026-09-20T20:32:27Z` 为用例 `mixed_32k_1024_c12` 创建 Run `prepare-stage-trace-20260920`（profile）。

- `2026-09-20T20:46:24Z` Run `prepare-stage-trace-20260920` 记录为 `pass`；正确性为 `pass`。Functional check passed; 48/48 warmup and 12/12 c12×512 sample. 157 pure decode steps per rank. TP0 median prepare19.55ms: state update0.78, input assembly4.81, dispatch/Mamba0.19, compress+attention metadata12.99, preprocess0.31. Eight-rank compress+attention median10.80-14.17ms; largest stage but includes several operations, not proven removable. Short diagnostic TPS471.1 is not official baseline.

- `2026-09-20T20:49:56Z` 为用例 `mixed_32k_1024_c12` 创建 Run `attention-builder-trace-20260920`（profile）。
