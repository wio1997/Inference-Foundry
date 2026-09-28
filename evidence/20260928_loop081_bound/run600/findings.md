# Run600 — same-W₀ Host preparation envelope, read-only

This reducer reuses Run341's natural warm48→measured48 c12 diagnostic. It hashes and validates 42 existing inputs (all8 marks and `_model_forward` call files plus scheduler admission and phase summaries), groups 32 rank-cohort histories, checks forward call ordinals/shapes and the final 96-token handoff request ID/order on all8, and measures rank-local Host time from first `execute_entry` to `runtime_built`. No service or NPU was started. This is **the Run341 W₀**, not Run239, Run597 or formal Run99.

| Measured cohort | All-rank Host first-entry→last-build envelope (ms) | Rank-local Host forward union (ms, range) | Rank-local Host outside-forward remainder (ms, range) | Rank-local last execute-entry→runtime-built (ms, range) | Scheduler `num_output_tokens` at handoff, sum |
|---|---:|---:|---:|---:|---:|
| 5 | 2879.432 | 2010.120–2285.376 | 593.776–862.447 | 19.575–118.332 | 181 |
| 6 | 3513.892 | 2299.350–2560.955 | 948.502–1200.686 | 20.053–111.006 | 175 |
| 7 | 3506.801 | 2251.938–2553.024 | 952.941–1246.293 | 19.809–118.286 | 183 |
| 8 | 3213.699 | 1947.628–2233.262 | 977.849–1253.201 | 19.797–121.471 | 162 |

The forward union is a **Host interval union within one rank**, not a sum of different ranks' maxima. The remainder is arithmetic in that same Host clock interval; it includes original scheduler work, seed/proposer submission, Runtime build, waits and other code, plus device activity that can be asynchronous. It is not idle time, a direct removable gap, or a device critical-path lower/upper bound. The final execute-entry→runtime-built interval starts before the current 96-token Target input preparation and ends after Python `build_extreme_runtime`; it is not a certified seed/KV-ready interval.

`cached.num_output_tokens` comes from the scheduler's state at the final handoff entry; the installed scheduler can include pending async output placeholders in this field. It is **not proof of how many token IDs were generated, worker-published, server-emitted, or client-received** before Runtime. The four sums must not be subtracted from the Runtime's 1024-token output to infer necessary cycles, and no acceptance/cycle optimization is in scope.

Bound effect: this narrows **Current Product preparation Host accounting** for an existing W₀ and identifies a non-forward Host/async region worth splitting in the next measurement. It does not tighten finite Resource/Hardware, Scheduling/Execution or Product E2E endpoints. The most valuable next acquisition is still one same-new-W₀ identity and all8 ready/completion timeline from request admission and residual prefill through initial DSpark seed/KV/state, first Target, Runtime and actual SSE/client delivery. Preserve fixed DSpark7 semantics and a compact full Runtime basis. Observer overhead needs A0/A1 admission before time conclusions. Astra's independent acquisition review is `../astra_product_frontier_next_review.md`.

Formal Current: **571.681 tok/s**. Strict finite Bound endpoints and numeric Current→credible-limit gap remain **null**.
