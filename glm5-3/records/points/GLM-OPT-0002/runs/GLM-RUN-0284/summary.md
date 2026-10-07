# Run284 — H13 full PD comparison INCONCLUSIVE; candidate off

Same Run283 workers and two retained target graph banks; no reload, new profile, parameter scan or recovery. Fixed ten fresh-salt requests complete: priming B short, A1/B1/A2/B2 warm+natural23 EOS, retained A short. Original SSE/exact IDs, stable all16 bank witnesses, native KV transfer, per-endpoint cache and terminal H6/H5/FULL health/idle independently pass. Raw measurement integrity is valid; the complete A/B/A/B comparison is inadmissible because B2 work differs.

| Complete phase | D wall ms | PD wall ms | P wall ms | TPOT ms | Drafts / draft tokens / accepted |
|---|---:|---:|---:|---:|---|
| A1 | 898.215137 | 1246.855076 | 348.639939 | 27.291881 | 11 / 11 / 11 |
| B1 | 938.197469 | 1290.625274 | 352.427805 | 25.295144 | 11 / 11 / 11 |
| A2 | 896.845528 | 1269.165869 | 372.320341 | 27.340859 | 11 / 11 / 11 |
| B2 | 901.330175 | 1353.569147 | 452.238972 | 27.384048 | 12 / 12 / 11 |

D savings A-B are -39.982332/-4.484647ms; A drift1.369609ms. PD savings are -43.770198/-84.403278ms; A drift22.310793ms. P contribution remains separately reported. The work-matched first pair has a 43.9282ms shorter first-to-last token arrival span but an approximately83.934ms later first arrival, yielding worse total D wall. This is a non-repeated steady-path signal, not KEEP. SSE grouping is not an engine-step trace; the additional B2 draft cannot be called an unnecessary trailing step.

All four complete phases have P local58/0 and external58/0; D local58/0 and external58/58. Fresh round namespace and raw cache deltas rule out inherited local prefix hits. D external100% is requested remote KV reuse on this tiny diagnostic, not formal93% hit-rate/SLA acceptance.

Terminal H13 mode0/observe0 transition9, H11off, H6/H5/targetFULL on, all16 healthy/idle and identical P249/D283 roots/workers. H13 does not enter the active research performance stack. Full API/formal80K input/600 output/93% prefix and declared KV-hit, SLA and final Current remain open. See measurement_reduced.json and independent_final_reduction.py; the earlier unexecuted formatting-only reducer draft is preserved separately.

H11/H12/H13 without repeatable complete-PD gain trigger Goal Review. Root accepts an offline pivot to the actual first-step critical path; no unchanged candidate rerun or favorable-pair search follows.
