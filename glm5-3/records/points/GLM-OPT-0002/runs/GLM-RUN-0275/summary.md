# Run275 — scoped K1 MTP graph matched complete PD

Completed 2026-10-07 10:03:53Z; source/spec immutable. Run273 actual correctness independently PASS; failed Run274 is retained separately (measurement-only P cache-query assertion). Same resident workers; H6/H5 and target FULL remain on; no reload, profile or scan. Fixed graph prime, A1/B1/A2/B2 each warm+natural23 complete, final warm: 10 fresh-salt requests.

| Complete request | D wall s | PD wall s | TPOT ms |
| --- | ---: | ---: | ---: |
| A1 MTP eager | 0.902043 | 1.268795 | 27.309137 |
| B1 MTP graph | 0.904356 | 1.393136 | 27.167849 |
| A2 MTP eager | 0.891747 | 1.364870 | 27.307136 |
| B2 MTP graph | 0.901369 | 1.281302 | 27.275200 |

Independent raw reduction verifies all ten SSE/event IDs/usage/EOS, recomputed SSE TPOT, scheduler deltas (complete11/11/11/11/0), fresh salts, P/D endpoint native counters and all16 successful KV transfers. All16 graph warm witnesses at transitions6/8/10 match persistent capture inputs and own outputs; eager7/9 and retained11 actual NONE. Complete witness snapshots unchanged from its warm; observer heavy D2H lies outside measured window. Same roots/workers/H6 cache maps/H5 policy/all16 health and idle, no recovery.

D savings −0.002313/−0.009623s versus A drift0.010296s. PD savings−0.124341/+0.083568s versus drift0.096075s; P contribution separately−0.122028/+0.093191s. Small TPOT changes do not yield repeated D or complete-PD gain. **INCONCLUSIVE; H11 off and not in research performance stack.** H6/H5 retained. Source graph support remains a proven scoped correctness candidate; do not repeat for a favorable result or call correctness timings performance.

Each complete P local58/0,external58/0; D local58/0,external58/58. D external100% represents requested P→D transfer; it is not prior-round local hit or formal93% cache evidence. No80K/600/93% or full API/SLA/Current claim. See measurement_reduced.json and FETCHED_TERMINAL_INPUTS.json for independent evidence/hashes; original raw remains local and on respective servers.
