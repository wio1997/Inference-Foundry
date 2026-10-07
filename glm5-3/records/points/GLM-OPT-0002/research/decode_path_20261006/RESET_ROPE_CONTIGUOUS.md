# H12: immutable RoPE cache layout

2026-10-07. Standard GLM-5.3 W8A8 PD, H6/H5 retained, target FULL, H11 off. Formal 80000 input / 600 output / 93% declared KV condition and complete API/SLA remain separate. No parameter scan or broad profile.

The previous submission hypothesis is closed as a large current starvation explanation: Run276's actual 16-rank target replay starts differ by 82–111us in steady steps. Selected non-wait hardware interval unions cover about 52.3ms of 55.4ms; the main CPU's 35ms synchronization overlaps device work. This does not prove every device task necessary or supply an overall lower bound. Old Run249 eager-host budgets cannot be transferred to current FULL.

**Hypothesis:** `_record_cos_and_sin_cache_interleaved` leaves immutable [1048576,64] BF16 lookup tables as half-width views with stride128. Each advanced row lookup materializes the entire table through Slice before selecting two rows. Make each table contiguous once during registration, before capture, to remove this repeated framework work without changing indexing or kernels.

**Evidence:** [root causal join](../../jobs/GLM53-CURRENT-FULL-PROFILE-OFFLINE-20261007/critical_path_reduced.json), exact task-ID/time matched CSV rows under that job's `rope_geometry/`, and [actual source](post_full_framework_sources/rotary_embedding.py). Rank0/13/15 × two steady periods give 24 joined Slice→CANN connection→Dequeue→Enqueue→ATen chains. Each has input [1048576,128], output [1048576,64], BF16 ND; four tasks/step materialize 512MiB. Period3 inclusive durations are 1.333594/1.325234/1.323494ms. These are observed work, not measured savings. Source-line attribution uses the unique two-unsqueeze pattern; the profile has no Python line stack.

Production scope is [two initialization copies](rope_contiguous_candidate/rope_contiguous.patch). Unchanged helper indexing preserves negative/empty/multidimensional/boolean and error behavior; copies into target persistent `_cos_mla/_sin_mla` are unchanged. Original repeat storage and new dense pair each have 256MiB live payload; initialization peak can add 256MiB and allocator reservation can differ. The native-index_select prototype is PARKED and will not be combined.

[CPU oracle](rope_contiguous_candidate/CPU_result.json) passes504 cases for actual registration, indexing/layout, persistent output identity, repeated-registration and payload accounting. [Independent review](ROPE_CONTIGUOUS_CACHE_REVIEW.md) and [Run277 review](RUN277_ROPE_LAYOUT_REVIEW.md) support this scope. Root accepts the graph lifetime conclusion: table lookup is in metadata construction outside the model wrapper; observed Slice tasks have MODEL_ID4294967295. Target graph binds distinct persistent outputs. MTP graph remains off. A diagnostic observer prebuilds both tables before any capture, selects only lookup globals, and retains unchanged output copies. Both arms pay the same observer branch; heavy witness work is confined to warm requests.

**Distinguishing evidence:** Run277 allows one owned D reload and four fixed small PD correctness requests, five including one failure recovery. No P restart, profiling, scan or extra workload. Verify actual all16 old/dense layouts, unchanged target FULL and H6/H5, exact IDs/EOS and native all16 transfer. After independent raw correctness reduction, freeze a same-worker A1/B1/A2/B2 full natural23 PD comparison, fresh local cache salts and per-endpoint counters, warm witness outside measured requests, exact aggregate acceptance work.

| Result | Decision |
| --- | --- |
| Correctness, actual layouts or target/stack lifetime fails | Preserve raw, one declared recovery, H12 off; no performance run |
| Correctness passes | Same-worker matched PD comparison; no additional profile needed merely to repeat the root evidence |
| Both matched pairs reduce D wall beyond A drift, PD wall beyond A drift and TPOT, with exact work/cache/outputs and valid terminal state | Add H12 to active research stack; formal Current remains open |
| Gain is absent, inconsistent, confounded or below drift | H12 off/INCONCLUSIVE; no favorable-workload replay; continue from the next source-proven current-framework waste |

The largest realized removable gap is still unknown. This is the largest clearly source-attributed redundant whole-table operation in the supplied post-FULL boundary, not a claim to explain every millisecond or reach SLA.
