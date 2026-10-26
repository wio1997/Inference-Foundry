# Run343 — original bulk with an EngineCore pre-submission fence

## Question and preflight

The current service enables `--async-scheduling`. `VllmConfig.max_concurrent_batches` is 2 for ModelRunner V1/PP1; `EngineCore.step_with_batch_queue` can `schedule()` and enqueue a successor worker `execute_model` before consuming the prior output. Scheduler advances computed/in-flight counts and async placeholders when scheduling, while the worker executes queued RPCs in order. A successor can therefore call ModelRunner `_update_states` before EngineCore handles a proposed partial return. A future segmented publication cannot safely cancel that RPC afterward.

Historical search `historical_search.json` at pinned old commit `db3beef` retrieves R36's async SchedulerOutput deepcopy case, whose large synthetic CPU cost rarely hit the warmed service and gave no stable c12 E2E gain. It provides a measurement caution, not a verdict on this queue fence. R06/R32 concern old DP2 prefill layout/padding, not Current DP1×TP8 continuation.

## Intervention and result

A reversible SHA-anchored patch to `core.py` conservatively fences a `SchedulerOutput` with `total_num_scheduled_tokens==96` **before** its execute RPC is submitted. While it is pending, `step_with_batch_queue` consumes existing FIFO outputs but cannot call a new `schedule()`; it releases the fence only after that output is consumed and `update_from_output` returns. A predecessor may already be queued. This is not a global queue-capacity-one mode. The async flag, original `FixedCohortServing` bulk path, Target, Draft and Graph remain unchanged. Patcher preview/installation/restoration and source hashes are recorded; borrowed `core.py` was restored to `3ae1381a6af841e21058c825702382dc66faae45c950ac5acb8495d2d3d05aad`.

One original frozen formal session completed 48×32K→1024 c12 warmup plus three 48-request repeats, all with 48/48 successes. Eight-rank Runtime reports passed **128/128** over 16 cohorts and all target Graph modes were FULL. EngineCore logged 16 acquire/release pairs, each candidate scheduled96, with one predecessor queued at acquisition, zero queued after candidate consumption, and `bulk=True` on every release. Thus all 16 observed handoffs were covered. The patch's single control path cannot call `schedule()` while its pending candidate exists; worker entry/exit sequence IDs were not separately logged.

| Formal repeat | fence-only TPS | Run99 prior-session TPS |
|---:|---:|---:|
| 1 | 565.154 | 612.962 |
| 2 | 595.734 | 567.573 |
| 3 | 575.709 | 571.681 |
| median | **575.709** | **571.681** |

The cross-session median difference is +0.705%, within the broad observed repeat variation. It does **not** establish a speedup or zero fence cost; no same-session paired A/B was run. The three measured runs used 1232/1204/1237 Target cycles; sums of their cohort maximum-rank Runtime walls were 69.998/70.163/69.859 s. The corresponding client-minus-Runtime envelopes were 16.973/12.344/15.518 s, descriptive cross-clock residuals rather than identified removable time. The output parser passed its formal correctness gates. An APIServer `EngineDeadError` occurred at 22:15:33 UTC after `summary.json` was written at 22:15:32.96 during service stop; it did not affect the completed benchmark. Stop returned each card to ~3.4 GB HBM and no service remained.

## Decision

**KEEP the pre-submission fence as a validated enabling mechanism for the next private publication gate**, not as a Product performance optimization. Run342's legal chain still lacks early output publication, real arrival before old-nine finish, Scheduler partial accounting, KV retention, and continuation. The next gate must operate under this fence and demonstrate these semantics before any b3 refill/seed or formal TPS claim. Its first version should queue newly arrived requests without running them and should measure old-nine pause/publication overhead. Numeric Hardware/Resource, Scheduling-aware attainable and Product E2E Bounds remain unknown; formal Current stays Run99 571.681 tok/s until a causally comparable intervention passes repeated frozen E2E.

Evidence: `analysis.json`, `summary.json`, `warmup.json`, `bench48_1..3.json`, `runtime/rank*_cohort*.json`, `patch_install.json`, `patch_restore.json`, `driver.log`, service log named in `loop074_run343_analyze.py`. Code: `scripts/loop074_run343_fence_patch.py`, `scripts/run_loop074_fence_run343.sh`.
