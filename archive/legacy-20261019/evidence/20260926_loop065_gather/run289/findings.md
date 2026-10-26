# Run289 invalid before H003; repeated Host allocator failure

Same frozen service configuration as Run288. Again failed after model/drafter load during vLLM KV block-table CpuGpuBuffer pinned Host allocation; this time ranks2/3/7 reported aclrtMallocHostWithCfg 207001. No Extreme handoff, Graph capture, target-layer branch or clients. This repeats the environmental blocker and is not a H003 correctness or scheduling result. Launcher terminated, official stop released all eight NPUs, and borrowed source restored to SHA 27cbbf92a02f16301aad2a35091e258b8459dc77744bd8294f2f11a4c126004e.

Read-only process audit found 336 orphaned multiprocessing.spawn workers, all parented by the dedicated container sleep-infinity init (PID3604827) with no running VLLM service and idle NPUs; summed RSS ~230GiB. Targeted cleanup of only these orphan workers recovered Host MemAvailable from ~190GiB to ~905GiB. The remaining defunct PID entries have RSS0. This is a strong causal environment explanation, though the exact CANN pinned allocator threshold is not measured. Standalone 2MiB pinned Host allocations succeeded after cleanup. Run290 retries once after this concrete repair, with a pre-collective fixed-shape guard; no product setting changed. Evidence: environment/orphan_workers_before.json and environment/orphan_cleanup.json.


## Late artifact correction

The Run289 in-container launcher survived outer cleanup and, when Run290 later became healthy, submitted its own 48+12 requests to Run290's service. Its warmup48.json/bench12.json files were written at the same timestamps as Run290's. They are cross-run contaminated and must not be treated as Run289 success: Run289's own service had already failed before handoff. This explains Run290's 120 POST requests and invalidates its client/cohort performance comparison.
