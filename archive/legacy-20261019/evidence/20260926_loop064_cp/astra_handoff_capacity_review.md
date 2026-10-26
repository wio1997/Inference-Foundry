# Astra High: incomplete handoff versus max-num-seqs

Read-only review during Run282. No running source/service/launcher was changed and no NPU work was started. This review does not decide KEEP.

## Facts

scripts/serve.sh uses max-num-seqs16, max-num-batched-tokens8192 and async scheduling; bench.py holds its concurrency12 semaphore through the complete HTTP response stream. The borrowed runner handoff needs use_spec_decode, input_batch.num_reqs==12, total_num_scheduled_tokens==96, non-null speculative metadata and a new request tuple. The num_reqs at this point is the real persistent input batch count, not a padded Graph bucket count.

The parent GPUModelRunner._update_states removes finished requests and requests not scheduled in the current step from input_batch (gpu_model_runner.py:1180-1230). The scheduler patch handles stopped requests, frees them and removes stopped_running_reqs before returning EngineCore outputs (patch_kv_delivery_preemption.py:1045-1095). Thus client request completion normally follows server-side completion bookkeeping; a semaphore slot becoming free does not by itself imply that a still-running old request remains beside a new one. FixedCohortServing parking happens inside an already-entered Runtime; it does not itself create extra scheduler requests or explain failure to enter that Runtime.

Scheduler max_num_running_reqs is a capacity limit. Setting it to16 does not force 12 requests to become16, nor guarantee a 12-request uniform decode step. Async scheduling explicitly skips running requests with pending output placeholders whose output limit is already satisfied (scheduler.py:483-504), and token budgets/prefill can produce different scheduled subsets and token counts. This supports admission/shape trajectory as a hypothesis, not a proven Run281 cause.

Run281 has only16 rank-cohort artifacts, corresponding to two complete Extreme cohorts. All60 client outputs being length-correct does not fill the other36 requests' execution-coverage gap. CP branch logs from capture and client TPS cannot establish that those other requests used Extreme.

## Judgment and confidence

The specific explanation “max16 admits four parked old requests alongside c12, so handoff misses12” is weak without an observed >12 input batch or stale-id evidence. Source behavior weighs against treating it as the default explanation. More plausible but still unproven: fewer than12 requests are simultaneously ready for uniform8-token decode, so the exact96 gate is never reached, or another predicate rejects the step. The existing gates must be measured before selecting a fix.

Changing max-num-seqs to12 is mathematically compatible with a c12 product capacity, but is not automatically performance-neutral or a coverage fix. It changes scheduler admission, max_num_reqs-sized metadata/DSpark/communication buffers, Graph bucket generation/capture memory and potentially Prefill scheduling/resource budgets. It may change the workload trajectory and Graph/allocator behavior even when peak client concurrency remains12. Existing Run99 comparisons do not isolate this configuration change. Do not combine it with CP overlap changes or interpret an improved handoff rate alone as proof that extra old requests caused the failure.

## Highest-information next action

Prioritize a bounded handoff predicate/shape audit over additional mixed-path Product TPS. Record per step/cohort: actual input request count, scheduled token total and per-request histogram, use_spec_decode, metadata presence, request-tuple previously-served predicate, prompt/decode state and token progress. Summarize reason counts and a few representative transitions after the run; avoid high-volume synchronous hot-path logging. If stale finished ids or >12 scheduled requests appear, retain scheduler finished/scheduled ids and placeholder counts for those specific transitions. Distinguish totals96 from actual uniform8-per-request geometry; the latter is what the fixed Runtime ABI needs.

Correlate those summaries with unique request ids and final Runtime coverage, rather than only counting output files. A fixed c12 diagnostic with confirmed handoff can still provide CP Graph/numerical evidence; incomplete warmup48 coverage prevents a formal all-Extreme E2E comparison.

If evidence then implicates admission capacity, compare unpatched CP/A0 with max16 versus max12 only, preserving dataset/client/protocol, and obtain a contemporary reverse/control repeat as needed. Require complete five-cohort coverage for warmup48+bench12, exact output/state/Graph gates, comparable prefix-cache state and recorded Prefill shapes. Separate any capacity-induced Graph/buffer effect from admission effect. Full formal repeat is needed only after coverage and correctness justify promotion.

Current root cause remains unknown. The priority is gate/shape evidence, not a presumption that max12 solves it. No numerical hardware or scheduling ceiling follows from this coverage failure.
