# Challenger package: H1 closure and native PP packing question

Rules / product / Current / history: same RESET.md and ASTRA_PACKAGE.md. CurrentNone / no PERF_KEEP. Read H1_DECISION.md only as Sol interpretation; independently check code and raw.

Actual source snapshots: native_scheduler.py (running eligibility loop near504, waiting admission), native_async_scheduler.py (next eligible=current_step+PP), native_pp_utils.py (FIFO reserve/consume even empty), ascend_patch_spec_pp.py (draft+accepted publication), native_model_runner.py (empty execute still updates PP state), v1_engine_core.py and native_multiproc_executor.py (lazy FIFO RPC).

Actual decisive raw locally: ../../runs/GLM-RUN-0242/sol_raw/core.jsonl, rank8.jsonl, rank0.jsonl, request0..3.events.json. They are copies of stopped epoch authoritative server host_trace files; trace_summary.json records identities. sol_correlation.json is mechanical locators only. Review especially batches18/19 and early-publication empty commits; do not infer device bubbles from empty scheduling or get_output wait. Device profiler failed duePROFILING_MODE dynamic; native log is server Run242/diagnostic_native.log. No actual device coverage yet.

Single next question H2: could native scheduler preserve PP eligibility while keeping independent request cohorts at separate PP phases, and does current greedy packing cause a measurable stage bubble? Conditional minimal native-control candidate only; no patch. Current all4 same phase means every other slot empty, but batch size versus PP overlap tradeoff unknown. Standard PD/Full Replica architecture unchanged.

Suggested low-cost discrimination: existing CANN msprof --dynamic=on --pid (actual container namespace PIDs) supports duration; profiler binary/help personally verified. After original service restoration, one ≤15s diagnostic window with4x64 direct-native streaming, all16 workers device task/communication coverage and clocks/IDs. No restart or parameter scan. Native device timestamps must establish which useful target/draft stages overlap; evaluate whether a phase-balanced candidate is worth correctness+matched full E2E. No actual gain claim.

Please answer standard challenger outputs (Gap ranking, KEEP/STOP/PIVOT, top3, stop list, smallest discriminator, likely misread) and explicitly challenge both H1 closure scope and phase-packing inference. Do not produce a performance patch or authorize a larger E2E from host waits alone.
