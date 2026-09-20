# DeepSeek Extreme P0 — Project State

Updated: 2026-09-20 16:06 UTC. Evidence maturity: **E2** for warm-cache DP1/TP8 service; prefill and critical-path mechanism remain unmeasured.

## Fixed contract

- DeepSeek V4 Flash W4A8 at `/data/yxy/DeepSeek-V4-Flash-0731-w4a8`; one host with 8 × Ascend 910B3; DP=1, TP=8; correct complete prefill/decode service.
- Reference: privileged `dsv4ab` container, vLLM 0.26.0, vLLM-Ascend 0.26.0rc1; source commits and image ID in `evidence/20260920_baseline/freeze.txt`.
- Primary measured workload: frozen 48-request dataset hash in `freeze.txt`, 32K input, 1024 output, concurrency 12, temperature 0, ignore EOS, 90% repeated prefix. **Warm-cache protocol:** run one full dataset pass before measuring; hold service/code/params unchanged. First cold pass is not comparable with steady repeats.
- Correctness gate for candidates: complete 4-prompt 128-token outputs plus exact short functional checks; numerical equivalence requires a separate stable comparison before KEEP. Exact long reasoning hashes are nondeterministic and invalid as a gate.
- Performance gate: same client and cache protocol; measure TTFT, TPOT, output TPS and noise. A KEEP needs gain larger than measured run spread and no material regression; otherwise INCONCLUSIVE/REJECT.

## Current baseline

- Baseline service was ready at 15:47:25 UTC, then stopped after evidence freeze. A diagnostic restart with `PROFILING_MODE=dynamic` began at ~16:06 UTC; it is loading on port 8080. Do not launch another service over it. Log: `logs/serve_dsv4f-w4a8_8npu_dp1tp8_mlen1M_nomooncake_P0-PROFILE-20260920.log`.
- Three corrected warm-cache runs: output TPS **543.65 / 523.15 / 545.85**, median **543.65**; TPOT mean **19.45 / 20.08 / 19.73 ms**, median **19.73 ms**; TTFT mean **1.12 / 1.37 / 1.33 s**, median **1.33 s**. Each run 48/48, 49,152 output tokens. TTFT variation is material; confidence in small gains is low.
- Prefix cache hit rate 99.75% during corrected runs. Speculative accepted tokens/draft 2.94 for first two; acceptance 42.0%, third 41.2%. NPU AICore utilization median 77%, p90 83% during first two runs.
- One earlier 48-request run is INVALID for TTFT/TPOT because the client omitted `reasoning` stream deltas. It is preserved in `evidence/20260920_baseline/bench48_invalid_1.json`; parser fixed at `b4baa57` and 2/2 short parser proof saved.
- Four deterministic long-prompt golden outputs and two short smoke outputs passed; hashes and raw outputs are committed under `evidence/20260920_baseline/`.

## Next loop

1. Loop 001 is pivoted after the invalid parser run; Loop 002 accepts the corrected baseline. Preserve both decisions and evidence.
2. Two disjoint cold 32K×128 c1 groups finished: TTFT 2662.7/2627.2 ms, second group 0/131404 prefix hits. System msprof captured HBM/HCCS/AICore only; application profile restart now loading. Profile/diagnose current DP1/TP8 **without treating older DP2/TP4 results as current truth**. Separate warm decode and cold prefill, inspect draft/target spans, HCCL, kernels and host; use profiler only outside official benchmark windows.
3. Rank the largest eliminable Gap. Historical evidence points to DSpark efficiency and TP communication/compute contention as candidates, but they are unproven for this run.
4. Freeze one mechanism hypothesis, minimal candidate, correctness, same-protocol A/B, KEEP/REJECT, update all four state files and commit. Continue while a testable high-value gap remains.

## Resume

Run `git log -3 --oneline` and `python3 scripts/taskctl.py resume --task-dir tasks/deepseek-extreme-p0`. Read only this file, `PERFORMANCE_MAP.md`, `ACHIEVABLE_BOUND.md`, `RESULTS.md`, and paths in the resume pack. Check `npu-smi info`, `docker ps`, port 8080, and source Git status before work. Framework trees are shared bind mounts; do not edit them in place during a live baseline. Current raw service log is in `logs/` (ignored); committed measurement evidence is in `evidence/`.

## Update 2026-09-20 16:30 UTC — Loop 003 pivoted after diagnostic

The diagnostic service with `PROFILING_MODE=dynamic` became healthy at 16:17:54 UTC and remains on port 8080. TP0 application profiles for four cold 32K→128 c1 and four exact-repeat warm 128-token c4 requests both passed 4/4. Raw traces remain on host; `evidence/20260920_diagnostic/app_profile_index.json` records paths and SHA256. Reproduce summaries with `scripts/analyze_profile.py`.

Cold TP0 event span 18.583 s: compute/copy union 10.829 s, HCCL union 6.386 s, all-op union 16.510 s. Warm TP0 event span 3.401 s: compute/copy union 1.688 s, HCCL union 1.165 s, all-op union 2.799 s. These are rank-local profiled activity spans, not E2E causal attribution. Communication is nearly serial with compute in the warm sample. FlashComm1 reduce-scatter/all-gather is the next falsifiable candidate.

Next: make a single-variable FlashComm1-off A/B while preserving DP1/TP8, W4A8, DSpark 7, graph mode and workload. Restart takes ~12 min. Check golden4 exact outputs, then three corrected full 48-request warm passes after a full-dataset warmup. KEEP requires robust TPS gain beyond the 4.3% baseline spread and no material TTFT/TPOT regression. Restore FlashComm1 if rejected. Do not launch another service over the current one.

## Update 2026-09-20 16:39 UTC — Loop 004 pivot

The isolated FlashComm1-off service failed during worker initialization before weights loaded or any request ran. `enable_dsa_cp=true` requires sequence parallelism; both must be disabled to test the non-SP path. The waiting benchmark runner was stopped, port 8080 is down, and all eight NPUs returned to ~3.4 GB idle HBM. The next loop will test the explicitly combined FlashComm1-off/DSA-CP-off path. Evidence: `evidence/20260920_flashcomm_off/startup_failure.txt`; full startup log is under ignored `logs/` with committed SHA256.

## Live checkpoint 2026-09-20 16:42 UTC — Loop 005 running

After Loop 004 pivot, launched valid coupled-path candidate with `FLASHCOMM1_ENABLED=false`, `DSA_CP_ENABLED=false`, DP1/TP8 and all other frozen serving settings. The API PID inside `dsv4ab` is 722334; service log: `logs/serve_dsv4f-w4a8_8npu_dp1tp8_mlen1M_nomooncake_P0-SP-DSACP-OFF-20260920.log`. A detached `scripts/run_flashcomm_candidate.sh` waits for readiness, checks golden4, warms the full dataset, then performs three 48-request passes into `evidence/20260920_sp_dsa_off/`. Check `runner.log` and `times.txt` there before restarting anything. At this checkpoint the service is still initializing and no result exists.

## Update 2026-09-20 16:55 UTC — Loop 005 pivot

The coupled SP/DSA-CP-off service is healthy on port 8080. The benchmark runner stopped at its strict golden4 gate; no full benchmark ran. All four long-prompt outputs differed from the baseline, but repeating golden4 on the same unchanged candidate also differed 4/4. Thus exact reasoning-text equality is not a valid correctness gate for this setup. Keep the service running; next loop establishes a functional check and measures E2E, with numerical equivalence still required before KEEP. Evidence: `evidence/20260920_sp_dsa_off/golden_check.json`, `golden4.json`, `golden4_repeat.json`.

## Update 2026-09-20 17:06 UTC — Loop 006 REJECT

The current healthy service is the SP/DSA-CP-off candidate (API PID 722334). Functional gate passed: four full 128-token long outputs and exact short answers 42, OK, 4. Full-dataset warmup then three 48×32K→1024 c12 passes yielded 524.41, 541.05, 534.88 tok/s (median 534.88), versus frozen baseline median 543.65. The -1.61% change is inside baseline noise; TPOT median 19.87 vs 19.73 ms and TTFT median 1.435 vs 1.332 s. REJECT this coupled path. Numerical equivalence was not established and is unnecessary for a rejected performance candidate. Evidence: `evidence/20260920_sp_dsa_off_perf/`. Next isolate `enable_dsa_cp=false` with FlashComm1 restored to true; compare under the same protocol. Stop only PID 722334 before restart.

## Live checkpoint 2026-09-20 17:08 UTC — Loop 007 running

The rejected coupled service was stopped and all eight NPUs returned to idle HBM. A new service is starting in `dsv4ab`, API PID 732566, with FlashComm1 true and only DSA CP false; all other serving settings remain frozen. Log: `logs/serve_dsv4f-w4a8_8npu_dp1tp8_mlen1M_nomooncake_P0-DSACP-OFF-20260920.log`. Detached runner `scripts/run_flashcomm_candidate.sh` is waiting for readiness; output directory `evidence/20260920_dsa_cp_off/`, runner PID 1955064. Check those files before restarting. No Loop 007 result exists yet.

## Update 2026-09-20 17:27 UTC — Loop 007 REJECT

DSA CP off alone, FlashComm1 on, passed the functional gate and three full warmed 48/48 runs. Output TPS 535.16, 518.10, 514.23 (median 518.10), down 4.70% from baseline 543.65. TTFT median 1.533 s versus 1.332 s; TPOT median 20.48 versus 19.73 ms. REJECT. The current service on port 8080 is this rejected candidate, API PID 732566; stop only that PID before restart. Next restore both baseline flags true and measure target/draft/host and cold-prefill critical path. No candidate has earned KEEP yet.

## Live checkpoint 2026-09-20 17:29 UTC — Loop 008 scope profiling

The rejected DSA-CP-off service was stopped; 8 NPUs returned to ~3.4 GB idle HBM. A baseline-flag service (FlashComm1=true, DSA CP=true, DP1/TP8, same model/source) is starting with torch-NPU profiler enabled for diagnostics only. API PID 739569, log `logs/serve_dsv4f-w4a8_8npu_dp1tp8_mlen1M_nomooncake_P0-SCOPE-20260920.log`. Detached `scripts/profile_scopes.py` PID 1976075 waits for readiness, warms 4 short prompts, then profiles warm 4×128 c4 and cold 2×32K→128 c1 in one scope window. Outputs: `evidence/20260920_scope_profile/run1/`; large raw profiler files are ignored under `raw/`. Check runner and service logs before restarting. No profile result yet.

## Update 2026-09-20 17:59 UTC — Loop 008 diagnostic accepted

Baseline flags restored and healthy on port 8080, with torch-NPU profiler enabled only for diagnosis (API PID 739569). The profile runner completed warm 4×128 c4 (4/4) then cold distinct 2×32K→128 c1 (2/2), and stopped profiling. Raw TP0 export and SHA256 index are in `evidence/20260920_scope_profile/run1/raw_index.json`; large raw files remain ignored. The profiled warm request window took 5.473 s versus 3.633 s in a separate unprofiled msprof window, so do not use profiled times as an official performance baseline.

TP0 warm device union 4.094 s, communication 2.269 s, compute/copy 1.876 s. CPU `prepare input` scope sum 1.320 s, `draft_token` 2.237 s; `aten::item` children within prepare sum 0.436 s across 1364 calls. These CPU scope durations include nested work and synchronization. Cold window 11.178 s, device union 9.481 s, communication 4.640 s, compute/copy 5.203 s. No speedup is claimed.

Next Loop 009: locate the repeated device-to-host scalar synchronization callsite with stack or targeted instrumentation, prove whether it blocks TP0 device progress, then make one minimal source change only if semantics are preserved. Compare unprofiled matched warm/cold workloads and run functional plus numerical correctness checks before KEEP. Current source trees are still clean.

## Live checkpoint 2026-09-20 18:02 UTC — Loop 009 stack profile loading

Loop 008 diagnostic committed at `2e0d4fa`; Loop 009 created at `9f6459f`. Baseline-flag profile service with `torch_profiler_with_stack=true` is starting in `dsv4ab`, API PID 747236, log `logs/serve_dsv4f-w4a8_8npu_dp1tp8_mlen1M_nomooncake_STACK-20260920.log`. Previous PID 739569 was stopped and all 8 NPUs were idle (~3.4 GB/card) before restart. Detached `scripts/profile_scopes.py --warm-only` runner PID 2003088 waits for readiness and will capture exact-repeat 4×128 c4; output `evidence/20260920_stack_profile/run1/`. Raw data under `raw/` is ignored. Check runner and service log; do not launch another service. No callsite conclusion or optimization result exists yet.

## Update 2026-09-20 18:17 UTC — Loop 009 pivot after profiler crash

The short warm stack profile completed 4/4 inference requests, but torch-NPU `with_modules=true` caused worker segfaults during `/stop_profile` (HTTP 500). The API process exited. Offline parsing recovered device kernels but lost FRAMEWORK operator events; no `aten::item` call stacks were exported. The first runner attempt failed only because a relative output path was resolved inside the container; `scripts/profile_scopes.py` now resolves it, and the retry reached profiler stop. Evidence: `evidence/20260920_stack_profile/run1/phase_times.json`, `stop_failure_excerpt.txt`, `service_log_sha256.txt`, runner logs. All 8 NPUs returned to idle ~3.4 GB/card. Next use targeted, reversible source instrumentation to identify item callsites without torch profiler. Do not treat the crash as a model correctness result.

## Live checkpoint 2026-09-20 18:19 UTC — Loop 010 item tracer loading

Stack profiler failure/pivot committed at `7dce14a`; Loop 010 created at `8fd9b07`. `vllm-ascend` source tree has one deliberate, uncommitted diagnostic patch to `vllm_ascend/worker/model_runner_v1.py`, frozen in root repo `patches/loop010_item_trace.patch` (commit `68f0ac5`). It wraps Python `Tensor.item` only during `prepare input` when `TRACE_PREPARE_ITEMS=1`, logging callsite, device, count, cumulative/max time every 8 steps; restore with `git apply -R` from the framework root after the run. The earlier stack profiler service crashed and all eight NPUs returned idle before this restart. Baseline flags, DP1/TP8, no profiler service is starting in `dsv4ab`, API PID 754292, log `logs/serve_dsv4f-w4a8_8npu_dp1tp8_mlen1M_nomooncake_ITEMTRACE-20260920.log`. Detached runner PID 2022323 waits for health, then runs short warmup and measured 4×128 c4 sets to `evidence/20260920_item_trace/run1/`. Do not start a second service. No tracer result yet.
