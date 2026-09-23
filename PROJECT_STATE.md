# DeepSeek Extreme P0 — Project State

Updated: 2026-09-23 03:04 UTC. Evidence maturity: **E2** for both Stock and the first full-serving Extreme Runtime under the frozen warm-cache DP1/TP8 protocol.

> Latest checkpoint (2026-09-23 03:04 UTC): Loop034 completed the first formal
> Extreme E2E A/B. All three measured runs passed 48/48 at exactly 1024 output
> tokens; median output TPS is 217.342 versus Stock 543.655 (-60.022%). The
> fixed serving boundary is correct, but sustained runtime decode is not yet
> competitive. Next profile the complete 1024-cycle runtime-owned chain.

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

1. Preserve Loop034 as the serving-correctness and formal performance point;
   do not repeat its warmup or three-run A/B without a new candidate.
2. Profile the complete runtime-owned 1024-cycle chain, not generic vLLM.
   Attribute the 53.373 s cohort wall across target replay, proposer, TP/EP
   communication, acceptance/state, output staging and Host launch/sync gaps.
3. Choose the largest causally removable region from that profile. Device
   residence, graph/persistent execution, communication overlap and
   cross-operator fusion/SuperKernel remain candidates, not assumptions.
4. Validate continuous correctness first, then repeat the frozen E2E protocol
   only for a structurally meaningful candidate.

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

## Update 2026-09-20 18:35 UTC — Loop 010 diagnostic accepted

The reversible tracer service completed two warm 4×128 c4 sets (4/4 each) and one cold distinct 2×32K→128 c1 set (2/2). `Tensor.item` within `prepare input` is concentrated at `vllm_ascend/attention/context_parallel/dsa_cp.py:1008`: `seq_lens_q.max().item()` on each NPU rank. In the same builder, `max_local_query_len` and `max_local_seq_lens` are already computed from CPU local metadata derived by the same `_build_local_token_metadata` formula. A second NPU `.item()` at line 1009 is cheap only after line 1008 synchronizes. TP0 cumulative snapshot deltas approximate 210 ms / 54 calls in the measured warm set and 1268 ms / 70 calls during the later cold set; the rank values differ. These are synchronizing call durations, not proven removable E2E time. Source is currently diagnostic-patched and service is healthy (API PID 754292). Next: stop only this service, restore tracer patch, implement CPU-max substitution with a runtime parity check for warm/cold and then matched unprofiled A/B before KEEP.

## Live checkpoint 2026-09-20 18:39 UTC — Loop 011 QLI CPU maxima candidate loading

Loop 010 accepted at `5984d4f`, Loop 011 started at `c8e73d4`. Tracer service PID 754292 was stopped, all eight NPUs returned idle, and diagnostic patch was reversed; the framework source is now modified only at `vllm_ascend/attention/context_parallel/dsa_cp.py` by `patches/loop011_qli_cpu_max.patch` (frozen at `3853228`). Candidate passes existing CPU local query/key maxima to QLI metadata; optional parity verification recomputes old NPU maxima only while `evidence/20260920_qli_cpu_max/verify.flag` exists. Baseline-flag no-profiler service is loading in `dsv4ab`, API PID 760651, log `logs/serve_dsv4f-w4a8_8npu_dp1tp8_mlen1M_nomooncake_QLICPU-20260920.log`. Detached runner PID 2040414 executes `scripts/run_qli_candidate.sh`: wait for health, verify warm twice + cold2 with assertions, remove flag, functional/golden checks, full-dataset warmup, three 48×32K→1024 c12 passes. Outputs under `evidence/20260920_qli_cpu_max/`; check `runner_top.log`, `verify/runner.log`, and `perf/runner.log` before restarting. No candidate result exists yet. Source is shared bind mount; preserve patch or reverse it explicitly.

## Live checkpoint 2026-09-20 18:58 UTC — Loop 011 mixed result, cold paired test pending

The QLI CPU-max candidate is healthy (API PID 760651), parity verified on warm twice and cold2: all eight ranks logged at least 192 successful CPU/NPU max comparisons with no mismatch, then verification flag removed. Functional gate passed. Full-dataset warmup and three 48/48 frozen mixed passes finished: output TPS 556.93 / 524.73 / 540.40, median 540.40 versus baseline 543.65 (-0.60%); no robust mixed gain. Two distinct cold unverified candidate groups (offset24-27 and28-31, 4×32K→128 c1 each) had mean TTFT 2338.29 and2359.84 ms versus older different-group baseline2627–2663 ms. This is only a cold-path signal until same-offset baseline is measured. Next stop only PID 760651, reverse `patches/loop011_qli_cpu_max.patch`, restart baseline flags/source with no profiler, and run the exact offset24-31 cold groups before verdict. Candidate evidence under `evidence/20260920_qli_cpu_max/`; no KEEP yet.

## Live checkpoint 2026-09-20 19:00 UTC — paired original-source baseline loading

Candidate data and four state files committed at `1582a38`. Candidate API PID760651 was stopped, all NPUs returned to idle, and `patches/loop011_qli_cpu_max.patch` was reversed; `git -C /data/wio/vllm_ascend_26/framework/vllm-ascend status --short` was empty. Original-source baseline-flag no-profiler service is starting in `dsv4ab`, API PID767712, log `logs/serve_dsv4f-w4a8_8npu_dp1tp8_mlen1M_nomooncake_QLIPAIRBASE-20260920.log`. Detached runner PID2059362 executes `scripts/run_qli_pair_baseline.sh`: wait for readiness, functional/golden, full warmup and three 48-request passes, then exact cold offset24-27 and28-31 groups at c1. Outputs under `evidence/20260920_qli_pair_baseline/`. Check `runner_top.log`, `perf/runner.log`, and `cold/times.txt`. Do not launch another service. Loop011 verdict pending paired data; do not mark candidate KEEP yet.

## Update 2026-09-20 19:20 UTC — Loop 011 paired result, pivot to prefill-only

Original-source paired service is healthy on port8080 (API PID767712); framework source trees are clean. After the same full-dataset warmup and three 48/48 mixed passes, original-source median output TPS537.60 vs QLI CPU-max candidate540.40 (+0.52%, noise); original-source median mean TTFT1262.74 vs candidate1413.44 ms (candidate +11.9%, uncertain cause). Same-offset cold prompts24-31 (8×32851 tokens, c1, 128 output) showed original-source mean TTFT2633.90 vs candidate2349.06 ms; all 8 candidate requests improved by205–343 ms, mean -284.83 ms (-10.81%). Original-source prefix cache hits remained 0 across262808 cold queried tokens. Runtime CPU/NPU QLI max parity previously passed >=192 per rank. This supports a real cold-path opportunity but does not justify the all-step candidate KEEP because of mixed TTFT uncertainty. Loop011 pivots to a prefill-only QLI CPU-max intervention, leaving pure decode unchanged, then repeats paired cold and mixed checks. Evidence `evidence/20260920_qli_pair_baseline/comparison.json`, cold metrics and all runs. Do not stop current service until next loop is ready.

## Live checkpoint 2026-09-20 19:23 UTC — Loop 012 original cold reference extended

Loop011 paired pivot committed at `6f47a23`; Loop012 started at `c06222a`. Prefill-only candidate patch is frozen at `patches/loop012_qli_prefill_only.patch` in `e1d3f76`; it is applied to `dsa_cp.py` on disk but the currently running original-source service PID767712 loaded before the patch, so it still executes the original code. Two more fresh original-source cold groups at offsets40-43 and44-47 each passed4/4 with mean TTFT2615.16/2621.18 ms, and prefix cache hits0/262808 queried tokens. Evidence `evidence/20260920_qli_pair_baseline/cold_more/`. Next stop only PID767712, confirm8 NPUs idle, start prefill-only candidate, run full warmup+3 mixed passes and cold offsets24/28/40/44, then compare. The framework tree now has the intended uncommitted prefill-only patch; do not reverse it before candidate launch.

## Live checkpoint 2026-09-20 19:26 UTC — Loop 012 prefill-only candidate loading

Original-source paired service PID767712 was stopped and all8 NPUs returned idle. The prefill-only QLI patch `patches/loop012_qli_prefill_only.patch` remains applied to the shared vllm-ascend `dsa_cp.py`; baseline flags DP1/TP8 no-profiler service is starting in `dsv4ab`, API PID774165, log `logs/serve_dsv4f-w4a8_8npu_dp1tp8_mlen1M_nomooncake_QLIPREFILL-20260920.log`. Detached runner PID2080004 runs `scripts/run_qli_prefill_candidate.sh`: wait for health, functional/golden checks, full-dataset warmup, three48-request mixed passes, then fresh cold offsets24/28/40/44 c1. Output `evidence/20260920_qli_prefill_only/`; check `runner_top.log`, `perf/runner.log`, `cold/times.txt`. Original-source paired cold references for all these offsets and prefix-cache metrics are committed (`6f47a23`, `b3c2b06`). No Loop012 result yet. Do not launch another service over PID774165.


## 2026-09-20 19:48 UTC — Loop 012 KEEP, prefill-only QLI

Loop 012 accepted the prefill-only QLI CPU-max change. Framework source commit `36589852a1eb8f5e842ad920f7c80ebdf1376ee9` is clean and reproducible by `patches/loop012_qli_prefill_only.patch`; pure decode retains the original NPU scalar path. Same 16 original-source and candidate cold 32851-token prompts after matched warmup all succeeded and all improved: mean TTFT 2626.03 to 2362.89 ms (-263.14 ms, -10.02%); both configurations recorded zero prefix cache hits. Functional gate passed. Three full 48×32K→1024 c12 passes gave candidate output TPS 547.55/538.53/553.35, median 547.55 vs paired original median 537.60 (+1.85%, within baseline noise); candidate median request-mean TTFT 1143.58 vs original 1262.74 ms. This is a cold-prefill KEEP, not a proven mixed throughput gain. Evidence: `evidence/20260920_qli_prefill_only/comparison.json`, functional and perf files, and Loop 012 TaskCtl verdict. Candidate service API PID 774165 remains healthy on port 8080 in `dsv4ab`, DP1/TP8, with baseline flags; do not launch over it. Next: diagnose warm mixed decode critical path for a meaningful TPS improvement, retaining this patch.

## Live checkpoint 2026-09-20 19:56 UTC — Loop 013 decode c12 profiler loading

Loop012 KEEP is committed at df30712 (framework 36589852). Loop013 investigates warm mixed decode, with no candidate change yet. Two dynamic msprof attach attempts were invalid: msprof refused live worker PID774466 even after absolute output path; logs are under evidence/20260920_decode_c12_profile/run1,run2. Candidate service PID774165 was stopped; all8 NPUs returned to ~3.4GB idle. Same DP1/TP8 prefill-only source and baseline flags service is loading with torch-NPU profiler without stacks, API PID780818, log logs/serve_dsv4f-w4a8_8npu_dp1tp8_mlen1M_nomooncake_PROFILEC12-20260920.log. Detached runner PID2105542 waits for health, full warmup, warm12×128 c12, then profiles 12×512 c12 via scripts/profile_decode_c12.py; output evidence/20260920_decode_c12_profile/run3/, raw profile separately ignored by Git. Do not launch another service or runner. Check runner log and phase_times before proceeding; profiler may take minutes to stop/export.


## Update 2026-09-20 20:24 UTC — Loop 013 c12 diagnostic pivot

Loop013 produced a valid 8-rank torch-NPU no-stack trace after two invalid dynamic-msprof attaches and one 1-token warmup harness error. The successful run used the kept framework commit 36589852, DP1/TP8, baseline flags, 48×128 full warmup, then warm12×128 and profiled 12×512 c12; all requests succeeded. Torch profiler starts/stops returned 200. Raw 8-rank trace (~22 GB after export) remains at evidence/20260920_decode_c12_profile/torch_raw/ and is Git-ignored; run4/raw_index.json hashes eight kernel CSVs and TP0 trace. Compact derived artifacts and analysis scripts are under evidence/20260920_decode_c12_profile/run4/ and scripts/. TP0 16.459 s sample: device busy13.038 s, HCCL union5.510 s, compute/copy union7.736 s; 170 draft host scopes sum9.036 s (median52.894 ms). Rank HCCL union varied2.236–5.843 s despite near-equal compute/copy7.63–7.75 s. This is diagnostic, not an E2E improvement or a proven removable gap. Loop013 verdict PIVOT: next inspect draft MoE/DSA host self-time and synchronization for an actionable minimal intervention. Current profiler-enabled service API PID780818 is healthy on port8080; no official benchmark should use its profiled request. Do not start another service over it. The earlier kept prefill-only source patch remains committed and clean.

## Live checkpoint 2026-09-20 20:30 UTC — Loop014 source audit

Loop013 PIVOT was committed at fd96bfc; Loop014 is active. Its first design-check Run inspected model_runner_v1.py and Loop013 host trace. In the sampled c12 decode, prepare-input median21.42 ms, typical nested aten::item0.70 ms, host self median9.57 ms. The prepare path calls _update_states, _prepare_inputs, Mamba preprocessing, _build_attention_metadata and _preprocess; no single function has yet been shown to be removable. Next introduce reversible env-gated perf_counter stage spans at these boundaries, restart only the current profiler-enabled API PID780818 after ensuring all8 NPUs idle, collect a short unprofiled c12 diagnostic, then decide whether a minimal correctness-preserving optimization is justified. Preserve prefill-only QLI commit36589852. Current service healthy at last check; profiler is enabled but inactive. Do not treat it as an official baseline server. TaskCtl resume-pack and latest commit are the recovery entry.

## Live checkpoint 2026-09-20 20:33 UTC — Loop014 stage tracer

Loop014 tracer frozen as patches/loop014_prepare_trace.patch, applied but uncommitted in vllm-ascend. Profiler PID780818 stopped; eight NPUs idle before restart. No-profiler DP1/TP8 tracer service API PID789878 loading, log logs/serve_dsv4f-w4a8_8npu_dp1tp8_mlen1M_nomooncake_TRACEPREP-20260920.log. Detached runner PID2135816 waits for health then functional check, 48x128 warmup and 12x512 c12 diagnostic. Evidence: evidence/20260920_loop014_prepare_trace/. Do not start another service. Analyze rank CSVs, then remove tracer patch before candidate E2E tests. Kept QLI commit36589852 must remain.

## Update 2026-09-20 20:47 UTC — Loop014 first stage trace

Loop014 first no-profiler stage trace completed on tracer service PID789878, source patch patches/loop014_prepare_trace.patch. Functional gate pass, warmup48/48, sample12/12 (12x512 c12). Raw rank CSVs (8/8, 3152 total rows) indexed and hashed at evidence/20260920_loop014_prepare_trace/run1/raw_index.json; stage_summary.json filters 157 pure-decode steps per rank by request monotonic window. TP0 median prepare19.55ms, including state0.78, input assembly4.81, dispatch/Mamba0.19, compress+attention metadata12.99, preprocess0.31. Eight-rank compress+attention median10.80-14.17ms. This is the largest host prepare substage, but includes required work. Source read shows _build_attention_metadata loops over KV cache groups and attention builders; next separate the compression-position setup from builder calls and builder types before code change. Current tracer service healthy at last check; patch remains applied, so revert explicitly before official A/B.

## Live checkpoint 2026-09-20 20:51 UTC — builder trace loading

Loop014 source tracer extended to per-attention-builder timing, frozen at patches/loop014_builder_trace.patch (supersedes first 17-line diagnostic patch). Applied to vllm-ascend source only; base kept commit36589852 remains. Syntax and diff checks passed. Previous tracer API PID789878 stopped; 8 NPUs returned idle. New no-profiler DP1/TP8 builder-trace service API PID796352 is loading, log logs/serve_dsv4f-w4a8_8npu_dp1tp8_mlen1M_nomooncake_TRACEBUILDER-20260920.log. Detached runner PID2153103 waits for health, functional, 48x128 warmup and 12x512 c12. Evidence will be run2/, raw2/ and builder_raw2/ under evidence/20260920_loop014_prepare_trace/. Do not launch another service or runner. After analysis restore source to commit36589852 before any official E2E A/B.

## Update 2026-09-20 21:08 UTC — Loop014 PIVOT

Loop014 PIVOT. Builder-level trace passed functional and 12/12 c12 requests; 143 matched pure-decode builder steps per rank. Each step builds eight DSA-CP groups. TP0 median attention metadata15.165ms, eight builders14.025ms, DCP0.002ms, residual1.117ms. First builder7.860ms; seven later builders0.725-1.096ms each. Shared local/ratio state already cached in dsa_cp.py; no safe >=5% mixed TPS change was found. Raw evidence indexed in evidence/20260920_loop014_prepare_trace/run2/raw_index.json, distilled builder_summary.json. Diagnostic service PID796352 stopped, all eight NPUs idle, model_runner_v1.py restored. Framework tree clean at kept QLI commit36589852; patches/loop014_builder_trace.patch reproduces the diagnostic. Next start Loop015 for cold prefill device critical path. No service currently running.

## Live checkpoint 2026-09-20 21:13 UTC — Loop015 scatter slot probe

Loop015 design-check passed for existing cold msprof kernel shape audit. ScatterNdUpdateSk dominant cache [34091,32,1,512], indices [8096,2], updates [8096,1,512]: 736 calls across four cold requests, median task 1017us. Compressor ratio4 shape [8096,4096]: 336 calls, median1503us. These totals include decode and are not TTFT savings. ScatterNdUpdateSk source uses deterministic LinearIndex/Sort/Scatter with SyncAll for duplicate indices. One-shot env-gated slot-index diagnostic patch is applied to framework device_op.py, frozen in patches/loop015_scatter_index_probe.patch; kept QLI source commit 36589852 otherwise. DP1/TP8 no-profiler probe API PID802820 is loading in dsv4ab; detached runner host PID2173051 awaits health, then functional check and one cold 32K->128 c1 request. Probe data: evidence/20260920_loop015_cold_kernels/probe/. Do not launch another service or treat probed latency as benchmark. Inspect per-rank valid slot uniqueness, then restore diagnostic source patch before official A/B.

## Update 2026-09-20 21:29 UTC — Loop015 PIVOT

One-shot cold 32K request passed functional gate and yielded one 8096x2 DSA compressor scatter slot mapping per rank. All eight rank captures had 8096 valid unique pairs, zero duplicates; the representative TP0 ordering is mostly contiguous within 32-slot blocks but not globally sorted. This proves only those eight calls, not a general uniqueness invariant. The diagnostic request succeeded with TTFT2471.76ms, but one-time CPU sync prevents E2E comparison. API PID802820 was stopped; all eight NPUs returned idle; device_op.py probe patch restored and framework clean at 36589852. Isolated one-NPU 25-call screen on captured TP0 indices and same 34091x32x1x512 float32 cache gave bit-equal SK/V2 output; V2 device median2.415ms versus SK1.427ms (69% slower). Reject V2 swap. Loop015 PIVOT, no kept source change. Evidence: evidence/20260920_loop015_cold_kernels/probe/ and scatter_v2_screen.json; scripts/bench_loop015_scatter_v2.py. Next Loop016 will inspect slot-mapping construction and a direct unique-index scatter with safe fallback, then require full correctness and paired cold E2E before KEEP.

## Live checkpoint 2026-09-20 21:43 UTC — Loop016 SWA scatter candidate

Correction to Loop015 attribution: the captured 8096-row [34091,32,1,512] call is SWA prefill KV scatter, not compressed KV scatter. It receives the base scheduler slot mapping formatted as [block,offset]. One cold c1 request had unique valid rows on all8 ranks. Source audit confirms single-request fresh block allocation and sequential positions make this exact cold path injective; broader chunk/prefix/spec paths still need stress evidence. Isolated screens using captured indices: torch_npu built-in scatter1.402ms vs SK1.429ms (little gain); contiguous index_copy0.510ms vs SK1.428ms; simulated page-interleaved cache stride32768 with per-call index flatten and index_copy0.694ms vs SK1.528ms, bit-equal. These are single-NPU screens, not E2E. Experimental flag-gated patch in vllm-ascend device_op.py and dsa_v1.py is frozen as patches/loop016_swa_prefill_index_copy.patch, source otherwise kept commit36589852. It selects only single unpadded SWA prefill via CPU metadata and falls back to SK for other callers; fast path needs a flag file. Syntax/diff checks passed, but service correctness and paired TTFT pending. No service currently running and eight NPUs idle as last checked. Next launch DP1/TP8 no-profiler service and detached scripts/run_loop016_ab.py. Do not claim KEEP until exact output parity, fast path traces on all8 ranks, zero prefix hits and paired cold A/B pass.

## Live checkpoint 2026-09-20 21:47 UTC — Loop016 service A/B loading

Flag-gated experimental framework patch remains applied and uncommitted, reproducible from patches/loop016_swa_prefill_index_copy.patch. DP1/TP8 no-profiler API PID809483 is loading in privileged dsv4ab; log logs/serve_dsv4f-w4a8_8npu_dp1tp8_mlen1M_nomooncake_INDEXCOPY-20260920.log. Detached host runner PID2207198 waits for health, then short functional gate, candidate long exact golden parity and8 alternated distinct-prompt cold baseline/candidate pairs. Runner: scripts/run_loop016_ab.py; evidence: evidence/20260920_loop016_direct_scatter/service_ab/. Candidate flag is absent until runner explicitly enables it. Do not launch a second service. After run inspect all8 fast-path trace files, prefix metrics, parity, paired TTFT; stop service and restore/keep source according to evidence. TaskCtl Loop016 A/B Run ongoing.

## Update 2026-09-20 22:08 UTC — Loop016 first A/B invalid, revised run queued

First full-service attempt API PID809483 loaded and short functional gate passed. The runner's strict four-prompt golden gate was invalid: reference used repeatRate0.9 dataset while runner used no-repeat dataset, so all prompt hashes mismatched. Correct-dataset retry matched all prompt hashes but long output hashes differed, as prior same-service repeats already showed nondeterministic reasoning text. No fast-path trace appeared (0/8), so no candidate timing was taken. A clean no-flag baseline on no-repeat dataset offsets8–15 succeeded8/8, mean TTFT2367.81ms, per-request saved at evidence/20260920_loop016_direct_scatter/service_ab/baseline8_offset8.json. API PID809483 stopped and all8 NPUs idle. Revised source patch patches/loop016_swa_prefill_index_copy_v2.patch uses single-request slot slice length guard, logs CPU metadata at activation, and remains uncommitted in framework; syntax/diff checks passed. New runner scripts/run_loop016_recovery.py will require8 rank fast traces, then compare the exact same offsets8–15 against the saved baseline in a new no-profiler DP1/TP8 service. Original A/B TaskCtl Run INVALID; A/B2 planned. No KEEP claim.

## Live checkpoint 2026-09-20 22:12 UTC — Loop016 revised A/B loading

Revised flag-gated source patch patches/loop016_swa_prefill_index_copy_v2.patch is applied to framework at kept base36589852. API PID815939 in dsv4ab is loading, no profiler, frozen DP1/TP8; log logs/serve_dsv4f-w4a8_8npu_dp1tp8_mlen1M_nomooncake_INDEXCOPY2-20260920.log. Detached runner host PID2226077 waits for health, then short functional, candidate probe with eight rank trace requirement, and same-prompt offset8–15 candidate against saved no-flag baseline. Runner evidence evidence/20260920_loop016_direct_scatter/service_ab2/; trace files written to service_ab/. Do not start another service. Candidate flag absent until runner toggles it.

## Update 2026-09-20 22:25 UTC — Loop016 actual DSA-CP path correction

Revised A/B2 API PID815939 passed short functional and one candidate probe, but produced 0/8 fast-path traces and 0/8 dsa_v1 metadata traces. Root cause: frozen service enables DSA context parallel; AscendDSACPImpl in vllm_ascend/attention/context_parallel/dsa_cp.py performs the actual SWA prefill scatter at line1522. dsa_v1.py path was unused. A/B2 TaskCtl Run INVALID, no candidate E2E result. Service stopped, all8 NPUs idle. Third experimental patch patches/loop016_swa_prefill_index_copy_v3.patch restores dsa_v1 and puts the CPU-only single-unpadded-prefill guard and metadata trace at the DSA-CP callsite; device_op fast path remains flag-gated and all other calls fall back to SK. Syntax/diff checks passed. New runner scripts/run_loop016_cp.py awaits service then requires8 rank activation before exact-prompt offset8–15 cold A/B. Baseline mean2367.81ms remains valid. Do not start over any active service; currently none.

## Live checkpoint 2026-09-20 22:29 UTC — Loop016 active DSA-CP A/B loading

Third no-profiler DP1/TP8 service API PID822276 is loading in dsv4ab, log logs/serve_dsv4f-w4a8_8npu_dp1tp8_mlen1M_nomooncake_INDEXCOPY3-20260920.log. Framework has applied uncommitted diagnostic candidate patch patches/loop016_swa_prefill_index_copy_v3.patch on kept base36589852. Detached runner PID2242486 waits for health, short functional, candidate probe and 8-rank fast-path traces, then same-prompt cold offset8–15 candidate. Evidence evidence/20260920_loop016_direct_scatter/service_ab3/; trace files under service_ab/. Candidate flag absent until runner enables it. Do not start another service. If runner fails, inspect its log, stop this API, restore or revise source, and record verdict.


## Update 2026-09-20 22:39 UTC — Loop016 REJECT, framework restored

The active DSA-CP SWA prefill index_copy candidate ran on all eight TP ranks and passed the short functional gate and eight 32851→128 requests. The same-prompt no-repeat offsets8–15 cold comparison had zero prefix-cache hits. Baseline mean TTFT2367.81ms versus candidate2438.00ms, +70.19ms (+2.9642% slower), 0/8 requests improved. This rejects the implemented specialization despite faster isolated operator screens; the isolated gain was not realized E2E. Prefix query count rose 274→295933 with hits0→0. Evidence: evidence/20260920_loop016_direct_scatter/service_ab3/summary.json and metrics_before/after.txt, eight rank traces in service_ab/. TaskCtl Loop016 rejected. API PID822276 stopped; eight NPUs idle. Framework restored clean at kept QLI commit36589852; rejected implementation remains reproducible as patches/loop016_swa_prefill_index_copy_v3.patch. Next Loop017 will examine Compressor ratio4 prefill cost and critical path before a bounded candidate. No service running.


## Live checkpoint 2026-09-20 22:46 UTC — Loop017 Compressor attribution

Loop017 started after Loop016 rejection. No service running; eight NPUs idle; framework clean at kept commit36589852. Existing original-source cold msprof op_summary was distilled by scripts/analyze_loop017_compressor.py into evidence/20260920_loop017_compressor/profile_clusters.json. The ratio4 Compressor [8096,4096;1024,4096] has four clusters of84 calls, one cluster per cold request, with ~126.4ms summed device task time/request and median1.503ms/call. Another [512,4096] weight shape has80 calls/request and ~40ms/task sum; [256,4096] has84 and ~29.8ms. These are occupancy, not proven TTFT critical-path savings. Next inspect stream order/dependencies and kernel bottleneck, then freeze a bounded candidate; do not modify framework before attribution.


## Live checkpoint 2026-09-20 22:50 UTC — Loop017 same-stream dependency

Frozen original-source cold msprof ordering, distilled by scripts/analyze_loop017_stream.py: all336 ratio4 main-shape Compressor tasks are immediately followed on the same device stream by ScatterNdUpdateSk and then SparseAttnSharedkv. Median Compressor-end to next-start gap2.87us; Compressor-start to attention-start1.84475ms. Evidence evidence/20260920_loop017_compressor/stream_order.json. This strengthens the serial per-layer opportunity, but global E2E savings still depend on cross-stream overlap and actual implementation. No framework changes or active service. Next inspect compressor kernel tiling and bound a small operator-level optimization, followed by full-service correctness/E2E.


## Live checkpoint 2026-09-20 23:59 UTC — Loop017 isolated operator baseline

All8 NPUs were idle and port8080 had no service before the isolated run. Added scripts/bench_loop017_compressor.py using the cold profile's exact Compressor tensor shapes, ratio4/coff2/cache_mode1 on one NPU. Correctly scaled synthetic activations/weights produced finite [2025,512] output; 25-call device median1.630ms, min1.499ms, wall median1.810ms. Evidence evidence/20260920_loop017_compressor/shape_matched_operator.json. The first unscaled random sample produced nonfinite output and is retained as shape_matched_operator_invalid.json, excluded from baseline. There is no reference-model correctness check or E2E claim. Source audit: arch32 tiling fixes mBaseSize128 for coff2, nSize2, 20 AIC blocks; kernel splits headDim512 over dBase64, giving two M groups and periodic SyncAll. Considering an isolated tiling sweep before any package replacement; installed custom operator and framework unchanged. One-NPU process finished, eight NPUs idle.


## Live checkpoint 2026-09-21 00:06 UTC — Loop017 isolated mBase256 build

Candidate changes only arch32 Compressor tiling for coff2: mBaseSize128→256, patch patches/loop017_compressor_mbase256.patch. Reproducible build script scripts/build_loop017_mbase256.sh copies csrc into /tmp and installs generated custom vendor package into a fresh ignored artifacts/ prefix, never /usr/local/Ascend or framework source. Build process host docker-exec PID2298364/container script PID2298389 is in progress; log evidence/20260920_loop017_compressor/mbase256_build.log; exact prefix in candidate_prefix.txt. Do not launch duplicate build. Framework remains clean at36589852; no service, all8 NPUs idle before build. Stock operator repeat with fixed seed has finite output and device median1.514ms on25 calls; evidence reference_screen.json, log reference_run.log. Stock output+state reference snapshot artifacts/loop017_compressor_reference_output.pt SHA256 e5dc8cfee73cec0bb80a5ce95dbf48ed3222a4d7f628ce6e381a615b56cbb0ec (66MB, ignored Git, regenerable with LOOP017_REFERENCE_OUT). Once build completes, isolate vendor per process, compare input fingerprint/output/state to this snapshot, then benchmark. No candidate performance or correctness result yet.


## Live checkpoint 2026-09-21 00:22 UTC — isolated build watchdog

The mBase256 isolated package build remains active: container script host PID2298389, ninja PID2298924; many bisheng compiler processes and >1500 objects generated. No installed vendor or framework source changed. Background watcher host PID2314657 runs scripts/watch_loop017_build.sh. It waits for build PID exit, then, only if isolated package lib exists, runs scripts/run_loop017_isolated_bench.sh against fixed reference; logs candidate_run.log and candidate_exit_code.txt under evidence/20260920_loop017_compressor/. Do not launch duplicate build or benchmark while watcher active. Stock reference snapshot/checksum and candidate prefix are in earlier checkpoint. All8 NPUs idle before this CPU build; check again before any service.


## Update 2026-09-21 01:10 UTC — Loop017 REJECT; Loop018 direction

Isolated Compressor mBase256 vendor package built successfully into an ignored private prefix. The fixed-seed stock exact-shape reference completed at 1.51436ms/device-call median. Candidate process loaded its isolated libcust_opapi.so, but did not finish the first screen or produce output after >8 minutes; it was interrupted (exit130). Correctness, comparative performance and causal attribution are invalid. TaskCtl Loop017 rejected at operational gate; this does not prove mBase256 tiling is intrinsically slower. No global vendor or framework install changed. Framework is clean at kept commit36589852; no service and all8 NPUs idle after interruption. Evidence: evidence/20260920_loop017_compressor/candidate_observation.json and candidate_run.log.

Read-only audit of saved c12 8-rank communication.json: identical collective counts/rank (25160 allGather, 7820 alltoall, 15980 reduceScatter) but apparent HCCL elapsed differs substantially by rank. Every included communication entry reports elapsed entirely as Idle Time, with zero Transit/Wait; these data do not identify removable wire or compute time. Audit: evidence/20260921_decode_comm_audit/collective_time_components.json. Next Loop018: measure the service-level decode critical path and arrival skew, then freeze an intervention only if the trace supports one.


## Update 2026-09-21 01:15 UTC — Loop018 PIVOT

Saved c12 TP8 collective sequences aligned exactly across all eight ranks (48,960 calls). Raw start spread median0.6105ms nearly matches end spread0.583ms, and rank6 appears earliest in48,829/48,960 calls. A sensitivity calibration using median shared-end differences estimates rank6 timestamp offset -579.172us relative to rank0; corrected start/end spreads still have medians0.2175/0.206ms. This is not independent clock synchronization. Together with all communication time classified Idle, the trace cannot support a removable collective-wait estimate or safe patch. Loop018 pivoted, no source or service changes. Next Loop019 targets measured warm decode host metadata builders in the active callsite. No service on8080, eight NPUs idle at last check.


## Live checkpoint 2026-09-21 01:19 UTC — Loop019 active

Created Loop019 after Loop018 pivot. Existing no-profiler pure-decode c12 trace (evidence/20260920_loop014_prepare_trace/run2/builder_summary.json) has143 matched steps on TP0. Eight AscendDSACPMetadataBuilder calls per step; group0:0 median7.859927ms versus each of the other seven ~0.75–1.10ms. Source vllm_ascend/attention/context_parallel/dsa_cp.py is active and already caches common device local, RoPE local and CPU local metadata across cache groups in build_req_metadata. The first builder additionally establishes input positions, cos/sin and shared metadata. Thus avoid a blind all-builder cache patch. Next instrument first-builder substage timings in a flag-gated local patch and no-profiler service run, then choose a semantics-safe minimal change if one stage dominates. Framework remains clean at36589852; no service on8080 and NPUs idle at last check.


## Live checkpoint 2026-09-21 01:25 UTC — Loop019 stage trace ready

Flag-gated diagnostic patch patches/loop019_builder_stage_trace.patch is applied only to framework dsa_cp.py on kept source36589852. It records four host timestamps around shared input/RoPE setup, DSA slot formatting and build_req_metadata, plus first-builder flag and shape. py_compile and git diff --check pass. No behavior branch changes when the environment flag is absent. Framework working tree is dirty only by this patch; do not start a different service or reset it while the Loop019 trace service is active. Trace output planned under evidence/20260921_loop019_builder_stage/raw/.


## Live checkpoint 2026-09-21 01:22 UTC — Loop019 service loading

No existing port8080 listener; all8 910B3 at ~3.4GB idle HBM before launch. Privileged dsv4ab diagnostic API PID837367 started with DP1/TP8, W4A8, 1M maxlen and flag LOOP019_BUILDER_TRACE_DIR. Log logs/serve_dsv4f-w4a8_8npu_dp1tp8_mlen1M_nomooncake_LOOP019-STAGE-20260921.log. Framework has only patches/loop019_builder_stage_trace.patch applied; active source dirty by design. Background runner PID2352861 (launch shell) executes scripts/run_loop014_prepare_trace.py with output evidence/20260921_loop019_builder_stage/run1/; runner log evidence/20260921_loop019_builder_stage/runner.log. It waits for health, runs short functional, warmup48x128, then no-profiler c12 12x512 stage sample. Raw per-rank CSV under evidence/20260921_loop019_builder_stage/raw/. Do not launch another service or reset framework before runner result. TaskCtl Loop019 run active-dsacp-builder-stage-20260921 created but not finished.


## Live checkpoint 2026-09-21 01:59 UTC — Loop019 first diagnostic done

First no-profiler DP1/TP8 stage trace finished: short functional passed, 12/12 sample requests succeeded. Across174 pure-decode steps/rank, first DSA-CP builder median total3.61–6.50ms, including build_req_metadata2.74–5.74ms and shared setup0.57–0.66ms; other builder median0.67–0.85ms. Evidence evidence/20260921_loop019_builder_stage/stage_summary.json, raw eight-rank CSVs and sample client. API PID837367 stopped after sample, all8 NPUs returned to idle. A second flag-gated patch patches/loop019_builder_req_trace.patch is applied in framework, expanding timing inside build_req_metadata (device-local, RoPE, CPU-local, max, compressor, SAS, QLI). py_compile/diff checks pass; no service running. Next relaunch same frozen service and sample with LOOP019_REQ_TRACE_DIR to isolate the dominant substage, then decide minimal candidate.


## Live checkpoint 2026-09-21 02:01 UTC — Loop019 request subphase service loading

First Loop019 diagnostic committed at cddd081. Second service API PID843805 loading in privileged dsv4ab, port8080; flags LOOP019_BUILDER_TRACE_DIR and LOOP019_REQ_TRACE_DIR point to evidence/20260921_loop019_builder_req/raw and raw_req. Runner PID2379121 waits for health, functional gate, full warmup and12×32K→512 c12 sample; output run2/, runner.log. Log logs/serve_dsv4f-w4a8_8npu_dp1tp8_mlen1M_nomooncake_LOOP019-REQ-20260921.log. Framework has expanded diagnostic patch patches/loop019_builder_req_trace.patch (uncommitted in its source tree), no behavioral optimization. Do not reset framework or start competing service. TaskCtl run active-dsacp-request-subphases-20260921 pending.


## GitHub sync note 2026-09-21 02:04 UTC

User pointed to https://github.com/wio1997/Inference-Foundry and asked why work is absent. The server repository previously had no remote; all commits through6643648 were local. Added origin URL and fetched remote main (1b0b48e; two README-only commits, unrelated history). Prepared a local merge worktree /tmp/inference-foundry-github-sync on branch github-sync-20260921, merge commit59b5148 joining remote README history and local project through6643648. No push has occurred. Await user answer to the explicit publish choice (main / review branch / hold). Do not force-push or publish without that answer; ongoing Loop019 work stays on local main.


## Live checkpoint 2026-09-21 02:13 UTC — Loop019 QLI subphase identified

Second no-profiler DP1/TP8 request-metadata trace finished: short functional passed, 12/12 c12×512 sample passed. Across169–170 pure-decode first-builder steps/rank, QLI subphase median1.735–4.306ms/rank dominates first build_req_metadata; device-local0.477–0.548ms, CPU-local0.255–0.292ms, SAS0.422–0.485ms. These inclusive host spans do not establish removable E2E cost. Evidence evidence/20260921_loop019_builder_req/request_subphases_summary.json plus all8 raw rank CSV and sample. API PID843805 stopped, port8080 down, all8 NPUs idle (~3.4GB HBM), framework restored clean at36589852. Loop019 remains active: decompose QLI decode path without repeating previously rejected all-step CPU-maxima experiment (Loop011); freeze only a causally testable minimal change.


## GitHub publication authorization 2026-09-21 02:14 UTC

User explicitly authorized push to GitHub main. After committing Loop019 second diagnostic at934fba2, update local github-sync-20260921 merge worktree from main, verify origin/main still1b0b48e, then push merge HEAD to origin main as a fast-forward. This preserves remote README and local history. Record pushed commit hash and verify remote SHA afterward.


## GitHub main published 2026-09-21 02:16 UTC

After explicit user authorization, merged local project and remote README histories and fast-forward pushed GitHub main to14baeb0d985dca31b4156cee0bc2149350cf4b79. git ls-remote origin refs/heads/main verified the same SHA; public raw PROJECT_STATE.md returned HTTP200. Server local main fast-forwarded to the same commit. No force push. Continue using origin=https://github.com/wio1997/Inference-Foundry.git; future pushes should first fetch and require fast-forward.


## Live checkpoint 2026-09-21 03:02 UTC — Loop019 QLI scalar/op trace loading

Loop011 established that replacing both decode QLI NPU scalar maxima with CPU maxima gave parity but no robust mixed TPS benefit (+0.52%, TTFT worse). Loop019 therefore decomposes QLI before another intervention. Flag-gated patch patches/loop019_qli_subphase_trace.patch applied only to dsa_cp.py; it timestamps first and second .item() and the following QLI metadata op without changing arguments. py_compile and diff check passed. No prior service on8080, 8 NPUs idle before launch. Privileged dsv4ab DP1/TP8 API PID850293 is loading, log logs/serve_dsv4f-w4a8_8npu_dp1tp8_mlen1M_nomooncake_LOOP019-QLI-20260921.log. Runner PID2416955 waits for health, short functional, full warmup and 12×32K→512 c12 sample; outputs evidence/20260921_loop019_qli_subphase/run3/, raw per-rank QLI traces in raw/, runner.log. TaskCtl run qli-scalar-versus-op-trace-20260921 pending. Do not reset patched source or launch competing service until runner result. GitHub main is configured/tracked and was synced at1931d54 before this new local work.


## Update 2026-09-21 03:11 UTC — Loop019 PIVOT, Loop020 next

Third no-profiler QLI trace passed functional and12/12 c12×512. Within155–156 pure-decode uncached QLI calls/rank, first q.max().item median0.402–3.716ms (rank-dependent), second k.max().item0.110–0.135ms, metadata op including clones0.356–0.411ms. Earlier Loop011 removed both scalars with numeric parity but mixed TPS moved only+0.52% (noise) and TTFT worsened. Thus the host QLI wait is real but E2E-removable value is unproven; do not repeat CPU-max change. TaskCtl Loop019 pivoted. API PID850293 stopped, all8 NPUs idle, framework clean at kept36589852. Evidence evidence/20260921_loop019_qli_subphase/ and patches/loop019_qli_subphase_trace.patch. Next Loop020: analyze DSpark proposer vs target verification host/device critical path and acceptance in frozen mixed c12 before any patch.


## Live checkpoint 2026-09-21 03:20 UTC — Loop020 DSpark stage service loading

Loop020 started. Saved TP0 c12 profiler has170 draft_token host scopes median52.894ms/sum9.036s; device events temporally inside them occupy union median41.860ms/sum7.352s (compute37.641ms, HCCL5.272ms). This is co-occurrence, not causal ownership: target async tasks may spill across host scope boundaries. Evidence evidence/20260921_dspark_audit/host_device_scope_overlap_tp0.json. Minimal diagnostic patch patches/loop020_draft_stage_trace.patch records _propose stages (inputs, attention metadata, step metadata, model call) without semantic changes. DP1/TP8 service API PID856647 loading in dsv4ab; runner PID2439916 awaits health, functional, warmup and12×32K→512 c12 sample. Log logs/serve_dsv4f-w4a8_8npu_dp1tp8_mlen1M_nomooncake_LOOP020-DRAFT-20260921.log; output evidence/20260921_loop020_draft_stage/run1/, raw/. Framework source dirty only by diagnostic patch; do not reset/start competing service before result.


## Live checkpoint 2026-09-21 03:36 UTC — Loop020 DSpark proposer stages

DP1/TP8 no-profiler DSpark diagnostic passed short functional and12/12 c12×512. For160 no-prefill proposer calls/rank, graph_values=[0] (eager). _propose median36.11–41.38ms/rank; run_draft28.43–32.69ms, build_draft_attn_metadata across all attention groups5.67–6.41ms, set_inputs1.47–1.66ms. These are inclusive host times and do not identify removable device work. Saved profiled c12 acceptance buckets have mean advancement3.54–3.64 tokens for 7-token drafts; evidence evidence/20260921_dspark_audit/acceptance_profile.json. API PID856647 stopped, all8 NPUs idle, framework restored clean36589852. Next attribute which device tasks are owned by run_draft (rather than temporally coincident), and inspect whether step0 metadata can safely avoid repeated work; no patch candidate yet.


## Update 2026-09-21 04:09 UTC — Loop020 PIVOT; Loop021 active

TP0 async-flow attribution now causally joins 121,038 launches whose CPU flow origin is inside the same-thread `draft_token` scope to device tasks at an exact timestamp match. Across 170 scopes, causal device-task union clipped to each host scope is median 51.661 ms versus 52.894 ms host duration; launched work extends to 63.259 ms median union beyond the scope. This rules out a large host-idle interpretation of the proposer span. Loop020 pivoted: DSpark is eager-only, acceptance advances 3.54–3.64 tokens per 7-token draft, and no safe scheduling patch exceeds noise yet. Loop021 targets exact CPU-parent/source attribution for repeated Index/Pad/Copy/event chains. No service on port 8080; eight NPUs idle; framework clean at kept commit 36589852a.


## Update 2026-09-21 05:10 UTC — Loop021 REJECT; Loop022 active

Exact TP0 CPU-parent → async-flow → device-task attribution matched all 121,038 draft-owned launches. The proposed Index/Pad/Copy fragmentation is too small as a standalone target: `aclnnIndex` clipped union median 0.545 ms when present and `aclnnConstantPadNd` 0.337 ms, below frozen mixed-run noise. A correction to Loop020 interpretation is required: the 51.661 ms causal union was dominated by `EVENT_WAIT`, not active compute. Outer `draft_token | wait_event` has 51.169 ms median clipped union across all 170 scopes; `vllm::moe_forward_shared | wait_event` has 26.595 ms. Event waits encode cross-stream dependencies and are not additive savings. Loop021 rejected; Loop022 maps MoE event producers/consumers and determines whether any wait lies on the main critical stream. No service on 8080, all eight NPUs idle, framework clean.


## Update 2026-09-21 06:10 UTC — Loop022 REJECT; Loop023 active

Event topology resolves the large draft waits. Two outer waits occur once per step on device streams 42/43 (median 53.104/54.095 ms) and immediately precede `MEMCPY_ASYNC`; they are auxiliary copy streams waiting for draft results, not main compute stalls. In shared-expert stream36, the apparent large wait is the first `before_routed_experts` dependency between layer invocations: previous qmatmul → next dynamic quant, median 25.734 ms. The three intra-call waits are only 0.191, 0.051 and 0.011 ms median. Default stream47 final waits are ~0.00002 ms, showing shared-expert overlap finishes before the main consumer. Loop022 rejected. Loop023 now tests k5 versus k7 speculative length because k7 advances only 3.54–3.64 tokens on average. No service active; NPUs idle; framework clean.

## Live checkpoint 2026-09-21 06:19 UTC — Loop023 k5 screen loading

After Loop022 rejection, a bounded speculative-length screen started. `scripts/serve_loop023_k5.sh` differs from the frozen launcher only by DSpark `num_speculative_tokens=5` (model `dspark_block_size=5`; SP is disabled). Privileged `dsv4ab` launches DP1/TP8, W4A8, 1M max length on port8080; service launcher host PID2529097, runner PID2529098. Log: `logs/serve_dsv4f-w4a8_8npu_dp1tp8_mlen1M_nomooncake_LOOP023-K5-20260921.log`; evidence: `evidence/20260921_loop023_k5_screen/`. Runner waits for health, passes correctness smoke, warms 48x128 c12, then runs 12x32K→512 c12. TaskCtl run `k5-screen-20260921` is active. Do not launch a competing service. Compare to k7 only after successful completion; no performance claim yet.


## Update 2026-09-21 07:07 UTC — Loop023 REJECT; Loop024 active

The k5 service never reached readiness. vLLM rejected graph shapes during TP8 initialization because `(k+1)=6` and sequence-parallel TP8 require a common divisible capture shape; runner timed out without correctness or benchmark. With model `dspark_block_size>=5`, there is no smaller valid k below7 under this invariant. Loop023 rejected as invalid comparison; port8080 is down and all NPUs are idle. Source inspection confirms k7/max_seqs16 reserves96 draft slots: max batched8192 becomes max scheduled8096 and emits an explicit suboptimal warning. Loop024 tests the minimal scheduler-only correction, max batched8288, restoring max scheduled8192 with k7 unchanged.

## Live checkpoint 2026-09-21 07:14 UTC — Loop024 8288-token screen loading

Loop023 k5 workers were explicitly terminated after their startup error left ~29GB/card allocated; only that failed experiment process group was killed. HBM returned to ~3.4GB/card. Loop024 relaunched successfully with valid k7 and the sole configuration change `max_num_batched_tokens=8288`; container service PID869225, host runner PID2554536. Runner waits for health, correctness smoke, 48x128 warmup and12x32K→512 c12 screen under `evidence/20260921_loop024_tokens8288/run1/`. Service log must confirm resolved max scheduled capacity8192 before performance is accepted. No result yet; do not start a competing service.


## Update 2026-09-21 08:14 UTC — Loop024 REJECT; Loop025 active

The k7/max-batched8288 service passed correctness and12/12 c12 screen; the8096 scheduled-token warning disappeared, confirming capacity restoration. Screen result:472.871 output tok/s, TTFT mean2007.49ms, TPOT mean17.300ms. This is +2.77% versus the median of six prior k7 diagnostic screens but -3.83% versus the closest Loop020 same-runner screen, within the frozen4.3% noise threshold. No expensive full benchmark; Loop024 rejected and service stopped. Loop025 now measures DSpark net value against a matched target-only no-spec screen before spending more effort inside proposer/verification. Framework clean; NPUs returning idle.

## Live checkpoint 2026-09-21 08:07 UTC — Loop025 target-only screen loading

Loop024 was rejected and its service stopped. A target-only control now launches with the frozen DP1/TP8 W4A8 configuration and the sole semantic change of removing `--speculative-config`; max batched tokens remains8192. Container API PID878239, host runner PID2600192. Evidence: `evidence/20260921_loop025_nospec/`; service log: `logs/serve_dsv4f-w4a8_8npu_dp1tp8_mlen1M_nomooncake_LOOP025-NOSPEC-20260921.log`. Runner performs the identical correctness,48x128 warmup and12x32K→512 c12 screen. Do not start a competing service. This experiment measures DSpark net E2E value; no result yet.

## Update 2026-09-21 09:09 UTC — Loop025 ACCEPT; Loop026 active

Matched target-only control passed correctness and12/12 c12 but delivered only208.047 tok/s and54.333ms TPOT. Matched k7 DSpark control delivered491.698 tok/s and17.554ms TPOT:2.363x throughput and67.69% lower TPOT, far beyond4.3% noise; short-screen TTFT was20.52% worse. Loop025 accepted: retain DSpark and optimize proposer/verification organization. DeepSeek V4 DSpark has exactly three draft layers. Loop026 tests a flag-gated middle-layer bypass; target verification preserves final model semantics even if proposal quality falls. A rough falsification gate estimates acceptance must remain above3.15 advanced tokens/cycle versus current3.64; this is not a performance claim and E2E decides. No service active; framework clean.

## Live checkpoint 2026-09-21 09:16 UTC — Loop026 middle-layer bypass loading

Framework carries two reversible changes on clean commit36589852: saved diagnostic `patches/loop020_draft_stage_trace.patch` and candidate `patches/loop026_dspark_skip_middle.patch`. The candidate is environment-gated and bypasses only layer offset1 of exactly three DSpark draft layers; first-layer projection and last-layer head remain intact. Target verification continues to protect final semantics. DP1/TP8 k7 service PID885161 loads with `VLLM_ASCEND_DSPARK_SKIP_MIDDLE_LAYER=1`; runner PID2637350 performs correctness, warmup and12x32K→512 c12 while recording proposer stages under `evidence/20260921_loop026_layer_bypass/raw/`. Do not launch another service or reset framework. Acceptance must remain above the rough3.15-token break-even and E2E must exceed4.3% before full benchmark.


## Update 2026-09-21 10:09 UTC — Loop026 REJECT; Loop027 active

The environment-gated middle-layer bypass passed functional correctness and12/12 c12 because target verification protects semantics. It reduced rank-median proposer `model_run` from31.106 to23.887ms (-23.21%) and total proposer from39.545 to32.298ms, but proposal quality collapsed: only0.363 accepted draft tokens and1.363 advanced tokens/cycle versus baseline3.54–3.64. Output TPS fell491.698→217.092 (-55.85%) and TPOT rose173.29%, nearly target-only behavior. Loop026 rejected; all three trained draft layers are necessary. Service stopped and framework restored clean36589852. Loop027 preserves exact three-layer semantics and tests the evidenced framework gap: fixed-shape DSpark is eager-only, so audit segmented graph/standalone proposer replay feasibility before any integration.

## Update 2026-09-21 11:07 UTC — Loop027 v2 DSpark graph startup probe

Source audit found a stronger bounded path than manually capturing the legacy proposer. The legacy `vllm_ascend/spec_decode/dspark_proposer.py` forces `use_cuda_graph=False`, while the same v0.26 source tree already registers `AscendDSparkSpeculator` under the v2 model runner and wires its query graph manager to the Ascend update stream. A startup/correctness-only probe therefore uses `VLLM_USE_V2_MODEL_RUNNER=1`, `FULL_DECODE_ONLY`, and DSpark `enforce_eager=false`, with DP1/TP8, k7, 8192 scheduler tokens, W4A8 model and 1M max length unchanged. API PID891755 is loading in privileged `dsv4ab`; log `logs/serve_dsv4f-w4a8_8npu_dp1tp8_mlen1M_nomooncake_LOOP027-V2GRAPH-20260921.log`. Source audit is `evidence/20260921_loop027_graph_feasibility/source_audit.json`; launcher is `scripts/serve_loop027_v2_graph.sh`. No performance claim exists. Next require successful load, graph capture evidence and functional correctness before any screen.

## Update 2026-09-21 11:18 UTC — Product boundary changed to specialized runtime

User direction freezes the final product as a DeepSeek V4 Flash W4A8 / 8×910B3 / DP1×TP8 / DSpark7 specialized runtime. vLLM/vLLM-Ascend is now reference implementation, correctness oracle and operator source, not an architecture constraint. The Loop027 v2 graph probe failed before capture at generic KV-group discovery (`No draft attention groups found`); all residual workers were force-cleaned and all eight NPUs are idle. Loop027 pivoted with no correctness or performance claim. `SPECIALIZED_RUNTIME.md` defines the minimum target chain, removal candidates and parity/E2E promotion gates. Next loop extracts and replays the exact fixed proposer→target verification→acceptance contract from the known-working legacy path.

## Update 2026-09-21 12:13 UTC — Loop028 fixed-cycle pointer-stability probe loading

Source contract V0 maps the target forward/verification, proposer preparation, exact three-layer DSpark, Markov argmax and state-publication chain; evidence `evidence/20260921_loop028_runtime_contract/source_contract_v0.json`. A diagnostic patch `patches/loop028_contract_pointer_trace.patch` records only tensor shapes, strides and device pointers across pure-decode calls, with no tensor values or D2H sync. It tests whether the working legacy path already reuses stable buffers suitable for a fixed replay contract. Privileged `dsv4ab` API PID897585 is loading with frozen DP1/TP8 W4A8 k7 settings; log `logs/serve_dsv4f-w4a8_8npu_dp1tp8_mlen1M_nomooncake_LOOP028-CONTRACT-20260921.log`. Do not launch another service or reset framework until the probe finishes. No correctness or performance result yet.

## Update 2026-09-21 12:25 UTC — Loop028 atomic fixed-cycle replay probe

The first pointer-only diagnostic was INVALID before model execution because its trace block was inserted into a profile helper rather than `_propose`; mixed worker import timing exposed `NameError: _loop028_trace_dir is not defined`. It yields no correctness/performance conclusion, and all workers/NPUs were cleaned. The corrected atomic patch `patches/loop028_fixed_cycle_replay.patch` places tracing at the actual proposer boundary and, once at c12, invokes the already-materialized `run_draft` closure twice without re-entering scheduler/request preparation. It uses the exact model, collectives, metadata objects and KV buffers, requires bit-equal draft tokens on every rank, then runs service correctness. API PID904009 and runner PID2746890 are loading; evidence directories are `raw/`, `replay/`, and `run2/`. Do not launch another service or edit the framework source until completion.

## Update 2026-09-21 12:52 UTC — Loop028 PIVOT and pause

Loop028 reached a bounded executable replay result. In the corrected c12 diagnostic, each of eight TP ranks recorded 172 warm pure-decode proposer calls. All observed shapes and strides were fixed. Ten traced fields kept one process-local address per rank; only `target_token_ids` and `target_positions` changed address. Re-executing the already-materialized real-weight DSpark proposer closure without scheduler/request preparation produced exactly equal `[12,7]` draft token tensors on 8/8 ranks. Short functional checks passed.

This establishes the proposer as a viable fixed-cycle replay segment. It does not satisfy the full standalone-cycle gate: target verification, accepted-token parity, KV mutation and end-of-cycle state mutation were not compared. JSON tracing and the inserted duplicate call invalidate the 412.50 tok/s decode sample for performance comparison; the accepted mixed baseline remains 543.65 tok/s and no KEEP was claimed.

The Loop028 verdict is PIVOTED with partial causal support. The service was stopped, port 8080 released, no NPU processes remain, and the diagnostic framework patch was removed; vLLM-Ascend is clean at `36589852a`. Work is paused by user request after this commit. On resume, extend the proven proposer segment through verification, device-side acceptance and explicit state/KV parity before claiming a standalone full cycle. Evidence: `evidence/20260921_loop028_runtime_contract/` and `tasks/deepseek-extreme-p0/loops/loop-028/`.
## Update 2026-09-21 14:35 UTC — Loop029 fixed runtime semantics proven, execution takeover next

Loop029 has moved the project onto a product-owned execution path. `runtime/fixed_decode.py` now owns a fixed c12 decode state and target-input materialization without `SchedulerOutput`, request objects, `InputBatch`, or `ModelRunner`; `runtime/greedy_accept.py` owns the frozen temperature-0 speculative acceptance rule. CPU and NPU semantic smoke tests pass for eight causally connected cycles.

The real 8×910B3 / DP1×TP8 oracle-shadow run passed 12/12 requests. On every TP rank, the runtime-predicted next target ABI was exact for 28/28 consecutive comparisons (224 total): `input_ids`, `positions`, `query_start`, `seq_lens`, and `slot_mapping`. Independent greedy acceptance matched the oracle's accepted tokens and accepted counts for 29/29 comparisons per rank (232 total). The measured 130.44 output tok/s is diagnostic-only because synchronous comparisons and per-cycle file writes are enabled.

This is not yet a standalone decode cycle: target forward, DSpark proposal, target KV writes, recurrent/GDN state, TP collectives, and model weights remain oracle-owned. Keep Loop029 active. The next minimum milestone is to give the runtime its own fixed target KV/recurrent buffers and direct target operator adapter, execute at least eight real target→accept→advance→propose cycles, and compare token/state fingerprints against the oracle on all ranks. Do not resume generic vLLM audit or local hotspot patching. The first fusion candidate is the now-proven fixed region spanning target argmax/greedy acceptance, accepted-count reduction, sequence/position advance, and next target-input preparation; evaluate fusion only after the direct operator boundary is executable.

Evidence: `evidence/20260921_loop029_standalone_v0/shadow_summary_attempt4.json`, `shadow4_accept_c12.json`, and TaskCtl run `run-20260921T143422Z`. The diagnostic service is stopped, all eight NPUs are idle, and the vLLM-Ascend checkout is clean at `36589852a`.

## Update 2026-09-21 15:23 UTC — product control plane and cache ownership boundary

`runtime/extreme_decode.py` now owns the continuous fixed decode stage order and imports no vLLM code. Standalone CPU and Ascend NPU processes completed eight cycles with explicit prepare-target, target, acceptance, state-advance and proposer stages; target and proposer each executed eight times. `runtime/assets.py` adds direct ownership checks for physical cache tensors and exact selected-element fingerprints derived from touched slot locations. These tests use deterministic operators and synthetic caches, so they prove control-plane independence and the ownership ABI, not real-weight correctness.

The only allowed vLLM role in the next integration is one-time bootstrap after weight/process-group/cache initialization. The transfer point is the oracle's physical `kv_caches` tensor tree plus already-bound layer operators; no ModelRunner, SchedulerOutput, request object or metadata builder may enter `ExtremeDecodeRuntime.step()`. Next bind the live target model, TP/EP collectives and DSA compressor/indexer/SWA caches, then run real-weight continuous cycles and exact touched-cache parity.

## Update 2026-09-22 08:16 UTC — first real-weight Extreme Runtime chain passed

Loop029 now has a true runtime-owned continuous decode chain. The one-time
bootstrap transfers the loaded DeepSeek V4 Flash W4A8 model, TP8/EP process
groups, 67 deduplicated physical KV/DSA cache tensors, fixed c12 buffers and the
DSpark7 proposer. Control is handed to `ExtremeDecodeRuntime` before the generic
`ModelRunner` target forward. After handoff the run records zero oracle target
calls and retains no `ModelRunner`.

Run17 passed on all eight 910B3 ranks: eight consecutive
prepare-target → target → acceptance → state-advance → proposer cycles, 163
accepted/emitted tokens, exact `num_computed_tokens = initial + emitted`, valid
acceptance counts and identical final rank state. Target execution currently
uses an eager direct model binding; graph/replay is deliberately deferred until
this owned path is profiled. The bootstrap still borrows prebuilt attention
metadata and the DSpark adapter still refreshes some common CPU mirrors, so the
serving shell is not yet fully independent.

The strict transaction replay gate was retired with evidence, not waived:
run16 restored all 69 captured state entries exactly (67 caches plus top-k and
MTP buffers), but the stock graph replay itself changed two of 96 argmax tokens
and accepted token values. Direct eager replay changed three argmax tokens while
preserving acceptance. This establishes intrinsic execution replay noise; the
accepted runtime gate is continuous state correctness and cross-rank agreement,
building on the already-proven target ABI and acceptance parity.

No performance result is claimed. The 12-request trigger intentionally aborts
the surrounding service after writing evidence, so its client timing is invalid
for comparison with the 543.65 tok/s stock baseline. Next profile only the
runtime-owned eight-cycle DAG, migrate remaining metadata/CPU refresh state into
fixed device buffers, and choose graph/replay, persistent execution and fusion
regions from that trace.
## Update 2026-09-23 06:36 UTC — Loop035 first continuous-state divergence

Loop035 profiled eight real-weight Extreme c12 cycles on all eight 910B3 ranks. Median NPU event time per cycle across ranks was target 45.07 ms, DSpark proposer 5.93 ms, acceptance 0.38 ms, target-input preparation 0.27 ms, and state advance 0.02 ms. This is a short diagnostic DAG, not a formal 1024-cycle E2E critical path. A warm Stock 12×32K→1024 single cohort completed 12/12 at 553.76 tok/s; Prometheus deltas gave 9151 accepted draft tokens across 3147 iterations, or 2.908 accepted drafts/iteration. Extreme's sustained output rate remains near 1.2–1.4/slot/cycle in the formal Loop034 result.

The first proven stale target state is DSA-CP derived metadata: at Extreme cycle 1, target positions and seq_lens advanced, but req_metadata.start_pos and cp_metadata.local_seq_lens retained bootstrap values. Rank-0 first request moved from position 32871/seq_len 32879 to 32872/32880, while those derived fields stayed 32871/32879. A diagnostic refresh made both fields current on all eight ranks, but eight-cycle accepted outputs stayed low (127 total; from cycle 3 roughly one/slot/cycle). That isolated candidate was reverted. Earlier per-group KV slot updates were invalid for compressed groups and also reverted; the start_pos-only refresh did not recover acceptance. Full long-range token-level Stock oracle parity remains open; cross-service token comparison is confounded by Stock self-run differences and unmatched full KV history.

Next discriminate target versus proposal with same-state evidence, then test the other DSA derived buffers, especially SAS/QLI metadata, without retaining generic ModelRunner in the product hot path. Keep the independent Runtime boundary and do not claim a throughput fix or rerun the formal 48-request A/B until semantic/acceptance recovery is shown. Evidence: evidence/20260923_loop035_diagnostic/ and TaskCtl Loop035 runs.

Loop035 follow-up: direct SWA slot mapping refresh passed a bootstrap equality gate on all bound SWA groups and completed eight real-weight cycles on all TP ranks, but yielded only 161 accepted outputs total (cycle 1 onward 1.25–1.83 outputs/slot). This isolated candidate was reverted. Across three diagnostic traces, every adjacent slot transition preserved committed draft, last accepted token, and num_computed exactly (84/84 each); the first draft/target prediction match often fell to 0–4/12 slots after cycle 0. The divergence is upstream of the product state carry, within target verification and/or proposal quality. Evidence: run10/summary.json and first_draft_analysis.json in the Loop035 diagnostic directory.

Loop035 oracle metadata control: a diagnostic callback rebuilt DSA target metadata each cycle using the existing builder and a private CPU sequence-length view. The first version wrote the shared DSpark host mirror and failed the local gate; its 211-output trace is invalid. The corrected DSA-only run passed all eight ranks, exact host mirror and local state checks, and produced 209 outputs over eight cycles (cycle means 1.67–2.42 after cycle 0). Rebuilding GDN metadata too passed the same gates and produced 215 outputs (cycle means 1.67–2.75 after cycle 0). These runs use direct eager target and unordered prompt admission; their output differences cannot establish causal acceptance improvement or Stock token parity. No builder callback was retained in the product Runtime. The next decisive gate is a same-state proposer comparison from a single target result and acceptance tensor; subsequent long token oracle and full-chain graph profile remain open.

Loop035 same-state proposal gate: after a single target pass and acceptance, a touched-cache transaction restored 69 entries exactly on all eight TP ranks. Extreme and Stock DSpark proposer then consumed the same target result and acceptance counts; their 84-token draft outputs matched 71/84 on each rank. Differences were confined to two request slots, both with one accepted output (6 and 7 draft-token mismatches). This is an observed first-proposal difference, but the snapshot excludes any proposer-private mutable state and Stock self-replay has not yet been checked. The next Run records exact input fields and a Stock repeat before assigning cause.

Loop035 proposer replay control (run15): all eight TP ranks matched the Product and Stock proposer input tensors and prepared indices/lengths exactly after one shared target/acceptance pass. The 69-entry target-asset snapshot restored exactly twice. Product versus Stock drafts matched 68/84 tokens, but Stock versus its own repeat matched only 67/84. The snapshot captured target positions only; the seven subsequent DSpark KV writes can lie outside it. This invalidates a deterministic Product/Stock parity claim from runs14-15. Run16 is extending the transactional snapshot to the draft group's future physical slots before repeating the gate. No throughput or semantic repair is claimed.

Run16 expanded the diagnostic snapshot to 16 cache tensors across DSpark draft groups 2 and 3. All eight ranks restored every captured entry exactly, yet Stock self-replay matched only 69/84 draft tokens. Source audit then found the snapshot used the KV config block size while DSpark's input expansion addresses each group with its kernel block size and may write beyond the first seven query slots. Run17 corrects the address formula and bounds the snapshot across the draft continuation. Run16 does not prove intrinsic nondeterminism or Product semantic divergence.

Run17 used DSpark's per-group kernel block size (32 for draft groups 2 and 3) and a wider bounded future-slot snapshot for 16 cache tensors. All eight rank snapshots restored exactly. Stock self-replay improved to 75/84 and Product/Stock to 67/84 draft tokens on this request cohort; all 12 first-draft tokens matched in both comparisons. Full draft parity is still not controlled, but this cohort gives no evidence of a first-token proposer split at the handoff. Run18 is collecting 1024-cycle top-level and DSpark-substage NPU event timings with diagnostic clones disabled. The low sustained acceptance and long token oracle remain unresolved.

Run18 completed the full 1024-cycle Extreme-owned c12 decode DAG on all eight ranks (FULL target graph, exact fixed state and host mirrors). Stage NPU event medians after 16 warm cycles: target 45.510 ms, DSpark 5.943 ms, acceptance 0.305 ms, prepare 0.210 ms, advance 0.020 ms. Inside DSpark, model 5.865 ms dominates. Rank-0 emitted 12,797 tokens in 53.571 s, a 238.99 tok/s standalone diagnostic; formal E2E remains Loop034 217.342 versus Stock 543.655. Crucially outputs per slot per cycle decayed to 1.099 by cycles 64–127 and exactly 1.000 in cycles 960–1023. Stock's first eight trace cycles were also modest (2.135 mean) but later windows reached 4.094 and long-run counters imply 3.908 outputs/iteration. Early-cycle comparisons are not a sufficient parity gate. Next test whether full DSA+GDN metadata rebuilding prevents the long-run collapse; then locate the first target/proposer token/state divergence with matched prompt and KV history. No structural optimization or E2E A/B is justified yet.

Run19 extended the oracle-only DSA+GDN per-cycle metadata builder refresh to 256 c12 cycles with a direct eager target. Eight ranks completed with exact fixed state and host mirrors, but outputs/slot/cycle still fell to 1.176 in cycles 64–127 and 1.036 in 128–191. The builder control alone does not prevent the collapse; unordered prompts and target mode prevent interpreting the small difference from Run18 as a causal improvement. The next isolated candidate is draft group 2/3 context slot mapping refresh using the group-specific kernel block size, with exact bootstrap parity required before any continuous decode.

Run20 isolated DSpark draft group 2/3 context slot mapping refresh under a bootstrap equality gate. All eight ranks passed 256 FULL-graph target cycles. At cycle 0 both groups matched the generic bootstrap mappings; at cycle 1 group 2 was stale in all 96 context positions while group 3 stayed exact. This proves an additional derived-state gap, but the isolated refresh did not recover proposal quality: outputs/slot/cycle were 1.013 in cycles 64–127 and 1.018 in 128–191. Do not KEEP the isolated refresh. The env-gated diagnostic remains temporarily available for a combined DSA+GDN builder plus draft-slot control; afterward remove rejected diagnostic code from the product handoff. Formal E2E A/B still waits on semantic recovery.

Run21 combined DSA+GDN per-cycle target metadata rebuilding with the parity-gated draft gid2 slot refresh. Eight ranks passed 256 direct eager cycles, but outputs/slot/cycle were 1.013 in cycles 64–127 and 1.003 in 128–191. This rejects the interaction hypothesis; the draft slot refresh was removed from the product handoff after preserving its patch and evidence. The next oracle-only discriminator is to run the reference BlockTable slot-mapping update for every target cache group before rebuilding metadata. This directly tests whether stale target physical KV write addresses, including compressed groups, drive the progressive collapse. The generic update is diagnostic only and must not enter the independent Runtime hot path.

Loop035 evidence correction (2026-09-23): Run22 exposed a missing diagnostic hook in DirectTargetHandoff.forward. Before commit a11a71b, assigning diagnostic_metadata_refresh had no effect. Therefore Run12, Run13, Run19, and the target-builder component of Run21 did not actually rebuild DSA/GDN metadata. Their completed cycles and acceptance/state observations remain valid, but prior claims that a builder refresh was tested or failed to recover acceptance are withdrawn. Run22 likewise completed cycles but its target-slot oracle was not exercised and was marked invalid. Run23 is the first connected callback test, with an explicit nonempty slot audit and cycle-0 equality gate.

Run23 (connected combined oracle) passed 8/8 rank local gates for 256 eager target cycles. All six target slot groups matched at cycle 0; five groups changed 96/96 slots by cycle 1, and the audit contained 24 entries/rank. Outputs/slot/cycle reached 7.480 in cycles 128-191 and 7.728 in 192-255, with long stretches where all 12 slots accepted all eight outputs. This is a strong combined-effect signal but suspiciously exceeds Stock's roughly 3.908 long-run output rate; false acceptance or degenerate token loops remain possible. Do not claim a semantic fix or E2E improvement. Run24 isolates connected DSA+GDN builder refresh without slot recomputation, requiring 256 callback invocations; a token-level oracle remains necessary.
