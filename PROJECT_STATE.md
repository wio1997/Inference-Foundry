# DeepSeek Extreme P0 — Project State

Updated: 2026-09-20 16:06 UTC. Evidence maturity: **E2** for warm-cache DP1/TP8 service; prefill and critical-path mechanism remain unmeasured.

## Fixed contract

- DeepSeek V4 Flash W4A8 at `/data/yxy/DeepSeek-V4-Flash-0731-w4a8`; one host with 8 × Ascend 910B3; DP=1, TP=8; correct complete prefill/decode service.
- Reference: privileged `dsv4ab` container, vLLM 0.26.0, vLLM-Ascend 0.26.0rc1; source commits and image ID in `evidence/20260920_baseline/freeze.txt`.
- Primary measured workload: frozen 48-request dataset hash in `freeze.txt`, 32K input, 1024 output, concurrency 12, temperature 0, ignore EOS, 90% repeated prefix. **Warm-cache protocol:** run one full dataset pass before measuring; hold service/code/params unchanged. First cold pass is not comparable with steady repeats.
- Correctness gate for candidates: deterministic 4-prompt, 128-token output equality with `evidence/20260920_baseline/golden4.json`, plus 48/48 benchmark success and no material regression.
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
