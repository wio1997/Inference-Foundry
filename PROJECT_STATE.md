# DeepSeek Extreme P0 — Project State

Updated: 2026-09-20 UTC. Task status: E0 → E1, live DP1/TP8 baseline service loading.

## Contract

- Model: `/data/yxy/DeepSeek-V4-Flash-0731-w4a8`, DeepSeek V4 Flash W4A8.
- Hardware: one S900K3-49 host, 8 × Ascend 910B3, DP=1, TP=8.
- Reference: vLLM 0.26.0 + vLLM-Ascend 0.26.0rc1 privileged container `dsv4ab`.
- Goal: correct complete prefill/decode service with materially better end-to-end performance than the frozen baseline; follow measured gap to achievable hardware bound.
- Protected workload: `GSM8K-in32768-num48-DeepSeek-V4-Flash-0731-w4a8-repeatRate0.9.jsonl`, 48 requests, concurrency 12, max output 1024, temperature 0, ignore EOS, prefix cache enabled.
- Required gates for any KEEP: same model semantics; deterministic output comparison or equivalent correctness check; same workload and runtime for before/after; performance gain outside observed noise; no major regression in TTFT, TPOT, output TPS.

## Current state

- Initial Git repository: `d79b82c`; frozen benchmark protocol: `0453007`.
- At 15:31 UTC, NPU 0–7 reported no running processes and ~3.4 GB idle HBM occupancy per card. Existing containers were left untouched.
- Baseline service started at ~15:35 UTC with `scripts/serve.sh`; log path is `logs/serve_dsv4f-w4a8_8npu_dp1tp8_mlen1M_nomooncake_P0-BASE-20260920.log`.
- Freeze data: `evidence/20260920_baseline/`; model weight inventory generated from safetensors headers.
- Existing `/data/wio/vllm_ascend_26` experiments are historical evidence only. Its older DP1/TP8 run was at different source commits; do not use it as the frozen baseline for this task.
- Framework source trees are shared bind mounts. Do not alter them until the baseline is complete and a clean isolated candidate path is prepared.

## Next actions

1. Poll service readiness and log errors. Confirm all 8 ranks and output correctness with a short deterministic request.
2. Run the frozen 48-request benchmark with `scripts/bench.py`; repeat or cross-check if metrics are noisy or protocol differs from historical aisbench.
3. Record device/host timing or profiling without inserting synchronization. Complete Performance Map V0 and Achievable Bound V0 with measured and unknown terms distinguished.
4. Select a single falsifiable high-value gap, then implement, test, benchmark, KEEP/REJECT, update four state files and commit.
5. Continue the next loop while a clear, testable high-value gap remains.

## Resume

`git log -3 --oneline`; read this file, `PERFORMANCE_MAP.md`, `ACHIEVABLE_BOUND.md`, `RESULTS.md`, then only referenced evidence. Check `npu-smi info`, `docker ps`, and port 8080 before launching anything. Do not kill unrelated processes. Each completed loop must leave a Git commit. Raw service log is in `logs/` on this host; distilled evidence must be committed under `evidence/`.
