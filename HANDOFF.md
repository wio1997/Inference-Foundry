# DeepSeek Extreme P0 — Short Handoff

## Entry

- SSH: `ssh 61.241.77.34-60008`
- Repo: `/data/wio/Inference_Foundry`
- GitHub: `https://github.com/wio1997/Inference-Foundry`, branch `main`
- Container: `vllm-ascend26-dsv4f-w4a8` (privileged, host network)
- Image: `quay.io/ascend/vllm-ascend:v0.26.0rc1`
- Model: `/data/yxy/DeepSeek-V4-Flash-0731-w4a8`
- Framework: `/data/wio/vllm_ascend_26/framework/{vllm,vllm-ascend}` with the
  Loop034 serving patches preserved in `patches/`

GitHub contains the project state, scripts and committed evidence. Continuing experiments also requires this server: model weights, image, framework checkout and NPUs are not stored in the GitHub repo.

## Resume

```bash
ssh 61.241.77.34-60008
cd /data/wio/Inference_Foundry
git pull --ff-only
cat AGENTS.md HANDOFF.md PROJECT_STATE.md PERFORMANCE_MAP.md ACHIEVABLE_BOUND.md RESULTS.md
python3 scripts/taskctl.py resume --task-dir tasks/deepseek-extreme-p0
npu-smi info
docker ps -a --filter name=vllm-ascend26-dsv4f-w4a8
git status --short
```

Loop034 is complete. The first full-serving Extreme Runtime passed the frozen
warm-cache `48×32K→1024, c12` protocol: warmup plus three measured runs all
finished 48/48 requests at exactly 1024 output tokens. Across all four
workloads, 16 cohorts produced 128 passing rank records; each cohort stayed in
the runtime-owned decode loop and returned one terminal bulk frame, with no
per-cycle ModelRunner/Scheduler re-entry.

Formal output TPS was `217.342 / 218.884 / 215.889`, median `217.342 tok/s`.
Median-of-runs TTFT p50 was `1942.120 ms` and TPOT p50 `53.298 ms`. This is a
valid same-protocol result and is `60.022%` below Stock `543.655 tok/s`.
Rank-0 cohort wall median was `53.373 s` for about 1025 cycles, so serving
bookkeeping is no longer the principal gap; sustained Extreme decode is.
Evidence is under `evidence/20260922_loop034_fixed_serving/e2e/`.

Next open a profile-driven loop over the full 1024-cycle runtime chain and
attribute target graph replay, proposer, TP/EP communication, acceptance/state
and launch gaps before choosing a structural optimization. Do not redo
Loop029–034 correctness or the formal A/B.

## Start baseline service

The container already exists. If stopped:

```bash
docker start vllm-ascend26-dsv4f-w4a8
```

After confirming all eight NPUs are free, start the frozen DP1×TP8 DSpark7 service:

```bash
docker exec vllm-ascend26-dsv4f-w4a8 bash -lc 'cd /data/wio/Inference_Foundry && MAX_MODEL_LEN=1048576 RUN_TS=RESUME-$(date +%Y%m%d-%H%M) bash scripts/serve.sh'
tail -f logs/serve_dsv4f-w4a8_8npu_dp1tp8_mlen1M_nomooncake_RESUME-*.log
curl -f http://127.0.0.1:8080/health
```

Startup normally takes several minutes. Do not launch a second service while port 8080 or an NPU process is active.

## Stop service

```bash
docker exec vllm-ascend26-dsv4f-w4a8 bash -lc 'pkill -9 -x "VLLM::EngineCor" 2>/dev/null || true; pkill -9 -x "VLLM::DPCoordin" 2>/dev/null || true; pkill -9 -x "VLLM::Worker_DP" 2>/dev/null || true; pkill -9 -x "VLLM::APIServer" 2>/dev/null || true; pkill -9 -x vllm 2>/dev/null || true'
npu-smi info
```

## Minimal prompt for a new conversation

> SSH `61.241.77.34-60008`, continue `/data/wio/Inference_Foundry` from `main`. Read root `AGENTS.md`, `HANDOFF.md` and the TaskCtl resume pack. Loop034 has completed the formal warm-cache A/B at median 217.342 tok/s versus Stock 543.655. Start the next profile-driven loop on the full 1024-cycle Extreme-owned chain, then choose the highest-value structural optimization from evidence. Do not redo Loop029–034 evidence.
## Loop035 checkpoint (2026-09-23 06:36 UTC)

The first short real-weight Extreme DAG profile found target ~45.07 ms/cycle and proposer ~5.93 ms/cycle (NPU event medians); the dominant gap remains low accepted outputs. Stock warm 12-request long decode accepted 2.908 drafts/iteration. At Extreme cycle 1, target position/seq_len advance while DSA-CP start_pos and local_seq_lens remain at bootstrap values. Updating only those fields did not recover acceptance, so the candidate was reverted. See PROJECT_STATE.md and evidence/20260923_loop035_diagnostic/. Next obtain same-state target/proposer discrimination and inspect SAS/QLI derived metadata. Do not repeat formal Loop034 A/B until acceptance and long token correctness improve.
