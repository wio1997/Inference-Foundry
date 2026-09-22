# DeepSeek Extreme P0 — Short Handoff

## Entry

- SSH: `ssh 61.241.77.34-60008`
- Repo: `/data/wio/Inference_Foundry`
- GitHub: `https://github.com/wio1997/Inference-Foundry`, branch `main`
- Container: `dsv4ab` (privileged, host network)
- Image: `quay.io/ascend/vllm-ascend:v0.26.0rc1`
- Model: `/data/yxy/DeepSeek-V4-Flash-0731-w4a8`
- Framework: `/data/wio/vllm_ascend_26/framework/vllm-ascend`, clean at `36589852a`

GitHub contains the project state, scripts and committed evidence. Continuing experiments also requires this server: model weights, image, framework checkout and NPUs are not stored in the GitHub repo.

## Resume

```bash
ssh 61.241.77.34-60008
cd /data/wio/Inference_Foundry
git pull --ff-only
cat AGENTS.md HANDOFF.md PROJECT_STATE.md PERFORMANCE_MAP.md ACHIEVABLE_BOUND.md RESULTS.md
python3 scripts/taskctl.py resume --task-dir tasks/deepseek-extreme-p0
npu-smi info
docker ps -a --filter name=dsv4ab
git status --short
```

Current checkpoint is active Loop034. Loop033 already promoted fixed target
graph replay: 64 real-weight cycles passed on all eight ranks, with exact state
and Host mirrors, 53.286 ms median cycle wall and 294.400 tok/s internal decode
window. This internal number is not comparable with the accepted Stock E2E
baseline `543.65 tok/s`.

Loop034 implements the first fixed serving shell: c12 remains inside Extreme
Runtime until the 1024-token limit, accepted tokens drain once from fixed device
history, and the serving control plane receives one bulk completion frame.
Bootstrap reserves 1024 KV/DSA lookahead tokens per request before handoff.
CPU exact-trim, varied-acceptance and serialization gates pass. Evidence is in
`evidence/20260922_loop034_fixed_serving/run1/`.

The formal warmup plus three-run 48×32K→1024 c12 A/B is prepared in
`scripts/run_loop034_extreme_e2e.sh`. At the latest checkpoint an unrelated
W8A8 service on port 8300 owns all eight NPUs. Do not terminate it. Once the
cards are stably free, run the launcher; it includes a 60-second safety check.
Do not redo Loop029–033 correctness/profile experiments.

## Start baseline service

The container already exists. If stopped:

```bash
docker start dsv4ab
```

After confirming all eight NPUs are free, start the frozen DP1×TP8 DSpark7 service:

```bash
docker exec dsv4ab bash -lc 'cd /data/wio/Inference_Foundry && MAX_MODEL_LEN=1048576 RUN_TS=RESUME-$(date +%Y%m%d-%H%M) bash scripts/serve.sh'
tail -f logs/serve_dsv4f-w4a8_8npu_dp1tp8_mlen1M_nomooncake_RESUME-*.log
curl -f http://127.0.0.1:8080/health
```

Startup normally takes several minutes. Do not launch a second service while port 8080 or an NPU process is active.

## Stop service

```bash
docker exec dsv4ab bash -lc 'pkill -9 -x "VLLM::EngineCor" 2>/dev/null || true; pkill -9 -x "VLLM::DPCoordin" 2>/dev/null || true; pkill -9 -x "VLLM::Worker_DP" 2>/dev/null || true; pkill -9 -x "VLLM::APIServer" 2>/dev/null || true; pkill -9 -x vllm 2>/dev/null || true'
npu-smi info
```

## Minimal prompt for a new conversation

> SSH `61.241.77.34-60008`, continue `/data/wio/Inference_Foundry` from `main`. Read root `AGENTS.md`, `HANDOFF.md` and the TaskCtl resume pack. Continue active Loop034. After confirming the external port-8300 W8A8 job has released all eight cards, run `scripts/run_loop034_extreme_e2e.sh` for the formal warm-cache 48×32K→1024 c12 A/B. Do not redo Loop029–033 evidence.
