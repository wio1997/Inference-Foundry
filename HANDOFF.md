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

## Loop035 checkpoint (2026-09-23 11:57 UTC)

The specialized Runtime 1024-cycle DAG profile (Run18) found target 45.510 ms,
DSpark proposer 5.943 ms and acceptance 0.305 ms median NPU time. Acceptance
fell to exactly 1.000 output/slot/cycle late in the run. Formal E2E remains
Loop034 Extreme 217.342 versus Stock 543.655 tok/s; no new formal A/B has run.

Run22 revealed that an intended bootstrap diagnostic metadata callback had
never been invoked. Claims of DSA/GDN builder effectiveness from Runs12/13/19/21
were corrected in PROJECT_STATE.md and RESULTS.md. The hook was connected in
a11a71b. Connected builder-only Run24 reached 1.836 output/slot/cycle in
cycles128-191; target-slot-only Run25 reached 1.065. Their combination Run23
reached 7.480, but this unusually high acceptance is not yet a semantic fix.

Exact API token captures for the same 12 prompts and 1024 outputs were made for
Stock and combined-oracle Extreme. Both returned 12/12 length-exact outputs,
but 0/12 matched across services. Crucially, Extreme self-repeat and Stock
same-service self-repeat also matched 0/12; Stock first output mismatch was
token 7-78. Thus cross-run generation is not a causal token oracle on this
configuration. Run29 used same-state target self-replay with exact restoration
of 69 touched cache/mutable entries: target argmax still differed at 1-9 of
96 positions per cycle, while all 12 acceptance counts stayed equal.

Run30/31 A/B/C controls were invalid due old metadata tensor aliases. Runs32-34
used a valid A/C self-replay control on all eight ranks. At cycle 1, combined
builder+slots gave A/B 75/96 and A/C 92/96 target argmax; builder path gave
A/B 64/96 and A/C 91/96; slot-only gave A/B 88/96 and A/C 88/96.
Thus the builder-derived DSA/GDN state is the first detectable target
prediction divergence; generic slot mapping alone did not exceed replay noise.
The builder path also rewrites SWA group-2 slots, so it is not purely metadata.
No semantic correctness or sustained acceptance fix is established. Next build
a minimal Runtime-owned derived-state updater from source, compare tensor/state
parity to the builder oracle, and validate continuous decode before formal E2E
A/B. No service is currently running. The framework diagnostic patch snapshot
is patches/loop035_current_framework_model_runner.patch.

Agent orchestration was repaired independently in b53b912. The active main
agent is GPT-6 Sol. The new scripts/delegate_zcode.py ran an actual read-only
DeepSeek Flash Zcode task; its session ID, provider model I/O trace, response
and exit status are in evidence/20260923_agent_orchestration/probe1/.
TaskCtl remains an evidence manager, not a model router.

## Loop035 native target metadata checkpoint (2026-09-23 14:06 UTC)

Runs35–37 are opt-in correctness diagnostics for a Runtime-owned fixed c12/TP8
DSA-CP metadata updater. Run35 was invalid due an outer-list traversal bug in
bootstrap source extraction. Run36 completed two cycles on all eight ranks:
45/55 fields exact at cycle0 and 42/55 at cycle1; SAS/QLI buffers and three
compressed state-cache slot mappings differed. Run37 added reference-to-reference
self-replay from the same restored old metadata tensors. SAS/QLI tails were
non-exact even on self-replay, but native SAS header index4 and QLI header
index12 differed before that noise on all ranks; the three compressed slots
were native-only differences at cycle1. Native semantic parity is not proven.
Evidence is `evidence/20260923_loop035_diagnostic/run35/` through `run37/`.

A source-backed binder correction now restricts raw SWA slot updates to layer
names ending `swa_cache`. Run38 is in progress with an operator argument trace
to locate the first SAS header mismatch. No new formal E2E A/B has run; the
Loop034 baseline remains Extreme 217.342 versus Stock 543.655 tok/s.
EOF'
Run38 traced all SAS metadata operator arguments on 8 ranks for c1/c4/c128.
Reference/native scalar arguments and tensor values matched; the sole input
difference was `cu_seqlens_q` dtype: reference int32, native int64. PyTorch
`cumsum` had promoted the fixed local query-start vector. The native Runtime
now requests int32 cumsum explicitly; Run39 is validating field parity. With
raw SWA writes restricted to actual `swa_cache` groups, Run38 no longer had the
three compressed state-cache slot differences seen in Run37. Do not infer
long-running acceptance or token correctness from this two-cycle field gate.
Evidence: `evidence/20260923_loop035_diagnostic/run38/summary.json`.
EOF'
Run39 completed the int32 correction on all eight ranks and two cycles. For
all c1/c4/c128 SAS calls, reference/native operator arguments and first 32
output values now match exactly. All 45 non-SAS/QLI fields in each cycle match;
SAS first difference is index 97 or later, QLI index 25 or later, in regions
that are unstable under reference self-replay. This passes the defined metadata
header/input gate, but full-buffer and target/acceptance semantic equivalence
remain open. Next perform same-state target reference/native/reference control
with exact old metadata and physical KV restoration, then continuous decode.
Evidence: `evidence/20260923_loop035_diagnostic/run39/summary.json`.
EOF'