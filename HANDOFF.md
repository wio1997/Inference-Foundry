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

Run38 traced all SAS metadata operator arguments on 8 ranks for c1/c4/c128.
Reference/native scalar arguments and tensor values matched; the sole input
difference was `cu_seqlens_q` dtype: reference int32, native int64. PyTorch
`cumsum` had promoted the fixed local query-start vector. The native Runtime
now requests int32 cumsum explicitly; Run39 is validating field parity. With
raw SWA writes restricted to actual `swa_cache` groups, Run38 no longer had the
three compressed state-cache slot differences seen in Run37. Do not infer
long-running acceptance or token correctness from this two-cycle field gate.
Evidence: `evidence/20260923_loop035_diagnostic/run38/summary.json`.

Run39 completed the int32 correction on all eight ranks and two cycles. For
all c1/c4/c128 SAS calls, reference/native operator arguments and first 32
output values now match exactly. All 45 non-SAS/QLI fields in each cycle match;
SAS first difference is index 97 or later, QLI index 25 or later, in regions
that are unstable under reference self-replay. This passes the defined metadata
header/input gate, but full-buffer and target/acceptance semantic equivalence
remain open. Next perform same-state target reference/native/reference control
with exact old metadata and physical KV restoration, then continuous decode.
Evidence: `evidence/20260923_loop035_diagnostic/run39/summary.json`.

Run40 completed an 8-rank same-state DSA-only target A/native B/restored
reference C control. Per cycle, A/B argmax matched 92/96 and 94/96; A/C
self-replay matched 93/96 and 92/96. A/B acceptance counts matched 12/12 in
both cycles; A/C differed in one count in cycle0. All 138 physical KV entries,
old metadata tensors and common fields were restored, and each rank passed the
host mirror/state gates. Four large reused metadata tensors were excluded from
snapshot, with A/C returning to normal replay noise. This supports isolated
native DSA target/acceptance parity for two cycles, not long token equivalence.
Run41 is now measuring 256 continuous FULL-graph cycles with native DSA,
including the new metadata stage in the Runtime DAG profile.
Evidence: `evidence/20260923_loop035_diagnostic/run40/summary.json`.

Run41 completed 256 native-DSA c12 cycles on all 8 ranks with exact state and
host mirrors. Its target graph flag was omitted: records explicitly show
`target_graph_requested=false`, `target_graph_mode=NONE`. This eager trace is
not comparable with Run18 FULL-graph stage times or formal E2E. It emitted
5,370 tokens/rank, with outputs/slot/cycle 1.940 in cycles128–191 and 1.729
in cycles192–255, but no long semantic oracle was run. Run42 is the corrected
FULL-graph 256-cycle native-DSA profile, with no extra diagnostic clones.
Evidence: `evidence/20260923_loop035_diagnostic/run41/summary.json`.

Run42 corrected the graph flag and completed 256 FULL-graph native-DSA cycles
on all 8 ranks with exact state and host mirrors. NPU event medians (after 16
cycles) were derived metadata 8.616ms, target46.201ms, proposer5.880ms,
acceptance0.331ms. Rank0 emitted4,894 tokens in15.830s (309.16tok/s short
standalone diagnostic). Outputs/slot/cycle were1.665 in cycles128–191 and
1.594 in192–255. This short run is not the formal 48-request E2E protocol;
acceptance remains low. Run43 tests native GDN prior accepted-count binding
alongside native DSA in same-state reference/native/reference target control.
Evidence: `evidence/20260923_loop035_diagnostic/run42/summary.json`.

Run43 did not reach target A/B/C: on all 8 ranks, bootstrap rejected the
assumption that the initial `attn_metadata` dict already exposed a GDN
speculative count view. It is INVALID and has no target/acceptance inference.
Source inspection identified the graph-stable `num_accepted_tokens` buffer on
`GDNAttentionMetadataBuilder`; bootstrap now hands that tensor over once,
without retaining the builder in Runtime. Run44 is redoing the two-cycle
combined DSA+GDN same-state control, including count values before reference,
after reference A, and after native B. Evidence: run43/summary.json.

## Loop035 GDN hypothesis correction (2026-09-23 15:50 UTC)

Run43 and Run44 both failed their explicit bootstrap gate before target work.
Run43 found no GDN count view in the initial metadata; Run44 found no GDN
metadata builder among active attention groups on all eight ranks. The model
config and `vllm/vllm/models/deepseek_v4/` source contain no GDN architecture.
Thus prior descriptions of a "DSA+GDN builder" control overstate what was
executed for DeepSeek V4 Flash: the connected callback rebuilt active DSA/SWA
metadata, while its generic GDN branch was not taken. No earlier acceptance
or target result is evidence about GDN. The optional GDN Runtime binding has
been removed. Return to DSA, SWA/compressed KV, target and DSpark proposal
causal controls. Evidence: run43/summary.json and run44/summary.json.

## Loop035 Run45 late native DSA control (2026-09-23 16:05 UTC)

FULL-graph 129-cycle A/reference, B/native, C/restored-reference target control sampled cycles 0,1,64,128. All 8 ranks passed physical KV restoration, metadata restoration, exact state advance and host mirrors. At cycle64 A/B argmax was 74/96 versus A/C 72/96; at cycle128 78/96 versus 76/96. Accepted-token equality was 90/96 versus 90/96, then 89/96 versus 91/96. Native DSA difference is within reference self-replay noise at sampled late states; reference self-replay is too noisy to certify long token equivalence. Sustained low acceptance remains unresolved. Next discriminate DSpark proposal and target/KV using effective tokens and controlled same-state replay. Evidence: `evidence/20260923_loop035_diagnostic/run45/summary.json`.

## Loop035 Run46 late DSpark proposer control (2026-09-23 16:22 UTC)

After 128 continuous native-DSA FULL-graph cycles, all 8 ranks passed same-state Product/Stock/Stock proposer input, prepare-field and physical KV restoration gates. Product/Stock first draft token was 12/12 equal; full draft 83/84 equal versus Stock self-replay 79/84. Thus sampled late Product proposer implementation is not the observed low-acceptance root. This does not compare full serving trajectories: next establish effective token-level Stock oracle and first target/KV state divergence from the shared initial state. Evidence: `evidence/20260923_loop035_diagnostic/run46/summary.json`.

## Loop035 Run47 Stock output self-control (2026-09-23 16:39 UTC)

Identical 12×32K→1024 c12 Stock requests were submitted twice to one service with temperature0 and ignore_eos. Both were 12/12 complete and 1024 tokens/request, yet exact output text matched 0/12; earliest differing character across requests was 38, longest common prefix ranged 38–920. Independent Stock-vs-Extreme output text cannot serve as a first-divergence oracle. Run47 is INVALID for that method. Next perform Stock/direct target A/B/C within one fixed KV state with Stock self-replay control, comparing first/effective target tokens, accepted counts and physical cache restores. Evidence: `run47/summary.json`.

## Loop035 Run48 diagnostic binding correction (2026-09-23 16:56 UTC)

Run48 Stock/direct/Stock target control reached its diagnostic bootstrap on all 8 ranks but failed before target A/B/C because the new mode did not request the per-group common metadata views from the Stock builder. No target conclusion follows. The diagnostic gate now captures those views when `EXTREME_STOCK_TARGET_ABA=1`; Run49 is starting with the same cycle128 and fixed cohort protocol. Product Runtime code is unchanged. Evidence: `run48/summary.json`.

## Loop035 Run49 late Stock/direct target control (2026-09-23 17:10 UTC)

At the 128th fixed Stock decode cycle, native-DSA direct target B was run between Stock target A and restored Stock C on the exact same physical KV and metadata state. All 8 ranks passed physical KV, metadata and common-field restore gates. A/B first target token matched 12/12, full argmax 96/96, accepted tokens 96/96 and all 12 accepted counts; A/C full argmax was 95/96. Stock counts were [6,4,6,8,7,6,1,7,6,4,8,1]. This excludes a direct target invocation discrepancy on that sampled Stock state. It does not prove continuous Extreme state evolution; next compare KV/DSA and state transition writes after the first target/proposer cycle against Stock, then trace first split. The intentional sentinel truncated client streams, so bench output is not throughput evidence. `run49/summary.json`.

## Loop035 Run50 post-target KV write control (2026-09-23 17:24 UTC)

At Stock fixed cycle128, Stock A/direct B/Stock C touched-cache writes were compared after each target from the same pre-state. All 8 ranks restored 69 selected rows exactly between calls; first target token matched 12/12 A/B and A/C. A/B and Stock A/C each differed in the same 19/69 cache rows, with broadly comparable element counts. Raw cache value equality is dominated by Stock self-replay noise and does not identify a direct target cache-write defect. Next implement a dedicated continuous lockstep harness to compare causal state/KV transitions and effective accepted tokens, rather than adding further ad hoc ModelRunner branches. Evidence: `run50/summary.json`.

## Loop035 Run51 late DSA field parity (2026-09-23 17:40 UTC)

After 129 native-DSA FULL-graph continuous cycles, reference/native/reference metadata comparison sampled cycles 0,1,64,128. On all 8 ranks at every sample, all 45 non-SAS/QLI fields matched exactly; each of three SAS operator calls had identical arguments and identical first32 output values. The remaining ten SAS/QLI tensor tails were unequal even in reference self-replay, so they cannot establish native divergence. Host mirrors remained exact. This narrows the continuous acceptance investigation to block allocation/physical KV or other long-lived binding state; next audit the fixed Runtime block table through 1024 output. `run51/summary.json`.

## Loop035 Run52 standalone KV allocation failure and protocol correction (2026-09-23 17:58 UTC)

An opt-in audit inside `FixedDecodeRuntime.prepare_target_inputs()` sampled 256 native-DSA FULL-graph standalone cycles. On all 8 ranks, the borrowed bootstrap block table supplied physical block0 to 66/96 target positions at cycle16 and 96/96 from cycle32, across 12 slots. Thus standalone state/host counters were insufficient: without allocator ownership, later KV writes alias block0. This Run is a correctness FAIL for the standalone no-reservation protocol. Crucial scope: Loop034 formal Extreme serving sets `EXTREME_RUNTIME_SERVE=1` and `EXTREME_RUNTIME_RESERVE_TOKENS=1024`; the vLLM scheduler uses that to preallocate lookahead blocks. Run52 cannot explain formal E2E low acceptance. Run53 now audits the reserved serving path with native DSA and reports acceptance windows before choosing a fix. Evidence `run52/summary.json`.

## Loop035 Run53 reserved-serving completion failure (2026-09-23 18:25 UTC)

Run53 used 1024-token scheduler reservation, native DSA and a fixed 12×32K→1024 FULL-graph cohort. All 12 clients returned exactly 1024 tokens, and eight rank gates passed. Audit found no physical block0 target positions through cycle128, but at cycle256 completed slot2 targeted block0 in all eight positions. Rank0 staged 2564 tokens internally for that slot and 2149 excess tokens across the cohort. The shell kept completed slots running until the slowest slot finished. This is a serving KV correctness failure despite client lengths; its single-cohort 366.47 tok/s with auditing is diagnostic, not a formal A/B. Run53 is TaskCtl FAIL. A fixed-slot parking/mask fix is in progress, with 64 extra reserved tokens for the final valid target boundary. Verify on all ranks before repeating formal 48-request A/B. Evidence: `evidence/20260923_loop035_diagnostic/run53/summary.json`.

## Loop035 Run54 fixed-slot parking (2026-09-23 18:36 UTC)

The dedicated Runtime now freezes completed slots' logical output counters and parks their fixed-shape target/proposer replay inside their own reserved KV region. The DSpark Host mirrors are committed and reset to the same parked position; serving requires 1088 scheduler-reserved tokens for a 1024-token output. The first real 12×32K→1024 c12 cohort passed 12/12 client lengths and all eight rank/Host gates. Rank0 excess staged tokens fell from Run53 2149 to 49. Physical block0 was absent at sampled cycles0,1,128,256, but the audit did not sample late cycles257–433; Run55 must sample that interval before formal A/B. Run54's 360.33 tok/s is a single audited cohort, not a same-protocol formal result. See `evidence/20260923_loop035_diagnostic/run54/summary.json`. A separate Zcode/DeepSeek read-only review produced a response but its wrapper timed out at 180 seconds; the main Agent reviewed two applicable guard gaps and added a scheduler-reservation switch check and a short-request parking clamp. This review is not a Runtime correctness gate.

## Loop035 Run55 late parked-slot KV audit (2026-09-23 18:49 UTC)

With 1088-token scheduler reservation and the completed-slot mask/parking fix, a real 12×32K→1024 c12 cohort finished 12/12 requests at exact length. All eight ranks passed Host mirror and execution gates. Physical target block0 and negative mappings were absent at sampled cycles 0,128,256,320,360,384,400,420 on every rank. Multiple rank0 slots show stable parked positions at cycles384–420, rather than advancing beyond reservation. Rank0 staged only 53 excess tokens at completion boundaries. This passes the sampled late KV safety gate. The single-cohort 359.51 tok/s is diagnostic; long token-level semantic equivalence and the acceptance gap versus Stock remain open. Next use valid reserved-serving same-state A/reference, B/native, C/reference target controls at later cycles; do not restart generic ModelRunner patch churn. Evidence: `evidence/20260923_loop035_diagnostic/run55/summary.json`.

## Loop035 Run56 invalid diagnostic gate (2026-09-23 19:03 UTC)

Reserved-serving native target A/B/C did not reach target calls: the reference DSA builder callback requires `EXTREME_DSA_BUILDER_ORACLE=1`, which Run56 omitted. All eight ranks failed the same explicit bootstrap gate. The client benchmark subsequently could not write its JSON because the diagnostic directory had not been created; that is not the rank root cause. Run56 is INVALID and has no target/acceptance inference. Run57 will add the existing builder flag and precreate the directory. Evidence: `evidence/20260923_loop035_diagnostic/run56/summary.json`.

## Loop035 Run57 reserved same-state native DSA target control (2026-09-23 19:17 UTC)

Run57 added the required reference builder callback and sampled target A/reference, B/native DSA, C/restored reference at cycles0,1,64,128,256 after continuous FULL-graph decode with 1088 scheduler-reserved KV tokens. All eight ranks passed physical KV/metadata restores, state advance and Host mirrors. At cycles64/128/256 A/B argmax matched 95/96,93/96,94/96 against A/C self-replay 94/96,93/96,93/96; accepted counts matched all 12 slots for both comparisons. The early c0/c1 small count differences need the self-replay control and do not establish a persistent split. Native DSA target behavior remains within reference replay noise on a valid reserved state. Client sentinel output and TPS are invalid. This does not prove continuous Stock-vs-Extreme token equivalence or explain the remaining acceptance gap. Next trace the first continuous Stock/Product state divergence in a dedicated lockstep harness, with integer positions/slot mappings/counts and cache ownership before expensive value comparisons. Evidence: `evidence/20260923_loop035_diagnostic/run57/summary.json`.

## Loop035 Run58 partial KV snapshot coverage audit (2026-09-23 19:37 UTC)

The transactional cache snapshot now reports partial out-of-range indices as skipped coverage, so a restore can no longer silently look complete when only some mapped rows were captured. A synthetic CPU case passed. Run58 repeated the reserved 257-cycle target A/reference, B/native DSA, C/reference control with this stronger gate. All eight ranks passed, with zero skipped/partial snapshots and exact restores at cycles0,1,64,128,256. At cycles64/128/256 A/B argmax was94/96 each; A/C self replay was95/96,93/96,93/96, and accepted counts matched. This confirms the sampled native DSA result under the stronger recorded-row coverage gate, while full mutable-state coverage and long Stock token equivalence remain open. The diagnostic client's sentinel TPS is invalid. Next inspect continuous Stock input/state and DSpark proposer trajectories; do not repeat the same isolated target parity sample. Evidence: `evidence/20260923_loop035_diagnostic/run58/summary.json`.


## Loop035 Run59–64 continuous state and early target gate (2026-09-23 21:17 UTC)

Run59's read-only Stock observer incorrectly indexed every KV group by absolute position; group1 has a 256-column table, so the observer aborted before inference. Run60 completed 12/12 clients but wrote no 256-cycle shadow file because the fixed c12 window ended earlier. Both are TaskCtl INVALID for their intended shadow gates. Run61 added periodic checkpoints and recorded 128 consecutive fixed c12 cycles on all eight ranks: target input ABI stayed exact, and group0/2/3/4/5 table-to-slot mappings matched with no negative physical blocks. Group1 geometry remains unverified. This does not compare Stock and Product physical trajectories. See run61/summary.json.

Run62 captured the first 32 Stock c12 cycles (target input IDs/positions, argmax, accepted tokens, next draft) on all eight ranks. Run63 captured the first 32 Extreme serving cycles and completed 12/12 requests at 1024 tokens. Only 3/12 slots had identical full target input and positions at cycle0 across these independent services. In aligned slots, Stock/Extreme draft and one target token differed, but Run28/47 already established cross-run Stock non-determinism; this paired trace is a clue, not a causal oracle. Run63 acceptance was 2.55 outputs/slot/cycle over cycles192–255 and 1.78 over cycles256–452 in this diagnostic cohort, not a formal throughput result. Evidence: run62/summary.json and run63/paired_summary.json.

Run64 reused the existing same-state Stock A / independent native-DSA FULL target B / restored Stock C diagnostic at fixed cycle1. All eight ranks passed KV and metadata restoration. A/B target argmax matched 89/96 versus A/C 88/96; A/B accepted tokens matched 96/96 and all 12 counts, versus A/C 91/96. Both A/B and A/C post-target writes matched 50/69 captured rows and had identical mismatch row names. Together with Run49/50 at cycle128 and Run57/58 in reserved Product state, this rules out a clear candidate-specific target/DSA split at sampled early and late states. The intentional sentinel makes client TPS invalid. Evidence: run64/summary.json.

Next implement a dedicated continuous Stock-authoritative lockstep observer that checks product state transitions and KV ownership/write coverage per cycle, with Stock self-replay as noise control. Reuse Run20's confirmed draft group2 slot staleness and Run23–25's combined-effect but semantically unverified controls; do not repeat their isolated experiments. Only after the continuous semantic/acceptance cause is identified should the prepared 48×32K→1024 formal A/B script run.

Agent routing: root AGENTS.md was read and now names GPT-6 Sol as default main Agent, Zcode/DeepSeek only for bounded read-only tasks, and TaskCtl as state/evidence manager rather than model scheduler. evidence/20260923_agent_orchestration/probe1/execution.json records a successful deepseek/deepseek-flash provider call (2 requests); run54_readonly_review records 11 provider requests but a 180-second wrapper timeout. These are actual route records, not merely configuration text.

## Loop035 Run65 continuous target-slot ownership audit (2026-09-24 02:16 UTC)

Run65 added a read-only, Runtime-side audit of all six KV-group slot mappings
for the first eight cycles of a real reserved-serving 12×32K→1024 cohort. All
8 ranks passed the 435-cycle serving and Host-mirror gates; all 12 clients
returned exactly 1024 tokens. At cycle0, the five groups with supported
absolute-position geometry matched their block-table-derived mapping. At
cycle1, groups0/2/4/5 each differed in 96/96 mappings on every rank, and
remained stale through cycle7; group3 matched throughout. Group1 has a
256-column table that cannot be indexed by the absolute positions in this
observer, so its mapping is unclassified. The diagnostic does not mutate KV
or attention state. This is the first repeatable continuous target-slot
divergence under valid scheduler reservation, but it does not yet prove that
every stale buffer is consumed by the active target path or that it causes low
acceptance. Source shows DSA builder formats common slot mappings and decode
operators consume SWA slot mappings; next map each group to live layer
metadata and perform a same-state causal A/B/self-replay with exact cache
restoration. Do not infer long token equivalence or rerun formal E2E yet.
Evidence: `evidence/20260924_loop035_diagnostic/run65/summary.json`.

## Loop035 Run66 slot-refresh same-state target control (2026-09-24 02:34 UTC)

Run66 used valid 1088-token scheduler reservation and sampled product cycles0/1
with target A/reference builder plus all-group slot refresh, B/native DSA
metadata with the existing mappings, C/restored reference. All eight ranks
passed touched physical KV restores, no skipped cache rows, state advance and
Host mirrors. At cycle1, reference refresh changed all 96 mappings in groups
0/1/2/4/5 and none in group3. A/B argmax matched 92/96, versus A/C
self-replay 91/96; accepted tokens matched 92/96 for both comparisons. At
cycle0, when no slots changed, A/B argmax was 93/96 versus A/C 90/96.
The candidate therefore has no clear direct target/acceptance signal beyond
reference self-replay noise in these two cycles. Four large metadata tensors
were omitted from metadata snapshots, so full mutable-state equivalence is
not certified. The diagnostic intentionally terminated client streams and
has no TPS validity. Continue with a dedicated continuous two-lane
state/KV-write trajectory check, especially the DSpark context cache and
physical ownership. Do not promote all-group slot refresh as a semantic fix
from Run65 or Run66. Evidence:
`evidence/20260924_loop035_diagnostic/run66/summary.json`.

## Loop035 Run67 DSpark context slot refresh (2026-09-24 03:10 UTC)

Source ties the previously observed stale DSpark draft gid2 mapping to the
input kernel in `dspark_proposer.py`; that kernel builds the per-layer context
slot mappings used by the Ascend DSpark context-KV scatter. An opt-in
Runtime-owned refresh now updates only draft gid2/3 mappings from the borrowed
block tables immediately before DSpark preparation. It retains no ModelRunner
and is disabled by default pending validation.

Two 12×32K→1024 cohorts in one reserved-serving service passed 12/12 exact
client lengths and 8/8 rank/Host gates each. Gid2 needed 96/96 slot updates
from cycle1 through the sampled first eight cycles, gid3 needed none, and
sampled physical blocks were nonzero. Cohort cycles were 300 and 325;
diagnostic output TPS was 467.92 and 484.46. Run65 without this opt-in took
435 cycles and 357.08 tok/s in its first diagnostic cohort. This is a strong
acceptance signal, not a formal matched A/B: the services were separate,
independent generations are nondeterministic, and the diagnostic has overhead.
Target group mappings still appear stale before target, so do not claim full
semantic repair. Next collect a same-code flag-off control, then an effective
long token/state oracle before promoting the fix or running formal 48-request
E2E. Evidence: `evidence/20260924_loop035_diagnostic/run67/summary.json`.


## Loop035 Run68 same-code flag-off DSpark control (2026-09-24 03:25 UTC)

Run68 kept the Run67 code, serving reservation, target graph, dataset,
12-request concurrency and two-cohort service, but disabled only the opt-in
DSpark context slot refresh. Both cohorts passed 12/12 exact 1024-token client
lengths and all 8 rank/Host gates. They needed 458 and 423 cycles, versus
Run67 flag-on 300 and 325 cycles (34.50% and 23.17% fewer cycles). Diagnostic
output TPS was 342.17/410.12 flag-off versus 467.92/484.46 flag-on. The
repeated controlled direction supports draft gid2 slot staleness as a major
continuous acceptance loss. Independent service generation remains
nondeterministic; no long token-level oracle equivalence is established. Do
not promote a semantic fix or run formal 48-request A/B yet. Next inspect
continuous DSpark KV write ownership and full-state snapshots, then use a
strict long-trajectory oracle with Stock self-replay. Evidence:
`evidence/20260924_loop035_diagnostic/run68/summary.json`.


## Loop035 Run69 DSpark context-write slot provenance (2026-09-24 03:45 UTC)

A dedicated read-only Runtime observer captured the actual DSpark context-slot
buffer after proposal in a flag-off 12×32K→1024 reserved-serving cohort. All
12 clients and 8 ranks passed exact-length/Host gates, with 458 cycles. At
cycle0 both draft groups matched the current block tables. At cycles1–7,
gid2 context slots differed from the current physical slot in 96/96 entries
on every rank, and matched the stale source mapping 96/96; gid3 remained
exact. No sampled draft physical block was zero or negative. The DSpark source
passes this context buffer to `precompute_and_store_context_kv`, which scatters
shared SWA KV using those slots. Combined with Run67/68, this establishes a
concrete stale-address path and its sustained acceptance cost. It does not
certify target writes or long token-level oracle equivalence. Next perform
strict continuous KV ownership/write trajectory and Stock self-replay control
before formal E2E. Evidence: `evidence/20260924_loop035_diagnostic/run69/summary.json`.


## Loop035 Run70 refreshed context-write slot control (2026-09-24 03:59 UTC)

Run70 repeated the Run69 read-only context-slot observer with only the
Runtime draft slot refresh enabled. Across cycles0–7, both gid2 and gid3
actual DSpark context-scatter slots matched their current block-table-derived
physical slots in all 96 positions on all 8 ranks. Gid2 source refresh
changed 96 slots each cycle from cycle1; no sampled physical block was zero.
The cohort passed 12/12 exact 1024-token clients and 8/8 rank/Host gates,
using 295 cycles and 473.02 diagnostic tok/s. Matched observer Run69 flag-off
needed 458 cycles and 341.43 diagnostic tok/s, with stale gid2 context slots
at cycles1–7. Thus the context write-address path and its acceptance effect
are established for the sampled early cycles. Full continuous KV state and
long token-level Stock oracle equivalence remain open. Formal 48-request A/B
must wait for that gate. Evidence:
`evidence/20260924_loop035_diagnostic/run70/summary.json`.


## Loop035 Run71 full product trace gate (2026-09-24 04:19 UTC)

A dedicated offline verifier checked every cycle of a valid refreshed-slot
12×32K→1024 reserved-serving cohort. Rank0 saved 294 consecutive cycles of
target input IDs/positions, target argmax, accepted token rows/counts, and next
draft. All 3,528 slot-cycles satisfied the independent Python greedy-prefix
rule and target-input/state ABI. The checker covered 3,039 active state
transitions and 477 completion-or-park transitions; each of 12 slots reached
1024 client tokens. All 8 ranks passed the original exact-length/Host gates.
Diagnostic throughput was 467.27 tok/s. This establishes continuous internal
Product acceptance and state-machine consistency; it does not compare target
KV values or long token sequence against a Stock same-state oracle. Next
complete physical KV ownership/write coverage and transactional Stock
A/Product B/Stock C replay with complete snapshot coverage. Evidence:
`evidence/20260924_loop035_diagnostic/run71/summary.json` and
`evidence/20260924_loop035_diagnostic/run71/trace_check.json`.


## Loop035 Run72 full DSpark write-address trajectory (2026-09-24 04:36 UTC)

A streaming read-only audit covered all 285 cycles of a refreshed-slot
12×32K→1024 reserved-serving cohort on all 8 ranks: 4,560 ordered
pre-target/post-proposer records, 12/12 exact-length clients, and 8/8
rank/Host gates. Draft gid2/3 actual context-scatter slots matched the
current physical block tables in every cycle. The audit found no negative,
block0, same-cycle duplicate, or cross-request owner-conflicting draft
address. This extends the early Run70 write-address result through completion
and parked slots. Generic pre-target mappings for groups0/2/4/5 were stale
from cycle1 through cycle284; group3 matched. Group0 physical block0 is also
present in Stock Run61, and group1 table geometry is not classified, so these
common-buffer observations do not prove active target writes are wrong.
The full-audit 409.40 tok/s is overhead-contaminated and not a performance
comparison. Physical KV values and long Stock token parity remain open.
Evidence: `evidence/20260924_loop035_diagnostic/run72/summary.json`.

`RuntimeAssets.snapshot_slots(strict=True)` now rejects incomplete cache
selection and invalid candidate slots, and captures full mutable tensor rows.
A CPU synthetic restore/out-of-bounds gate passed. This is a prerequisite for
the next same-state Stock/Product/Stock transaction; the caller must still
supply a complete target and DSpark write-set manifest before claiming full
snapshot coverage.


## Loop035 Run73 cache/metadata alias inventory (2026-09-24 04:57 UTC)

The one-time Runtime handoff emitted cache/metadata pointer manifests on all
8 ranks. Their structures matched: 67 target cache tensors, all mapped to a
KV group. Group0 owns c4 attention/indexer caches (16 tensors), group1 c128
attention (8), groups2/3 SWA and DSpark draft (8 each), and groups4/5
compressor state caches (17/10). The three compressed attention/indexer
metadata sources have no direct slot tensor. Their actual scatter/state
addresses derive from compressor metadata, start position and block tables;
source shows SWA scatter uses req_metadata.slot_mapping, which the native
updater refreshes. Therefore the stale generic pre-target mappings observed
in Run65/72 do not by themselves establish active target write corruption.
The current cache snapshot helper maps physical cache tensors to groups but
its common-slot selection does not certify compressed/state writes. Next
construct a complete candidate write-set from active req_metadata and
compressor output, then use strict snapshot coverage for same-state Stock
A/Product B/Stock C. Run73 passed 12/12×1024 clients and 8/8 rank/Host gates;
its TPS is diagnostic. Evidence:
`evidence/20260924_loop035_diagnostic/run73/summary.json` and
`evidence/20260924_loop035_diagnostic/run73/manifest_check.json`.

## Loop035 Run74 write-set audit (2026-09-24 05:00 UTC)

Source-backed physical writes: group0/1 compressed scatter slots come from compressor_metadata outputs, group2/3 SWA and DSpark slots from active req metadata, and group4/5 compressor state pages from start_pos plus state_block_table. The current strict snapshot validates supplied candidates but cannot certify compressed/state coverage from generic group slots. Before Stock A/Product B/Stock C, construct per-cache write candidates from active source metadata, verify all 67 physical caches and mutable aliases, and restore the four large metadata tensors omitted by Run66 or prove them immutable. Run74 is a design check only. See evidence/20260924_loop035_diagnostic/run74/write_set_audit.md.

## Loop035 Run76 alias correction (2026-09-24 05:25 UTC)

Run75 short request was INVALID because serving requires max_tokens=1024,
but all 8 manifests were captured before request execution. Run76 found the
old cache_slot_specs positional binding mismatched the sorted kv_caches tree:
31/67 view group labels were outside actual aliases; 42/67 views alias
multiple groups; 67 views use 46 storage allocations. The earlier Run73/74
per-cache group ownership inference and Run66 cache snapshot completeness
must not be used. Source write-address formulas still apply per actual layer
alias. Next implement page-level strict snapshots from the union of each
view actual layer aliases and block tables, then continuous Stock/Product
oracle. evidence/20260924_loop035_diagnostic/run76/alias_check.json.

## Loop035 Run77 strict page snapshot (2026-09-24 05:26 UTC)

RuntimeAssets.snapshot_pages now accepts explicit physical page IDs for
each named cache tensor view and snapshots full mutable rows. Synthetic
shared-storage restore plus missing/out-of-range/empty rejection passed.
It is not yet used by a live oracle. Next derive the per-view page union
from all actual layer aliases and compressed/state/SWA write producers,
then execute the strict NPU transaction.

## Loop035 Run80 strict target candidate pages (2026-09-24 06:18 UTC)

Run80 passed a fixed 12x1024 diagnostic cohort and all eight rank/Host gates.
At target cycles0/1, source-derived page candidates from 170 actual layer
aliases covered all 67 physical cache views with strict NPU snapshot and no
skips. Twenty compressed source layers had zero writes at each sampled cycle;
all 67 views still had pages through shared aliases. This is candidate
coverage only. Before Stock/Product/Stock same-state oracle, compare
compressor_metadata side-output slots to candidate pages, restore all
physical pages and mutable metadata aliases, then run A/C self replay.
Runs78/79 were invalid diagnostic wiring/accounting attempts.
Evidence: evidence/20260924_loop035_diagnostic/run80/summary.json.

## Loop035 Run81 compressor operator page parity (2026-09-24 06:39 UTC)

Fixed 12x1024 c12 diagnostic completed 12/12 requests. On all eight ranks at target
cycles 0/1, strict pre-target snapshots covered 67/67 physical cache views
through 170 actual layer aliases, with no skipped views. The exact
`torch.ops._C_ascend.compressor_metadata` slot side output was compared with
the source-derived compressed page candidates for each of three unique
metadata signatures per rank-cycle (48 checks total): zero missing actual
pages and zero extra candidate pages. c4 produced valid compressed writes;
c128 produced none at the sampled cycles. This validates the candidate write
page set for sampled target cycles only. It does not establish restored KV or
metadata state, Stock token equivalence, or long-run acceptance. Run81
464.73 tok/s is diagnostic and must not be compared to the formal Loop034 A/B.
Next: include all mutable metadata aliases in the transactional snapshot,
verify exact restoration after live target, then run continuous same-state
Stock A/Product B/Stock C with A/C self-replay floor.
Evidence: `evidence/20260924_loop035_diagnostic/run81/summary.json`.

## Loop035 Run82 live same-source target replay (2026-09-24 07:00 UTC)

Fixed c12 diagnostic completed 12/12x1024. On all eight ranks and target
cycles0/1, 67/67 cache views plus 76 deduplicated attention metadata tensors
(597,641,348 bytes per rank snapshot) restored exactly after the first target.
No captured metadata tensor changed during target. The same source target
replayed from restored state nevertheless changed 6/96 and 4/96 argmax
positions (cycle0/1 respectively); acceptance counts matched 12/12, but
cycle1 accepted token IDs matched only 95/96. All eight ranks reported the
same comparison counts. This is an observed same-source replay floor or
uncaptured state, not evidence of Stock/Product semantic divergence. Run82
444.35 tok/s is diagnostic. Next locate the mismatch positions and logits
margins, take A/B/C same-source replay samples and validate a self-replay
floor before Stock/Product/Stock comparison. Evidence:
`evidence/20260924_loop035_diagnostic/run82/summary.json`.

## Loop035 Run83 strict triple target self replay (2026-09-24 07:16 UTC)

Fixed c12 12/12x1024 completed. At sampled continuous target cycles0/1,
all eight ranks restored 67 cache views and all 76 metadata tensors exactly
between A/B/C direct target executions. Rank results were identical. Target
argmax equality A/B, A/C, B/C was 92/96, 92/96, 91/96 at cycle0 and 88/96,
94/96, 90/96 at cycle1. Accepted-token equality was 93/96, 95/96, 94/96
and 89/96, 96/96, 89/96 respectively; counts also varied across replays.
Mismatch top-2 logits and request/position coordinates are in rank artifacts;
the largest top-1 margin at a changed argmax was 1.0. Therefore strict
captured-state restoration does not make target/acceptance deterministic, and
an isolated candidate mismatch must be judged against same-source A/C noise.
This repeats no further as a standalone control. The next investigation is
continuous Stock-authoritative state/DSpark/KV lockstep and first sustained
acceptance split, using this replay floor. Diagnostic 461.41 tok/s is not
formal A/B. Evidence: `evidence/20260924_loop035_diagnostic/run83/summary.json`.

## Loop035 Run84 continuous target write-page trajectory (2026-09-24 07:41 UTC)

Run84 completed 12/12x1024 and all eight Runtime/Host gates. A compact,
read-only audit sampled every target cycle0-255 on every rank: 2,048 strict
pre-target snapshots, each covering all 67 physical cache views from actual
layer aliases with 69 captured entries including mutable tensors, zero skips
and zero out-of-range pages. The actual compressor_metadata side-output was
checked for three signatures per rank-cycle (6,144 checks): no missing actual
page and no extra candidate page. Unlike Run81, c128 wrote during this
trajectory: first valid write at cycle8, 132 valid cycles and 198 valid slot
rows per rank across the 256-cycle window. The full Runtime cohort took 303
cycles; 47 late cycles were outside this audit. This certifies the sampled
target write-address candidates, not KV values, post-target restoration or
Stock token equivalence. The 128.83 tok/s result includes 2,048 snapshot
clones and is not a performance measurement. Run72 already covers full
DSpark gid2/3 context write addresses. Next compare continuous Stock-owned
state and physical KV/DSpark trajectories using these certified write sets;
do not repeat standalone target replay or run formal 48-request A/B yet.
Evidence: `evidence/20260924_loop035_diagnostic/run84/summary.json`.

## Loop035 Run85 Stock-authoritative acceptance (2026-09-24 08:08 UTC)

A 12x1024 c12 Stock-only continuous shadow compared Extreme greedy acceptance
with Stock's actual sampler using the same target logits and draft IDs over
160 cycles on all eight ranks. All 122,880 output cells and every accepted
count matched. The target integer ABI and five supported generic KV group
mappings matched; group1's compressed table geometry remains unsupported by
this generic checker. This establishes the acceptance implementation for this
frozen greedy contract, not independent Product/Stock logits or KV-value
equivalence. Zcode was actually invoked as deepseek/deepseek-flash in read-only
plan mode for a bounded source audit, timed out after 180 s, and supplied no
usable analysis; the main Agent completed and reviewed the audit.
Evidence: evidence/20260924_loop035_diagnostic/run85/summary.json.

## Loop035 Run87 serving decode DAG stage profile (2026-09-24 08:46 UTC)

A dedicated FixedCohortServing diagnostic exporter captured per-cycle NPU event
intervals for all eight ranks over 301 continuous c12 cycles; 12/12 requests
produced exactly 1024 tokens. Across steady cycles1-300, stage medians were
target47.702 ms, derived target metadata8.673 ms, proposer6.359 ms,
acceptance0.337 ms, prepare target0.207 ms and state advance0.024 ms.
Within proposer, DSpark model5.939 ms dominates. Target remains the largest
serial stage, while metadata is the second-largest. Run86 completed 12/12 but
was marked invalid as a profile because its serving path did not persist the
existing event markers; Run87 added the exporter in dedicated Runtime code,
without a generic ModelRunner change. Rank0 accepted-count windows rose from
2.07 (cycles0-7) to 5.20 (128-191); late decline follows parking of completed
slots and is not evidence of live-request acceptance collapse. Diagnostic
469.17 tok/s includes instrumentation and one cohort, not formal A/B.
Evidence: evidence/20260924_loop035_diagnostic/run87/summary.json and dag/rank*.json.
Next: use the measured target and metadata stages to choose a structural
optimization, while retaining the independent KV-value/Stock semantic caveat.

## Loop035 Run88 strict Stock/Product/Stock at cycle128 (2026-09-24 09:13 UTC)

A diagnostic-only Stock ABA hook now uses the existing source-backed TargetPageAudit
manifest rather than the incomplete generic slot snapshot. Stock advanced 128
continuous fixed decode cycles, then all eight ranks captured all 67 physical
cache views (69 entries including two mutable buffers), all 76 attention
metadata tensors and actual compressor slot candidates; all three restores
were exact. At the same frozen state, Stock A versus Product B and Stock A
versus Stock C each matched 92/96 target argmax positions. Accepted cells
matched 92/96 AB versus 95/96 AC. Product B alone differed at request slot2,
first target position, changing one accepted count. This is a possible first
semantic fork, not yet proven because exact same-source target replay has
known nondeterminism. Post-target cache values differed widely on both AB
(16,437,317 elements) and Stock AC (16,443,266); AB is not worse by this
aggregate and raw mismatches cannot be ascribed to Product. Next obtain
same-state repeated Stock and Product target samples with top-2 logits at
the disputed coordinate, preserving this strict snapshot/restore gate.
Evidence: evidence/20260924_loop035_diagnostic/run88/summary.json.

## Loop035 Run89/90 bounded same-state repeat (2026-09-24 09:46 UTC)

Run89 reached Stock cycle128 but the optional five-pass diagnostic retained
additional full KV post-write copies and OOMed on a 1 GiB comparison
allocation. It is invalid for semantic inference. Run90 compared and
released A/B/C post-write copies before repeating the targets. On all eight
ranks, Stock advanced 128 continuous cycles, strict snapshots covered all
67 cache views (69 entries) and 76 metadata tensors, and all five restores
were exact. Rank metrics were identical across ranks. Against Stock A,
Product B matched 92/96 argmax positions and Stock C 93/96; Product B/B2
matched 92/96 and Stock C/C2 94/96. The sole position where both Product
replays agreed and all three Stock replays agreed on another token was
request slot7, draft position2. Stock A/C/C2 tied tokens 270 and 4888 at
top-2 logits; Product B/B2 selected 4888 by 0.125. All argmax disagreement
positions had top-2 margins at most 0.5; Stock self-replay also changed
tokens. This does not establish a high-margin Product semantic fork or exact
long-run token parity. The real first question is numerical/tie behavior
under the same captured state, not the fixed greedy acceptance algorithm.
Keep formal A/B gated pending continuous semantics decision; no more
standalone self-replay controls without a concrete new state hypothesis.
Evidence: evidence/20260924_loop035_diagnostic/run90/summary.json.

## Loop035 Run91/92 late continuous semantic gate (2026-09-24 10:22 UTC)

Run91 attempted a Stock fixed-c12 ABA sample at cycle256 with the frozen
1024-token output cap. All 12 requests completed, but the Stock scheduler
changed shape after the first completion, so no fixed 12x8 sample existed
at cycle256; Run91 is invalid as an ABA diagnostic and its throughput is not
formal. Run92 changed only the diagnostic request cap to 2048, keeping all
12 slots live through cycle256. On all eight ranks, strict five-way
Stock A/Product B/Stock C/Product B2/Stock C2 snapshots covered 67 cache
views, 69 entries and 76 metadata tensors with exact restoration each time.
There was no stable Product-only argmax position. Product B matched Stock A
accepted output 96/96, while Stock C matched 92/96. All argmax differences
had top-2 margin <=0.25. Together with Run85 same-logits acceptance parity,
Run72 full DSpark gid2/3 write addresses, Run84 target page trajectory and
Run87 phase profile, this passes a bounded continuous semantic/acceptance
gate for formal performance evaluation. It does not prove independent
long-sequence token identity: Stock self-replay is non-deterministic and
Run92s 2048 cap is a diagnostic control, not the frozen formal workload.
Evidence: evidence/20260924_loop035_diagnostic/run92/summary.json.
The next formal Extreme script now enables the causal gid2 slot refresh.
Use the existing reliable Stock 543.65 tok/s baseline; do not rerun Loop034.

## Loop035 Run93 formal Extreme serving A/B against frozen Stock (2026-09-24 10:51 UTC)

After the bounded continuous semantic gate (Run85/90/92), the exact frozen
48x32K-to-1024 c12 warm-cache protocol was run once for warmup and three
official Extreme measurements with EXTREME_DSPARK_SLOT_REFRESH=1. All three
completed 48/48 requests at exactly 1024 tokens: 517.880, 525.417 and
534.744 tok/s, median 525.417. The comparable Loop034 Extreme median was
217.342, so the slot-refresh version improves output TPS 141.747% (2.417x).
It remains 3.355% below the reliable frozen Stock 543.655 tok/s baseline;
Stock was not rerun. All 128 rank/cohort rows pass, all host mirrors exact,
FULL target graph, zero post-handoff ModelRunner cycles and oracle target
calls, with 2,048 slot-refresh audit entries. Rank0 cohort cycle median
fell from 1025.0 to 300.5; rank0 median staged output/slot/cycle is 3.427
versus the original 1.194. Continuous acceptance window medians across
16 rank0 cohorts are 3.519 at cycles8-63, 4.257 at64-127, and 4.163
at128-191. Run87 DAG profile now points to target47.702 ms/cycle and
derived metadata8.673 ms as remaining serial stages, versus proposer6.359
ms and acceptance0.337 ms. This passes Loop035s frozen acceptance and
cycle-attribution goal. Exact independent long token identity is still
limited by Stock self-replay nondeterminism, and the present performance
still misses the P0 above-baseline goal.
Evidence: evidence/20260924_loop035_formal/run93/summary.json and
tasks/deepseek-extreme-p0/loops/loop-035/comparisons.jsonl.
Next optimization should test the dedicated target/metadata critical path
with a frozen correctness gate, not resume generic ModelRunner patches.

## Loop036 Run94 invalid control and requested pause (2026-09-24 11:17 UTC)

After Loop035 Run93, Loop036 was frozen to remove the per-cycle
`local_seq_lens.max().item()` sync from the dedicated target metadata updater.
A source audit in `tasks/deepseek-extreme-p0/loops/loop-036/doc/source_audit.md`
found that the two scalar max-K attributes only feed fallback branches when
actual per-request length tensors are absent. The opt-in
`EXTREME_TARGET_METADATA_STATIC_KV_MAX=1` candidate sets the scalar attrs
to 1; `EXTREME_TARGET_METADATA_SHADOW=1` compares the generated SAS/QLI
metadata with the old dynamic call. Both flags default off, so Run93 behavior
is untouched.

Run94 attempted a 12-request 2048-token continuous shadow to cover 256 live
cycles, but the installed Extreme serving contract explicitly rejects
`max_tokens != 1024` at `model_runner_v1.py:3379`. The service exited with
that ValueError/EngineDeadError before any metadata shadow marker. TaskCtl
marks Run94 INVALID; 11/12 client success in the partial bench is not a
correctness or performance result. No shadow parity or stage timing claim
exists yet. The prepared `scripts/run_loop036_metadata_profile.sh` is not
executed. Next session should either construct a 1024-token-valid continuous
control (cohort shape may change before cycle256) or an explicitly diagnostic
guard bypass, then verify 8-rank metadata parity before measuring the static
path. Do not repeat Loop034 or Run93 formal E2E until there is a validated
structural improvement.

The user requested that work stop after Run94 and resume in a new dialogue.
At handoff no benchmark or vLLM service remains running. The heartbeat
automation is paused. Agent routing: a concurrent edit to root
`AGENTS.md` appeared at 11:12:54 UTC during Run94. Its current text names
GPT-6 Sol as primary, Astra Medium/High as optional independent perspectives,
and DeepSeek/Zcode for bounded verifiable work, including complex tasks when
clearly scoped. The user subsequently directed that the revised
`AGENTS.md` be committed and pushed; its broader routing guidance is now
authoritative. `docs/agent_orchestration.md` was reconciled in the same
commit. Earlier default-model
change alone did not implement actual routing; `scripts/delegate_zcode.py`
was subsequently added. Run85 really invoked `deepseek/deepseek-flash`
through Zcode (return code 124 after 180 s timeout); no DeepSeek conclusion
was accepted. TaskCtl records state, not model dispatch. No subagent was
called in this Loop036 session. The user explicitly requires every subsequent
Run record, TaskCtl resume state, HANDOFF update and necessary small evidence
summary to be committed and pushed to this GitHub repository so a new
dialogue can resume from `main`.

## Loop036 Run95 host-launch invalid (2026-09-24)

Run95 corrected the request cap to the 1024-token serving contract but was
invoked on the host. `scripts/serve.sh` failed before model start because the
Ascend toolkit environment is available inside container
`vllm-ascend26-dsv4f-w4a8`, not on the host. TaskCtl records INVALID; no
metadata parity or performance inference. Next invoke the corrected shadow
script inside the container as Run96.

## Loop036 Run96 shadow failure (2026-09-24)

The first legal container-side 12×1024 shadow reached the target metadata
updater. On all eight TP ranks the `ratio=4` SAS static/dynamic comparison
failed during the first update and aborted the service. TaskCtl records FAIL;
partial client outputs and TPS are invalid. The first shadow only compared
full 1024-element outputs and did not record mismatch positions or a dynamic
self-replay control. Given prior SAS tail nondeterminism, next run must compare
dynamic A/static B/dynamic C on identical input and report first differing
index and header parity before accepting or rejecting the scalar candidate.
`evidence/20260924_loop036_metadata/run96/summary.json` records the compact
evidence. Static candidate remains opt-in and unvalidated; no profile/E2E.

## Loop036 Run97 continuous A/B/C metadata gate (2026-09-24)

The legal 12×1024, c12, TP8 DSpark7 Run97 completed 299 Runtime-owned decode
cycles on all eight ranks; 12/12 clients and all rank records passed at exactly
1024 output tokens. Every cycle checked dynamic A/static B/dynamic C metadata.
SAS positions 0–96 and QLI positions 0–24 matched exactly; markers at cycles
1/64/128/256 exist on every rank. Full 1024-element buffers are not equal,
but dynamic A/C self-replay also differs after those stable headers. This is a
bounded stable-field parity gate, not full-buffer exact parity or independent
long-token semantic proof. Shadow instrumentation invalidates its TPS as a
performance comparison. Source audit still finds the static scalar fallback
unreachable with actual per-request length tensors. Compact evidence:
`evidence/20260924_loop036_metadata/run97/summary.json`. Next run the
prewritten no-shadow static DAG profile, compare with Run87 under the same
single-cohort contract, and only proceed to formal E2E if timing gain is
material and correctness remains intact.

## Loop036 Run98 no-shadow DAG profile (2026-09-24)

Static metadata with shadow off completed one legal 12×1024 c12 cohort: all
8 ranks ran 300 continuous cycles and passed Runtime gates; all clients
finished at exact output length. The median derived metadata event stage
fell from matched Run87 8.6730 to 0.6674 ms/cycle, a reduction of 8.0056 ms
(92.3%). Target was 46.560 ms and proposer 6.391 ms. The single-cohort
517.63 tok/s is diagnostic, not the formal product result. Run97 stable
header parity and the source fallback audit support a formal E2E test; full
SAS/QLI tail identity remains limited by dynamic self-replay noise. Next
run the frozen warm-cache 48×32K→1024 c12 warmup plus three official
measurements, with no Stock rerun. Evidence:
`evidence/20260924_loop036_metadata/run98/summary.json`.

## Loop036 Run99 formal static-metadata E2E (2026-09-24)

The frozen warm-cache 48×32K→1024 c12 protocol ran one warmup and three
measured Extreme runs with static metadata and no shadow. All measured runs
completed 48/48 requests at exactly 1024 tokens; all 128 rank/cohort Runtime
records pass. Output TPS: 612.962 / 567.573 / 571.681, median 571.681.
This is 5.155% above the reliable frozen Stock 543.655 and 8.806% above
Run93 Extreme 525.417. Stock and Loop034/Run93 were not rerun. This is the
first valid same-protocol above-Stock P0 result, but its 5.2% margin is
modest and the target remains the dominant 46.56 ms/cycle stage. Stable
metadata fields passed the 8×299-cycle Run97 A/B/C gate; full AICPU output
tails remain nondeterministic even under dynamic self-replay, so independent
long-token identity remains a limitation. Evidence:
`evidence/20260924_loop036_metadata/run99/summary.json`. Next update the
Performance Map and investigate the 46.56 ms target critical path, with no
repeat of reliable formal baselines unless a new candidate clears correctness.

## Loop037 opened (2026-09-24)

Loop036 is technically satisfied but TaskCtl verdict is PIVOTED because
invalid setup and full-buffer shadow attempts remain preserved. The next
bounded Loop037 will attribute the remaining 46.560 ms/cycle target graph
stage with a legal one-cohort Runtime-only NPU profile across 8 ranks. It
must separate compute/communication overlap and device gaps before choosing
another optimization. No target performance claim exists yet.

## Loop037 Run100 invalid profile window (2026-09-24)

Both legal 12×1024 cohorts and 16 Runtime rank records passed, but the
`/start_profile` request blocked roughly 14 seconds and only returned as the
sampled cohort finished. Eight offline-parsed rank traces contain zero
`extreme::target` scopes and zero kernel rows. TaskCtl marks Run100 INVALID
for target attribution; its diagnostic TPS has no comparison role. Raw trace
remains at `evidence/20260924_loop037_target/run100/profile/`, compact
reason at `run100/summary.json`. Run101 must activate profiler before sending
the sampled cohort, then stop after a bounded number of decode seconds.

## Loop037 Run101 oversized target profile (2026-09-24)

Run101 started profiler before a legal 12×1024 cohort, so its window did
include execution; the cohort completed 12/12. Eight seconds generated roughly
13.6 GiB raw trace across eight TP ranks. Offline `torch_npu.profiler.analyse`
warned parsing could exceed 30 minutes per rank and had not finished rank0;
the parser was terminated, with raw files preserved under
`evidence/20260924_loop037_target/run100/profile/*20260924133739*_ascend_pt`.
TaskCtl marks profile attribution INVALID though request correctness passed.
No kernel/communication conclusion. Next use a subsecond window while a
48-request workload keeps c12 decode active through the profiler start RPC.
Compact evidence: `evidence/20260924_loop037_target/run101/summary.json`.

## Loop037 Run102 wrong-phase short profile (2026-09-24)

Subsecond profiling during a legal 48×1024 workload produced parsable traces
(~28 MB raw/rank), and all 48 requests completed. Yet all eight rank traces
contain zero `extreme::cycle` and zero `extreme::target` scopes; the window
landed at a cohort boundary and showed only a few DSpark layer scopes.
TaskCtl marks target attribution INVALID, correctness PASS. Parsed raw trace
is under `evidence/20260924_loop037_target/run100/profile/*20260924134249*_ascend_pt`;
compact count summary is `run102/summary.json`. Next start the profiler RPC
immediately after launching one legal 12-request cohort, so its ~13-second
handshake returns while that same cohort remains in Extreme decode.

## Loop037 Run103 early prefill profile (2026-09-24)

Run103 again completed legal 12×1024 requests, but `/start_profile` returned
immediately this time, so the 0.5-second trace captured prefill. All eight
parsed ranks have zero `extreme::cycle` and zero `extreme::target` scopes;
TaskCtl marks target attribution INVALID, correctness PASS. Raw traces are
under `evidence/20260924_loop037_target/run100/profile/*20260924134850*_ascend_pt`,
compact counts in `run103/summary.json`. The next window should request
profiler start about 3 seconds after cohort launch; both immediate and delayed
RPC return should then fall within the 20-second continuous decode interval.

## Loop037 Run104 oversized stop window (2026-09-24)

The 3-second delayed start attempt completed a legal 12×1024 cohort, but
`/stop_profile` returned about 23 seconds after the stop request, expanding
the intended 0.5-second capture to a multi-GiB trace across 8 ranks. It has
not been parsed to target operator attribution; TaskCtl marks the profile
INVALID, request correctness PASS. Raw trace:
`evidence/20260924_loop037_target/run100/profile/*20260924135309*_ascend_pt`.
Compact evidence: `run104/summary.json`. Runs100–104 demonstrate that HTTP
profiler activation/deactivation latency is unstable relative to a 20-second
cohort. Next use a Runtime cycle marker to start/stop a bounded 8-rank trace
inside a selected cycle window, or a profiler mode that can export only those
cycles. Keep the verified Run99 formal 571.681 tok/s as current P0 result.

## Loop038 Run105 launch setup invalid (2026-09-24)

The cycle-scheduled profiler implementation was committed at d5e2651 and
Run105 was recorded before launch. Its direct script invocation failed with
exit 126 because the new script lacked executable mode. No service, workload,
or profiler started. TaskCtl marks Run105 INVALID. Run106 will invoke the
script through bash and preserve the legal 12x1024 TP8 profile protocol.

## Loop038 Run106 bounded cycle profile (2026-09-24)

Run106 completed one legal 12×1024 c12 TP8 cohort: 12/12 exact-length
requests, 8/8 Runtime gates, 289 cycles/rank. Each rank captured target
scopes at cycles64–65; raw profiles were only ~5.6 MB/rank and parsed in
~6 seconds/rank. The per-target profiler device-total median is 51.144 ms
(range46.255–56.502), while CPU scope median is only 6.103 ms. The nested
wait_event median is 50.134 ms and nested HCCL all-gather sum is 0.247 ms;
CPU-scope timestamp clipping misses asynchronous graph execution and is not
valid target device attribution. Across the entire two-cycle trace, grouped
matmul kernel sums have median20.592 ms, but include target and proposer.
No target-only optimization bound is established yet. The single-cohort
508.89 tok/s includes profiler overhead and is not a Stock comparison.
Evidence: evidence/20260924_loop038_cycle/run106/attribution.json. A next
pass must link graph kernels to the target replay or use NPU event stage
boundaries before choosing a kernel edit. Service was stopped after Run106.

## Loop038 Run107 synchronized target window (2026-09-24)

Run107 completed a legal 12×1024 c12 cohort, 12/12 exact-length requests,
8/8 Runtime pass, 301 cycles/rank. Only cycles64–65 used NPU synchronize
around target, so client 494.03 tok/s is diagnostic and cannot be compared
with Stock. All eight cycle64 traces have exactly 2,836 target kernels;
rank1 cycle65 includes an extra 143 and was excluded. In 15 canonical
windows the median target device interval union is50.281 ms, compute union
39.932 ms, communication union11.547 ms, compute/communication overlap
1.292 ms. The 86 grouped-matmul kernels sum9.966 ms/cycle (kernel sums
are not additive across streams). This provides a bounded candidate with
roughly 10 ms/cycle upper limit, not an achieved saving. Raw profiles are
about5 MB/rank, parsed successfully. Evidence:
`evidence/20260924_loop038_cycle/run107/target_window.json`.

Loop038 technical attribution goal is met and TaskCtl verdict is PIVOTED
because the invalid Run105 launcher remains in its history. Loop039 is
active: inspect actual W4A8 target grouped-matmul call sites/shapes and
backend alternatives, establish same-state eight-rank correctness, then
measure stage reduction before any formal 48-request E2E. Current accepted
formal product result remains Run99 median571.681 tok/s, +5.155% vs Stock;
no new formal E2E or Stock baseline was run. Service was stopped and NPUs
are idle after Run107. The last agent work used the default Sol main role;
no Astra or Zcode call was made in Loops037–038.

## Loop039 Run108 source and trace map (2026-09-24)

Read-only audit confirms the model config has43 hidden layers and, in each
of the eight canonical rank cycle64 target windows, exactly43 GMM1 fused
SwigluQuantWeightNzV2 kernels plus43 GMM2 GroupedMatmulWeightNz kernels.
Rank0 summed durations are6.425 and3.646 ms. The borrowed implementation
enters `DeviceOperator.npu_grouped_matmul_swiglu_quant` and
`DeviceOperator.npu_grouped_matmul_gmm2` from `moe_mlp.py`; this is a
source/trace mapping, not proof that a replacement will save time. Exact
runtime tensor shapes and a semantics-preserving faster backend are still
unmeasured. TaskCtl Run108 PASS (design-check only), evidence:
`evidence/20260924_loop039_gmm/run108/source_audit.json`.

## Governing objective clarification (2026-09-24)

User reaffirmed the frozen-product objective: DeepSeek V4 Flash W4A8 on
8×Ascend 910B3, DP1×TP8, DSpark7; correctness is mandatory, and repeatable
formal E2E is the final judge. Keep removing framework, Host, scheduling,
communication, synchronization and execution redundancy until the product is
as close as practicable to the credible hardware/model achievable limit.
Do not preserve genericity or abstractions without product value. Loop039’s
5 ms target-stage / 15%-over-Stock gate is a local falsifiable screen, not
a terminal target; the grouped-matmul hypothesis can be rejected or outranked
by a larger cross-module opportunity. The current credible whole-product
hardware bound is unknown. Run99 remains the official571.681 tok/s result.

## Loop039 Run109 active branch correction (2026-09-24)

Run108’s 43+43 target kernel counts remain valid, but its GMM1 Python
call-path attribution was too broad. The frozen quant description has
group_size=0, so W4A8 weights are per-channel; the active GMM1 branch is
`moe_mlp.py` custom `grouped_matmul_swiglu_quant_v2`, and GMM2 uses
`DeviceOperator.npu_grouped_matmul_gmm2`. Checkpoint packed-I8 layer0
expert0 shapes are w1/w3 [1024,4096], w2 [2048,2048]. Exact routed live
input shapes, graph addresses and a faster parity-preserving replacement
remain unmeasured. Do not use Run108’s GMM1 `DeviceOperator` mapping.
TaskCtl Run109 PASS design-check. Evidence:
`evidence/20260924_loop039_gmm/run109/active_branch.json`.

## Zcode execution trial Run110 (2026-09-25)

After updating AGENTS/docs to prefer Zcode for bounded low-risk execution,
Run110 delegated a read-only environment check with explicit inputs and
acceptance fields. Zcode exit0 after188.845s/27 provider requests, but
headless build mode denied docker exec, npu-smi and local HTTP. The wrapper
recorded configured `deepseek/deepseek-flash` and observed_model=null; the
model cannot be independently attributed from this run. TaskCtl marks Run110
INVALID. Sol directly rechecked HEAD1b7c0e0, 8 profile window files, 8/8
idle NPUs and stopped service. Do not delegate service/NPU operations to
this Zcode configuration until permission behavior is fixed and actual-model
provenance is verifiable; direct execution is the current fallback.

## Zcode DeepSeek connectivity Run111 (2026-09-25)

At user request, a no-tool, no-file-change Zcode plan-mode ping returned the
exact sentinel `DEEPSEEK_PING_20260925` with exit 0 in 14.735 s and one
provider request. The local Zcode model-I/O record for that session identifies
request provider `deepseek`, request model `deepseek-flash`, and response
modelId `deepseek-flash`; the sanitized evidence is
`evidence/20260925_loop039_delegate/run111/model_io_summary.json`.
TaskCtl Run111 PASS for short-text connectivity only. Run110 remains INVALID:
headless build-mode Bash permissions denied docker/NPU/HTTP operations. This
ping does not establish tool execution viability, Runtime correctness, or
performance. Loop039 product work resumes from Run109 live routed shapes and
same-state eight-rank correctness; Run99 remains the official E2E result.

## Zcode headless permission root cause (Runs112-114, 2026-09-25)

Run112: `--mode build` / Bash `pwd` succeeded with exact cwd and exit0.
Run113: isolated direct `--mode yolo` / one read-only Docker status command
succeeded and returned `running`. Run114: `--mode build` / Docker status
was denied before command execution with `No permission client configured
for Bash`; Zcode process exit0 did not imply tool success. All three local
model-I/O records identify DeepSeek/deepseek-flash. Retrospective Zcode runtime
log audit also confirms Run110 made27 DeepSeek model streams; its task result
stays INVALID because26 Bash and one WebFetch permission requests were
denied. Root cause: headless build mode has no approval client for commands
that need approval; simple preapproved commands work. CLI headless prompt
defaults to yolo, while the wrapper explicitly selected build. The exact
prior successful invocation mode is not independently known. See
`docs/agent_orchestration.md` and
`evidence/20260925_loop039_delegate/run11{0,2,3,4}/`.
No product benchmark or Runtime change was made.

## Loop039 Run115 live shape envelope (2026-09-25)

A temporary, shape-only probe at borrowed `quant_apply_mlp` ran the legal
12×1024 DP1×TP8 DSpark7 diagnostic workload. All12 requests succeeded;
8/8 workers wrote 33 unique shape signatures with the same shape set.
The 96-token target graph (top_k=6) maps to MoE input int8 [576,4096],
per-token scale float32 [576], local expert count32 with count-mode
group_list int64 [32]. Packed grouped weights at the call site are w1
int32 [32,4096,512] and w2 int32 [32,2048,512], with corresponding
per-channel scale tensors. This is a static envelope: the probe did not
copy live per-expert token counts during graph replay, so effective active
work per rank remains unknown. Run115 diagnostic output TPS544.725 is not
formal E2E and is not compared to Stock. The borrowed source was restored
byte-identically (SHA256 5b537cc4...), the service was stopped, and 8 NPUs
returned to idle HBM. TaskCtl Run115 PASS profile only. Evidence:
`evidence/20260925_loop039_gmm/run115/shape_summary.json`.

Next: use Run107 target trace and this shape envelope to estimate how much
of ~10 ms/cycle grouped-matmul duration can actually be removed for the
frozen product. In parallel assess communication and non-GMM target costs;
choose one bounded same-state correctness experiment before any E2E.

## Loop039 Run116 opportunity audit (2026-09-25)

Sol recomputed 15 valid synchronized Run107 windows: median GMM kernel
duration sum9.96625 ms, communication union not overlapped by compute
10.2475 ms, and non-GMM compute-union lower bound29.97925 ms. These are
diagnostic interval/sum quantities, not directly realizable E2E savings.
GMM remains a bounded candidate but is not established as the largest
product opportunity. A requested Astra Medium second view agreed with the
caution, but no service-side model ID was exposed; its model identity is
unverified and that review is advisory only. Run116 PASS design-check.
Next experiment must first find a semantics-valid per-channel W4A8
substitute and observe real expert counts, then eight-rank same-state A/B/A
parity and GMM interval / full target-stage reduction. If the candidate
cannot plausibly save >=5 ms/cycle locally or the saving does not reach the
target stage, pivot to communication and non-GMM attribution. This local
screen is not the product's terminal goal. Formal E2E remains Run99.

## Continuous execution rule (2026-09-25)

User clarified that evidence, TaskCtl, recovery pack, Git push, interim
conclusions, and service cleanup are checkpoints, not handoff triggers.
After each, Sol revisits the maximum removable Gap and next_action and
continues when high-value work is clear and unblocked. Run116 already
identifies comparable GMM and exposed-communication candidates; proceed
to discriminating evidence and Sol's own decision. Stop only for the
specific external, user-information, irresolvable architecture, material-risk,
or forced-runtime conditions recorded in AGENTS.md. Product objective,
model division, correctness and formal E2E standards are unchanged.

## Loop039 Runs117-119 communication versus GMM adjudication (2026-09-25)

Run117 shows the first target reduce-scatter in synchronized Run107 has
20.533/9.370 ms start skew across ranks for cycles64/65, but only
0.010/0.017 ms end skew. Early ranks wait for late arrivals inside the
collective; measured HCCL duration is not intrinsic transfer cost.
Run118 finds the same signature in Run106 without target synchronization:
first-collective start skew10.193/9.735 ms and end skew0.0065/0.01175 ms.
Rank skew is already present at prepare_target entry
(9.637/10.510 ms), before target execution. The cycle64 proposer-end
skew10.466 ms propagates to the next prepare entry10.510 ms, but these
two profiled cycles cannot establish a steady product proposer cost.

Run119 corrects that inference with eight-rank Run98 steady cycles64-255:
target event median46.575 ms, proposer6.400 ms, DSpark model5.987 ms.
Formal Run99 cohort wall median57.044 ms/cycle is consistent with this
scale. Sol decision: the exposed-communication ~10 ms from Run107 is a
peer-wait symptom in diagnostic windows, not a demonstrated independent
HCCL transfer opportunity. Keep target GMM as the next bounded
semantics-preserving candidate; do not assume its ~9.97 ms kernel sum can
all be removed. The larger non-GMM target compute remains under review.
No new E2E was run. TaskCtl Runs117-119 PASS as profile/design audits.

## Loop039 Run120 checkpoint (2026-09-25)

Run120 online group_list capture was INVALID because the benchmark launcher
sent max_tokens=384, while the frozen Extreme serving path requires exactly
max_tokens=1024. EngineCore raised that guard; 11/12 partial request success
and 7.38 tok/s cannot be used. No cycle64/65 counts were captured. Both
temporary instrumented sources were restored to their recorded original
SHA256, the stop script released all eight NPUs to below 3.5 GB, and the
root-cause excerpt is saved. Next action: Run121 with the exact 1024-token
request contract, same bounded probe and validation. This checkpoint does
not end the execution turn.

## Loop039 Run121 live expert counts (2026-09-25)

Run121 obeyed the exact 1024-token serving guard: 12/12 requests succeeded.
Temporary probes saved 16 group_list snapshots (8 ranks, cycles64/65).
Each rank retained 86 refs; ordinals0-42 are unchanged graph-build refs,
whereas ordinals43-85 change between the two cycles on all eight ranks.
For each of the 43 live layers and both cycles, all 8 ranks together route
exactly 576 assignments (96 tokens × top_k6). On 688 rank-layer samples,
active experts range 7-29 of 32, median15, mean15.663; global layer
active experts median121 of 256. Actual local tokens per layer median69,
mean72, range23-186. These are valid live routing facts, not a claim that
the present GMM computes empty experts or that half its time is removable.
The probe's 526.360 tok/s is diagnostic and not a formal E2E verdict.
Both patched sources were restored to the exact original hashes; stop
script released all eight NPUs below 3.5GB. Next Run122 checks whether
the real count sparsity offers any extra GMM opportunity beyond what the
existing fused operator already handles.

## Loop039 Run122 GMM path audit (2026-09-25)

Run107's 15 synchronized target windows split the 9.96625 ms median GMM
kernel sum into GMM1 fused SwiGLU/quant 6.3785 ms and GMM2 3.58775 ms.
The borrowed Ascend A8W4 GMM1 kernel computes each expert's M from the
live group_list and skips MatMul when M<=0; Run121's 15/32 median active
experts therefore do not provide a new 50% skipping opportunity. This
source audit does not prove GMM2 behavior or bound all possible replacement
kernels. No semantics-valid operator replacement with >=5 ms/cycle saving
has been identified. Next: source-backed ranking of the ~30 ms non-GMM
compute families before allocating a costly eight-rank GMM A/B/A run.

## Loop039 Run123 non-GMM ranking (2026-09-25)

In Run107's 15 valid synchronized target windows, stable non-GMM kernel
sums are quant matmul4.8205 ms, Compressor3.36575 ms, HC pre2.961 ms,
scatter cache2.34275 ms, then smaller DSA/indexer families. These sums
are not independent removability or stage critical-path reductions.
No single non-GMM family already establishes a >=5 ms/cycle opportunity.
Before pivoting Loop039, Run124 will test the existing fused W4A8 GMM1
on one 910B3 with real Run121 expert counts versus controlled layouts.

## Loop039 Run124 invalid GMM1 smoke (2026-09-25)

A single-910B3 call to the product fused GMM1 custom operator failed at
the aclnn argument check before timing: synthetic int32 W1 had 1-D
storageShape; WeightNzV2 requires 5-D NZ storage. No latency evidence.
Process exited, no serving service remained, all NPUs at idle power.
Run125 will construct product-compatible NZ storage and retry.

## Loop039 Run125 invalid NZ cast (2026-09-25)

Run125 still did not time the fused GMM1: torch_npu warned
allow_internal_format=False, npu_format_cast yielded format2, and
WeightNzV2 rejected its 1-D storage. No service remained. Run126 will
enable internal format before allocating W1 and verify format29.

## Loop039 Run126 invalid missing scale bias (2026-09-25)

Run126 reached W1 NZ format29, then WeightNzV2 required
weightAssistMatrix for A8W4. No timing. Borrowed W4A8 weight preparation
builds w13_scale_bias per expert/output channel; Run127 will supply a
[32,4096] float32 scale-bias tensor as the assist matrix.

## Loop039 Run127 product-shape GMM1 smoke (2026-09-25)

With torch_npu internal format enabled, synthetic W1 in NZ format29,
[32,4096] float32 assist matrix, and product logical shapes, the fused
WeightNzV2 GMM1 returns correctly shaped tensors for real Run121,
same-token dense, and single-expert count vectors. First-call wall
includes compile, so Run127 establishes interface validity only.
Run128 will use device-event A/B/A timing; no serving service ran.

## Loop039 Run128 single-NPU GMM1 route sensitivity (2026-09-25)

Synthetic product-shape W4A8 fused GMM1 with 60 valid tokens and 50
device-event samples per A/B/A block: real 15-active-expert distribution
median0.37385/0.38910 ms, dense 32-expert0.45886 ms, single-expert
0.33789 ms. The operator already responds to group-list sparsity.
The values are eager, synthetic-weight, one-card diagnostics and cannot
be scaled to Run107 graph GMM1 sum or formal E2E. No legal faster
replacement has been established. Next Run129 revisits steady product
rank dispersion to rule out the profiled HCCL wait as the larger gap
before Sol's priority decision. Read-only second-view review requested;
unverified model provenance cannot enter formal evidence.

## Loop039 Run129 unprofiled rank dispersion (2026-09-25)

Run98 unprofiled steady cycles64-255: target median46.5868 ms,
proposer6.4062 ms; median per-cycle rank duration spreads are 0.0812
and 0.0603 ms respectively. These duration data have no synchronized
absolute timestamps and cannot exclude a fixed cross-rank arrival
offset. Run107 HCCL wait should not be interpreted as independently
removable transfer without further evidence, but communication remains
an unresolved candidate. A second-view read-only review agrees on this
qualification; its actual service model ID is not independently
verifiable, so model_verified=false and Sol owns the decision.

## Loop039 PIVOT and Loop040 opening (2026-09-25)

Run130 is Sol's evidence review. A requested Astra Medium read-only
second view has no independently visible actual service model ID:
model_verified=false; its advice is not formal evidence. Sol pivots
Loop039 because GMM1 already skips empty experts and no legal faster
replacement has shown a plausible >=5 ms target-stage reduction.
This is a priority decision, not proof that GMM is at hardware bound.
The Run107 HCCL peer-wait remains unresolved as intrinsic transfer
versus fixed arrival offset. Loop040 first maps the stable 4.8205 ms
quant-matmul family per call and exact shape using existing Run107 trace,
then tests a concrete semantics-preserving repeated-projection/fusion
hypothesis. Correctness and official E2E gates remain unchanged.

## Loop040 Run131 quant-matmul trace map (2026-09-25)

Across all 15 valid Run107 synchronized rank-cycle windows, target
contains exactly 236 QuantMatmulWeightNz kernels and 43 fused GMM1 layer
markers. Per-layer ordered bins match the frozen config's c4/c128
alternation: two initial c0 layers 4/5 calls, 21 c4 layers six calls,
20 c128 layers five calls, plus one tail call. Quant family kernel sum
median4.8223 ms/cycle; c4 layer median120.28 us, c128 103.83 us.
This is device-order association, not exact Python module/shape mapping.
Run132 will use a temporary borrowed W8A8 apply probe only at 96-token
capture to test same-input repeated quantization; legal 12x1024 requests,
source restore and service stop are mandatory. No formal E2E yet.

## Loop040 Run132 W8A8 call probe (2026-09-25)

Legal 12x1024 requests succeeded on eight ranks. Every rank logged
899 W8A8 apply calls with x[96,4096] and weight[4096,512]:
43 target self_attn.wkv module prefixes occurred twice each in graph
setup, and three MTP wkv prefixes occurred 271 times each during the
run. This probe covered only wkv, not the other 236 quant-matmul target
kernels; cross-layer repeated tensor addresses cannot prove same
semantic input because graph memory is reused. The 556.756 tok/s
diagnostic value is not formal E2E. Borrowed source hash restored,
stop script released all eight NPUs below 3.5GB. Next Run133 is
a source-backed accounting of direct DSA CP quant calls and shared
quantization before another costly service capture.

## Loop040 PIVOT and Loop041 opening (2026-09-25)

Run133 source audit under frozen DSA CP/FlashComm1 flags explains the
obvious quant projection pairs: two wq_b chunks and the c4 indexer
query share one qr dynamic quant; wq_a uses local tokens, wkv uses
gathered cache tokens, and weights_proj is unquantized. Run131's
236 quant kernels are real, but no same-input duplicate quantization
candidate is established. Loop040 is PIVOTED, not a proof that quant
matmul is at bound. Loop041 will map actual HC pre/post, residual
clone/copy, RMSNorm and cache-write device costs before changing model
semantics. A >=2 ms unprofiled target-stage improvement plus legal
E2E gain is an incremental product gate; ultimate >=15% over Stock
remains the cumulative target. No new formal E2E was run.

## Loop041 Run134 HC/copy/cache census (2026-09-25)

Run134 audited all 15 valid, synchronized Run107 target windows offline.
Each contains 86 HcPre (kernel sum median 2.962 ms), 86 HcPost
(0.688 ms), 130 RmsNorm (0.940 ms), and 126 ScatterNdUpdateSk
(2.344 ms). These kernel sums overlap and are not removable wall time.
The decoder source clones hidden state twice per layer, but replay shows
only one CPU aten::clone and copy kernel names cannot be attributed to
those source clones. The available HcPreInvRms op accepts x/epsilon
only, so it cannot directly replace HcPreV2's HC transform. No safe
HC/copy edit is established. Run135 maps cache scatter ownership and
checks for duplicate or unnecessary writes; service stays stopped.
This commit is a checkpoint, not an execution stop.

## Loop041 Run135 decision and Loop042 opening (2026-09-25)

Run135 mapped 126 cache scatter kernels per target in every one of
15 valid windows. Ordered counts are 1,1 for the initial c0 layers,
4 for each of 21 c4 layers, and 2 for each of 20 c128 layers.
The frozen DSA CP source accounts for one SWA write per layer, one
compressed-KV write for ratio>1, and two indexer K/scale writes for
c4. No duplicate-write candidate exists from these counts; removing
one would risk persistent cache semantics. Loop041 PIVOTED without an
implementation or E2E claim. Loop042 now tests whether the roughly
10 ms apparent communication exposure is a removable inter-rank phase
skew or a critical-path-neutral wait. First Run136 uses existing
traces and DAG timestamps; no service restart yet. Every checkpoint
requires reassessing the highest gap and continuing while unblocked.

## Loop042 Run136 phase audit (2026-09-25)

Run136 reanalysed Run106's two unsynchronized profiled cycles.
Prepare entry skew was 9.637 and 10.510 ms; first reduce-scatter
start skew 10.193 and 9.735 ms, while end skew was only 0.0065 and
0.0118 ms. The latest rank changed from rank3 to rank0. Run98 steady
cycles64-255 show tiny per-cycle rank duration spreads (target median
0.0812 ms, proposer 0.0603 ms), but save no shared absolute timestamps.
Therefore an unprofiled steady phase offset and its effect on cohort
wall remain unknown. Run137 will use opt-in lightweight host timestamps
at stage boundaries in a legal eight-rank c12 run, without profiler or
forced NPU sync. Only then choose a scheduler or communication edit.

## Loop042 Run137 legal steady host phase capture (2026-09-25)

Run137 completed all 12 exact-1024 requests on eight ranks; each rank
recorded 294 cycles. Across cycles64-255, shared-host absolute stage
marks show median begin skew3.944 ms, after-target skew0.991 ms, and
after-proposer skew3.928 ms. The latest begin rank changes across
cycles, mainly rank3/4; target execution narrows rank phase, proposer
widens it again. Cohort host wall median58.130 ms. The timestamps are
host marks after asynchronous graph dispatch, not direct NPU collective
or removable time. Diagnostic TPS529.742 is not formal E2E. Temporary
runtime/fixed_serving source restored to recorded original SHA256 and
stop script returned all NPUs to about3.4GB idle. Run138 will locate
proposer host blocking and quantify candidate host critical-path time.
No scheduler edit is justified by phase skew alone.

## Loop042 Run138 proposer source/timing audit (2026-09-25)

Run138 found Run137 Host target/proposer scope medians4.872/41.992 ms,
while separate Run98 NPU event medians are target46.575/proposer6.400 ms.
The Host target+proposer rank medians are tightly near47 ms: the 42 ms
Host proposer scope is not removable CPU time. Run106 two profiled
cycles put median50.201 ms in borrowed _propose/model, with each outer
handoff wrapper <=1.284 ms. The more meaningful latest-rank end cadence
in Run137 is median54.082 ms and total-span mean54.877 ms/cycle; its
58.130 ms cross-rank cycle envelope is not a steady throughput period.
Astra Medium read-only review challenged the inference of a removable
phase gap and recommended a no-barrier wall/thread-CPU subphase capture.
Requested Astra model is not independently verified by tool metadata.
Run139 will locate wait versus CPU activity before any scheduler edit.

## Loop042 Run139 INVALID instrumentation mismatch (2026-09-25)

Run139 started eight-rank serving, but the first request raised
AttributeError because the temporary timing wrapper assumed the Step3p5
method `_build_step_attn_metadatas` existed on the actual
AscendDSparkProposer. It does not. EngineCore died; 11/12 partial
responses and 7.27 tok/s are invalid, and no subphase trace was
produced. The patch record confirms original SHA256 restored for both
borrowed files; stop script released all eight NPUs to idle. Run140
will instrument actual `set_inputs_first_pass`,
`build_draft_attn_metadata`, `_runnable`, and
`compute_draft_token_ids` methods after preflight verification.
This failure is a TaskCtl checkpoint, not an execution stop.

## Loop042 Run140 and Loop043 opening (2026-09-25)

After invalid Run139 was preserved, Run140 corrected temporary wrappers to
the actual AscendDSparkProposer methods and passed 12/12 exact1024
requests on eight ranks, with 290 cycles/rank. Across 1536 steady
rank-cycles, proposer model Host wall/thread CPU medians are
39.571/39.544 ms; nested `_runnable` 30.297/30.283 ms,
`build_draft_attn_metadata` 6.306/6.294 ms, and
`set_inputs_first_pass` 1.085/1.083 ms. Nested values are not
additive. Latest-rank proposer-end cadence median53.824 ms and
mean54.741 ms/cycle. Diagnostic TPS534.467 is not formal E2E.
Patch hashes restored and stop script released eight NPUs to idle.
Loop042 PIVOTED: phase skew alone does not support a scheduler edit.
Loop043 audits a product-specific DSpark fixed replay/execution boundary
against actual dynamic metadata and KV state before implementation.
Astra High read-only architecture review was requested; its model ID
is configured, not independently verified. Sol retains final decision.

## Loop043 Run141 DSpark replay boundary audit (2026-09-25)

Run141 confirmed the actual AscendDSparkProposer inherits eager
llm_base_proposer, rather than Step3p5. `_runnable` contains dynamic
context-KV input preparation and writes, three-layer model forward,
gather/LMHead, then a fixed seven-step Markov correction. The Markov
head uses replicated weights and has no attention metadata, KV write
or inter-rank communication inside that tail. It mutates logits via
`add_`; every replay must refresh raw logits, and output buffers must
remain live through acceptance. Run98 device stage medians sum54.199 ms
versus Run140 latest-rank proposer-end span mean54.741 ms/cycle, but
those are different cohorts and cannot be subtracted as a gain bound.
Astra High read-only review recommends timing the four actual segments
before a replay implementation; requested model is not independently
verified. Run142 is the next legal no-barrier eight-rank capture.

## Loop043 Run142 decision and Loop044 opening (2026-09-25)

Run142 passed 12/12 legal exact1024 requests on eight ranks and captured
64 consecutive steady DSpark segment samples per rank. Median Host wall/
thread CPU/device event times: context-KV2.992/2.992/0.733 ms,
three-layer model23.930/23.901/2.795 ms, gather0.775/0.774/0.055 ms,
LMHead0.663/0.663/0.411 ms, seven-step Markov2.655/2.655/0.695 ms.
Event sums can overlap and are diagnostic. The safe Markov replay
boundary is <1 ms of measured device interval and its Host work is
largely concurrent with target; no compelling exposed >=5ms savings.
Source hashes restored and service stopped, idle8 verified. Loop043
PIVOTED without implementation or formal E2E. Loop044 returns to the
~46.6 ms target device stage and will reconstruct exposed per-family
critical path before selecting an edit. Astra High was read-only input,
actual service model not independently verified; Sol made the verdict.

## Loop044 Run143 target timeline coverage (2026-09-25)

Run143 swept device intervals in all 15 valid synchronized Run107 target
windows. Median device union50.314 ms. Timeline-exclusive coverage:
GMM1 6.379 ms, GMM2 3.134 ms, quant matmul3.253 ms,
compressor3.366 ms, HC pre2.961 ms, cache scatter2.343 ms,
other compute14.099 ms. Communication10.248 ms exclusive includes
peer wait. This is a descriptive trace decomposition: deleting a
family would shift dependent kernels, so exclusive coverage is not
causal latency gain. Run98 unprofiled target median remains46.575 ms.
Run144 will split the 1413 other-compute kernels by exact operation and
source role before choosing a product-specific intervention. No service
run or formal E2E in Run143.

## Loop044 Run144 other-compute source grouping (2026-09-25)

Across 15 valid Run107 target windows, the 1413 other-compute kernels
have 48 exact names. Largest median exclusive timeline coverage is
VllmQuantLightningIndexer 1.7465 ms, generic MatMul 1.6893 ms,
SparseAttnSharedkv 1.6203 ms, transpose batch matmul 1.4643 ms,
MoeInitRouting 1.0553 ms, and rotary 1.0375 ms. Source mapping places
the indexer/attention in DSA and routing in MoE; generic MatMul and
rotary are shared implementations with unresolved unique callsites.
No single other-compute name has >=5 ms coverage. Summing mandatory
DSA stages is not a causal savings estimate. Run145 will discriminate
communication peer wait against GMM compute on matching windows.
This was offline only; service remains stopped.

## Loop044 Run145 communication versus GMM (2026-09-25)

On the same 15 valid Run107 target windows, 260 HCCL kernels per
window total median 11.281 ms, while 86 GMM kernels total 9.966 ms.
The first reduce-scatter dominates communication dispersion. Across
ranks its start skew is 20.533/8.650 ms in cycles64/65, but its end
skew is 0.010/0.0135 ms. The latest rank enters this collective in
~0.035 ms, whereas earlier ranks wait. The other 259 HCCL kernels sum
roughly 4.8–5.7 ms per rank. Reducing early-rank wait alone cannot move
collective completion. This profiled synchronization is diagnostic;
it does not prove the scheduler has no room. GMM is the larger stable
compute exposure, but 9.966 ms is not a removable bound. Run146 will
audit product GMM hardware efficiency and exact code path before an
edit. Offline only; service remains stopped.

## Loop044 Run146 product GMM traffic estimate (2026-09-25)

Run115 product shape has W1 packed 8 MiB and W2 packed 4 MiB per
expert. Run121 live routing has median672.5 active expert-layer pairs
per rank-cycle, corresponding to 7.881 GiB packed weights if each
active matrix is read once. Combining with Run107 GMM median9.966 ms
would imply ~849 GB/s, conditional on actual memory loads and across
different diagnostic cohorts. This is not measured HBM traffic or an
achievable bound. GMM1 source skips zero-M experts; GMM2 calls
torch_npu grouped matmul. Run107 profiling requested task_trace and
has no AI-core memory counters. Run147 will profile a one-card
product-shape GMM1 route with memory-access counters. Service remains
stopped.

## Loop044 Run147 invalid profiler launch (2026-09-25)

The shell could not open profile.log because the Run147 evidence parent
directory did not exist. Docker exec and the GMM profiler script never
started, so Run147 has no counter or correctness result. TaskCtl marks
it invalid. Run148 creates the directory before execution; service
remains stopped.

## Loop044 Run148 GMM1 memory counter (2026-09-25)

One-card Level1 MemoryAccess capture succeeded after the invalid Run147
launch. Four product-shape synthetic-weight calls using a real Run121
route (60 tokens, 15 active experts) have median118.743 us GMM1 kernel
duration, 129681 KB main-memory read and 2012.5 KB write. Read traffic
is 1.055× the packed W1 bytes for 15 active experts; effective
in-kernel read is 1118 GB/s. This supports weight traffic as the
dominant cost and confirms zero experts are largely skipped in this
synthetic case. It is not device peak or an eight-rank achievable
bound, and differs from the live target route. Run149 will profile
GMM2 with the same route. No serving service ran; NPUs idle.

## Loop044 Run149 invalid GMM2 attribution (2026-09-25)

The one-card GMM2 profiler executed, but median kernel duration was
~794 us versus ~83 us per live Run107 layer. Its synthetic call omitted
the W4A8 per-channel w2_scale_bias passed by the product source.
The counters therefore cannot bound live GMM2. TaskCtl marks Run149
invalid for product attribution and preserves the profile. Run150
will pass bias and feed GMM1-produced activation/scale. No serving
service ran; eight NPUs returned idle.

## Loop044 Run150 corrected GMM2 counter and next target gap (2026-09-25)

Adding product W4A8 bias2 and GMM1-produced activation/scale corrected
the one-card GMM2 path. Four samples have ~68 us median, ~66 MB
main-memory read and ~1 TB/s effective in-kernel read. Counter read
is close to the packed W2 bytes for 15 active experts. This is
comparable order to Run107 live ~83 us/layer, unlike invalid Run149
~794 us. Together with Run148, the two GMM kernels mostly read active
packed weights and already avoid empty experts. We have no demonstrated
semantics-safe >=5 ms GMM edit under frozen W4A8. This is not a
hardware peak proof. Sol pivots Run151 to the DSA Compressor/indexer/
attention/transpose sequence for a source-backed removable
intermediate or fusion opportunity. Serving remains stopped, NPUs idle.

## Loop044 Run151 DSA chain audit (2026-09-25)

All 15 Run107 valid windows show 43 sparse-attention calls: two c0
layers without compressor/indexer, 21 c4 layers with two distinct
compressors and one lightning indexer, and 20 c128 layers with one
compressor. The 62 compressor calls are exactly c4×2 plus c128×1:
attention compressed KV and indexer compressed state are distinct.
Per-window medians are Compressor3.366 ms, Indexer1.746 ms,
SparseAttn1.621 ms. Every transpose batch matmul follows an HCCL
alltoall after sparse attention, so it is not a pre-attention DSA
fusion step. Source and trace do not show a redundant compressor or
semantics-safe large DSA fusion. The first script assertion on
metadata-kernel count failed because metadata kernels are not present
at every compressor call; corrected script classified only actual
compressor/indexer kernels and passed. Run152 audits repeated TP8
collectives for coalescing. Offline only, service remains stopped.

## Loop044 Run152 TP collective adjacency (2026-09-25)

In 15 valid Run107 windows, 33–41 per-window allGather pairs are
strictly adjacent (median36). Their payloads are BF16 49152 elements
and FP32 3072 elements. The second FP32 calls total median0.257 ms
per window. Different dtypes prevent trivial coalescing, and other
collectives interleave dependent compute. The first script assertion
incorrectly expected 43 strictly adjacent pairs; the corrected
audit reports the actual range. No safe large TP8 collective edit
emerged. Independent read-only bound review then highlighted a
possibly larger formal E2E range gap: Run99 client request duration
exceeds FixedCohortServing.run wall by 11–17 s across repeats.
That is only accounting until stages are localized. Run153 will
reconcile existing Run99 timestamps and tail utilization offline.

## Loop044 Run153 E2E scope reconciliation and Loop045 (2026-09-25)

Run99 formal client median85.978 s versus four rank0 FixedCohortServing
walls69.040 s leaves16.938 s (19.7%) outside the internal timer.
Across 12 cohorts, subtracting maximum client TTFT from each
request-envelope-minus-runtime difference leaves median0.140 s
(range0.074–0.172). The gap is primarily in first-token scope,
not established wasted serving time. Stock baseline on another date
had cohort maximum TTFT median1.514 s versus Run99 Extreme3.922 s;
that comparison motivates direct capture but is not causal.
The timer starts after build_extreme_runtime. Existing JSON lacks
common-clock admission, prefill, handoff, construction and publication
marks; per-slot count_history was not persisted, so tail compaction
benefit cannot be quantified. Loop044 PIVOTED and Loop045 opened for
one legal diagnostic capture. Independent Astra High review was
read-only; configured model cannot be independently verified and Sol
owns this decision. No formal Run99/Stock rerun.

## Loop045 Run154 cold boundary and parked-slot capture (2026-09-25)

A legal eight-rank 12×1024 diagnostic completed 12/12 with eight
boundary and runtime files. Same-host clock medians: client-to-first
worker execute0.383 s; first execute-to-handoff5.954 s;
handoff-to-runtime-build0.083 s; build_extreme_runtime0.041 s;
FixedCohortServing16.636 s; worker publication-to-client end0.175 s.
First-token scope is mainly pre-handoff prefill/model execution in
this cold-service run, not runtime construction. Parked slots account
for 14.484% of fixed slot-cycles; this is wasted-shape exposure,
not achievable speedup. Diagnostic TPS527.999 is not formal E2E.
All sources restored to original SHA and eight NPUs idle. A later
EngineDeadError in service log occurred after the 12 successful
responses; preserved excerpt, no claim about its cause. Run99 formal
was warm-cache, so Run155 will warm with 48 legal requests before
measuring one 12-request cohort in the same service. No reliable
formal E2E rerun.

## Loop045 Run155 warm boundary (2026-09-25)

Legal 8-rank same-service 48×1024 warmup plus 12×1024 measured
diagnostic: 60/60 requests succeeded and all 40 rank-cohort runtime
records passed. Warm measured client envelope20.779 s and diagnostic
TPS591.369; neither is formal E2E. Cohort5 boundary starts with one
trailing warmup execute before measured client start; analyzer selects
first execute at/after measured start. Same-host medians: client-to-
first execute0.241 s; first execute-to-handoff3.621 s; runtime build
0.00264 s; fixed serving16.728 s; publication-to-client end0.184 s.
Runtime construction cannot explain a multi-second E2E gap. Measured
parked slot-cycles18.525%, exposure only. Temporary source restored,
service stopped, 8 NPUs idle. Next Run156 is offline to discriminate
prefill work and tail opportunity before another NPU intervention.

## Loop045 Run156 cohort accounting (2026-09-25)

Offline Run155 8-rank audit: warmed cohorts3/4 each schedule about
50.6k prefill tokens across 11 calls and take3.66 s from first
execute to handoff. Measured cohort5 schedules only1.63k tokens
across12 calls yet takes3.62 s. Volume alone cannot explain the
pre-handoff delay. Per-call admission/tokenization, Host and device
time are unresolved; direct stage timing is the next high-value probe.
Parked slot exposure spans9.7–19.5% across five cohorts; it is not
a measured compaction speedup. Run157 will inspect source and design
minimal timing probes before another NPU run.

## Loop045 Run157 pre-handoff timing design (2026-09-25)

`bench.py` launches12 requests concurrently; Run155 measured client
starts differ by milliseconds. The worker boundary logs execute entry
but not exit, so each ~0.4 s start spacing can be worker work or
inter-call scheduler/admission wait. Run158 will temporarily wrap
NPUModelRunner.execute_model to record same-host entry/exit on all8
ranks in a legal warmed service. No device synchronization; in-call
wall is not device-only time. Design evidence in run157.

## Loop045 Run158 worker call timing (2026-09-25)

Legal 8-rank same-service48×1024 warmup plus12×1024 measured
diagnostic:60/60 successful, all40 rank-cohort records and8 rank
method logs complete. Measured pre-handoff median3.433 s, including
10 prefill execute_model calls totaling2.928 s and inter-execute
gaps0.478 s; sample_tokens totals0.447 s within those gaps. Final
handoff preamble0.026 s. The large latency is inside worker calls,
not primarily scheduler/admission idle. Python wall is not NPU
device-only time. Diagnostic measured TPS622.761 is not formal E2E.
All temporary source restored; service stopped and8 NPUs idle.
Next Run159 inspects prefill execute_model internals and designs a
minimal Host-versus-device timing probe before runtime changes.

## Loop045 Run159 prefill stage probe design (2026-09-25)

Source inspection identifies execute_model preparation, normal model
forward and post-process boundaries. Run160 will timestamp these
inside the worker without synchronization, excluding the final
Extreme handoff call. Python wall will locate Host-facing stage
latency but cannot establish device kernel time. Probe design and
source anchors are in run159 evidence.

## Loop045 Run160 invalid patch collision (2026-09-25)

No service or NPU work. Stage and timing temporary patches shared
the same backup path, causing timing install to fail before startup.
Trap restored stage and boundary patches to original SHA and stopped
service. Run160 is invalid in TaskCtl; failure logs preserved. Run161
will give patches distinct backup paths and repeat the legal probe.

## Loop045 Run161 prefill stage split (2026-09-25)

Legal8-rank48×1024 warmup and12×1024 measured diagnostic passed
60/60, all40 rank-cohort records. Across ranks, measured prefill
execute_model median3.397 s, with `_model_forward` Python wall3.122 s
(~91.9%) and preparation0.258 s. The large pre-handoff gap is
inside model-forward invocation, not mainly request admission or
Runtime construction. This is Python wall, not device-only time;
forward can include NPU work, HCCL and waits. Diagnostic TPS580.055
is not formal E2E. All patches restored and service stopped with8
NPUs idle. Run162 should obtain device-level prefill evidence before
any code change.

## Loop045 Run162 invalid Host pinned-memory startup (2026-09-25)

The rank0 single-forward profiler was installed but service failed
before health check or benchmark. Worker TP6 KV block-table pinned
Host buffer allocation raised `torch.OutOfMemoryError` from
`aclrtMallocHostWithCfg` error207001. At failure Host had about16 GiB
free and full71 GiB swap, but root resource ownership is not yet
proven. No NPU trace or performance result exists. Runner terminated,
its detached health-check shell cleaned, source patches restored to
original SHA, service stopped and8 NPUs idle. Run162 is invalid in
TaskCtl. Next Run163 is read-only resource audit before considering a
new service attempt; do not kill unrelated processes.

## Loop045 Run163 Host resource audit (2026-09-25)

Read-only audit of the stopped dedicated
`vllm-ascend26-dsv4f-w4a8` container found3646 live cgroup PIDs:
3270 multiprocessing.forkserver processes (~716 GiB summed RSS),
327 spawn processes (~194 GiB), eight `python3 -` (~59 GiB), plus
trackers/init. Host swap71 GiB is full; cgroup reports ~985 GB
current and no configured memory.max. These residual processes create
strong pressure and plausibly explain Run162 pinned Host allocation
failure, but direct allocator causality is unproven. No active service
or NPU process. Run164 will restart only this dedicated stopped
container and verify resource recovery before retrying the profiler.

## Loop045 Run164 dedicated container resource recovery (2026-09-25)

`docker restart --time 10` first exited1 because daemon did not
receive an exit event, but its stop attempt removed most residual
processes. Explicit `docker stop --timeout 30` then `docker start`
both exited0. Dedicated container cgroup fell from3646 to1 PID;
Host memory used fell from841 GiB to20 GiB and swap from71 GiB
to749 MiB. Mounts, original model-runner/fixed-serving SHA, stopped
service and8 idle NPUs verified. Run164 passed overall with the
initial daemon error preserved. Run165 will retry the prefill profiler
as a new legal 8-rank diagnostic.

## Loop045 Run165 rank0 prefill device trace (2026-09-25)

After dedicated-container recovery, legal8-rank48×1024 warmup
plus12×1024 measured diagnostic passed60/60 and all40 rank-cohort
records. One rank0 measured prefill `_model_forward` was profiled.
CANN step trace: stage444.180 ms, compute39.597 ms, exposed
communication2.862 ms, Free/no recorded device task401.721 ms
(90.44%). Independent kernel-interval union agrees after removing
264 HCCL/Aiv duplicate labels. This is one rank, one call with
profiler initialization and explicit sync; no formal E2E or speedup
claim. It strongly motivates Host/launch/dependency attribution,
not GMM/HCCL kernel tuning for this prefill call. Diagnostic
TPS635.156 is not formal. Sources restored, service stopped,8 NPUs
idle. Run166 will inspect CPU/API and all device-task timing offline
before selecting a prefill execution intervention.

## Loop045 Run166 Host-to-device gap correlation (2026-09-25)

Offline exact-timestamp analysis of Run165's one rank0,83-token
profiled prefill forward:2825 kernel rows,2561 unique intervals,
444.124 ms first-to-last device span and401.665 ms without a
recorded kernel. All kernel starts match HostToDevice flow endpoints.
Conservatively using the earliest matching flow for each next task,
357.603 ms (89.0%) of device-free gaps lie before the reported Host
flow start;44.062 ms lie afterward. This strongly favors Host
submission pacing in the profiled call over pure device compute or
HCCL. The profiler perturbs this matched shape (Run161 unprofiled
rank0 83-token `_model_forward` ~0.320 s versus profiled stage
0.444 s), so357.6 ms is not removable-time or formal E2E proof.
Run167 requests Astra High independent architecture/bound review
before Sol selects a correctness-gated prefill execution experiment.

## Loop045 Run167 independent architecture review (2026-09-25)

Configured GPT-6 Astra High read-only reviewer recommends a narrow exact-state, eight-rank genuine prefill `_model_forward` capture feasibility experiment, with full metadata and prefill write-set restoration and A/B/A-prime correctness. Existing DSA CP graph capture rejects prefill; do not force decode dispatcher. Run165/166 Host-flow evidence is one perturbed rank0 call and cannot establish an E2E speedup. Sol accepts source/write-set feasibility audit as the next step. Reviewer model ID is configured but not independently visible in the execution record. Details: `evidence/20260925_loop045_boundary/run167/architecture_review.md`.

## Loop046 Run168 prefill capture source audit (2026-09-25)

Read-only audit confirmed the stock DSA CP graph capture rejects prefill, while a separate exact-state closure must preserve actual forward context, rank-local metadata, all SWA/compressed KV/compressor/c4 indexer writes, async gather and connector order. The A5-only full o-proj weight pointer switch is not expected on 910B3. Capture compatibility and shape reuse remain unproven. Run169 should quantify natural prefill batch-state frequency and unprofiled latest-rank cost from existing legal evidence before building a costly full-state graph probe. See `evidence/20260925_loop046_prefill/run168/source_audit.md`.

## Loop046 Run169 natural prefill shape frequency (2026-09-25)

Offline eight-rank Run158/161 measured warmed cohorts have10/12 pre-handoff calls respectively; every unpadded token count is unique within each cohort. Seven/eight large calls consume median2.729/2.928 s of their respective measured wall metrics, but these are opportunity ceilings, not removable time. Across runs only token counts8 and83 repeat, with metadata equivalence unproven. A separate exact-state graph for each one-off shape has no demonstrated amortization. Sol shifts next to locating the scheduler/admission cause of repeated expensive forwards and a bounded coalescing test; no graph or E2E gain claim. Evidence: `evidence/20260925_loop046_prefill/run169/frequency_analysis.json`.
