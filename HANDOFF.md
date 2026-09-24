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
