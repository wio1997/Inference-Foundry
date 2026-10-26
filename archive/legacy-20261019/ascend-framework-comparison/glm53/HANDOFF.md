# 当前恢复点 2026-10-19T09:15:43.025775+00:00 — 用户要求暂停

L22 两框架诊断、profile 与采集均已结束；当前无模型/benchmark/profile/collector 执行。最新用户要求“最新的这个跑完就先停一下 总结归纳 然后到github”。只完成本地总结和 GitHub 发布，然后保持暂停；禁止旧 launcher 重放或自动开始下一候选。服务、缓存和当前 L17 D 运行时保留；objective_complete=false。

归纳见 [diagnostics/model-profile/C32-L22-pause-summary-20261019.md](diagnostics/model-profile/C32-L22-pause-summary-20261019.md)；验收见 results/C32-L22-terminal-primary-acceptance-v1.json。SG stop HTTP500 与控制失败保留，32 rank 文件完成另证；VL 32 原始 rank 完整但仅两个代表解析。不构成新正式性能结果。旧 SSH 阻塞及下方“继续执行”指令均为历史。

SG 实际6793/完整资格6492，native pending128（旧64+新64）；vLLM 三诊断192/57696另列。所有原始报告和历史 ledger 保持原字节。GitHub 当前发布收据另存，旧收据不代表本快照。

下方为历史记录；仅此当前头与 STATE.user_pause 决定是否执行。

# 当前恢复点 2026-10-19T08:21:11.157894+00:00

L22 vLLM真实终态：32logical/64backend/19232output，全部计数终态与前缀命中通过。两D scope start/stop均200，原生文件完成性待独立采集。客户端C32/真实80K/600不变，采样按DP4/10与总D>=8，实际触发12；不能称全D32或性能提升。

L20/L21/L22本轮vLLM实际合计192backend/57696output，独立ledger见 results/model-request-ledger-vllm-L20-L21-L22-diagnostics-v1.json。原SG6729/6492/1432790与64pending字节不变。无新增SG记账；启动前读其独立run目录防重放。

下一步：接受L22 vLLM原生trace和D-idle，执行已冻结L22 SG一次，再采集匹配batch/context的trace。持续授权有效，无需父/用户审批；仍需继续优化，objective_complete=false。旧SSH阻塞已历史化。实际资源经ZCode；无跨对话消息/Git/浏览器/系统更改。

下方为保存的历史记录。

# Current state - 2026-10-19T07:44:52Z (L20 cold-cache diagnostic adaptation in progress; vLLM v2 counts valid, no usable C32 profile)

- Root status is `L20_COLD_CACHE_DIAGNOSTIC_ADAPTATION_IN_PROGRESS`; `objective_complete=false`. No active model/benchmark/profiler; a metadata-only known-key check is in progress.
- vLLM v2 `20261019T-L20-vllm-C32-paired-diagnostic-v2` is terminal: true coordinator DONE, `client_rc=0`, `host_wait_timeout=false`; report `runs/20261019T-L20-vllm-C32-paired-diagnostic-v2/diagnostic-result.remote.json` SHA-256 `0c867a9cc19873d77a3277c7ee78aa9f35fbdcc0084e9c77354e78524177db49`. Actual 32 logical / 64 backend / 64 count-terminal-qualified / 19232 output (P32 + D19200), 0 pending, 0 retries.
- No valid C32 profile: 32/32 P native `cached_tokens=0`; 605/605 D metric pairs complete; sampled D peak 4 running / 4 active DP; `NOT_STARTED_ADMISSION_TIMEOUT`; 0 profile start/stop, no new trace. Independent review `results/C32-L20-vllm-v2-independent-diagnostic-review-v1.json` SHA-256 `49cff99e0fc553d2c495cc06692a8f8670e9947aaf357609e3e8a0ad25e47baa`, decision `COUNTS_AND_LIFECYCLE_VALID_DIAGNOSTIC_NO_USABLE_C32_PROFILE`.
- No performance improvement/regression attribution and no replacement of historical vLLM `530.8125387945` TPS / `19.1526558011` ms TPOT or SGLang `411.9606331005` TPS / `37.9931225359` ms TPOT.
- Cold-cache read-only receipt `runs/20261019T-L20-vllm-cold-cache-readonly-v1/receipt.json` SHA-256 `90d4ec4217299990a14d3d75a19ec0205a077b599b80dd85fe0432b8e96c2cb8`: four CID image/argv checks pass, prefix caching still on, D16 native running/waiting all 0, 0 new model/SDK; cache miss cause UNKNOWN. Referenced as-is, not modified.
- Metadata-only check `runs/20261019T-L20-vllm-known-key-metadata-v1` (directory absent at this checkpoint) is in progress: read-only `DistributedObjectStore.init(0, init_bm=False)` / `batch_is_exist` / `batch_get_key_info` / `close`; 0 models, no BM/NPU allocation; result UNKNOWN, not pre-written as success.
- SSH was restored this round (password auth; real remote true + L20 request + read-only 4-node success). The old external SSH transport block is historical. Historical L17 actual320 is not a current active request.
- Ledger: new `results/model-request-ledger-L20-vllm-diagnostic-v2.json` SHA-256 `a768c1abacec46c9de69ea2207a3abd33781c619c6a680e2237ec139d7328226` is a separate vLLM partition: 64 actual attempts / 64 count-terminal-qualified / 19232 output / 0 cache-qualified; native full-SDMA not newly collected; cumulative vLLM history null. The SGLang ledger `results/model-request-ledger-through-L17-formal-v2.json` SHA-256 `b86b1b11d582522df50ed15835f29d72373c8e015662cf413e61f418d8a48cfb` is byte-identical and remains the root ledger (6729/6492/1432790/64 pending); vLLM counts were not added.
- SGLang v2 remains `NOT_STARTED` because the vLLM sampling gate failed and work pivoted to cache/admission evidence, not because of waiting for new user authorization.
- Next: primary accepts the metadata key/source evidence, decides the minimal cache/concurrency-gate correction, then continues the actual paired diagnostic. Ongoing authorization remains valid; no user/parent approval required; no cross-conversation message.
- Before snapshot and exact before/after SHA: `runs/20261019T-L20-vllm-diagnostic-checkpoint-zcode-v1/`.

以下为更新前 HANDOFF 全文（2026-10-19T07:16:03Z 版本，SHA-256 `2d3263deeb2f790f8f8694dc3c5be279dcd0d06866fe2ddeff4eddbef2ae2b39`），逐字保留为历史（原文未改动）：

---

# Current state - 2026-10-19T07:16:03Z (L20 external SSH transport blocked: actual TCP refusal, zero model dispatch)

- Root status is `L20_EXTERNAL_SSH_TRANSPORT_BLOCKED_NO_MODEL_DISPATCH`. Primary blocker review `results/C32-L20-control-channel-blocker-independent-review-v1.json` SHA-256 `526f1f9c611ce8c6c27d2e2412b15cdbce4bdd76945893511e4842726bf3ecd5`, decision `EXTERNAL_SSH_TRANSPORT_BLOCK_CONFIRMED_ZERO_L20_MODEL_DISPATCH`; global `objective_complete=false`.
- The actual blocker is external TCP refusal at `61.241.77.34:60011`, not merely a missing local socket file: the latest direct attempt was refused before authentication; earlier a foreground master authenticated and a multiplexed true returned rc0 in 0.973 s, then the session exited 255 with the server not responding. The password was previously verified valid; no new password request or credential persistence. The async user question on restoring the forward or providing a new stable endpoint remains pending; do not ask again.
- The single L20 vLLM launcher exited 1 locally at `AssertionError: foreground authenticated control socket missing`, before SSH, lock, prechecks or coordinator. No local run directory or active lock was created; model requests 0, profiles 0, SSH 0, SDK init 0, prechecks 0, and planned 32/64/19232 were not booked. SGLang cohort `NOT_STARTED`; no retry, replay, repair or socket substitution.
- All prior executors are terminal and active execution is none. Frozen L20 settings/review/source and L17 results remain byte-identical; all 8 primary-review input SHA-256 values were re-verified unchanged.
- Ledger remains unchanged: 6729 backend attempts = 6492 qualified + 173 old unqualified + 64 pending native; qualified output 1432790. Zero new model requests and zero new profiles were booked as actual.
- L18 accepted analysis applies only to the existing real C32/80000-input L13 profiles ranks 0/16 (NoCopy off), not to new performance evidence: the four longest H2D API waits are 71-78 ms, overlap >94% with recorded noncontrol NPU tasks (predominantly nondefault-model graph subtasks), and return 0.24-0.33 ms after the final model task. This does not show avoidable pure copy, physical utilization, pure compute or a causal framework delta.
- L19 readiness remains: the actual vLLM native profile is configured; both vLLM APIs 43 (DP0-7) and 45 (DP8-15) are required as disjoint local 8-DP control scopes under hybrid-LB/local_engines_only fanout, one start and one stop each, with no double start; SGLang uses one control scope API36 via FanOutCommunicator. No existing profile traces were found (0 on nodes 43 and 45).
- The initial vLLM worker environment shows `TASK_QUEUE_ENABLE` unset, but that does not establish the operative inherited value; no flag change is admitted.
- Corrected authority: the ongoing human optimization authorization remains valid and does NOT require new human or parent approval. The earlier "requires new explicit authorization" wording referred only to an internal new primary freeze and is not a current blocker.
- After stable SSH/forward restoration, the primary autonomously creates a new versioned once-only entry and review for the failed-local vLLM launch (never replay the old once-only entry), then executes the existing real 32x80K/600 paired cohorts for vLLM and SGLang through ZCode. No full 128 formal and no serial long warm.
- No optimization promotion, no new records booked as actual, and no new performance, SLA or causal claim.
- Before-update full copies and exact before/after SHA-256: `runs/20261019T-L20-external-transport-checkpoint-zcode-v1/before/` and `runs/20261019T-L20-external-transport-checkpoint-zcode-v1/receipt.json`.

以下为更新前 HANDOFF 全文（2026-10-19T07:09:15Z 版本，SHA-256 `0238f667cdc9a1e2ea4e740f3a8cdc96a98de235a5944779a9fd3f1ee9ee9948`），逐字保留为历史（原文未改动）：

---

# Current state - 2026-10-19T07:09:15Z (L20 vLLM diagnostic launch failed before coordinator: exact control socket missing)

- Root status is `L20_VLLM_DIAGNOSTIC_LAUNCH_FAILED_SOCKET_MISSING`. The user renewed authorization; the old L17 closeout "no new run" statement is historical only and is superseded. Global `objective_complete=false`.
- Exact frozen bytes were verified before execution: controller `scripts/glm53_C32_L20_paired_diagnostic_controller_v1.py` SHA-256 `e073caa002e588ad422676d8d560aa1154464cdaafa0177d9d4f98ff6957defd`; settings `config/C32-L20-paired-diagnostic-v1/vllm-cohort-settings-v1.json` SHA-256 `e7819e07c3fd344d3cc1b62d875aa5b9606483e341b8fdbcbb7a7f08323c9f5b`; actual review `results/C32-L20-paired-diagnostic-actual-review-v1.json` SHA-256 `5f7cec9773bec8c81d44e6274197b153acc7696e767be10a9971f7fa388c7be3`; guards `runs/20261019T-L20-primary-pre-freeze-fixes-v1/final-local-guards.json` SHA-256 `d206f35f468c795969729ea74a7bc8598d789ca0ea0b14dc63f66b0aa6084967` with vLLM rendered CPU-host SHA-256 `95abcb519c2385b17d27abdc8396b7b2f526eb2aabb82bf9383b27adb88858fe` and authorization/prerequisite PASS. The exact frozen launcher was then executed exactly once from the project root.
- The single launch returned rc 1 with `AssertionError: foreground authenticated control socket missing` at `main()` line 445, before settings authorization, prechecks, lock acquisition or coordinator detach. The expected socket `/Users/wio/work/sglang-glm53-pd/ssh-control-l20-foreground-20261019` was absent before launch and after a 90 s bounded local wait. No socket was created, replaced, repaired or substituted; no second launch was attempted.
- Concrete terminal state: no coordinator PID or session, no `runs/20261019T-L20-vllm-C32-paired-diagnostic-v1` directory, no active lock, no `controller-progress.json`, no remote contact. Actual counts: 0 model requests, 0 SSH, 0 profiles, 0 SDK init, 0 prechecks. Planned 32 logical / 64 backend / 19232 output are not booked. SGLang cohort is `NOT_STARTED`. Raw outer stdout/stderr/exit and the launch record are preserved in `runs/20261019T-L20-vllm-diagnostic-executor-v1/`.
- L18 existing-trace primary analysis is accepted: `results/C32-L18-existing-trace-primary-analysis-v1.json` SHA-256 `b8f51edde10a4fc63615a08b6d3e9473469d5e51302da7a1f98376fd5ca4b414`, decision `ACCEPTED_BOUNDED_DIAGNOSTIC_PIVOT_TO_PAIRED_DEVICE_EVIDENCE`; scope is the existing real C32/80000-input L13 profile ranks0/16, not a new performance measurement and not current L17 paired evidence.
- L19 readiness source facts: `results/C32-L19-vllm-profile-readiness-review-v1.json` SHA-256 `b64cbf4b3dcf7aa1b735a07542b52d37ac2b5752ef8e1574bbf5e575b007c787`, decision `APPROVED_READONLY_CURRENT_VLLM_D_PROFILE_READINESS`, source SHA-256 `f2894c4409ad714432a2fd319927373af97c81e4af181f4c0bf4f44ee9635e71`, host-code SHA-256 43 `19efa04fef42caad3fd265b22427e9e1bf4788c25cf451de75541a5ad9cd8239` and 45 `a7632b52d86ffce4ed4ec2da98a9137f3737a7fb8344e51f2c835b89612424e9`, with zero inference/profile/service/SDK init. Profile scope config review v2 is `results/C32-L19-vllm-profile-scope-config-review-v2.json` SHA-256 `c2b72b62165746a1a91d5897e47c0e1d58911a4d1e6632dc7c00268b11b02f5c`.
- New approved L20 loop pointer: `results/C32-L20-paired-diagnostic-actual-review-v1.json` SHA-256 `5f7cec9773bec8c81d44e6274197b153acc7696e767be10a9971f7fa388c7be3`, decision `APPROVED_BOUNDED_PAIRED_C32_DIAGNOSTIC`; run IDs vLLM `20261019T-L20-vllm-C32-paired-diagnostic-v1` and SGLang `20261019T-L20-sglang-C32-paired-diagnostic-v1`. The vLLM attempt is terminal FAILED; the approval remains unexecuted and is not a completion.
- L17 latest valid measurement, target miss and ledger 6729 are unchanged: measurement `results/sglang-C32-L17-native-formal-measurement-v2.json` SHA-256 `906c4e4084bb42cf7268af5c96af67961b51a5ff9230260931bc3930e5e7d8a7`; target missed true; ledger `results/model-request-ledger-through-L17-formal-v2.json` SHA-256 `b86b1b11d582522df50ed15835f29d72373c8e015662cf413e61f418d8a48cfb` (6729 backend attempts). No new actual counts were produced.
- Next: primary reviews the terminal launch failure and decides whether to restore the exact foreground control socket and authorize a new single launch. No retry, replay, repair, socket substitution, new profile outside the exact script, full formal 128, host log scan, Git/provider/system change, or SGLang cohort before that.
- Before-update full copies: `runs/20261019T-L20-vllm-diagnostic-executor-v1/before/STATE.json` SHA-256 `7941806445c257395e939484e71f97f8e169ca60a748b956fa663f8a07b503cb`; `.../before/STATUS_BRIEF.json` SHA-256 `0ec8356738d76fb749ad8398da3f775ee2c832541ea08a0b37835d340e6fbd77`; `.../before/HANDOFF.md` SHA-256 `be8e3ec1a57a64efb7aecb1074e3ec0fa4322b41cd14d9328b12dfa309fd40a6`.

以下为更新前 HANDOFF 全文（2026-10-19T06:22:06Z 版本，SHA-256 `be8e3ec1a57a64efb7aecb1074e3ec0fa4322b41cd14d9328b12dfa309fd40a6`），逐字保留为历史（原文未改动）：

---

# Current state - 2026-10-19T06:22:06Z (L18 existing trace transfer stopped at missing control socket)

- Root status is `L18_EXISTING_TRACE_TRANSFER_FAILED`. The single authorized collector `scripts/collect_C32_L18_existing_trace_pair_v1.py` (SHA-256 `73b81df5b237483eb76da738fb6f932c50c4705cc9b89b624b3e0b3dddf8af26`) exited rc 1; stdout is empty and the raw stderr is preserved at `runs/20261019T-L18-existing-trace-executor-v1/collector.stderr.txt`.

- Stop condition: the required control socket `/Users/wio/work/sglang-glm53-pd/ssh-control-l17-recovery-20261019` is absent; `AssertionError` at line 61 `assert Path(SOCKET).is_socket()`. No SSH, kubectl, remote trace read, run directory, retry, repair or socket substitution occurred. Actual counts: 0 model requests, 0 new profiles, 0 SDK init, 0 remote writes.

- The collector source hash and review `results/C32-L18-existing-trace-transfer-review-v1.json` (`APPROVED_EXACT_TWO_EXISTING_TRACE_READS`, SHA-256 `bfd6c8a39db50901b2c83d7b195f5af19f058b7918035e48926584681cbaff24`; host-code SHA-256 36 `f761ead707ca9534f0361fef184174a5fc3f4136eb50ed0116f05aeab35b0bc7`, 41 `e75b20ac938dee20d9d7b78e1713c5246bc92455edea438205e44a427347f325`) matched before execution.

- L18 diagnostic remains uncollected: distinguish a host synchronous wait draining prior device work from avoidable extra preparation, reusing the real C32 80K-base L13 trace; no vLLM paired trace exists yet. Evidence maturity is E3 for the bounded SGLang profile only; causal delta is UNKNOWN.

- Loop record `diagnostics/model-profile/loops/L18-D-native-critical-path-attribution-v1.json` (SHA-256 `00e96dc4822b42f8a7cbf777c0a24e98e409355afd96a00dbc6f30eb2d5489d7`); receipt `runs/20261019T-L18-existing-trace-executor-v1/receipt.json`. L17 measurement, ledger and target-miss records are unchanged; no runtime, performance, causal or SLA conclusion is claimed.

- Next: primary parses the accepted L15 API/flow/device-timestamp evidence and actual installed vLLM source, then chooses the minimal paired evidence if needed; no full formal rerun now. User renewed optimization authorization continues; the prior L17 closeout no-new-run statement is historical scope only.

以下为更新前 HANDOFF 全文（2026-10-19（L17 formal v2 closed）版本，SHA-256 `b4898a42577cb30a1929e392f23f1c1a5543f9668e89b949e75207f39f4be898`），逐字保留为历史（原文未改动）：

---

# Current state - 2026-10-19 (L17 formal v2 closed: valid measurement, decode target missed)

- Root status is `L17_FORMAL_V2_VALID_MEASUREMENT_DECODE_TARGET_MISSED`. L17 formal v2 is a valid C32 measurement, but the decode optimization is NOT accepted or promoted. `objective_complete=false` globally.
- Primary decision `results/C32-L17-native-formal-decision-v1.json` (SHA-256 `37b6897820aeecb8f02eee10d8bed3149e936088bf329ee17fabcf3275d04f49`) and primary independent review `results/C32-L17-formal-primary-independent-review-v1.json` (SHA-256 `5d504da9a50e331eea6bb4ee1f0d6f0f5d614f945391f39c7858d8050611863d`) accept the actual measurement, not the optimization.
- Measurement `results/sglang-C32-L17-native-formal-measurement-v2.json` (SHA-256 `906c4e4084bb42cf7268af5c96af67961b51a5ff9230260931bc3930e5e7d8a7`): 128/128 formal requests, 76800 formal D output tokens, full clock 186.42558009 s, TPS 411.9606331, TTFT P50 16.6566555 s, TPOT P50 37.9931225 ms, SLA false. 320 qualified backend attempts / 76992 output tokens (160 P + 32 warm D + 76800 formal D), all derived from the canonical raw rows (`runs/20261019T-L17-sglang-formal-C32-native-v2/formal-result.remote.json`, SHA-256 `f60db145b162582cc0716e0d25360affbc34d42099f269a84fb196f2c1f676ac`).
- Frozen target: >=5% median TPOT improvement AND TPS no regression. Observed versus historical L12D: TPS +5.66505%, median TPOT +3.640395% (worse), TTFT P50 -6.19797%. Target missed; no decode optimization promotion; one-cohort reliable gain INCONCLUSIVE; no causal regression, statistical significance or optimum claim.
- Native evidence `results/sglang-C32-L17-native-formal-native-v2.json` (SHA-256 `0cba128df67878cd4c9783845edf3e31347810b0e5025fd8d1dde0670139d1f7`): 32 P readers (160 new good batch_get each), 32 D typed SDMA destinations, 3456 descriptor batches, 2808364105728 descriptor bytes (NOT wire bytes; `physical_wire_bytes=null`), 248 original metric pairs / 496 HTTP rows, all32 Graph rank deltas positive, MTP5, D storage nulls 0, raw fields unchanged.
- Runtime: D36/41 remain the L17 NoCopy candidate; rollback was NOT performed; no model parameters changed. Historical L12D (`389.8740711045039` TPS, SLA false) remains the retained best accepted decode optimization/reference, not the current runtime. Original L17 formal v1 remains historical terminal FAILED before formal dispatch (32/32 warmups, 64 actual backend attempts, 64 output, formal 0); its 64 warmups remain `pending_native` and are not failed HTTP.
- Ledger `results/model-request-ledger-through-L17-formal-v2.json` (SHA-256 `b86b1b11d582522df50ed15835f29d72373c8e015662cf413e61f418d8a48cfb`): 6729 backend attempts = 6492 qualified + 173 old unqualified + 64 pending native; qualified output 1432790; pending native output 64; 460 disjoint unique RIDs. The old ledger `results/model-request-ledger-through-L17-interrupted-warmup-v1.json` (SHA-256 `4c2cd04961806c217b0c8be3b892ad61c88cfda816f44f4174ea8bc8bad2f32a`) is unchanged.
- Current comparison `results/C32-L17-current-comparison-v1.json` (SHA-256 `09b12f0b9d08931f82caf2509bb3c360eef569a582b9f03c6a1bdaabb9b8dda2`): L17 versus unchanged L12D and vLLM C32 references, with exact values and deltas.
- No active execution: all coordinator/postexecutors are terminal, SSH is restored, and there is no current blocker. No new optimization deployment, repeat or rollback is authorized by this closeout.
- Next: primary review of the existing L15 native AscendCL/API correlation and source evidence. Zero model/profile before any new frozen candidate.
- Current docs: `STATE.json` SHA-256 `5a2de13859b0c21254934d65f61905b4069cd609e5e902e446fd2900056e4432`; `STATUS_BRIEF.json` SHA-256 `7aa65849c54e549f2850a04382cba3305f41e0bff67e3667ae1a7004757247db`; loop v12 `diagnostics/model-profile/loops/L17-D-native-eager-live-inputs-v1.json` SHA-256 `64e69bc3bceadbae1f36b44412b3c6e32b44df81d1afa34889344177f6e43481`; `RECOVERY_LATEST.md` SHA-256 `90d267c5e48e6aecba2e323e042d44dc3d701fbf6b819472bd7583308fbd872d`; `PERFORMANCE_CHECKPOINT.md` SHA-256 `98cf22a9e02cd0682241d2e7d3d598a67707ba6d4375d162d6b7e294a87bbd0e`; `PLAN.md` SHA-256 `26c5d7a52c5a606798d7102ee65d7c539b2d77245b8505d39a0caa11bbb562fc`.

以下为更新前 HANDOFF 全文（2026-10-19T03:10:56Z 版本，SHA-256 `e63851d8567c53c408308c56f1fc497f0a9ca81a8c39a832ed5b58ea03dfe288`），逐字保留为历史（原文未改动）：

---

# Current state - 2026-10-19T03:10:56Z (formal v2 detached coordinator started once)

- Fresh pre-formal four-node baseline is independently accepted: `results/C32-L17-native-pre-formal-acceptance-v2.json` (SHA-256 `0d674588f84a248c117d832e87a738c80964e71549641b1a9437bf7072b99cde`), decision `ACCEPTED_FRESH_PRE_FORMAL_BASELINE_ONLY`; capture receipt `62c85e2bbc2aa9fb4cb11b70ecfe952a7ef9417fc4a4eafdf36e91a5cd892909`, nodes 34/35/36/41 all rc0, model_requests 0, SDK init 0. This is baseline identity/scope acceptance only, not performance.
- The frozen v2 source set was verified before launch: freeze review `916550bca77b243239116d5a0874064b727dae78794ba109b317ab7d8b7b5e81`; wrapper `1fb0955cda214b56e4acad249c8837622c7d5328f14b7a7cee2a683564dfe7ef`; coordinator `b16bc944b7e9ba5a1aedf56095421f2839f7adb6c76cf534c0be24cb1b0e2d92`; runner `7c65c3f27ebc959090efb8c63912cbeb50afe6505ab6972bd74031cf8be330a4`; contract `c2f03fd0ae43ff3cc88b98138d4a408fca23845d55c0bd992ec07909d77bbe74`; controller `ffe433832a056d7368bb8d131a94a899dafbe9a67a2d27435708da3bea5c2290`; settings `2ccd33f9b2b4cc7ec963928181d27e55ca013c138efae5a3d6ef3eb3eea08b12`; actual review v2 `04cf0862b5665b1cfbbfbf49cb24e0f422c5a96251ff5e57e173ddee1e0d223d`; executor authorization `149307f4377e6808a2e8b4ea1fa00cb6db79bc382a336e4939e44af64edf3f82`.
- The exact frozen wrapper (`/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/bin/python3.9 .../scripts/launch_C32_L17_formal_v2_detached_once.py`) was executed once, rc 0, with no retry. It created `runs/20261019T-L17-formal-v2-detached-executor-v1` under the exclusive lock and started detached coordinator PID 57128 (PPID 1, new session, stdin DEVNULL, stdout/stderr files; wrapper PID 57127). The single post-launch `ps` at 2026-10-19T03:09:23Z returned rc0. This is STARTED, not completion; no terminal report or actual counts are claimed.
- Original formal v1 `20261019T-L17-sglang-formal-C32-native-v1` remains terminal **FAILED before formal dispatch**: 32/32 warmup logical requests, 64 actual backend attempts, 64 output tokens, formal 0. It is unchanged and was not replayed.
- New v2 actual counts are `PENDING_TERMINAL_REPORT`; planned 320 backend attempts are not booked as actual. The single post-launch controller-progress read found the new run file absent and is recorded as `PRECHECK_NOT_OBSERVED`; no polling or timer.
- Ledger is unchanged (`results/model-request-ledger-through-L17-interrupted-warmup-v1.json`, SHA-256 `4c2cd04961806c217b0c8be3b892ad61c88cfda816f44f4174ea8bc8bad2f32a`): 6409 attempts, 6172 qualified successes, 173 old unqualified attempts, pending native 64 with 64 output, qualified output 1355798. It excludes the new in-flight v2 cohort.
- Next action: the primary independently verifies that PID 57128 survives ZCode exit and awaits the same run's true terminal state; then `clientauditv2`, `postnativecap5host`, `independentmetrics`, `nativeauditv2`, `primarydecision`. No replay/retry/cleanup/polling/timer; no performance or SLA claim.
- Loop record updated to v11 (`diagnostics/model-profile/loops/L17-D-native-eager-live-inputs-v1.json`, SHA-256 `d35c44c13a7bd361961f821e7cf6280656afdc87182bc76368a40a612ae0dc99`); STATE SHA-256 `b57e29adb2d8ecc877ced0dad4b8665d5fe182dc7ebf7cf0b32a80f6abc2cdd6`; STATUS_BRIEF SHA-256 `b621cf18305e12631f03c3730377aef5975a000baf1882c61a841e1dcd1e7711`.

以下为更新前 HANDOFF 全文（2026-10-19T03:03:17Z 版本，SHA-256 `3dbe526042352e394ed827d00bb863481117ebd2456c9b35d947498913478334`），逐字保留为历史（原文未改动）：

---

# Current state - 2026-10-19T03:03:17Z (after one authorized pre-formal capture)

- SSH transport is restored on the fresh socket `/Users/wio/work/sglang-glm53-pd/ssh-control-l17-recovery-20261019`; the fresh baseline capture used it. The prior password/banner blocker is historical and no longer a current blocker.
- Original formal RID `20261019T-L17-sglang-formal-C32-native-v1` is terminal **FAILED before formal dispatch**. The cache-gate RPC disconnected (`EOFError`), the client waited the 300 s external exact-key/storage-readback gate and ended `TimeoutError`. Independent review: `results/C32-L17-original-formal-failure-independent-review-v1.json` (SHA-256 `e9bb41f13c3504bbc38eb8aa55bbdae38b2a407ce621c1745064bfb5ce760b93`). This is a failure qualification, not performance acceptance.
- Actual original counts only: 32/32 warmup logical requests passed, 64 actual backend attempts, 64 actual output tokens (P32+D32), formal requests 0. The original CPU client container and docker client PID are absent. `hostcontroller.client_rc` remains null in the original file and was not rewritten.
- One authorized pre-formal baseline capture completed: `runs/20261019T-L17-sglang-native-pre-formal-v2`, receipt SHA-256 `62c85e2bbc2aa9fb4cb11b70ecfe952a7ef9417fc4a4eafdf36e91a5cd892909`; nodes 34/35/36/41 all rc0; model_requests 0, SDK init 0, profiling false, service changes 0, CPU client collection false, max SSH 2. rc0 is snapshot completion only, not model/native qualification or performance.
- Ledger updated to `results/model-request-ledger-through-L17-interrupted-warmup-v1.json` (SHA-256 `4c2cd04961806c217b0c8be3b892ad61c88cfda816f44f4174ea8bc8bad2f32a`): 6409 backend attempts, 6172 qualified successful inferences, 173 old unqualified attempts, pending native 64 with 64 output; qualified output 1355798; planned320 not booked. The earlier functional 76/21640 qualification remains accepted history.
- Replacement `20261019T-L17-sglang-formal-C32-native-v2` remains **DRAFT, not started, not approved**; its actual review does not exist. planned320 is not actual.
- Next action: primary independent acceptance of the four-node snapshot, then freeze the v2 actual review, settings and independent detached-executor authorization; only then may ZCode start the replacement once. No retry/cleanup/remediation/launcher replay, no new performance or SLA claim.
- Loop record updated to v10 (`diagnostics/model-profile/loops/L17-D-native-eager-live-inputs-v1.json`, SHA-256 `b2648d7799da0e0beee3c3349dadfff774226dd47d74194321a3b06a2522db34`); STATE SHA-256 `c873bc092f87fcc0fe0b89ca22ea19613f757d11b59c46afb26f9823dcc92d0c`; STATUS_BRIEF SHA-256 `31d5211343b33d4bf2f1a825c67e73a25baa3dd00767d254e86814e1650e8011`.

以下为更新前 HANDOFF 全文（2026-10-19T02:18:36Z 版本，SHA-256 `b0ba4166a881e3e8ab7d769507ddf78bb05d6e70515b832809563aa0fb30591b`），逐字保留为历史（原文未改动）：

---

# Current authoritative recovery reconnect state — 2026-10-19T02:18:36Z

- The user has confirmed that SSH entry `61.241.77.34:60011` is valid and requested fresh password authentication. Do not ask the user to confirm again whether the entry is valid.
- Under that authorization the primary made two fresh independent SSH reconnection attempts (15-second default and 45-second with `-F /dev/null`), not a reuse of the old multiplexed master. Both exited 255 with `Connection timed out during banner exchange`. In the 45-second attempt TCP connected and the client version was sent, but no server version/banner was received; no password prompt appeared; authentication was not reached; no password was entered or persisted.
- This recovery round did include primary SSH reconnect attempts. Only the earlier ZCode subtask was zero-SSH; do not describe the whole recovery round as zero-SSH.
- Original formal RID `20261019T-L17-sglang-formal-C32-native-v1` remote state remains **UNKNOWN**. No remote failure, completion or timeout is inferred from the local SSH timeout.
- No model/NPU/SDK request, remote mutation, launcher replay, new load test, polling/timer or Git operation was performed in this round. The prior balance/HTTP402 error is historical only, and ZCode completed this local indexing work.
- Reviews verified: `results/C32-L17-formal-recovery-password-reconnect-review-v1.json` (SHA-256 `f319922702b624a95f19b7c4724eb09b09d0611160e85c864a8e85fec6f296e8`) and `results/C32-L17-formal-recovery-fresh-master-probe-review-v1.json` (SHA-256 `6af97e5cfaf6f784cf417eff7f0ea5f185d302cad0e96f0858c7bd4f75ec00b4`).
- Probe v4 `scripts/inspect_C32_L17_interrupted_formal_readonly_v4.py` (SHA-256 `c1c7aa5e1018272ed94415357338910cb61c1f9f829b0fc4de641c46fd08c63f`) replaces the pending v3 only; v3 must not be executed. v4 requires the independent freshly authenticated control socket `/Users/wio/work/sglang-glm53-pd/ssh-control-l17-recovery-20261019`; that socket does not exist and v4 has not been executed.
- Current blocker: the user-confirmed endpoint has no SSH banner/sshd response before the password prompt; SSH forwarding/sshd reachability must be restored, or a working alternate authorized entry provided.
- Next action: restore banner/sshd response on the confirmed entry, or provide an alternate authorized entry. Once the handshake reaches the password prompt, authenticate freshly and create the independent control socket; then ZCode executes v4 once as a read-only original-client probe. Do not replay the original launcher. No new performance or SLA conclusion.
- Ledger unchanged (`results/model-request-ledger-through-L17-functional-v1.json`, SHA-256 `2f62a6225ba37eeb60f4e028ca59da68afedb98de57429254d3c8ade902724f5`): 6345 backend attempts, 6172 qualified successful inferences, 173 old unqualified attempts, 1355798 qualified output tokens.
- Running container identities are last-known inherited from `STATE.json`; live container/model health was not verified this round.

以下为原 HANDOFF 全文，保留为历史（原文未改动）：

---

# Current authoritative recovery state — 2026-10-19T02:13:13Z

- Local-only recovery indexing completed. No SSH/NPU/SDK/model request, no remote mutation, no launcher replay and no Git operation was performed in this round.
- L17 native functional acceptance remains independently passed: 76 backend attempts / 21640 successful output tokens (short 64/64/19232 plus long 12/12/2408). This is unchanged.
- Original formal RID `20261019T-L17-sglang-formal-C32-native-v1` remote state is **UNKNOWN**. The last local 29/32 prefix warmups (29 passes) is historical local progress only, **not** an offered or actual total. planned320 is not booked.
- Recovery review: `results/C32-L17-formal-recovery-transport-review-v1.json`, SHA-256 `3d023aa1142af6c4408d81422864e00130d2abbc8ad7988d98f823d2df2483a9` (verified, along with all frozen hashes and sources).
- Current SSH entry `61.241.77.34:60011`: TCP connected, but no banner within 15 seconds; multiplexed mapping timed out at 30 seconds; authentication was not reached. The password was not tested or persisted.
- ZCode completed this local index successfully. The prior HTTP402 is retained as historical only; it is not the current blocker and no execution-mode choice is pending.
- Do not assert that the remote model is alive or that the original client failed, completed or timed out. Those are UNKNOWN.
- Current blocker: SSH forwarding entry unresponsive; original formal remote state UNKNOWN.
- Next action: wait for the user to confirm restoration of the SSH forwarding entry or provide a new entry. Then ZCode executes `scripts/inspect_C32_L17_interrupted_formal_readonly_v3.py` once (SHA-256 `6b575ef88bb0e0545a4176369e6445edc7772a01271b72200e4980464693fc1c`), after verifying the original CPU43 report and exact owned process. Do not replay the original launcher. No polling or timer. No new performance or SLA conclusion.
- Ledger (verified from `results/model-request-ledger-through-L17-functional-v1.json`, SHA-256 `2f62a6225ba37eeb60f4e028ca59da68afedb98de57429254d3c8ade902724f5`): 6345 backend attempts, 6172 qualified successful inferences, 173 old unqualified attempts, 1355798 qualified output tokens.
- Running container identities are last-known inherited from `STATE.json`; live container/model health was not verified this round. D36 `fe95e2fc...c89bb37` (service 117), D41 `c5e28348...8c82b9` (service 116), P34/P35 and Meta retained as last-known.

以下为原 HANDOFF 全文，保留为历史（原文未改动）：

---

# Current interruption — 2026-10-19 UTC00:14

ZCode exited with HTTP402 `Insufficient Balance`. The local formal launcher/controller is no longer present; no local `launcher.exit`, terminal RPC receipt or complete formal report exists. Last local progress was 29/32 successful prefix warmups. **Remote CPU43 client state is unknown; do not replay the original launcher.**

Incident: `results/C32-L17-formal-zcode-controller-interruption-v1.json` (`04a9031c…`). The 76 completed functional requests remain independently qualified; the through-functional ledger excludes the interrupted/ongoing formal run. Planned320 is not actual. Await the user's temporary-primary-executor authorization or restored ZCode balance, then inspect the exact original owned CPU43 run and preserve its artifacts before proceeding. All frozen settings/reviews/source and model-service identities remain unchanged by this local note.

Previous checkpoint (historical; superseded only by the interruption above):

# Current authoritative checkpoint

## L17 native functional: accepted; formal run is running (WARMUP)

The current runtime CIDs 34 `a6639da9...902179`, 35 `94dd0021...7c57f1`,
36 `fe95e2fc...c89bb37`, 41 `c5e28348...8c82b9` passed the full native
functional review:

- decision `results/C32-L17-native-functional-decision-v1.json`
  (SHA `eff7674514df1d0e29607aade0f440edc1c8f0d01b561b7d79ca7cb61a066d87`),
  `decision=ACCEPTED`, `functional_ready=true`, `formal_ready=true`.
- independent typed-SDK review
  `results/C32-L17-post-functional-independent-native-review-v1.json`
  (SHA `e8e2bfe3010881af89e6c3e2623c3f3b9d2bc367a11aaa38300e62c12477ba3c`).
- functional result
  `results/sglang-C32-L17-native-functional-v1.json`
  (SHA `ffeaf2b369dd77631216f85500d9fb9763613f7605d9cef542dda988e1bc0c02`);
  readiness `results/sglang-C32-L17-native-readiness-v1.json`
  (SHA `a233245e6daf23f2756932765c0e3f5c92d8361ce23c16235803fc361b14b9a4`).
- actual functional totals: **76 backend attempts / 76 successful / 21640
  output tokens** (short 64/64/19232 plus long 12/12/2408). Zero retries,
  zero new profiles, zero formal requests.

Native evidence accepted for this functional scope:

- 32 D native no-copy `bs2` workers, 32 productive D Graph ranks and 32 D
  typed-SDMA destination devices.
- 460 post-startup PD write batches and **107980359680** descriptor bytes.
- `physical_wire_bytes=null` and `native_logical_request_ID_join=null` are
  preserved as null; no physical wire-byte or per-request native join is
  invented.
- P this round has **16 positive new cache readers at +5 each, not 32 new
  readers**. All32 P native MTP/mapping first-use proof remains verified.
  The canonical formal phase must still qualify all32 actual fresh readers.
- all32 P native I/O + existing MC mapping reuse, all32 P packed MTP
  first-use, and all32 D Ops/native Graph are verified.

## L17 formal: running, performance pending

Formal RID `20261019T-L17-sglang-formal-C32-native-v1` is still running.
The controller state is `FORMAL_RUNNING`; the single preserved
`controller-progress.json` observation was phase `WARMUP` with concurrency
32, 9 completed real prefix warmups and 9 warmup passes
(SHA `3618b704033bb282ec83e8555c2a3917113d7646b4bfee595e03f4e540010c70`,
232 bytes; snapshot in
`runs/20261019T-L17-functional-qualified-checkpoint-local-v1/controller-progress-observed-once.json`).
The live controller file continues to advance and is not re-read, sampled
or used for counts. The 9/9 observation is progress, **not** an
offered-request total.

Frozen settings and review are indexed only, not modified:

- settings `config/C32-L17-native-runtime-v1/formal-C32-native-v1.json`
  (SHA `1c9b9e9b2ed33f13647eecfd7282e01de661a7b4a8324146b996a1c8121bee56`).
- actual review `results/C32-L17-native-formal-actual-review-v1.json`
  (SHA `043b8804aca24d51f8a28636f1eb61b1062e77d84948916a783292d8b7083553`).

Planned formal values (128 formal requests, 32 warmup logical requests,
320 backend attempts) are **excluded** from every actual total. No client
terminal report or per-request result from this run has been counted. No
performance or SLA claim is made; `SLA_pass` remains unevaluated for L17.

## Model-request ledger (appended, not overwritten)

Previous ledgers are unchanged:

- `results/model-request-ledger-through-L13-D-diagnostic-v1.json`
  SHA `0c466880bb9d0686762413f08b8682935247bef7e4ad2ea9144a2d70b7aba7e1`.
- `results/model-request-ledger-through-L17-short-client-pending-v1.json`
  SHA `db2e90551b5e06aad8ce95dcfac92d7212b59f997b4032c7a14acb7737d54043`.

New ledger:
`results/model-request-ledger-through-L17-functional-v1.json`
(SHA `2f62a6225ba37eeb60f4e028ca59da68afedb98de57429254d3c8ade902724f5`).

- backend attempts `6269 + 64 short + 12 long = 6345`.
- qualified successes `6096 + 64 resolved pending + 12 long = 6172`.
- qualified output `1334158 + 19232 + 2408 = 1355798`.
- old unqualified remains **173**.
- pending native is now **0 attempts / 0 tokens**; the 64 short attempts
  and 19232 tokens are counted once as qualified.
- short 512/600 whole group (64 attempts, 19232 tokens) is functional.
- long 4 × 80000/600 contributes 4 requests / 2400 D tokens to
  `all_formal_shaped` (**not** primary formal); the remaining long P,
  control and prefix-warmup output is **8 tokens**, counted in
  `warm_and_P_control`.
- `all_formal_shaped` becomes 2212 requests / 1298400 D tokens;
  `functional_output_tokens` 54174; `warm_and_P_control_output_tokens` 3224.
- partition check: `1298400 + 54174 + 3224 = 1355798`.
- 64 short + 12 long unique RIDs = 76, no duplicate
  (all-76 RID set SHA `c6664458c1da6508d098c4a20922711345c7b95f52407344fa5a1de6ca297a47`).
- long run: 12 model backend calls plus **2 metadata-only SDK queries**
  (`init_bm=false`, `NPU_devices=false`); the metadata queries are not
  model requests. 0 retries, 0 new profiles.

## Preserved startup observation and scope localization

The original bounded startup observation is preserved unchanged: both
`observe-36` and `observe-41` ended `rc=1` at the 1800s bound with
`native_audit_errors` 7/6, no `GLM53_L17_D_OBSERVE=` summary and no
`startup_pass`. Models were left running; no cleanup, rollback, replay or
repair was issued.

Independent startup acceptance remains separate and does not rewrite that
observation: `results/C32-L17-native-startup-acceptance-decision-v1.json`
(SHA `4f0f032c7b5ec54e35f94c990399188edb31d60fb101a98e727a6a5bd5fa938d`),
decision `ACCEPTED_STARTUP_NATIVE_PREREQUISITE_ONLY`, originally
`functional_ready=false` / `formal_ready=false`; the later functional
acceptance supersedes those scope-limited flags.

The former 13-row issue is localized, not unknown:
`results/C32-L17-native-observer-scope-error-localization-v1.json`
(SHA `ec7a99f0199931bc37ce382b506f8a636d262633843ffe5aa45d73f5e43a79f1`)
shows exactly 13 non-SDK hook import-installation metadata rows.
First-use independent review:
`results/C32-L17-native-first-use-startup-independent-review-v1.json`
(SHA `6e3b5de908a6b8fcdcbb7bee4ea796474547ed8b9b6414f9cea8f221f7e85ae5`).

`/proc` initial environment and the actual getter are different scopes:
the raw initial bytes show all six expected flags absent/false
(0 true / 96 false per node), missing keys stay missing and are not
filled. The live first-use rows prove `native_no_copy_get=true` and
`clone_distinct=true` on all32 composite `(node, namespace_pid)` workers.
The empty initial flags do not contradict the actual getter true values.

## Historical L12D measurement

The accepted L12D measurement (TPS `389.8740711045039`,
`SLA_pass=false`) remains historical and unchanged. The current D is the
L17-D native-eager-v2 candidate, so the current runtime must **not** be
claimed to match or equal the L12D accepted baseline.

## Next action

Wait for the already-running formal RID to produce its actual terminal
report. Then run client, native and independent metrics audits and a
primary decision. Do not count planned 320 backend attempts or observed
warmup progress as offered totals. No performance or SLA acceptance is
allowed before those audits. No retry, replay, repair, cleanup or rollback.

## Preserved original D-start checkpoint

- executor run `runs/20261019T-L17-D-models-and-observe-v1/receipt.json`
  (SHA `8323f780c12b7e56eb84eacbc21d7a449bcd11f8307791c159e9d1d28934f04b`),
  `phase=FAILED_OR_PARTIAL_STATE_PRESERVED`, `pass=false`,
  `error=AssertionError()`, `cleanup_or_rollback_executed=false`.
- launcher `scripts/start_C32_L17_selected_D_models_and_observe_v1.py`
  (SHA `4af8f3787f06c8497e849f43fb795a8ea8fd58444028ce2711a8c82d079224cb`);
  log `runs/20261019T-L17-D-model-start-executor-v1/launcher.log`
  (SHA `78e19b44728a6554d7fb1683aa28cb72c8916eddafcfe103c67b213263c9c2eb`),
  exit `1`.
- authorization `results/C32-L17-D-model-start-actual-review-v1.json`
  (SHA `877fdce565b9b4b01dc909917b3c7d013d3497236b65804767197c1aa1888178`),
  `APPROVED_FOR_ACTUAL_D_MODEL_START_AND_OBSERVE_ONLY`.
- D start table: node 36 / `fe95e2fc...9bb37` / service 117 / provider PID 8
  / model PID 107; node 41 / `c5e28348...8c82b9` / service 116 / provider PID 7
  / model PID 106. Both `MODEL_RUNNING`, one start signal each, ports
  29440/30600 free at start.
- retained progress: node36 health 200, idle, registered 16, Graph ranks 16,
  native Ops workers 16, `native_audit_errors=7`, `warm_all16dp=true`, eager
  `COMPLETE_ALL16`, 16 bs2 draft-decode PIDs, `done=false`; node41 the same
  except `native_audit_errors=6` and `warm_all16dp=null` (node36-only check).
- preserved historical L12D accepted result: TPS `389.8740711045039`,
  full-clock `196.98668286` s, TTFT_P50 `17.757243875414133` s, TPOT_P50
  `36.65860450806373` ms, `SLA_pass=false`, largest gap `D decode`;
  vs L10 TPS +`26.20757259854496`%, TTFT -`40.51670798525417`%, TPOT
  +`0.2936857162019413`%; vs vLLM TPS -`26.551457885694774`%, TTFT
  -`16.089087065792963`%, TPOT +`91.40219972003096`%. Historical only; the
  current D is different and no match is claimed.

See `STATE.json` and the L17 loop record (v7,
`diagnostics/model-profile/loops/L17-D-native-eager-live-inputs-v1.json`,
SHA `0495da50fce632403e6ac32615dfe04a6001a7410edbe7b885f52a5ca1b9fece`).
