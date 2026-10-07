# Performance Research Reset — H9 sparse target replay ordering

2026-10-07, following Run265 and the independent [source review](stream_ordered_replay/STREAM_ORDER_REVIEW.md). Unique hypothesis: remove the unnecessary additional target pre-replay host wait on the reviewed sparse SFA path. No kernel/arithmetic, parallel-layout or MTP algorithm change. Active verified research stack remains H6/H5; final Current=None and complete API/SLA/stability acceptance remain open. 本阶段没有新增代码级性能 KEEP。

## Product and evidence

GLM-5.3 W8A8 standard P→KV→D, uploaded artifact `/data/tiankuan/wio/GLM-5.3-w8a8`, model glm-53, two16-rank roles, protected P249. Formal workload is80000 input/600 output/93% shared prefix and separately93% declared KV hit-rate condition, with actual hits reported separately. Run266 is a bounded code discriminator, not that formal workload.

H6 Run256 and H5-on-H6 Run261 repeated complete PD improvements are retained, not added arithmetically. Run262 H8 remains off/inconclusive. Run263 numerical inference was confounded by failed KV. Run264 established internal fail-closed but dropped empty API error. Run265 completed with canonical500/DONE, no failed-request model execution, all16 clean transfers, exact8-token short and23-token natural EOS complete PD, dynamic targetFULL replay and eager MTP. Pure logical-KV and API terminal repairs are retained in both arms. Run265 instrumented latency is not an H9 gain or a matched graph configuration gain.

Current resident D265 root3490723/start309425199/boot65c53cfa; P249 root1916718/start303049910/boot6d9cf06f. Fresh controller death/locks/source/ownership/health/idle checks are required before any device action. Local branch glm5-3-autonomous-20261001 HEAD ea554328, private server base bf07ebe2; this candidate is identified by frozen patches and source hashes, not a false installed commit claim.

## Source cause and minimal change

`BreakableACLGraphWrapper._replay` host-synchronizes before normal FULL replay. For this resident SFA path, actual SFA graph-param update is a no-op and ModelRunner excludes sparse/compressed host graph-param updates. Actual H6 native `NPUGraph::replay` enqueues `AclmdlRIExecuteAsync` on the current stream, preserving RNG and blocking-mode behavior. Base replay retains `get_offloader().sync_prev_onload()`.

Parent personally read the inherited `GPUModelRunner.synchronize_input_prep`: prior event synchronization protects reused CPU staging, and event record follows preparation. That protocol, all input-copy/rotary/MC2 mask ordering, HCCL captured dependencies, output readiness/copy fences and graph workspace lifetimes remain unchanged. Same-stream device ordering does not replace CPU staging lifetime protection; both are preserved.

[Two-file production diff](stream_ordered_replay/stream_ordered_replay.patch): runner recomputes a per-forward opt-in for FULL/glm_moe_dsa/ASCEND_SFA/use_sparse/noncompressed/noDeviceMetadataExecutor; breakable wrapper skips only the non-draft target's additional host wait. Missing/false guard, dense/compressed/device-metadata, NONE/PIECEWISE and draft retain prior behavior; capture unchanged. No arbitrary concurrency/offload acceptance claim. Code guard is semantic, not request/token/bucket hardcoding.

[CPU1152](stream_ordered_replay/CPU_result.json) actual runner/base/wrapper AST validates1150 parity branches,2 intended skip branches,7 dynamic condition revocations and draft role revocation. CPU queue/native operations are doubles. Run266's test-only mmap selector has [4608 actual shim AST cases](../../runs/GLM-RUN-0266/CPU_selector_result.json) matching original mode0 and production candidate mode1. Only mode transitions emit a tiny all-rank CPU witness; same-mode measured requests write none. Device correctness and exposed gain remain unknown.

## Reuse and priority

Reuse Run265 exact clean replay/KV/source evidence, Run261 fixed PD client, cumulative scheduler MTP counters, ownership/controller and matched decision. Historical graph/host studies (e.g. PK123) warn that FULL envelopes may include host sync and cannot transfer an old eager budget to current FULL. No previous verified GLM removal of this boundary was found in the relevant ledger; H9 is a new intervention. No new profiler or parameter scan.

Review finds no source-level blocker on this resident path, while ranking remaining FULL cost unknown: eager MTP, input/metadata, target replay, output and scheduler boundaries. This wait is the clearest presently removable code boundary, not a proven global maximum. Its cost may be absorbed by downstream dependencies. A fixed matched intervention resolves this unknown directly.

## Frozen discriminator

Hypothesis: the additional pre-replay host wait delays useful submission and removing it yields repeatable complete-PD savings with identical effective work.

Distinguishing evidence: one owned D reload, fixed FULL_DECODE_ONLY/CompilationNONE/bucket2/eagerK1MTP/H6H5 throughout both arms. Remove Run265 graph/transfer observers and controlled fault injection; pure KV/API fixes retained. Same workers/common library, test-only1-byte policy switches0/1. No source/config reload between arms. Existing native INFO transfer logs establish all16 complete-request transfers, golden IDs/content/EOS and external KV counters, cumulative MTP signatures, actual eligible target FULL replay witnesses. No costly per-step observation.

Correctness: mode0 and1 short2334/8 and complete58/natural23, then fixed A1/B1/A2/B2: one excluded short warmup, two measured shorts, one measured complete EOS each.20 normal requests plus at most one terminal policy warmup (21 budget), one recovery short only if needed. All transfer/source checks occur after requests, outside measured intervals.

Decision table:

- Any wrong IDs/EOS/KV/state, ineligible guard, missing rank or unmatched effective work → stop candidate, H9off, preserve H6/H5 and pure correctness repairs; record failure/inconclusive/NOT_APPLICABLE precisely.
- Two complete D-wall savings each exceed within-mode D drift, both TPOT and PD wall improve, short median improves in both pairs, complete/short cumulative MTP work matched → retain H9 as limited-workload research candidate; no formal API/SLA/Current promotion. Continue remaining path on H6/H5/H9.
- Correct and matched but gain absent or not repeated beyond drift → H9off/inconclusive or reject; keep H6/H5, do not repeat same intervention without new code/evidence; continue the next source question.

Recovery: only owned D tree, at most one quiet fixed validated FULL baseline reload, original runner/wrapper and pure KV/API repairs, H6/H5 on, one golden short. No candidate retry and no unrelated service operation. If recovery fails, preserve facts and do not start an untracked owner. P249 never retired. Public GitHub workload-only push is separately awaiting public-scope authorization and does not gate research.
