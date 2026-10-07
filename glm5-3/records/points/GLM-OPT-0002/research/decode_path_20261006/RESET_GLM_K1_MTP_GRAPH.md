# Performance Research Reset — scoped GLM K1 MTP Graph

2026-10-07, after Goal Review172 and user MTP Graph steering. Sole active performance hypothesis is draft eager host submission. Async early publication is parked; H9/H10 off. Keep verified H6/H5 research stack, target FULL_DECODE_ONLY bucket2/CompilationNONE and pure logical KV/API repairs. Current=None, full API/formal SLA/stability/complete-stack-vs-stock E2E open. 本阶段没有新增代码级性能 KEEP。

## Product and evidence

GLM-5.3 W8A8 standard P→KV→D, /data/tiankuan/wio/GLM-5.3-w8a8, glm-53. Formal workload80000 input/600 output/93% common prefix, separately declared93% KV condition. AISBench round salt/per-endpoint counters have been fixed and deployed166; GitHub research51789da8/main74910eab verified. No formal benchmark or measured93% claim. Existing historical raw/specs unchanged.

P249 remains protected, root1916718/start303049910/boot6d9cf06f. D267 root1760510/start309909425/boot65c53cfa, all16 H6/H5; previous controller GLM-RUN-0267-CLOSURE completed, fresh death/locks/source/worker/health/idle checks required before action. Current local/private commit and all frozen source hashes recorded separately from server166 trackedbf07ebe2; protected dirty files never reset.

Actual D proposer edcc55d...212–232 unconditionally disables GLM draft graph. Target FULL is already active; speculative enforce_eager=false alone is ineffective. Source-backed selected opportunity: repeated draft model/logits/sample Python submission is avoidable using the existing independent merged ACLGraphWrapper. Current exposed draft host time is unknown; Run249 profileON eager265ms/token cannot be used as the current removable budget. No kernel, model computation, MTP algorithm or communication deletion claimed.

Root personally read actual upstream DeepSeekMTP83a52a68..., proposer, runner, DCP first-pass, SFA/DCP metadata/forward/composer, rotary and wrapper. K1 uses one tensor-only predictor layer and tuple logits/recycle; first pass preserves two-row/one-index shapes; capture and real first-step metadata use persistent rotary/seq/query/slot/replicated/local buffers. Model-level blocker not found. All HCCL, target synchronize, offloader capture/join, CPU input/output, KV and error boundaries preserved. Graph/draft registries are already distinct; no invented registry collision or withdrawn SP-enabled shape objection.

## Code and independent review

Only proposer changes. Explicit opt-in GLM/K1/greedy draft/noncompress/SPoff/LMHeadTPoff/TPDCP16/PCPDP PP1/static EPLB capability; each call additionally admits uniform2B/B indices/greedy/no LoRA/history/generators/grammar/decode-only. All unsupported requests preserve native eager. Keyword input/device metadata/cache/indexer/topk signatures close generic wrapper's empty-positional-args gap; changed pointer/shape/branch falls back before graph submission. Actual SFA composed main plus independent indexer key/scale caches are included after Challenger found that omission. Captured output is cloned on main stream for owned lifetime; no early success publication.

[Review](MTP_GRAPH_PATCH_REVIEW.md) accepted after the concrete indexer correction; source review is not device proof. CPU actual candidate AST:1024 runtime cases,15 static exclusions,19 independent pointer-change fallbacks, dynamic values/DecodeOnly-vs-SpecDecoding signature agreement and output-after-next-replay ownership pass. Tensor/ACL/stream behavior is doubled. Capture/HCCL/numerical/multi-rank correctness and actual benefit remain unknown. No patch installed at Reset creation.

## Smallest decisive diagnostic — Run268

Hypothesis: scoped greedy K1 draft can capture and replay with refreshed dynamic metadata and exact standard PD outputs on H6/H5. This prerequisite distinguishes valid graph support from flag-only/unsafe guard deletion; it is not a performance comparison.

One owned D reload: unchanged target FULL bucket2, H6/H5, pure KV/API and native library; only proposer support and speculative enforce_eager=false plus explicit opt-in. Same-worker selector0/1 chooses native eager or scoped draft graph. Startup captures the independent draft entry; thin observer verifies all16 actual captures/replays, backend/classes, persistent kwargs/cache/indexer pointers, first two real position/seq/slot/index values and owned outputs. Its deliberate CPU observation overhead precludes performance conclusions. No profiler or parameter scan.

Normal request budget4: existing golden2334/8 selector0, same golden selector1, existing58/natural23 EOS selector1, selector0 terminal warm. All request salts are unique and passed identically to P/D. Exact IDs/content/usage/EOS, externalKV counters and native all16 successful transfer required. Acceptance/rejection counts recorded; observe rejected input only if naturally present, no outcome-driven repeat. These fixtures do not claim cross-block/reordering/all API/formal80K600 coverage.

Decision table:

- All16 real graph replays with capture-stable consumed pointers, changed positions/seq/slots, owned output, exact gold/complete EOS/KV success, health/idle → limited graph-support correctness; keep final selector0 pending a separately frozen matched A/B/A/B complete-PD test on these same workers. No research performance promotion from this diagnostic.
- Capture/communication/numerical/KV/error/owner failure or missing all16 graph path → preserve first failure; selector0 and, if needed, one owned recovery to H6/H5/target FULL/native eager MTP. No parameter retry or silent continuation after failed capture.
- Contract mismatch before graph submission → retain precise reason/pointer/branch as source evidence; continue the same scoped implementation from that evidence rather than pretend graphs ran.

Recovery budget one owned D reload and one golden short, pure repairs/H6/H5 retained. P never retired. Existing controller error remains original FAILED. Fresh lock/source/death/ownership checks and immutable source/spec pins precede actual execution; cross-host epoch reads explicitly use167 RPC. No broad NPU Run, formal AISBench test or performance gain claim is authorized by a successful diagnostic alone.
