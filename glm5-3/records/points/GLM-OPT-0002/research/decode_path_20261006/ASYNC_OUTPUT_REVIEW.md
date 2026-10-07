# Independent Challenger: actual asynchronous output path

2026-10-07. Offline source/raw review; no patch, service or device operation. Trigger: H8/H9 no repeatable new gain and H10 actual-path mismatch. H6/H5 remain the verified research stack. This is a route recommendation, not PERF_KEEP.

## Verdict and ranking

**PIVOT from synchronous early-MTP ordering to a bounded asynchronous completion-boundary investigation.** The largest removable cost on current FULL-target/eager-MTP/async execution is **unknown**. Source supports an unnecessarily broad dependency for *copying sampled tokens*, but does not yet support removing that dependency from *publishing the whole result*.

Priority, not measured savings ranking:

1. `NPUModelRunner.sample_tokens` → `AsyncGPUModelRunnerOutput`: distinguish sampled-data readiness from iteration completion. This is the highest-value next code question because the constructor's copy-stream wait includes already-submitted MTP work although sampled IDs already exist. It is not evidence that all MTP time can disappear.
2. Remaining eager MTP submission / next-input metadata preparation: necessary model work mixed with potentially removable host execution. Current FULL-path per-step budget and exposed fraction are unknown; old Run249 numbers cannot rank this against item 1.
3. Async batch queue/output consumption and deferred grammar: potential publication/scheduling lag, unknown magnitude. Do not replace with a presumed synchronous three-RPC diagram.

KEEP H6/H5. STOP H10 as a current-path candidate; it was NOT_APPLICABLE, not a failed performance comparison. Keep H8/H9 off absent genuinely new evidence. Incomplete API/SLA/Current does not justify terminating research.

## Facts personally checked

- `runs/GLM-RUN-0267/retained_witness.json`, H10_rows: all 16 ranks report async=true, early=false, eligible=false, mode0; natural complete result has 23 expected IDs/stop and 58 external hits. Graph target FULL, MTP eager. The separate graph-mode witnesses do not themselves contain async status.
- Run262 A1/B1 complete D wall 4.589576343/4.639556873 s; A2/B2 4.757268705/4.222879752 s. One favorable pair is not repeatable improvement. Run266 corresponding D wall .912435913/.911552784 and .901706371/.893635256 s: tiny differences do not establish the proposed host-sync benefit. These are different execution epochs, not a stack-versus-stock comparison.
- `early_padded_mtp/original_model_runner_v1.py:2589–2833`: sample; async bookkeeping; padded proposal; draft-copy decision; KV finalize; output construction; optional state/routed handling; only then construct AsyncGPUModelRunnerOutput. `_bookkeeping_sync:2867–3000` async branch adds placeholders, snapshots request maps/invalid indices, and does **not** execute synchronous sampled-token `_to_list`.
- `vllm__vllm__v1__worker__gpu_model_runner.py:259–355`: output constructor retains sampled/logprob/routed device references; copy stream waits on current stream, copies data and optional fault result, records event. `get_output` synchronizes that event before parse/release/return. It therefore currently includes the preceding MTP stream prefix, not merely sampler completion.
- Actual padded proposer `padded_mtp_order_sources/vllm-ascend__vllm_ascend__spec_decode__llm_base_proposer.py:1973–2033` clones sampled IDs before invalid-row index filling. Runner 1840–1955 builds separate next-token/count tensors and invokes `_propose`; this supports concurrent read-only copying of the original sampled tensor on this path, not universal safety for every proposer.
- Runner 1705–1728 already copies valid counts on a side stream before MTP, retaining its own event; `_correct_optimistic_seq_lens_cpu:1683–1703` waits before consuming those counts. Draft CPU copy at 1995–2036 is skipped for ordinary async requests but retained for structured/history/PP consumers.
- InputBatch 1030–1096 consumes the sampled-copy event to repair prior-step history, including request reorder, invalid/KV-failure placeholders and optimistic acceptance. The early event must not masquerade as count, draft or full-iteration completion.
- Worker executor 949–1019 queues outputs **after the worker method returns**, then its output thread calls get_output and converts exceptions to FAILURE. Core 624–746 schedules nonblocking work, consumes FIFO futures, processes aborts before scheduler update, and has a separate deferred-grammar draft-validation path. `post_step` skips ordinary draft retrieval under async. Actual max-concurrent/batch-queue occupancy in Run267 is not established by async=true alone.

## Safety boundary: early copy is not early success

A defensible narrow design starts an owned sampled/logprob transfer after sampler production (or after async bookkeeping for a smaller change) and before padded MTP. It attaches that transfer to the output object only at the existing return point. Keep strong references to source and CPU destination through completion; on a later proposal exception the pending transfer still needs lifetime ownership until its event completes. A normal-scope local variable that disappears during unwinding is insufficient proof. Preserve row snapshots, invalid-row parsing, history mapping and logprob semantics.

**Do not simply substitute this early event for today's output-ready event.** Today's late wait also provides iteration device completion before successful scheduler consumption. Scheduler `_free_request_blocks` (actual scheduler 2341–2352) may free KV blocks after finish; whether every subsequent remote transfer/reuse is independently fenced against still-running MTP is unproven. An early SUCCESS could also precede an asynchronous MTP/HCCL failure previously surfaced through the late dependency. This is a concrete missing ownership/error proof, not a claim an observed race already occurred.

First safe boundary: separate early sampled-copy readiness (usable by history consumer) from a late iteration/error gate for external publication. Preserve late routed-expert snapshot copies, optional EP-fault checks and connector finalize. A late gate may erase most apparent publication benefit; that is precisely why “MTP wait duration” is not removable budget. Current NPU constructor does not explicitly pass check_ep_fault, so do not invent an existing EP-fault check, nor delete the class contract for other consumers.

Keep CPU staging `synchronize_input_prep` (GPU runner 3865–3877), valid-count/DCP correction events, draft CPU event, current-stream target→MTP ordering, private routed snapshots and graph-buffer lifetimes. Same-stream ordering does not protect CPU writes to pinned staging or external KV transfers. Do not alter `num_tokens_no_spec` placeholder insertion or DCP metadata construction to obtain overlap. Unsupported proposer/output modes should retain the original path.

## Top 3 questions and smallest distinguishing evidence

1. Does post-MTP sampled-copy readiness actually delay a needed consumer after worker return, or is next-step device/host work already dominant? Record per-iteration sampler-ready, MTP completion, current copy-ready, worker-return/output-thread enqueue/dequeue and scheduler-future-consumption boundaries, plus actual queue occupancy. Use a single bounded diagnostic only if authorized, no broad profile. Do not force synchronization at every marker; defer event observation and distinguish device event intervals from host wall timestamps.
2. Can result success precede MTP completion without KV reuse, asynchronous error or output-order changes? Until block ownership/error consumers close this proof, keep a late completion gate; CPU-only delayed-copy/exception/abort/reorder ownership tests can falsify unsafe designs without a model Run.
3. Once an early copy exists, does CPU history readiness or D2H overlap matter under actual greedy versus grammar/history paths? Preserve each consumer's distinct readiness. No independent third candidate is activated by this question.

Decision: if copy readiness is already hidden at its first consumer, STOP early-copy performance route and prioritize measured MTP/next-input submission. If exposed but late completion is required, test only the smaller copy-overlap mechanism. Only if an explicit ownership and error contract permits early publication should a larger scheduling change be considered. No source-only result establishes gain or maximum ranking.

## Stop list / likely misread

No sync H10 retry on async workers; no H8/H9 repetition without new mechanism; no removal of staging/count/draft/HCCL fences; no early SUCCESS that bypasses MTP/KV failure; no eager-profileON budget transferred to FULL runtime; no summed overlapping scopes, SSE chunks treated as engine steps, or independent gains added. The most likely mistake is equating earlier token-copy completion with safe earlier iteration completion, then crediting the entire overlapped MTP interval as an E2E saving. The second is reading the method name `_bookkeeping_sync` as proof this branch synchronizes.
