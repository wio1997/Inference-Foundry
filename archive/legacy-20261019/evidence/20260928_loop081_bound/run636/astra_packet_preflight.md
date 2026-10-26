# Run636 Astra High physical-boundary packet source preflight

**Verdict: source-anchor plan accepted; no live observer implementation is approved yet.** The next packet must distinguish actual value/stream readiness from Host submission. Keep the DSpark7 algorithm, acceptance trajectory and output/model work fixed within any timing comparison. No service/NPU workload, installed-source edit or instrumentation was executed for this review; only this report was written.

## Whole-Product priority

**If only one new physical packet can be acquired, default to an adjacent Runtime-cycle producer→consumer packet**, provided the existing Run610/613 ownership can support a bounded observer that passes preflight. Measure actual Target outputs, DSpark result, state/metadata completion and their next consumers, including inherited side-stream waits. A selected KV row is insufficient as the primary packet.

Run99's roughly69s Runtime and Run606's8.334–8.984s preparation NONE Host cumulative time are different W0s and observers and cannot be used as a quantitative gap comparison. There is also same-W0 evidence: Run606's four built→allrank existing-sync Host envelopes sum to approximately69.2803s, while its own Target NONE Host calls total8.334–8.984s/rank. These remain broad current Host scopes, not additive necessary work or recoverable budgets. They support keeping Runtime as a major uncertainty, not a precise benefit ranking.

Preparation NONE becomes the first live packet if Runtime producer/consumer closure cannot currently be authenticated with bounded observer cost, or if a same-W0 sensitivity/coverage audit shows one preparation boundary can resolve a larger uncertainty per acquisition. It has simpler eager source hooks and24 observed calls (four prefill-only, twenty mixed), making it a reasonable low-cost first physical timing stratum. That conditional information-value choice must be recorded explicitly; it must not follow merely from NONE dominating preparation Host time. The rest of this note supplies its minimum implementation anchors and the Runtime alternative.

## Exact ordinary Target NONE anchors

Paths prefixed A are under `/data/wio/vllm_ascend_26/framework/vllm-ascend/vllm_ascend`; V under `/data/wio/vllm_ascend_26/framework/vllm/vllm`. Line numbers refer to the SHA-pinned sources below.

| Boundary | Exact source anchor | Admissible observation and missing closure |
|---|---|---|
| Ordinary call identity | A/worker/model_runner_v1.py `execute_model`1794; `_prepare_inputs` call1935; batch mode selection1947ff; actual forward context3649ff and `_model_forward` call3673 | Assign a fresh ordinary ordinal tied to request-ID multiset, scheduled/cached suffix state, actual mode, shape and metadata. Do not use the next cohort filename as ownership. |
| Input production | `_prepare_inputs`885ff, including input-ID assembly1365–1366 and metadata GPU copies1041/1163–1168; final call arguments at3673–3675 | A marker after all known input producers on their stream is a queued producer marker, not Host-certified input readiness. Audit later input/position/metadata preparation between1935 and3673. Input-ready requires actual first device consumer plus its incoming stream dependencies. |
| First Target consumer | A/models/deepseek_v4.py model forward1104ff; embedding1116–1117 | Trace the actual embedding/input use for the selected NONE call; preserve mode checks so the observer cannot enter a captured FULL body. Position/KV/metadata consumers can have different first use and readiness. |
| Local auxiliary values | A/models/deepseek_v4.py1136–1140, after selected layer output and `hidden_states.mean(dim=1)` | Record one generation per selected aux tensor, after its true producer queue point. These values arise before full-model return. The next local consumer may be TP all-gather; distinguish local aux from globally gathered aux. |
| Main final output / additional gather | Same file1153–1160 residual all-gather/buffer copy,1170–1174 HC head/norm/return; A/worker/model_runner_v1.py4568–4579 model run and optional hidden/aux gather | The MTP stash is current implementation work, not automatically a required DSpark7 value. Actual branch and consumer must be observed. Host model return is not device completion. |
| Hidden/aux all-gather | A/worker/model_runner_v1.py4485–4519 `_all_gather_hidden_states`, list and tuple helpers; gather4501 only if shape test passes | Bind input/output tensor role, pointer/allocation generation, shape and actual branch. Record per-output local/gathered readiness; do not replace it with one barrier for all outputs. |
| Logits producer | A/worker/model_runner_v1.py3979–3980 index/select plus `compute_logits`; A/models/deepseek_v4.py1281–1286 logits processor | Record submission and downstream producer marker for the actual returned logits, including its backend collectives. An output Python tensor object is not a completion certificate. |
| Logits first consumer | A/worker/model_runner_v1.py4080–4081 `_sample`; actual sampler branch4310–4340 | Hook the selected sampler/rejection-sampler operation and its actual stream. `_sample` begins with asynchronous output-token bookkeeping, so its Host entry alone is not the first logits read. |
| Gathered aux first ordinary consumer | A/worker/model_runner_v1.py1663–1668 or1685–1689: actual slice/index plus `torch.cat`; `_propose`1691ff | Which branch executes depends on spec metadata. Link each aux generation to actual indexing/cat, then the resulting target_hidden_states to Draft; declaration of `_propose` arguments alone is not a physical read. |
| Ordinary proposal and output-copy boundary | A/worker/model_runner_v1.py4092–4107 helper, actual callers4154/4158; draft CPU-copy stream1785–1791 | Preserve the existing wait-stream and event-record behavior. Copy readiness differs from Draft output value readiness and client publication. |

The collection pattern from Run606 can carry this identity through a scoped ContextVar. Its `execute_mark_count` is len(all marks); Run633 proved that the current packet's call prefix happens to be all execute markers. Prefer an explicit monotonically increasing execute ID for a new implementation and validate it against the legacy count/Request-ID join. Do not silently reuse an ordinal after a flush or bind a previous-cohort execute to the next cohort.

## HCCL and side-stream closure

The source entry points are V/distributed/parallel_state.py `GroupCoordinator.all_gather`670ff/`_all_gather_out_place`686ff and reduce-scatter701ff; V/distributed/device_communicators/base_device_communicator.py all_gather199–222 invokes `dist.all_gather_into_tensor`214, and reduce_scatter232–263 invokes `dist.reduce_scatter_tensor`259ff. A/distributed/device_communicators/npu_communicator.py all_to_all58–87 invokes `dist.all_to_all`85. These are candidate submission anchors, not proof that every call in the selected path uses them or that Python exposes the real HCCL device stream.

Bind the observed callable/backend and same-call native identity, group/rank, operation, tensor extent/dtype and producer/consumer generation. The ProcessGroup/HCCL implementation may launch on an internal stream. Python current_stream alone cannot identify its completion. Prefer existing native task/flow, event generation and stream wait evidence with the installed profiler method; if a real completion or consumer dependency is not exported, mark that edge UNKNOWN. Do not invent an HCCL stream ID from another run, infer compulsory bytes from algorithm bandwidth, or treat collective duration as pure network service.

A/attention/dsa_v1.py contains optional current multistream waits:1902/1906,1912/1929,1935/1963; prefill-related compressed-KV waits2189/2201; decode-related2487/2499; and additional2929/2948/2963/2973/2982 paths. These locations are **conditional candidates**, not a claim that all execute in the selected NONE call. Capture branch, existing stream/event identity and event generation only for observed paths. The tail current-stream wait can establish dependency inclusion when it is actually executed; presence in source is insufficient. A physical layer/Graph completion marker must cover every side stream whose outputs the claimed consumer uses.

## Runtime alternative: preferred when one packet is affordable

Repository source anchors:

- `runtime/target_adapter.py:64–77`: Target binding forward, separate hidden/aux outputs and logits computation. Boundary records here avoid inserting Python callbacks into the captured Target Graph. Native Graph ownership must still connect internal output production to the returned values; the outer end marker cannot prove all outputs become ready simultaneously.
- `runtime/extreme_decode.py:430–440`: Target→acceptance;449–450 state advance;454–460 proposer;471–478 next-draft copy/commit; next step372ff and metadata commit395ff. Observe two adjacent cycles plus predecessor/successor boundary lineage. Do not assert closure at Draft return without tracing draft commit and next Target's real read.
- `bootstrap/vllm_dspark_handoff.py:186–207`: existing count-copy side stream waits on current, records event, and the following commit uses the **existing** event.synchronize. Tag the event generation with its producing cycle; reusing one event object does not mean the same generation. Record Host wait begin/end and upstream queue identity without adding a wait.
- Same file312ff `_execute`, aux pack358–365 and `_propose`375–392: link Target aux to the actual cat/index consumer, Draft result to commit and next Target use. Keep context/query/KV dependency edge UNKNOWN where no row/allocation lineage is available; the broad readiness packet can still be useful without pretending that a missing row edge has been disproved.

The explicit profiling-sync-target branches at `runtime/extreme_decode.py:431–435` must remain disabled. Existing Run610 profiler.stop at cycle66 fenced execution; do not place a stop fence inside the timing window and transfer that duration to observer-off Current.

## Product publication and clock packet

Reuse the admitted Product chain rather than create a new publication definition: A/patch/platform/patch_kv_delivery_preemption.py `update_from_output`833ff and actual `request.append_output_token_ids`1195; V/entrypoints/openai/chat_completion/serving.py yielded content753, final usage799 and DONE834; client request start, received SSE payload hashes, DONE and request-end as separate timestamps. Retain generated/prebulk accepted/Runtime-bulk/API-counted/yield-associated/client-received ledgers and placeholder counts. A generator yield is not socket send or receipt.

Host markers use one recorded monotonic clock with client/server namespace/boot identity. Keep outer clock, measured wall and request/DONE endpoints distinct as in Run631. Device intervals remain per rank unless an applicable common device-clock certificate exists. Do not repeat the Run610 RAW-versus-MONOTONIC mistake or silently adopt its1ms mapping assumption as a universal bound. Queued events establish ordering on an authenticated stream; reading their elapsed time after the existing terminal synchronization avoids new timing-window waits but does not by itself map them to Host time.

## Observer controls and admission gates

Preallocate a bounded record/event pool before the measured window; arm only a declared ordinary NONE stratum or adjacent Runtime slice, with a complete low-volume whole-W0 identity ledger. Store compact numeric metadata in memory; avoid JSON serialization, file I/O, tensor clones, payload transfers, device `.item/.cpu/.tolist`, new streams or allocator growth in the hot path. Inspect tensor metadata without reading tensor values where possible. Flush at an already existing drained boundary and record that the flush perturbs later Host publication if it occurs before client completion. Prefer post-measurement export for timing transfer. Preserve exact source/script restore and all8 stop/idle verification.

Do **not** add device/current-stream/event synchronize, Work.wait, wait_stream/wait_event, HCCL barrier, or a model rerun to make a proof easier. Existing synchronization may be observed at its original location. New cross-stream waits alter the schedule being measured. An event recorded after submission must be called a marker until actual native producer/consumer and stream inclusion are established.

Before live: compile/import and CPU fault checks; exact source anchors/hashes; branch ownership checks; bounded pool overflow rejection; no-event-reuse generation ambiguity; ordinary FULL graph mode/dispatch unchanged; no unintended graph recapture or graph-pool allocation change; all8 identity/correctness/output/Basis admissions; and explicit cleanup plan. This note is not a substitute for review of the actual patch.

For OFF/ON/OFF, retain the frozen warm48→measured48 c12 protocol and source/config identities. No trajectory is forced to match by changing sampler, Draft/Target, acceptance or cycle count. Compare exact admitted W0/Basis/output where available; differing natural trajectories are separate observations and cannot be subtracted as observer cost or claimed equal-work timing. Require the ON effect to be bounded relative to matched observer-off variation and the size of the uncertainty the packet is intended to resolve. If there is no same-work control, admit topology/ownership only and keep timing transfer UNKNOWN. Do not select the fastest OFF/ON repeat or erase failed acquisition status.

Reject timing/Bound promotion on any source drift, missing rank/request/ordinal, wrong mode/shape, count/output/KV/state divergence, unmatched event generation, unknown producer stream, unsupported clock conversion, truncated pool/trace, or observed new synchronization/Graph behavior. A missing edge does not invalidate unrelated identity evidence but blocks a complete Scheduling claim. If observer cost or closure effort exceeds the expected information value, PIVOT to the other broad packet or reuse existing evidence; do not continue a chain of local certificates without updating the whole-Product uncertainty budget.

## Source pins

| Source | SHA256 |
|---|---|
| A/worker/model_runner_v1.py |004dbd0d5b1a5c3f9533fa1d12327bd0ae421fe24bb2674544b2c4a25d74aaba|
| A/models/deepseek_v4.py |11dd3e983dc1d628bf42a95581f9617861471b1f658739db44c20df886147247|
| A/attention/dsa_v1.py |88a0cd69dbbba476d0ac179431568590e6e876134390fba77f22721069f491cb|
| V/distributed/device_communicators/base_device_communicator.py |c4fafc71bbb3a7652ecdf425bf116e3a9ad203ba44baf005cbab5db88a8b8221|
| A/distributed/device_communicators/npu_communicator.py |e91429f5cfba0b8af002b7c781db795d4826d7d96e4f08a9684664eb6d7ed275|
| A/patch/platform/patch_kv_delivery_preemption.py |5ba894b9ec003502c008e12281c260787ed64a419327c451c8e67374bf525e8d|
| V/entrypoints/openai/chat_completion/serving.py |860e27d548ec7bb0497a46e0356a5acc5500998944cffa5e67e336041693c067|
| runtime/target_adapter.py |c46d60318098077064c0ff57eae98bc937dcf943bda531918388be4e9128acd5|
| runtime/extreme_decode.py |eb4b5142cc475a51e06c6eec8238e9cfeb918b568e8a5f27b0a4bc7d26963499|
| bootstrap/vllm_dspark_handoff.py |fe7039361894acd70666f72c3089efd635404f1fe9cbc685db21272d946a317e|
| scripts/loop081_product_patch_run602.py |a80aa76e2396a0da4f5fbb881d64dc84a0bcc177e08facda417c7e88dcce25e7|

Formal Current571.681tok/s and all strict/conditional numerical Bound endpoints remain unchanged/null. This source preflight supplies executable anchor choices and admission obligations, not physical ready times or live readiness.

## Final review of Sol's executable source preflight

**SCOPED PASS as a source-only plan; status remains NOT_LIVE_READY.** Reviewed generator `scripts/loop081_runtime_packet_preflight_run636.py` SHA256 `ad87df39254008886cbaf19184972429100a3d428cf120ef5bde1e78e2af54bb`; JSON SHA256 `9c779f24d77ff6a49c0d9c83340ed0077c0c98663e9dd08b69721057c0524c9c`. I independently executed it with writes intercepted in memory and reproduced its JSON byte-for-byte. All seven pinned source files and listed anchor-presence checks pass. Anchor membership is a recovery/preflight aid, not unique patch-placement validation or a physical dependency proof.

The initial packet incorrectly combined producer-ready with Host forward issue, and grouped hidden/aux before logits. The final revision correctly separates asynchronous Host issue from device ready→first native read, main-hidden→logits→acceptance from aux→DSpark, and adds acceptance→state→Draft query inputs plus Draft-result→commit-copy→next Target read. It makes metadata scheduling conditional and preserves Host park/invalidation. These corrections prevent false necessary serialization and close the identified coarse current-path omissions. Detailed tensor/storage generations and native readiness remain to be measured.

Runtime-first conditional priority and the preparation fallback match this independent review. Its same-Run606 Host accounting is properly qualified; it does not subtract cross-W0 durations or claim a measured largest removable gap. Preflight now asks for an implementable ownership-binding method; actual completion evidence is explicitly a posterior admission obligation. Existing synchronization is preserved, while new measured-path synchronization is forbidden. This avoids making the intended acquisition logically impossible by demanding its result before running.

The JSON's last-cycle staged-output edge is a **terminal frontier**, not a claim that only the last cycle produces Product outputs. The implementation must retain `fixed_serving.py:140–141` per-cycle token/count staging, `:144` parking, final CPU copies`:162–163`, Scheduler clipping/prebulk ownership and actual publication. Parking`:87–117` can invalidate metadata`:99`, park proposer state`:108` and rewrite counts`:114`; next-step preparation selects scheduled commit versus rebuild at `extreme_decode.py:390–412`. These must be represented whenever the chosen stratum crosses park or completion. An all-active two-cycle slice cannot certify tail/park cost for the full Product.

No complete execution DAG, value-ready certificate, observer effect threshold, live patch or numerical Bound is created by this source gate. Its four endpoint fields remain null and Formal Current571.681tok/s is unchanged. The next authorized work is the bounded reversible observer implementation and concrete preflight, selected using the whole-Product uncertainty budget; no new service is justified solely by successful anchor presence.
