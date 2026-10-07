# Independent GLM K1 MTP Graph source review

2026-10-07. Trigger 6: Graph / MTP / lifecycle boundary. Offline only; no implementation, service or device operation. H6/H5 and current target FULL remain. Async early publication parked. This review supports a scoped support implementation investigation, not performance KEEP or unconditional removal of GLM's guard.

## Conclusion

**KEEP the MTP Graph direction as the sole next research route.** Existing code contains substantially more K1 graph machinery than the forced-eager warning suggests. I found no source proof that GLM greedy K1 + TP16/DCP16 + SPoff + noncompress SFA is intrinsically uncapturable. Conversely, setting speculative enforce_eager=false alone cannot enable it: actual proposer 212–232 unconditionally disables graphs for GLM.

The minimum change is an explicit, per-call scoped admission contract plus closure of captured-pointer/output ownership; it is not “delete the guard.” Unsupported requests must continue through native eager MTP, preserving dynamic service functionality. Gain and largest removable timing are unknown.

## Sources personally read / positive evidence

Paths below are relative to this review directory.

- `padded_mtp_order_sources/vllm-ascend__vllm_ascend__spec_decode__llm_base_proposer.py`: init 180–292; wrapper 579–610; dummy 623–806; `_propose` 827–1147; merged 1204–1420; first-pass 1570–1675; update 2266; metadata builder 2335–2410.
- `.../vllm__vllm__v1__spec_decode__llm_base_proposer.py` (base input/get-position interface); actual Ascend eagle proposer (inherits this implementation).
- Actual `attention__sfa_v1.py` build/build_for_drafting/capture 280–405 and update 556; `attention__context_parallel__sfa_cp.py` DCP persistent view construction 640–846; `graph_sources/rotary_dynamic_source.json`.
- `graph_sources/vllm-ascend__vllm_ascend__compilation__acl_graph.py` 133–307 and separate target/draft GraphParams; `stream_ordered_replay/original/{breakable_aclgraph,ascend_forward_context}.py`; actual Ascend deepseek_mtp override and deepseek_v2 indexer handling; runner draft copy/count consumers.

Facts:

1. Once enabled, proposer wraps `_run_merged_draft` in its **own ACLGraphWrapper** (579), dispatches with target uniformity, and uses separate preallocated input_ids, positions, hidden_states and token_indices_to_sample. `_propose` copies first-pass data into those buffers before the graph call. K=0 already returns an empty draft; K1 exits after the first model/logits/sample (around 1407), bypassing multi-step metadata mutation.
2. A critical counterexample to an old blocker: non-DSpark first-step `build_draft_attn_metadata` invokes **builder.build**, not build_for_drafting. SFA therefore uses draft_index=None and `get_cos_and_sin_mla(..., use_cache=True)`. Capture SpecDecoding also calls build. Thus K1 does not automatically capture ephemeral use_cache=False rotary tensors. Later draft steps do, but are outside K1 scope.
3. Proposer copies query_start_loc, seq_lens and slot_mapping into per-step persistent groups. Actual DCP SFA copies replicated block tables, replicated slot mappings and local sequence lengths into persistent builder buffers. Dynamic positions/lengths do not by themselves require recapture when shapes and pointers stay within the admitted bucket.
4. SFA update_graph_params is pass. That removes an argument-update task dependency only if all consumed metadata really uses the persistent buffers. It is not evidence that arbitrary metadata objects or auxiliary streams are safe.
5. Target and draft wrapper entries are instance-local; GraphParams globals have distinct target/draft storage. The code passes is_draft_model=True during dummy/real proposal. No evidence here of inevitable target registry collision. Do not invent a collision solely from shared process globals; actual captured task handles still need attribution.
6. Proposer explicitly shares the target topk_indices_buffer and step0 resets skip_topk(False) before capture-time model invocation; K1's graph must contain that actual indexer work. Python set_skip_topk calls are not replayed, so correctness relies on the captured operations rather than Python flags changing each replay. Target→draft→next target ordering protects shared writes only under the maintained stream/communication dependency contract.

## Necessary boundaries and unresolved evidence

**Scoped eligibility:** require actual GLM model/backend, K==1 this call, greedy/non-probabilistic draft behavior, admitted uniform decode descriptor, matching TP/DCP and noncompress/SPoff assumptions; exclude unsupported prefill/mixed/capture shapes, LoRA/adapters and dynamic sampling modes until separately supported. Preserve eager fallback rather than reject legitimate public requests. Capture must use the same greedy branch, output rank/shape and model implementation as replay. The current wrapper key is a batch descriptor, not a sampling/config fingerprint.

**Pointer closure:** ACLGraphWrapper's debug input-address check examines positional `args` only (around 163/244). Proposer calls with keyword arguments; an empty matching address list proves nothing about captured inputs. Record/assert every actual tensor consumed by the draft graph: persistent IDs/positions/hidden/index selection, rotary, seq/query, local and replicated block/slot tables, DCP lengths/masks, shared index/cache and outputs. Check changing positions, block boundaries, rejected tokens and request reordering. Do not rely only on Python metadata identity.

**Output ownership:** wrapper returns the fixed captured output on replay. Existing eager output is newly allocated. Runner `_copy_draft_token_ids_to_cpu` uses a separate stream for grammar/history/PP, and the next replay can overwrite a graph output allocation. A main-stream consumer of prior drafts is naturally ordered, but a side-stream copy needs an explicit completion/ownership contract. A narrow implementation can retain eager fallback for those branches initially, or introduce an owned output copy/event before reusing the graph output. Do not add a blanket synchronization unless required. CPU staging events, accepted-count event/DCP correction, draft-copy event and existing output/error publication boundaries stay intact.

**Capture/runtime agreement:** dummy uses K1's two-token target-style batch descriptor; real proposer must retain that shape after DCP first-pass transformation. Persistent capacity alone does not prove query totals, selected-token indices or output shape agree. Existing SPoff sizing avoids the withdrawn physical16/fifth-request blocker. Any required sizing change must follow actual transformed inputs, not that obsolete premise.

**Model body not fully closed:** local `vllm-ascend__...deepseek_mtp.py` delegates forward to upstream DeepSeekMTP. The full actual upstream model implementation was not found among the files inspected; its model-level state/topk dispatch/captured sampling branch remains a specific missing source item. Deepseek_v2's MTP indexer comments establish intended non-skipped first-pass work, not complete GLM draft graph safety.

**Streams/errors:** keep offloader capture/join fences, HCCL dependencies and target/draft ordering. ACLGraphWrapper FULL draft/eagle may already omit its host synchronize; that is existing behavior, not an extra optimization to combine. No early result publication. A capture/replay failure cannot silently continue with potentially modified KV/model state; diagnostics must preserve failure and use the owned recovery procedure.

## Ranking / Top 3 / smallest evidence

Priority ranking is architectural opportunity, not measured removable milliseconds:

1. Capture the actual greedy K1 draft model + logits/sample work while retaining per-step dynamic input/metadata preparation. This can eliminate repeated draft Python submission; its exposed current cost is unknown.
2. Close graph output reuse and metadata pointer lifetimes. This is a correctness prerequisite, not a second performance candidate.
3. Only after those are understood, evaluate residual preparation overhead. Do not reactivate early publication or unrelated local optimizations in parallel.

Smallest next evidence is CPU/source capture-vs-replay contract checking, including per-call selector fallback, keyword tensor inventory and delayed side-stream output ownership; then, if Root authorizes a bounded device diagnostic, one scoped capture/replay correctness request with all-rank graph attribution, dynamic position/seq/slot/index snapshots, native KV success and exact target/draft output evidence. Observe both an accepted and rejected-token path when available without outcome-driven repetition. A single clean request cannot establish dynamic API coverage or performance; it can falsify the concrete implementation. No run is performed or authorized by this review.

Stop list: no bare guard deletion as support proof; no speculative flag-only claim; no old SP-enabled sizing objection; no claim K1 rotary is necessarily ephemeral; no pointer proof from empty args; no dropping staging/HCCL/KV/error fences; no random/probabilistic replay under greedy capture; no combined H9/H10 reactivation; no performance inference from successful capture.

AISBench namespace/cache-condition work is independent: no claim that its declared 93% condition is observed, and no relation between those counters and graph correctness is assumed.
