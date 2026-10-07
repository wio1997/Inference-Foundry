# Independent review of scoped MTP Graph candidate

2026-10-07. Read actual direct diff against `mtp_graph_candidate/original_llm_base_proposer.py`, candidate, actual SFA/DCP implementation, runner cache binding, upstream MTP and wrapper consumers. No installation, code patch, service or device execution. CPU fixtures are not device proof.

## Finding requiring a concrete correction

**Captured cache identity contract omits the independently bound indexer cache.** Candidate `_glm_k1_graph_contract` signs `layer.kv_cache` and `impl.topk_indices_buffer`. `_glm_k1_layers` explicitly excludes DeepseekV32IndexerCache layers. Actual SFA forward calls `_compose_sfa_kv_cache` (`vllm-ascend__...attention__sfa_v1.py:1406–1464`), which additionally reads **`impl.indexer.k_cache.kv_cache`**, including indexer scale when enabled. These tensors are kernel inputs even though they are absent from the main MLA tuple.

Concrete counterexample: capture once, replace only the separately bound indexer key/scale allocation, leave main MLA/cache metadata unchanged, then invoke again. Candidate signature remains equal; replay reads the old captured indexer allocation while eager would read the new allocation. This is not evidence that live Run buffers currently rebind, but is a precise hole in the promised address-change→eager contract.

Minimum fix: include the actual independent indexer cache tensor tuple (when has_indexer) or the full normalized composed-cache tensor set in the signature. Add a CPU contract case that changes only each indexer key/scale pointer and expects fallback before graph submission. Preserve topk buffer identity separately. No kernel change required.

## Checks that do not justify rejection

- Initialization uses inherited attributes after super and runner is assigned; use_compress/parallel data are defined before the gate. No demonstrated runner.dynamic_eplb ordering error; Root's actual runner ordering resolves that concern.
- Static opt-in and per-call conditions are substantially narrower than bare GLM guard deletion. The two dispatch decisions use `use_call_graph`; eager fallback does not require the graph-only DCP block-table clone. Unsupported calls still prepare inputs and run the original merged function in NONE mode.
- Signature fallback temporarily sets current ForwardContext mode to NONE and restores it in finally. No graph is submitted by this fallback itself. Remaining post-call update is SFA's no-op. I found no specific state-loss bug in this path.
- Dummy ForwardContext num_actual_tokens=0 versus real2 is not the same as SFA metadata num_actual_tokens. Dummy common attention metadata sets its token count from num_tokens. ForwardContext updates the persistent MC2 mask prefix on every context entry (actual ascend_forward_context 206–214), so differing mask contents need not change captured pointers or host branches. SFA forward does not branch on metadata.num_actual_tokens in the inspected decode implementation. Do not reject capture from the 0→2 difference alone.
- K1 dummy and admitted real one-request bucket2 have two model rows, one sample index and [1,1] output. DCP first-pass helper retains those shapes. The signature verifies relevant decode branch counts and excludes compact-prefill gather metadata.
- Current GLM binding uses upstream bind_kv_cache calling each layer's bind method, not the DeepSeekV4 `[kv_cache]` special case. SFA expects flat main tensor tuples and independently composes its indexer tuple. Tensor normalization in the candidate is not itself a demonstrated current-type error.
- Main-stream `result.clone()` addresses the previous concrete overwrite risk: subsequent graph replay overwrites its own output, not the owned returned clone; copy is ordered before the next replay on that stream. It does not justify removing existing D2H allocator/lifetime/event protocols, but those are unchanged. The scope also excludes history/grammar cases that normally request delayed draft CPU copies.

## Remaining bounded validation, not invented blockers

Contract evaluation is rank-local. Current stable allocations should keep selection consistent, but record all-rank graph/fallback selection; do not claim arbitrary mixed eager/graph ranks and lazy capture are proven equivalent. No actual mismatch is established here.

Persistent SFA seq/query/slot/rotary and DCP replicated/local buffers are covered by the listed signatures; no-op graph-param update cannot compensate for an omitted input, hence the concrete indexer correction above. Wrapper positional-args validation remains ineffective for kwargs, but this candidate's explicit signature is a meaningful remedy once the cache omission is fixed.

The signature permits DecodeOnly or SpecDecoding without including the enum in the signature. In the inspected current SFA decode body this is not a demonstrated distinct captured branch; do not manufacture a failure from that alone. Any future backend-specific state-dependent branch must be added or excluded. Current exact metadata type prevents silently admitting a different metadata subclass.

Recommendation: fix the indexer-cache signature omission, preserve all current guards and synchronization, then continue the bounded correctness investigation under Root's frozen decision. This review does not approve deployment, prove dynamic replay or declare performance KEEP. H6/H5 stay retained; H9/H10 and early publication stay off/parked.
