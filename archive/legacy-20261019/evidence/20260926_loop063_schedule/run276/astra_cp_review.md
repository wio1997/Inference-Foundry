# Run276 — Independent Astra High Resource/Scheduling review of frozen DSA CP Target

Read-only source and evidence review; no NPU service or code mutation. Reviewed the frozen enable_dsa_cp path, `context_parallel/dsa_cp.py:1400–1805`, and Run247/266/267. Its override `AscendDSACPImpl._forward` executes the frozen Target; `dsa_v1.py` CV three-block multistream decode is **not** this path. Any assumption that the CP Target already has that CV overlap is invalid.

## Semantic DAG and removable ordering

- Local Q uses rank-local hidden and local RoPE; it does not require the full hidden AllGather. SWA KV uses gathered hidden and its own KV cache/slots. These can in principle overlap if the Graph/HCCL capture path supports safe ordering.
- For c4, indexer cache update writes its own key and scale caches; indexer query/QLI then reads those caches plus local Q, hidden and QLI metadata. Main compressor reads gathered hidden, its separate state/cache and compressor metadata, then scatters compressed KV. It does not consume QLI indices. Sparse attention must wait for local Q, SWA KV, indexer topk and main compressed KV.
- Current CP code serially calls `_update_indexer_cache`, `_indexer_select_topk`, then main compressor/scatter. The indexer-query→main-compressor edge is implementation order; the two branches have a real fan-in at sparse attention. Same-token layer recurrence after attention/output remains semantic.
- Existing shared-expert overlap and its AllGather/reduction are already Current. Earlier EVENT_WAIT totals cannot be called removable time. No claim that the full collective sequence can be reordered safely.

## Highest-value minimal causal candidate

At c4 Target layers only, after `_update_indexer_cache` completes, fork the unchanged main compressor and its scatter to a private stream while the main stream executes the unchanged indexer query/QLI. Join before `notify_kv_cache_written` and sparse attention. Preserve all TP collective calls and ordering. Use A0 original serial, A same helper with immediate join (stream overhead), B fork/join overlap. First prove one representative c4 layer captures and is numerically correct; then cover all 21 c4 layers.

Same-state A/B/A self replay must compare actually written indexer/main KV pages, topk indices, attention output, logits, acceptance and final state under the established noise floor. Candidate output must truly flow into Target. Audit storage intervals, Graph pool/workspace and metadata cache tensor lifetime; a source-level distinct variable name alone does not prove non-aliasing. Measure all-rank layer fork→join, complete Target and cycle, including rank rendezvous and resource contention. If promising, repeated frozen 48×32K→1024 c12 formal E2E and correctness decide KEEP.

A purely algebraic optimistic local screen is `overlap saving <= min(indexer branch duration, main compressor+scatter duration)` before event overhead or resource contention. Run256 c4 main compressor task sum ~1.283 ms for 21 calls is only a scale reference and excludes full scatter; there is no justified numeric E2E upper bound yet. Main compressor/QLI can contend for AIV/HBM/Cube, and Graph capture or hidden shared workspace may defeat the candidate. Failure would reject this one schedule, not establish the Scheduling-aware ceiling.

Second candidate: overlap local query with hidden AllGather. The code already uses async gathering for a Prefill condition, but decode FULL Graph and HCCL capture legality require independent proof.

Confidence: semantic independence high; safe Graph concurrency medium; net Target/Product E2E benefit unknown.

## Sol source follow-up while Run277 loads

The non-A5 `DeviceOperator.unpack_dsa_forward_kv_cache` maps main compressed KV/state to tuple slots 0/2; `unpack_dsa_indexer_kv_cache` maps indexer state/key/scale to slots 3/4/5 (`device_op.py:813–835`). This is further source evidence for separate logical cache tensors, not a physical storage proof. `get_or_compute_compressor_metadata` caches by `(cache_group_key, type(metadata))` in forward context (`dsa_v1.py:69–120`); a fork must validate whether indexer/main reuse that cached output and ensure the auxiliary stream reads it only after producer completion. Full storage interval and Graph-pool/workspace audit remain required.
