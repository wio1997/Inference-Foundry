# Run168 genuine prefill capture feasibility and write-set audit

Read-only. Source: `/data/wio/vllm_ascend_26/framework/vllm-ascend/vllm_ascend/worker/model_runner_v1.py` and `.../attention/context_parallel/dsa_cp.py`; no service/NPU run.

## Boundary and graph support

- Real serving forward is called inside `set_ascend_forward_context` and `maybe_get_kv_connector_output` at model_runner_v1.py ~3673. `_model_forward` ~4544 executes the model and, under flash_comm_v1, gathers main and auxiliary hidden states. A standalone closure must retain this context, input/position storage and all metadata. Sampling is outside the closure and must be compared separately.
- DSA CP metadata builder advertises `UNIFORM_BATCH` (~274), but `build_for_graph_capture` (~1047) raises for prefill. The existing dispatcher is not a safe prefill implementation. A separate guarded capture path is required.
- Prefill metadata construction (~677-810) calculates rank-local QSL/sequence lengths, start positions, SAS/QLI, compressor row cardinalities, block tables, slot mappings and cos/sin. `max_local_query_len` and `max_local_seq_lens` come from CPU metadata; exact-state capture can freeze them, whereas general replay needs explicit refresh and shape keys.

## Mutable state and control

- `_forward` (~1400-1640) writes SWA cache; c4 `_update_indexer_cache` (~1677) writes indexer state/K/scale/full caches; c4/c128 compressor writes state and compressed KV. All touched pages and state must be snapshotted/restored before each A/B/A-prime execution. Existing decode prestate is not sufficient by inspection.
- Prefill can launch async hidden-state all-gather and wait on its handle (~1432,1507). `wait_for_kv_layer_from_connector` and `maybe_save_kv_layer_to_connector` bracket attention forward. These are capture compatibility questions; no unsupported-operation claim is made without an actual capture attempt.
- Full o-proj weight switching in dsa_cp.py is guarded by `AscendDeviceType.A5` (~1127); this 910B3 product path should not incur that particular pointer mutation. Verify at runtime before relying on it.
- `compressed_kv.numel()==0` and indexer `kv.numel()==0` are shape-based Python branches. The branch and compressor cardinality must be fixed for the exact state; this audit does not prove replay across different prefill batches.

## Product value check

Run161 measured 12 pre-handoff forward calls per rank, with eight large calls around 320-382 ms and four smaller calls. Run165 changed the measured sequence to four calls (83,8,923,96) under profiling, so the profile is unsuitable for predicting capture reuse or coalescing benefit. Before implementing a full capture/state restore, measure natural batch-state repeat frequency and an unprofiled latest-rank prefill launch cost by shape; reject a one-off graph if capture setup exceeds the number of reusable calls.

Sol decision: exact-state graph capture remains technically plausible but needs a runtime preflight and expensive complete prefill write-set restoration. Next perform an offline Run169 workload/state frequency and prefill cost bound using existing legal diagnostics, then choose whether to build a guarded capture probe or test bounded admission coalescing. No correctness or speedup claim from Run168.
