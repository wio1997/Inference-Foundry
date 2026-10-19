Primary review of local remaining-pipeline drafts while model observation runs.

The actual new model coordinator `start_C32_L17_selected_D_models_and_observe_v1.py` writes receipt keys:

- `pass=true` and phase `STARTUP_OBSERVED_AWAITING_INDEPENDENT_CAPTURE_AND_FUNCTIONAL` only after both native observers return success.
- `starts` keys `start-36` and `start-41`, integer rc0.
- `observe` keys `observe-36` and `observe-41`, integer rc0 (not numeric node keys).
- `native_startup_summaries` uses those same observe-prefixed keys; each value is the observer native result with startup_pass=true/formal_ready=false.
- `external_model_requests=0`, cleanup_or_rollback_executed=false and all4 runtime_CIDs.

Startup capture must consume this actual schema, never invent a top-level startup_pass or an observe map keyed36/41. Require actual pass/phase and both rc0 entries before capture. The raw capacity-v2 file contains D-only provider rows; helpers needing all4 must derive a view from immutable accepted P capacity19355aab… plus D capacitydbf1e19e…, without changing either source. The actual model coordinator already makes that derived view for cleanup template generation.

Per-node capture has16 actual D SDK workers. A per-node coverage status string must describe local16; only a composite decision joining node+namespacePID for both nodes can claim global32. Namespace PIDs may overlap between nodes. Preserve all actual first-use out_cache_loc shapes/dtypes/strides without claiming every padded-bs2 row represents two real active requests or noncontiguous geometry. Existing C32 functional real inferences/metrics/Graph/MTP/native transport proof remain required before formal.

No models/clients/capture action is approved by this note. Remaining local draft executor must not change the concurrent model-start receipt, STATE, HANDOFF, source capacities or frozen coordinator.
