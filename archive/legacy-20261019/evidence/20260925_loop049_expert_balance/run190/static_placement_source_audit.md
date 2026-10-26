# Run190: static expert map source audit

Read-only audit of borrowed vLLM/vllm-ascend trees. No service was started and no source was changed.

- `vllm_ascend/eplb/core/eplb_utils.py:27-45,74-116` reads `expert_map_path` as per-layer/per-device logical expert lists and constructs `global_expert_map`, `_expert_map`, and `log2phy` for Ascend execution.
- `vllm_ascend/ops/fused_moe/fused_moe.py:437-465,764` assigns execution `_expert_map` from the configured placement and passes it to the MoE runtime. A comment at 463-465 explicitly says the upstream `ExpertMapManager` physical map remains during checkpoint loading.
- `vllm/model_executor/layers/fused_moe/routed_experts.py:932-1072` uses `EplbState.build_initial_global_physical_to_logical_map`, which lays out physical expert slots in original logical order with redundant copies at the end. The mapping is independent of `expert_map_path`.
- `vllm_ascend/worker/model_runner_v1.py:5497-5507` disables only the model-loader EP weight filter when static EPLB is enabled; it does not reorder weights. `patch/platform/patch_fused_moe.py:68-79` enables EPLB capacity but does not change loader placement.

Therefore a non-identity 8-rank, 32-expert-per-rank `expert_map_path` has an execution/weight mismatch in the inspected path. It is not a safe zero-code optimization switch. A correct static placement would require integrating the map into checkpoint loading or physically moving all affected expert weights and scales after load, plus an 8-rank exact correctness gate. The conservative conclusion is source-path evidence, not a runtime reproduction. Run189's impossible ideal 2.0-2.2 ms/cycle balance screen and separate-service 0.39-0.44 ms GMM rank spreads do not justify that architecture change yet. Next, an offline cross-cycle placement simulation can bound attainable balance without touching the service.
