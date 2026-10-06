# Actual eager and ACLGraph producer boundaries

Follow-up [actual SP/source correction](GRAPH_APPLICABILITY_CORRECTION.md): current Run249/256 is SP-off, proven by original startup lines. The alignment blocker in the final paragraph is conditional SP-enabled evidence and does not apply here. No graph code patch is justified from it. Existing NONE/FULL producer and Mooncake/drafter boundaries below remain source facts; current-path padding and dynamic buffer contracts need separate validation.

这是现有 Run249/254 归因的只读源码核对，不是 graph 配置实验或第二个活动性能候选。实际源码原字节 SHA 保存在 graph_sources/source_identity.json、hook_source_identity.json 和 mooncake_hook_identity.json。

当前 enforce_eager=true、graph=NONE；AscendModelRunner `_use_aclgraph`（713–720）和 batch dispatcher（3139–3175）都明确保留 eager 入口。target backend SFA 声明 UNIFORM_BATCH，DCP metadata capture 支持 DecodeOnly/SpecDecoding；DCP dispatcher 还要求 prompt 全部 computed。不能据此声称当前 target 每步需要新增某个框架修复才能使用 graph，也不能把关闭 enforce_eager 当代码优化。

| Existing path | MLA host re-entry per replay | Producer mechanism |
|---|---|---|
|NONE or wrapper-mode mismatch|Yes|Original eager model/handler/native operator chain|
|Breakable non-FULL|Yes, whole decorated mla_forward|Other captured segments can bypass their producer; entire MLA remains eager|
|FULL, including breakable FULL|No on a hit|Captured target ops replay without invoking their original per-op Python/C++ producer|

Actual breakable_cudagraph.py100–115 bypasses the eager-break mechanism under FULL. Non-FULL add_eager captures a callable, replay195–214 invokes it again. ACLGraphWrapper capture187–189 calls the runnable, while replay266–267 only replays the saved graph and returns its existing output. BreakableACLGraphWrapper does not add a per-layer PD callback. Wrapper76–82 explicitly leaves dynamic input buffer copying to the outer runner; descriptor match is not sufficient dynamic metadata correctness. The actual DCP builder810–816 copies current local sequence lengths into a persistent buffer, but all captured inputs, request churn, MTP rejection and mixed batches still need validation.

FULL skips Python SFA maybe_save_kv_layer_to_connector1708; the actual attention/utils.py486–500 would call the connector save_kv_layer. However **current MooncakeConnectorV1 wait_for_layer_load1680–1682 and save_kv_layer1684–1688 are pass** (actual SHA `f30572890014d0244bf31d8e15dc5d93fb84147df37e0b2c1d1b01cbba6dd533`). This removes the suspected missing per-layer save/load callback as a current blocker. Runner's per-step start-load context and MTP-after-forward finalize remain necessary. Other connectors with real per-layer side effects cannot inherit this conclusion.

GLM llm_base_proposer.py212–231 explicitly disables drafter graph regardless of target graph. A full target graph therefore does not eliminate the entire target+MTP round producer. There is only one draft attention/MoE layer per round here versus 78/75 target layers; its existence alone cannot explain all the hundreds of milliseconds of target within-layer supply.

A narrow direct handler call only avoids the outer Python→boxed dispatcher→Python crossing. It retains actual handler lookup/metadata, inner MC2/GMM/DCP host calls and all per-op allocations/preparation. MoE-LoRA dual-stream, fake/mutation schema, TorchDispatch modes and graph/compile paths require preserving their registered opaque op. The 86.255% inclusive PythonKernelHolder share is not the removable wrapper cost.

Follow-up actual sizing/key/capture consumers now establish a concrete code blocker under TP16/maxseq4/MTPK1: default logical bound8 has no aligned size16; explicit16 is excluded from separate FULL keys; capture dummy16 schedules only8 valid tokens and asserts. See [source and CPU reproduction](GRAPH_ALIGNMENT_SOURCE.md). This refines the earlier unknown required-code-path conclusion without changing its measured-gain limit. No flags are changed and no graph NPU Run is queued. H6 remains the only candidate under native build; graph/handler review continues as source attribution, with dynamic metadata correctness and benefit still unverified.
