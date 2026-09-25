# Run211: prefill MoE-only source closure

Read-only source review. No service, patch, NPU experiment or performance claim.

The observed first eager prefill in Run208 is padded88/one request on all8 ranks, but the actual local MoE tensor row counts and routing fields were not captured. Do not equate the runner padded token count with MoE input shape.

Upstream vllm/model_executor/layers/fused_moe/runner/moe_runner.py:152-178 defines vllm.moe_forward_shared as a custom op that resolves a registered layer and delegates to layer._forward_impl. Lines 71-89 can increment forward_context.moe_layer_index on the string-name path. Lines 640-685 show outer MoERunner.forward transforms and reduction/combine work outside this custom-op boundary. Replacing only the custom op replay must preserve outer code and advance the Host layer index exactly once if the active implementation uses that path.

Borrowed vllm_ascend/ops/fused_moe/fused_moe.py:1000-1012 wraps the body in sequence-parallel local-size context. Lines 719-809 consume _EXTRA_CTX communication method, flashcomm and routing flags, perform prepare/apply/finalize and optionally mutate EPLB load counters. Frozen serve logs say dynamic EPLB false, but actual per-layer flags and LoRA context must be checked on the live path. Lines 823-951 create/wait stream events, run the shared expert auxiliary stream and join it; these dependencies are required. Current FULL decode graph proves some graph support but does not prove this prefill shape/collective branch can capture.

The custom op does not receive attention metadata or KV cache and the reviewed MoE body shows no direct DSA/KV writes. It is narrower than whole-prefill capture. However, complete inputs include hidden_states, router_logits, optional shared_experts_input/input_ids, layer identity, forward-context layer index and DP/SP local-size context, plus configured routing/communication mode and weights. Scratch/output aliasing and mutable state need live confirmation.

The stock ACLGraphWrapper at vllm_ascend/compilation/acl_graph.py:60-190 explicitly does not own or refresh replay input buffers. A diagnostic graph must allocate persistent copies, refresh tensors before replay, keep outputs/scratch alive, preserve all8-rank collective order, and account for capture warmup. The first test must be one layer and one signature, shadow-only, not a serving replacement.

Decision gate: source closure is bounded enough for a live read-only ABI/state probe. Run212 will record one selected layer at first padded88 eager prefill on all8 ranks: tensor shape/dtype/stride/address, layer name/index, sequence-parallel sizes, active communication/EPLB/LoRA flags, and before/after mutable counter hashes. If any unbounded hidden state or DSA/KV dependency appears, reject before graph capture. Only after this probe passes should Run213 attempt an isolated one-layer same-state graph capture.
