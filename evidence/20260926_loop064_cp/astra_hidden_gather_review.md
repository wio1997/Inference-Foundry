# Astra High follow-up: CP hidden AllGather versus local Q

Read-only source review; no NPU execution or running-source modification. Conditional next candidate if the c4 compressor/QLI fork does not establish benefit. This is not a KEEP decision.

## Established facts

In context_parallel/dsa_cp.py:1436-1510, decode enters synchronous maybe_all_gather_and_maybe_unpad before computing q_a, qr_local, q_b, Q RMS and local RoPE. The asynchronous hidden-gather path exists but is gated by has_prefill. All local-Q arithmetic consumes hidden_states_local, local weights and local RoPE; none consumes the gathered tensor. The gathered result first feeds hidden_states_cache and wkv. Thus gather-completion -> local-Q-start is an implementation edge, not a semantic RAW edge. Semantic-independence confidence is high.

The normal custom op in ops/register_custom_ops.py:39 calls tensor_model_parallel_all_gather(x,0) and then removes _EXTRA_CTX.pad_size. GroupCoordinator.all_gather resolves to its communicator; NPUCommunicator does not override all_gather, and DeviceCommunicatorBase.all_gather uses dist.all_gather_into_tensor with the same device_group. distributed/utils.py:17 all_gather_async likewise uses dist.all_gather_into_tensor(output,input,group=group.device_group,async_op=True). With self.tp_group=get_tp_group(), both have the same TP participants and underlying collective family; this is not a proposed EP group or new communicator. The difference is the wrapper/allocation path and completion dependency timing.

For fixed local [12,4096] BF16, input is 98,304B/rank and gathered [96,4096] is 786,432B. These are tensor payloads, not wire bytes. Confirm actual input shape/dtype/stride, pad_size and the mapped graph collective before using those numbers. Run250's size signature alone does not uniquely identify the particular AllGather among repeated same-size operations.

## Minimal causal test

Limit to one explicitly Target layer, 96-token decode, is_draft_model=false, unchanged TP8 and padding semantics. Use the existing async helper in both controls. A launches gather then immediately waits before local Q; B launches the identical gather, computes unchanged local Q, and waits immediately before the first full-hidden consumer (wkv). Keep unpatched A0 to identify any wrapper/allocation/capture-path effect separately. Do not combine with the c4 compressor fork. No additional HCCL group or broad stream rewrite is needed initially.

Capture A and B separately with recorded source/mode/graph identity. The existing prefill implementation is evidence of semantic/API intent, not proof that delayed Work.wait has the required FULL Graph replay semantics. A run which only returns successfully or prints a Python branch marker is insufficient. Show in the captured/replayed device timeline that local Q begins before the gathered-result dependency completes, and wkv begins only after that dependency. Check exactly one hidden gather, unchanged subsequent collective order, payload, buffers and layout. Keep input/output alive through the completion dependency; if static output storage is introduced, use the same storage policy in A and B and account for its memory cost.

Use same-state A/B/A control for gathered full hidden, q/qr, downstream cache writes, logits and acceptance, with the established self-replay noise floor. Validate all ranks and the latest-rank complete Target/cycle. No query-side collectives appear explicitly in this block, but confirm the selected wq_a/wq_b implementations retain the frozen local/replicated contract. Do not broaden to all43 layers before one-layer capture/dependency and correctness are established.

## Risks, interpretation, and bound impact

HCCL async progress/capture semantics may still make main Q wait implicitly, or use AIV/HBM resources also needed by Q quantization. Query matmul offers potentially complementary Cube work, but full-Q includes vector and memory phases; interference can erase the nominal overlap. Earlier ranks may hide peer wait while the latest rank gains almost nothing. Preserve same-group collective order on every rank; inspect next-collective arrival rather than interpreting isolated duration changes as transport improvement. Run267's profiler arrival skew is not product headroom.

Resource work and nominal communication bytes are unchanged: this tests Scheduling Bound. For a fixed layer with measured comparable branch costs, min(T_gather_exposed,T_local_Q) is a no-contention local saving screen. It is not a numerical bound until actual exposed service/dependency costs and resource interference are measured. A legal faster schedule would add a demonstrated engineering point, not establish the shortest attainable schedule or a physical TPS ceiling. Failure of this candidate would not refute the broader split at qr_local -> main-Q completion / indexer-query preparation.

Priority: after c4-fork feasibility evidence, this is a source-backed low-scope alternative whose key uncertainty is graph-visible HCCL completion/compute overlap. Measure that uncertainty before expanding the implementation. Numeric Target/E2E gain remains unknown.
