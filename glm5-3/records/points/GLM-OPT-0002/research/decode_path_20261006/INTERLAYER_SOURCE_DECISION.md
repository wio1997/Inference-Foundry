# Remaining inter-layer host region: source partition and rejected shortcuts

H6 is a closed, matched intervention, not a claim that the entire step is explained. [Run258](../../runs/GLM-RUN-0258/summary.md) directly measures the stock main-thread task clock from first dispatch entry to final combine return. D15 wall2135.588653ms/CPU2120.452730ms partitions exactly into native dispatch/combine331.196320ms CPU, routed MLP between dispatch-return/combine-entry153.028100ms, and combine-return/next-dispatch-entry1636.228310ms. D13 CPU2141.533280ms has337.569740/153.567460/1650.396080ms. Prefix and final suffix are excluded; four inter-round intervals are included. These are measured CPU contributions with observer overhead, not deletable budgets. In particular, D15 predicate257.133130ms is77.64% of native CPU, about12.13% of this measured envelope, not77% of the model.

[Existing Run249 disjoint source partition](interlayer_supply.json) uses380 MoE/native scopes and the actual Ascend override. D15's first-dispatch/last-combine profiler-ON window2824.677670ms has744.634071ms local pre-enqueue exposure:

| Source-contained phase | Scope wall ms | Local pre-enqueue exposure ms |
|---|---:|---:|
|Native dispatch|203.617220|127.549270|
|Routed expert MLP between dispatch/combine|225.580380|13.658833|
|Native combine|187.673060|111.881394|
|379 post-combine MoE tails: finalize and shared output|520.381100|204.649196|
|391 MLA attention scopes between MoEs|1059.604030|171.858087|
|379 next-MoE prepare/route/gate prefixes|293.231150|72.461822|
|Other norm/control/inter-round work|334.590730|42.575469|

Each column is internally disjoint and conserved. Do not add the exposure column to wall, rescale either column into Run258 CPU, or extrapolate to profiling-OFF265ms/token. A long source region can contain necessary computation and host submission. The mixed unexported MLA parent72.723ms is not a measured wrapper-only cost.

## Post-combine is not just finalize

Actual `AscendMoERunner._forward_impl`, vLLM0.27.1 branch, calls `routed_experts.forward_impl` including finalize **before** `ascend_shared_experts.forward`. The upstream default shared-before-routed helper is not the selected override. Current model SPoff, shared-overlapfalse, physicalTP16 selects shared TENSOR_PARALLEL. Shared W8A8 performs dynamic quant→gate-up quant matmul→dequant/swiglu/quant→down quant matmul. Each has a real output consumer. The separate shared TP sum happens after the outer `vllm::moe_forward_shared` returns.

[Post-combine reducer](reduce_post_combine.py) asserts all380 sequences, one gather,16 gather output copies, one dynamic quant, two shared matmuls, one activation and three shared waits per MoE. This larger window includes the final380th tail, so D15 exposure205.475746ms differs from the379-tail envelope204.649196ms by0.826550ms.

| D15 post-combine stage | Scope wall ms | Local exposure ms |
|---|---:|---:|
|Finalize allocation/split before gather|70.457020|16.907225|
|List gather and output materialization|209.725870|117.132083|
|Finalize tail/shared entry|59.400480|21.078968|
|First shared same-stream wait and subsequent setup|23.561610|15.524756|
|Shared quant/gate/activation/down, including two later waits|147.665850|34.730006|
|Shared tail/event cleanup|10.962340|0.102708|

Nested H4 copies:6080 scopes,112.573880ms wall/63.957957ms exposure **within** list-gather117.132083ms. Nested H5 shared waits:1140 scopes,48.434930ms wall/15.236537ms exposure **within** the shared stages. These subsets cannot be added to the stage totals. H4/H5 have already supplied independent interventions; no retry or speculative stack follows from this decomposition. Allocation, gather setup, event preparation and Python/native object teardown remain mixed in the between-marker residuals; deleting all of them would be unsupported.

## Collective semantics, with producer-side timing

[Unique CANN→Dequeue→connection-flow→Enqueue joins](collective_producers.json) classify all1901 standard HCOM records on both D13/D15. Classification uses **producer enqueue** source containment; asynchronous consumer CANN launch membership can misclassify adjacent layers.

800 TP all-reduces split attention output395, TP-sharded shared expert output380, first-three dense MLP15 and target/draft embedding10. Routed finalize uses380 token-shard gathers; the shared weights are TP-sharded and require a different sum. `MoERunner.forward` and Ascend `_fused_output_is_reduced` make shared-output reduction and final combined-output reduction mutually exclusive. There is no observed duplicate final reduction to delete. DCP395 gathers split79 compact-KV221184-element calls and316 later4608-element packed-Q calls, with316 output/LSE all-to-alls. Ten target/draft logits gathers remain. This establishes necessary dependencies in the current implementation, not a globally optimal communication design.

## Autograd naming does not establish lost inference mode

[Name reclassification](autograd_name_reclassification.json) preserves the original Run254231 `Autograd`/53 `ADInplaceOrView` counts. The231 uppercase-name samples are registered CompositeAutograd implementations except one sampled set_autograd_meta; Python tensor API bindings also reside in a `torch::autograd` namespace. None of these counts is exclusive backward-graph cost.53 AD wrapper samples and12 as_view metadata samples still require their ordinary-tensor/view semantics to be respected.

[CPU discriminator](dispatch_tls/CPU_result_A.json) runs actual handler/resolver/direct registration AST with CPU views,36 cases, no model/NPU initialization. It compares ordinary/requires-grad/inference tensors under normal/no_grad/inference modes, direct/boxed outer entry and direct/boxed nested leaf. Boolean inference/grad state and included/excluded key sets stay identical at caller/body/leaf/return; values, aliasing and unchanged inputs pass. Ordinary tensor views remain ordinary tensors even inside inference mode, and ordinary tensor key sets carry AD keys; inference tensor views stay inference tensors. This refutes the tested CPU boxed-entry state-loss explanation. It does not uniquely explain every actual PrivateUse1 stack or authorize stripping view/version semantics. No guard/bypass patch or live diagnostic is warranted by this evidence.

Accept Challenger's disjoint-window, nested-subset and TLS limits. **No newly identified specific code removal larger than H6 is established.** The largest remaining region is eager inter-layer host work, but no new capture/replay implementation, V2 migration, blanket collective removal, stacked H4/H5 or graph/config sweep is justified merely by its total. Continue source-based dynamic eager/replay and lifecycle work; a future diagnostic must distinguish a concrete candidate. Current=None; formal H6 remains scoped POSITIVE/PARKED; 本阶段没有新增代码级性能 KEEP。
