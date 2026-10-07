# Independent MTP Graph v3 composition review

2026-10-07. Offline only; no deployment/run/frozen-input changes.

Verified runner SHA256 `4c90054ad40f4c430c2562bc1603aacb497684429988bf1a60edce585d6d2923`; proposer `963f02e57937474f7e545f648e1ac6e905ed898dd4d626d44f502e667492e194`, unchanged from v2. Read complete runner diff, actual load order, proposer load completion, Run271 first-error raw and nested-wrapper CPU source/result.

## Conclusion

The one-condition runner change addresses the demonstrated nested-capture defect. Runner4102 calls drafter.load_model; proposer constructs its outer merged ACL wrapper; runner4196 subsequently used to wrap drafter.model again in Breakable. Actual Run271 raw shows outer ACL capture→merged model→inner Breakable capture→torch.accelerator.empty_cache→capture-forbidden synchronization. V3 suppresses only this later inner wrapper when `_glm_k1_graph` is true. It does not suppress target wrapping or other drafter paths. Do not repair this failure by disabling allocator checks or relaxing graph capture mode.

## Flag and lifetime invariant

In current proposer init, `_glm_k1_graph=True` implies `use_cuda_graph=True` and compilation mode has FULL graphs. The same config controls the outer wrapper creation at load_model's end. Personally checked load_model349–614: no successful early return bypasses that block, and use_cuda_graph is not reset in between. An exception aborts load and never reaches runner's later wrapping node. Thus on normal load return the flag implies `_glm_k1_acl` exists; the CPU test assumes that implication, and actual source supplies the missing proof.

NONE fallback remains the same raw model invocation: old Breakable.__call__ returned its runnable immediately under NONE; v3 directly retains that model. For scoped graph operation the outer wrapper owns merged model/logits/sample capture and the owned output clone remains. Non-FULL fallback must not be described as inner-graph execution; removing the inner wrapper intentionally leaves it eager on this scoped path.

## Offloader boundary must be stated accurately

The **outer ACLGraph** preserves capture-before `get_offloader().sync_prev_onload()` and capture-after `join_after_forward()` (actual ACLGraph180–194). Its replay section244–270 has **no** offloader sync_prev_onload call. Actual Breakable replay421 does have that call. Therefore “all offloader fences are equivalent” is not supported by these sources.

For a frozen **no-op/disabled weight-offloader** configuration this difference supplies no missing dependency, and no extra sync is recommended. Record the actual disabled/no-op offloader scope before a correctness diagnostic. If non-noop offloading is intended, its replay-time prefetch dependency is not closed by this patch; either exclude it from scope or explicitly establish the correct replay fence. This is a scope condition, not evidence that current resident weights have an offloader race. KV transfer is a separate subsystem and its existing lifecycle is untouched.

## Evidence and decision

Nested-wrapper CPU result passes two actual load-order cases, 24 scope cases, original nested-cleanup counterexample, retained target wrapping and NONE raw-model equivalence. Test source executes real load/wrap and Breakable dispatch/empty_cache AST with model/graph doubles; it does not establish device capture or numerical output. Its explicit `_glm_k1_acl` dummy is backed by the source invariant above, not independent constructor execution.

No additional concrete composition blocker found within the frozen no-offloader scoped path. Continue only the bounded correctness investigation after owned recovery and reset. Preserve first capture/replay failure; no performance comparison follows from this review. H6/H5 retained, H9/H10/early publication off. This review is not deployment authorization or PERF_KEEP.
