# Attribution after Run251: contemporaneous gap owners and single-stream events

Offline `reduce_gap_owners.py` intersects each disjoint exposed pre-enqueue gap with the **currently active** main-thread CPU scopes. It uses shortest active scope as an explicit deterministic convention and avoids adding nested durations. It does not call the next enqueue name the preceding cause. No Python stack is exported (`torch_profiler_with_stack=false`). MoE/MLA outer scopes can contain Python preparation not represented by an inner operator.

| Existing profON rank | Pre-enqueue gaps | Inside MoE | Inside MLA | Outside those scopes |
|---|---:|---:|---:|---:|
|D13|416.999ms|301.903ms|85.901ms|29.196ms|
|D15|747.341ms|531.027ms|173.377ms|42.937ms|

D15 shortest active scopes include MC2 dispatch123.193ms, combine111.406ms, outer-only MLA73.560ms, outer-only MoE73.286ms, copy50.017ms, empty_tensor35.218ms, SFA output/LSE merge31.799ms. These columns are exclusive local exposed **wall scope** coverage, not CPU execution or removable code budgets. `gap_scope_owners.json` preserves all categories and input hashes. H4's specific removed copies/final splits intersect63.958/5.733ms of these gaps; propagated acceleration need not equal this instantaneous intersection.

## H5 source path and actual stream evidence

Actual fused_moe.py0.27.1 creates before-routed event. moe_comm_method.py creates before-dispatch and before-combine; the active W8A8 moe_mlp.py creates before-GMM2. SharedExpert.forward consumes before-routed/GMM2/combine. There are380 MoE scopes ×4 records/3 waits. All1520 records join from PyTorch scope→Enqueue→Dequeue→CANN→hardware **stream47** on each measured rank.1140 waits join through the queue but have no exported TASK rows. The targeted original SQLite/raw CANN excerpt confirms aclrtStreamWaitEvent inside one Dequeue; absent TASK duration is not filled as a device wait or redundant barrier.

Installed utils.npu_stream_switch(enabled=False) returns nullcontext. The actual shared overlap flag is false and constructor copies that fixed flag. All these explicit events have current-stream producers/consumers; necessary HCCL/DCP internal stream joins remain elsewhere. Full-source reference search found only the known routed→shared event consumers, plus a LoRA before-GMM2 producer that is retained unchanged. No LoRA is enabled in this workload.

|D15 explicit MoE API|Count|ON inclusive wall|Local exposed pre-enqueue intersection|
|---|---:|---:|---:|
|Event::record|1520|85.085ms|16.882ms|
|Event::wait|1140|48.435ms|15.237ms|

The133.520ms API sum is slightly larger than H4's127.203ms removed scope total; H5's32.119ms local exposure is **smaller** than H4's69.691ms. Neither comparison ranks OFF E2E savings. H5 is the next concrete residual redundant path after the H4 comparison, not a claim of globally maximal speedup.

[Patch](moe_event.patch) adds a conditional event helper in existing utils, changes producer calls, makes the three mandatory shared waits None-safe and asserts that a true cross-stream consumer has its input event. Arithmetic, collectives, output tuple/layout and existing overlap=true operations are preserved. CPU source-normalization checks verify five entire ASTs match originals outside these event edits; policy/None/true missing-event checks passed. Independent [review](ASTRA_MOE_EVENT_REVIEW.md) supports the fixed-config eager W8A8 scope, without claiming import/device correctness or KEEP. A config refresh that retains old shared instances must not silently create an inconsistent cross-stream consumer; the assertion preserves failure rather than dropping the dependency.

[Reset169](RESET169.md) froze one candidate, no H4 stacking, native correctness→same-resident A/B/A/B→complete natural-EOS fixture in Run253. Actual controller and phase completed, exit0; original running handoff is preserved as a handoff. Native source import/dependency176 cases passed before one D reload. All64 resident witnesses and original SSE/arrival checks passed. Complete23-token D generation TPOT improved4.71/3.26%; D wall improved4.83/3.41%, with P contributions separated. This is positive scoped code evidence; formal PERF_KEEP/workload/SLA promotion remains INCONCLUSIVE/PARKED, Current=None. [Detailed decision and drift](H5_DECISION.md). No NPU profile or parameter sweep was added.

## Residual MC2 frontend and profiler limits

Targeted original CANN_API query on D15 producer thread shows dispatch/combine workspace queries380 each total9.957/9.963ms. This does **not** explain their203.617/187.673ms inclusive PyTorch frontend scopes. CANN contiguous workspace queries1510 total45.187ms and inplace-copy workspace queries6692 total42.587ms are also real nested frontend work, not additive to enclosing PyTorch scopes. The remaining output/descriptor/operator preparation cannot be attributed to workspace alone or declared removable without identifying its exact dependency/lifetime/API path.

[Run252](../../runs/GLM-RUN-0252/summary.md) confirms mains are predominantly on CPU in the unprofiled client window but includes SHM recent-read spin and runtime active waits. It does not identify a Python hot function, a compute floor, per-round OFF decomposition or exact Scheduler/Executor/ModelRunner durations. These limits remain explicit while source/raw attribution continues.
