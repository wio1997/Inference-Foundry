# Run581 — first production collective dependency slice, source gate

## Question and admitted scope

Frozen DSpark7 acceptance, cycle trajectory, output semantics and model work remain fixed. Run579–580 measure an independent-ready GMM × HCCL fixture, not a legal production schedule. The next Scheduling/Execution Bound question is whether a real Target collective has a movable producer-to-consumer interval in the selected FULL96 graph. Select one narrow slice: layer0 DSA `wo_b` local partial output → TP collective → attention output write → first `hc_post`. This is an identity gate before timing or schedule intervention.

`source_gate.json` records SHA256 for 13 installed/source/prior files, checks nine source predicates and validates the Run378 ledger's 265-call kind count (RS87, AG135, A2A43). The script is read-only and does not start service or use an NPU. Run378 ordinal3 is only a **candidate** for this source edge. Call order and shape cannot certify its live tensor identity.

The installed `dsa_cp.py` calls `self.wo_b` when full-weight gathering is disabled and copies its result into attention output. `deepseek_v4.py` sends that output to the same layer's first `hc_post`, then the FFN branch. If the loaded `wo_b.custom_op` is `SequenceRowParallelOp` and its non-fused FlashComm branch is selected, `linear_op.py` produces local `output_parallel` and passes it into TP `reduce_scatter`. There are alternative `DSV4OProjRowParallelOp`, fused MMRS and no-FlashComm branches. Run576's `full_gather=false` and Run580's service config make the sequence branch plausible but do not log the loaded custom op or exact native collective. This gate therefore **does not certify the complete dynamic edge**.

## Prior transfer

- Historical R20: DP2×TP4/EP8 prefill hidden AG overlapped local Q and improved three exact-32K short-service TTFT pairs by roughly 48–59 ms; actual E2E improvement was smaller than the profiled overlap. This is a scheduling mechanism prior, not a DP1×TP8 FULL96 decode result.
- Historical R27: EP8 MoE finalize RS led into residual/next-layer data use under its prefill shape; buffer/AIV alternatives did not improve that case. Its EP8 topology, shape and predecessor differ from this TP8 layer0 output edge.
- Current Run292–293 already show a legal *different* layer2 hidden-AG/local-Q window and roughly 10 µs local advancement to the following collective, without a proven complete Target/cycle E2E gain. It should not be repeated as this slice.
- Run508 closed the selected MLA update zip loop at zero iterations. Run502/510 support Graph identity/general terminal order, but their post-drain argument dump is not typed last-writer or device time evidence. Run579–580 whole-chain negative overlap cannot decide a shorter, dependency-preserving interval.

## Next live admission

1. In one ordinary selected FULL96 capture and replay generation on all eight ranks, record only scalar and tensor descriptors: actual layer0 `wo_b.custom_op` class/prefix/quant method, oproj TP/SP/DSA CP flags, FlashComm/MMRS/pad, TP group/rank order, selected Graph entry/generation and actual replay ordinal. Abort the edge if its branch differs.
2. Bind local partial result, exact collective input/output, attention copy destination and `hc_post` input with shape, dtype, stride, storage range and generation. Compare overlapping stores/copies before naming a last writer. Establish native collective identity by same-generation trace/task plus source-correlated payload; do not infer from ordinal3 alone.
3. Only then time rank-local producer-ready → collective submit/completion → copy/consumer-ready with event/trace methods that survive Graph capture. Do not call `npu_stream` getter in the hot path or insert synchronizations into the measured interval. If native collective completion cannot be separated, report the weaker caller-stream-consumable endpoint.
4. Preserve the guarded 48 warmup + 48 measured c12/1024 client/Runtime/output ledger, source restoration and all8 idle gates. A matched same-state self replay screens marker perturbation. Profiling time is diagnostic; correctness and repeated formal E2E decide any later intervention.

## Bound effect

Run581 closes a source and evidence-selection ambiguity, not a timing bound. Current Formal remains 571.681 tok/s. Strict Resource/Hardware, Scheduling/Execution and Product E2E endpoints and the numerical Current→credible-limit gap remain unknown. Even a complete layer0 edge must not be multiplied by 43 or treated as a Product saving. The Resource track still needs compulsory fixed-work/traffic and exact-board cumulative capacity; this source gate does not narrow those numerators or denominators.

Independent Astra High review selected and challenged this slice; its conclusion is reflected above. The next high-value action is the live branch/typed identity gate, before any optimization candidate.
