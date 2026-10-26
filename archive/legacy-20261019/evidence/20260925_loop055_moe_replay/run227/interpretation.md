# Run227: all43 first88 MoE Graph rejected by memory gate

Formal launch reached healthy DP1TP8 DSpark7 service. Legal max_tokens1024 warmup48 completed48/48, 563.818 tok/s diagnostic. A captured layers0-16 on every rank; capture of layer17 failed on all8 with torch_npu ExpandableSegment AclrtReserveMemAddress OutOfMemoryError (207001). A requests failed0/12; B parity and G/E stage phases never ran, so there is no correctness or performance claim for all43. This is a capture-memory feasibility failure under the frozen service memory contract, not evidence that graph outputs are wrong. Cleanup stopped the service and restored exact borrowed source SHA.

The all43 bank is not product-feasible with one graph per layer under current memory. Earlier Run225 four-layer stage difference16.661ms per first88 forward, but Run222 observed first88 only once per cohort and top shape coverage is sparse. A smaller 12-16-layer bank has limited amortizable E2E headroom and cannot be assumed to scale linearly. Sol pivots to measured decode/DSpark DAG and exposed communication, then ranks community MRV2 draft graph reuse against Current-to-Achievable bound. Any future MoE graph bank requires memory accounting, same-state logit/accepted-token gate and formal E2E.

Community audit: `community_mrv2_audit.md`, official main pinned SHA2bb3f44716f3505d2723a4e5badb10211a6c5589.
