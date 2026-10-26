# Run585 preflight — one fixed-W₀ BF16 `wo_a` work witness

This is a source-only plan from Astra High's independent Resource review. Do not install while Run583 owns the service or borrowed source files. A Run583 all8 branch/restore admission is a prerequisite.

## Scope and conditional conclusion

Select layer0, first measured cohort, cycle1, slot0, Target verification position0 and exactly one local `wo_a` group-row. The observed production path is `vllm_ascend/attention/context_parallel/dsa_cp.py` `_restore_tp_head_layout` → BF16 `o_proj_input.view(num_tokens,o_proj_groups,-1)` → `npu_transpose_batchmatmul` → reshape, with weight transformation in `vllm_ascend/ops/linear.py:480–499`. Under the observed TP8/o_groups8 mapping, each rank owns one `[4096,1024]` group, but rank-to-group ownership must be checked dynamically. Expected input `[96,1,4096]`, weight `[1,4096,1024]`, output `[96,1,1024]`; reject rather than silently adapt if production geometry differs.

Separate three results: (1) this replay executed the group projection; (2) the selected semantic Target evaluation is part of frozen diagnostic `W₀_Run585`, not parked/padding/duplicate physical work; (3) an **ordinary dense online evaluation class** with declared window-entry cache and permitted reuse requires that subexpression to be newly evaluated. Only after all three may the scoped conditional `W⁻` candidate be `2×4096×1024 = 8,388,608` conventional ops. Do not multiply by ranks, layers or physical96 rows. This is not yet a strict architecture-wide lower bound or finite Hardware time endpoint, because arbitrary algebraic variants and exact-board cumulative `C⁺/B` remain unresolved.

## Capture and joins

- Reuse the frozen 48+48 c12 client/admission, Run576 request/cohort and route joins, Run583 Graph capture serial/actual measured replay ordinal, Run569 arithmetic and the guarded stop/restore controller. First Graph capture/replay in warmup does not establish measured ownership.
- In Runtime outside compiled decoder forward, bind measured request/generation, slot, absolute/relative cycle, Target input token/position, active mask, query prefix, num-computed, preceding acceptance/count, seed anchor and immutable semantic evaluation key. Check the selected row is active and not padding.
- At the production BF16 call capture TP member/rank/group mapping, `full_gather`/`olora_tp`/CP branch, native attention static branch plus dynamic query-prefix, source rank0 and post-A2A row/head identity, quant method, weight shard and transform, input/weight/output dtype, shape, stride, storage and generation. Bind each to selected FULL96 Graph generation and *measured* replay ordinal.
- Preallocate isolated, stable private scratch before capture. Copy only selected input/output and necessary A2A head slices inside the custom op so later Graph-pool reuse cannot erase witness. Before selected cycle, poison scratch outside Runtime; after replay completion, synchronize once and export. Scratch alias/writeset checks are mandatory. These extra copies perturb the workload, so do not compare its latency or TPS with uninstrumented Current.
- Preserve complete diagnostic acceptance/count/output ledger and source/helper/binary SHA plus stop, all8 idle and source restore. Name the captured trajectory `W₀_Run585`, not historical Run99 W₀. Equality of independent runs' output tokens cannot prove same-state KV/Draft trajectory.

## Preflight and failure cases

Offline patch/temp-copy restore, `torch._dynamo.is_compiling()` guard and CPU reducer negative tests precede any live run. Avoid Run582's compiled decoder hook (`os.getenv`→logger fullgraph failure), files or `data_ptr` calls inside compiled forward, `torch._dynamo.disable`, device synchronization inside Graph and post-Target reads of temporary production tensors.

Reject warmup as measured, mismatched request/slot/cycle, duplicate or stale replay/generation, Graph submission without completion, inactive/padded row, unexpected pad/branch, swapped rank/group order, wrong BF16 shape/shard, stale query-prefix, A2A permutation, scratch alias or no overwrite, pre-window cached result, missing source/restore SHA and any attempt to promote the physical96 rows or a single group into total model work. This run needs no A0/A1 timing control because it makes no latency claim; future timing or same-W₀ cross-arm claims require matched state controls.
