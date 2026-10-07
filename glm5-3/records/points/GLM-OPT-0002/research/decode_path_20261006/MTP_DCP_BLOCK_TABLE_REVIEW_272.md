# Independent Run272 DCP block-table review

2026-10-07. Offline only. Read all16 `runs/GLM-RUN-0272/witnesses/h11_mode1_transition*_rank*.json`, actual proposer ordering and SFA-DCP builder/consumer. No Run, deployment or frozen-input changes.

## Facts and diagnosis

Independently compared capture_signature against both runtime signatures in all16 mode1 witnesses. All32 rows report actual_replay=false. Every row has exactly one differing scalar at `[2,2,4,1,0]`: the data_ptr for `metadata.dcp_context.block_table`. Each rank's first and second runtime addresses also differ. This is evidence of pointer-contract rejection and eager fallback, not failed numerical graph replay. Exact short outputs therefore do not validate MTP replay numerics.

Actual FULL `_propose` adjusts the common block table to `num_reqs_padded * decode_threshold` rows. `_adjust_tensor` uses F.pad when enlarging it. SFA-DCP `_build_with_metadata_view` first takes `_get_dcp_local_block_table(full_table,num_reqs)` and ultimately stores that view in DCPContext.block_table. The sparse decode consumer later passes this exact table to DeviceOperator's SFA call (sfa_cp1237).

The proposer currently builds that metadata before copying/rebinding common.block_table_tensor to block_table_tensor_clone. Rebinding common metadata cannot redirect the already-created tensor view. Dummy step0 also captures a view of runner's original table, whereas the existing clone branch only handles draft_index>0. These source paths directly explain the observed identity difference. The replicated table field is already builder-owned/persistent, which explains why only the local DCP table differed.

## Proposed minimal correction

Support the scoped fix: for `_glm_k1_graph` + FULL, copy the current table into the already allocated persistent block_table_tensor_clone **before** dummy step0's builder and **before** real build_draft_attn_metadata, then bind common.block_table_tensor to the corresponding persistent view. Both paths must use the same allocation and column stride. This is a framework metadata storage fix, not a kernel change or removal of the safety guard.

Required details for the actual diff:

- Copy the current source values, including its applicable padded rows, not merely replace the pointer. Validate row/column capacity; preserve dtype/layout and the builder's logical `num_reqs`. A two-row padded table does not mean two requests.
- Keep the builder's local-column slice and replicated-view derivation unchanged. They must both derive from the just-updated persistent source; do not bind only one late field while leaving the other based on stale values.
- Preserve all previous captured pointer comparisons. The expected success is stable pointers plus changed block IDs when inputs change, not equality obtained by deleting the DCP field from the signature.
- For this K1 path, bypass the old post-build clone/copy branch once the stable binding is already in place. At best it becomes redundant self-copy; avoid making overlap/self-copy semantics part of the proof. Other models and K>1 retain their existing behavior.
- Source may already alias the persistent destination in a repeated preparation path; avoid zeroing the destination before consuming an aliasing source. Clear only the inactive tail after copying, if a consumer can read that tail. Current DCP local view slices logical rows and bounded columns, so indiscriminate full-buffer clearing is unnecessary.

## Ordering/lifetime assessment

The allocation is made during non-profile dummy initialization before capture, and persists on the proposer. Same current-stream copy→metadata tensor preparation→graph replay orders writes for the present single-worker submission path; subsequent overwrite must remain after previous use on that stream, with existing HCCL/graph dependencies retained. No CPU staging or side-stream protocol should be removed. The table is device metadata, not a new pinned CPU buffer.

K1 excludes later speculative iterations that intentionally clone/mutate tables; restricting the change avoids altering their alias assumptions. The eager fallback should still consume the table bound for that call and preserve correct values if a different pointer/branch guard rejects graph use. No failure-state or KV ownership semantics are changed by this proposal.

I found no source-supported blocker to this narrow placement fix. Its final implementation still needs direct diff review. CPU evidence should model builder-retained views: demonstrate old late binding leaves the original view, and new capture/real preparation yields the same pointer/shape/stride while different source block IDs reach the stored view. Include padding/no-padding and fallback, input read-only, capacity and alias cases. A later authorized device check must establish actual replay and output correctness; successful capture and exact eager fallback outputs are insufficient. Performance script remains parked.

## Final v5 / observer-v6 diff review

Personally verified final proposer SHA `9bd8d2fc6ad292921df6b49125de993ec68567471075480f49872c3de53a55dd` and observer SHA `6a1355c8a95c8b0e5847ddbbf7d01343cbd86c372e0389e50f9d065d7c2cfc7e`. Read `mtp_graph_candidate_v5/delta_from_v4.patch`, helper/call-site enclosing branches and `check_block_table_CPU.py`.

**No concrete source blocker found in this placement fix.** Dummy helper invocation is inside the enclosing FULL metadata branch and only at draft_index0; the persistent clone is allocated before entering that branch. Runtime invocation precedes build_draft_attn_metadata. Scoped FULL bypasses the old late copy; non-scoped models and scoped NONE retain their prior branches. Capture/runtime DCP local slices therefore reference the same target allocation, while the copy refreshes its contents.

Helper checks 2D rank, row capacity and equal columns, takes the source-row prefix and copies without modifying the source. It skips copy only when data_ptr and stride match; given the actual int32 block-table sources and equal view shape, that is the required same-view alias case. It does not clear destination before reading, so avoids the identified alias destruction. No tail clear is required by the inspected consumer, which slices logical num_reqs and bounded columns; runtime padded rows carry F.pad zeros when present. Arbitrary overlapping offset/stride aliases are not covered by this helper/test, but no such source path is established in this scoped producer. Do not generalize its alias support beyond current inputs.

The CPU test genuinely uses Torch tensors and AST of the staging helper and actual local-view function. It verifies five changing F.pad allocations, stable capture pointer/live values, padding, unchanged source, same-view alias, unpadded source, old retained-view failure and shape rejections. The reported `alias_self_copy_skipped` flag is stronger than its direct value-only assertion: the test does not instrument copy_ count, though the inspected helper condition establishes the skip for this exact alias. This is not a functional blocker.

At review time no `block_table_CPU_result.json` is present locally. The first missing-snapshot-source attempt is an infrastructure failure and must remain recorded; **Torch CPU verification is pending**, not claimed passed here. Completing the staged dependency and this existing CPU check is the remaining offline validation item. Observer source retains the prior reviewed admission-only snapshots; no pointer guard was weakened. Device stream/graph correctness and actual replay remain unproven; exact eager fallback outputs from Run272 cannot fill that gap. Matched performance remains parked.
