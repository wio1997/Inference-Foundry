# Current-FULL local MC2 preparation review

2026-10-07. Offline only; no candidate installation or Run. H12 Run280 remains INCONCLUSIVE/off; H6/H5/targetFULL retained.

Read new local_prepare patch/candidate, exact54e original, routed_experts prepare→routing→quant→finalize consumers, token_dispatcher's x_active_mask use, actual ACLGraphWrapper, base BreakableCUDAGraphWrapper and Ascend subclass, runner load_model wrapper installation, and rank15 period3 CSV/task join.

## Semantics

No concrete source blocker found in the local-row transformation for the existing valid two-dimensional hidden/router input contract. For target padded length L divisible by TP, rank r receives exactly rows[r*(L/TP):(r+1)*(L/TP)] of the globally padded tensor. The helper computes that intersection; all-empty peers get fresh zeros, partially present shards pad only their valid prefix, full shards clone. That preserves zero router logits for masked peers and input isolation rather than aliasing caller hidden state.

Mask selection uses the quotient/remainder rule matching tensor_split, independently of the fast gate. Global padded_hidden_states_shape remains L×H, self.num_tokens remains original n; finalize still allocates a fresh global gather target then trims to n. No collective, mask bit, group or token order changes. SP/replace_allreduce remains its old branch. Non-divisible/no-padding/TP1 paths preserve old geometry; replacing tensor_split with narrow is a view-selection implementation change whose exact empty/noncontiguous cases belong in the current oracle.

Fresh new_zeros avoids sharing a mutable global zero tensor. In FULL capture it becomes a graph-owned allocation plus captured zero-fill work: each replay must actually reset it before dispatch, not rely on capture's initial zero contents. Normal native allocation/fill capture semantics make this plausible, but CPU3888 from the old implementation does not prove the new native zero-fill path, its layout, or dynamic graph lifetime. Existing correct kernels suffice; no kernel change is proposed.

Routing still examines local zero logits before MC2 consumes x_active_mask, just as before. Do not skip routing/dispatch for empty peers: collective participation is required. Do not replace clone by an alias on valid peers; original global padding made input independent and finalize explicitly documents shared-expert corruption risk from reused storage.

## Evidence and priority

Current trace proves repeated device work in the target graph, not old eager launch starvation. Rank15 exported task396389 is a static captured MemSet with Model49; geometry reports global16-row materialization although each TP peer uses one row. Source makes unnecessary global materialization concrete. It is reasonable to advance **one bounded native correctness check after the new CPU oracle**, provided execution/graph comparison design is first settled. Old Pad+MemSet inclusive3.66–4.05ms is not a saving estimate: candidate still needs fresh allocations/zero-fill or clone, and the rank determining completion can change.

## Graph-bank finding — blocker to a naive same-worker A/B

The actual current runner installs one BreakableACLGraphWrapper around target. Base wrapper entries are keyed only by BatchDescriptor (lines285,326–329), with one global graph pool; Ascend subclass additionally uses module-global graph parameter/task/workspace registries. Plain ACLGraphWrapper similarly has one concrete entry per BatchDescriptor. No source-read supported A/B bank identity or candidate selector exists. The existing capture has already recorded the old target prepare kernels: switching a Python prepare implementation during replay changes only uncaptured/eager work (including MTP), so it is an invalid target comparison.

Two Python dictionaries alone are insufficient to establish safe banks: capture output/workspace pools, attention task groups, update metadata registries, shared model buffers and graph-held addresses must be isolated or deliberately shared under a proved protocol. clear_graphs+recapture is also a lifecycle operation, not the existing same-worker mode toggle. I did not find a ready supported two-bank comparison API. Thus an honest native correctness test requires capture with the candidate installed; an honest performance comparison currently requires separate controlled reload/capture epochs **or** an independently reviewed bank implementation. Do not silently activate such an architectural extension or propose a large Run from this review.

## Smallest next evidence / stop list

Finish CPU byte/layout/input-nonmutation oracle for new_zeros, uneven/empty slices and fallback, using the exact prepare consumer geometry. Then decide whether the bounded reload/capture correctness cost is justified; there is source support for that decision, but no device/gain claim. Keep A/B design unresolved rather than pretending existing toggle support. Stop repeated H12 pairing, raw inclusive-to-saving conversion, no-op Python switches under FULL, removal of collectives on empty ranks, and immutable zero reuse. Largest realized removable gain remains unknown.
