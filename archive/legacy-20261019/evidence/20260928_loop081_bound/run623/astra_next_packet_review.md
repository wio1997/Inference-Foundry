# Run623 Astra High source-only next-packet gate

Repository HEAD at review: `9e939cc89c0eba5d97643085d4f0afbc3e7e5e80`. **DESIGN recommendation only; NOT LIVE-READY.** No patch, service or NPU experiment. Reuse Run614 gates, Run621/622 geometry and V3.42; do not repeat their measurements.

## Exact source inputs

All paths below are under `/data/wio/vllm_ascend_26/framework/vllm-ascend/` unless noted.

| Source | SHA256 |
|---|---|
| vllm_ascend/models/deepseek_v4_dspark.py | b459fa373e89da1722085cffa63a9959502dd29fb1a0cf4b8f073e431d811867 |
| vllm_ascend/attention/dsa_v1.py | 88a0cd69dbbba476d0ac179431568590e6e876134390fba77f22721069f491cb |
| vllm_ascend/device/device_op.py | 67dda24f28de847f9aee87db9714e5d2293ebb46309ad13152538cf63ca5f1b4 |
| vllm_ascend/spec_decode/dspark_proposer.py | e9163996db794c5fba777ea55a5374283780a2fd3404778f5369caea76f9344f |
| vllm_ascend/spec_decode/dflash_proposer.py | 6be46559f9f869861c676efb4531b0c01eef8a0445521ecadf5c1fec1c4412f8 |
| csrc/torch_binding.cpp | 20e1532b72600794ee3b5a39dbc8d6bd2eb89193e961664f5f47282a3028f2ce |
| Foundry bootstrap/vllm_dspark_handoff.py | fe7039361894acd70666f72c3089efd635404f1fe9cbc685db21272d946a317e |
| Foundry run614/typed_kv_packet_gate.json | 73ed975c9bab9eb47ea877741f223e62e9eda1de94c56d76010ef259c3d85f79 |

## Minimum hook addresses and dynamic arguments

1. **Context writer:** `deepseek_v4_dspark.py::_store_standard_swa_kv` around192–203, immediately around `DeviceOperator.dsa_kv_compress_scatter(swa_kv_cache, shared_kv, slot_mapping)` **after** 1D→2D formatting. Carry layer/prefix role from `precompute_and_store_context_kv` loop; context lineage originates `dflash_proposer.py::build_model_inputs_first_pass` around255–271. Record actual cache, update tensor and final slot tensor descriptors, update row count, source positions/request mapping and ownership token.
2. **Query writer:** `dsa_v1.py::_forward_prefill`2086; `_forward_decode`2414; enabled multistream prolog writes separately at1946. Record actual branch and `multistream_dsv4_dsa_overlap`; a hook covering only2086 misses the alternative. The common non-A5 sink `device_op.py:710–712` calls `torch.ops._C_ascend.npu_scatter_nd_update_sk(cache,slot_mapping,x)`. Use this as the operand boundary with an explicit context/query ContextVar tag; never infer role from ordinal. Untagged writes to a watched cache invalidate completeness.
3. **Reader:** `dsa_v1.py` comp_ratio<=1 prefill `return attn_op(...)` around2096–2113 and decode `attn_output=attn_op(...)` around2538–2557. Capture the **final actual keyword dictionary**, not handoff common metadata: q, ori_kv, optional cmp_kv/indices, ori_sparse_indices presence/content, ori_block_table, cu_seqlens_q, optional cu_seqlens_ori_kv/seqused_q, seqused_kv, sinks, sas metadata, cmp_ratio, masks, windows, layouts and return_softmax_lse. Also capture op implementation identity, layer, branch, stream and exact tensor views.
4. Non-A5 selector is `device_op.py::get_dsa_sparse_attn_op`→`_C_ascend.npu_sparse_attn_sharedkv`. The binding at `torch_binding.cpp:1096–1128` supplies actual ori_kv.stride(0) to ACLNN. Pin this source but do not claim it is the loaded eager kernel without same-acquisition image/tiling identity.

**Run621 limitation:** it captures before handoff common refresh/prepare_inputs. Its common96-element flat slot view is not the formatted scatter ABI: non-A5 formatting at device_op.py844–847 yields [block_idx,offset]. A -1 flat slot becomes an invalid block/offset pair; preserve actual values and validity, never reinterpret every offset as a valid write.

**Branch distinction:** comp_ratio<=1 alone does not identify the read domain. DSpark noncausal decode can pass `dspark_swa_indices` built at dsa_v1.py437ff/1424ff. The source sparse path reads a bounded prefix until first negative, with physical slot IDs; do not replace it with the dense sliding-window formula or block-table remap. Prefill adds cu_seqlens_ori_kv=actual_seq_lengths_query through extra kwargs; decode does not add that default. Actual sas_metadata controls execution coverage. Run619 pure helper is reusable only after its explicit ABI/range preconditions and loaded-path limits are met.

## Smallest useful same-acquisition packet

Keep all8 cohort5 cycles64/65 plus existing predecessor/successor lineage. Tag all context/query writers and readers for layers43–45, but retain only one predeclared candidate context source row per layer/cycle and one predeclared request/query reader. A fixed source row/request may be invalid or not consumed: report NO_WITNESS, do not alter acceptance or choose on Host after synchronizing.

- At each actual writer, asynchronously snapshot **all final index pairs for that small call** to distinguish duplicate destinations and intervening overwrites. Geometry/caps must be preflighted; do not assume96 from Run621. At reader, snapshot actual query prefix lengths/seq length and selected branch data before mutation. Sparse case: selected q row of indices including the complete relevant prefix/terminator, with source width cap2048; dense case: only required block-table columns with column identities plus exact lengths/window. Capture full small1024-int32 sas metadata if that is the actual ABI object. Sparse membership does not require copying the whole block table.
- Selected payload checkpoints: immediately after context writer and immediately before its candidate reader. Six selected row instances × two1024-byte snapshots = **12,288 bytes/rank**. This replaces ambiguous “pre/post” wording with exact endpoints. An additional pre-context old-row snapshot for later freshness analysis costs another6144 bytes/rank; context input/projection outputs are extra and need actual dimensions. Metadata, index arrays, tags, allocator rounding and workspace are separately bounded.
- Preallocate scratch; enqueue copies/gathers on the **existing owner stream**, after producer/before later overwrite. No .item/.cpu to select rows, no new wait/event dependency across production streams, no device-wide synchronization, no added model/HCCL call. Any device-side selection must be bounds-safe with an exported validity flag; clamped diagnostic gather must never alter production indices. Export after existing drain. Even async observer kernels perturb scheduling; durations remain diagnostic.
- Freeze enqueue token {run,rank,cohort,cycle,layer,context/query/read,ordinal}; join exact same-acquisition torch_to_npu start/finish flow and native stream/task. Do not borrow Run610/611 flow or Host-time binning.

## Generation admission and fail-closed limits

A host counter is an **enqueue write token**, not storage/content generation by itself. Certify a selected last-writer only if (a) actual cache byte interval/lifetime matches, (b) writer destination is valid and unambiguous, (c) all intervening writers to that interval are covered, (d) native order is established by same-stream FIFO or already existing recorded dependencies, and (e) the chosen actual reader has that row in its eligible domain and execution coverage. Duplicate unordered scatter destinations are UNKNOWN. Equal snapshot values do not prove writer identity; legal query overwrite changes the last-writer and may yield NO_CONTEXT_WITNESS while packet identity still passes.

Run621 repeated ids/pointers are not a lifetime certificate. Use the production cache owner/registration and exact tensor identity, audit rebinding/storage-resize paths over the bounded window, and reject any ownership ambiguity. Do not manufacture continuity by retaining a tensor beyond production lifetime. If source coverage cannot exclude deallocation/rebinding or untagged alias writes, allocation/native-write evidence is still missing and the result stays conditional.

Admission outcomes must separate: descriptor/flow packet PASS; selected current writer→eligible-reader witness PASS/NO_WITNESS/UNKNOWN; physical-first-read UNKNOWN absent loaded implementation proof; fresh unavoidable arithmetic UNKNOWN absent entry-credit proof. No witness is not proof of independence. A demonstrated current storage hazard is not proof every legal runtime must serialize this way: versioning, renaming or prefetch may remove the storage dependency. Conversely value provenance can establish a true producer requirement, while independent operations may move around it.

## Priority challenge: Resource/Hardware versus this packet

This is the smallest presently justified Scheduling acquisition, but its likely gain is **one dependency certificate**, not a numerical whole-system interval. Do not serially expand toy row observers indefinitely or label one edge a critical-path lower bound.

Run614 Resource review still identifies two larger unknowns: architecture-wide compulsory work/traffic after entry/reuse credit, and an applicable exact-board cumulative capacity envelope C_plus/B. A new authoritative applicable board/clock/interface certificate would have higher strict-Hardware value and should preempt this packet if obtainable. Existing Run462/590 searches are already exhausted without new access; another ordinary peak microbenchmark cannot certify an upper service cap and should not displace this packet on that premise.

For the user's **engineering achievable interval**, explicitly separate feasibility calibration from strict ceilings: use existing whole-pass basis and measured kernel/resource service in a resource-constrained DAG with documented reuse/overlap assumptions, then identify the largest interval sensitivity. A matched sustained fixed-shape mixed compute/HBM/HCCL service experiment is worth priority only when that sensitivity analysis identifies an unmeasured capacity/interference cell and prior evidence does not already cover it. Source-only review cannot establish that such a specific unmeasured cell dominates now. No new per-kernel scan or speculative hardware number is recommended.

Historical priors checked on demand: PK-001/R20 trace overlap not fully reaching E2E, PK-002/R21/R28 cross-stream contention, PK-088 exact HCCS inventory versus freshness/capacity limits. They constrain evidence interpretation, not architecture or current KEEP/REJECT.

**Strict Resource/Scheduling/Product endpoints and numerical gap remain null; Formal Current571.681 unchanged.** Proceed to a bounded implementation preflight only after concrete metadata/selection/lifetime/flow budgets exist; this source review is not live approval.
