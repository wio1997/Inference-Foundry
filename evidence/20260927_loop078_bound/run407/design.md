# Run407 — Astra independent minimal row-identity closure design

Status: DESIGN ONLY. Source inspected read-only on 2026-09-27. No source edits, service launch or NPU work performed. This file is the sole requested output.

## Verdict

A complete source proof plus an actual runtime branch/group/Graph map **can in principle** prove router row identity without adding a token-ID collective at every layer. The inspected FlashComm1 and DSA CP transformations have explicit ordered maps; a transpose in this path is not by itself a candidate-row permutation bug.

That proof is **not complete for Run403**. Neither Run404's structural gates nor Run405's identical all-rank routes establish the active chain at all 43 routers. Python source alone does not establish opaque native operator semantics or that the replayed compiled graph matches the inspected branch. Keep Run405 retained unions conditional on the slot-major layout. Run407 should first close the source/branch certificate and add identity instrumentation only where an actual mapping remains unproved. Do not promote finite Algorithm/Resource, Hardware or Scheduling bounds.

## 1. What the inspected source supports

Framework root: /data/wio/vllm_ascend_26/framework. Paths below are relative to its vllm-ascend/vllm_ascend directory unless explicitly upstream.

| Boundary | Observed source map | Runtime premise still required |
|---|---|---|
| Target input | fixed_decode builds [last token, seven draft tokens] per slot; handoff constructs arange(96) logits indices | Capture actual input row labels and actual index tensor, not construction intent alone |
| Embedding | ops/vocab_parallel_embedding.py:_forward_origin calls maybe_pad_and_reduce; under FlashComm this returns a contiguous TP token block. _forward_embed_tp instead gathers rank-local inputs into static capacity blocks then reduce-scatters and trims | Concrete embedding forward_type, comm-group ordered ranks, capacity and input distribution; do not substitute the origin proof for embed_tp |
| FlashComm gather | ops/register_custom_ops.py:_maybe_all_gather_and_maybe_unpad_impl gathers dim0 and tail-unpads; EP/DP branch gathers padded DP blocks and copies each rank's live prefix in rank order | FlashComm, enable_sp/enable_sp_by_pass, is_ep_comm, DP metadata, pad_size and ordered group identities |
| FlashComm scatter/residual | _maybe_pad_and_reduce_impl tail-pads and reduce-scatters dim0; _maybe_chunk_residual_impl pads and takes torch.chunk(...)[tp_rank] | Identical row identities in corresponding reduced contributions, actual TP rank order, residual shape branch |
| Explicit MoE sequence parallel | Upstream model_executor/models/utils.py:sequence_parallel_chunk_impl pads and narrows from tp_rank * ceil(N/TP); model MoE's final dim0 gather trims to num_tokens | use_sequence_parallel_moe is a separate flag from FlashComm1; capture both |
| DSA CP query partition | attention/context_parallel/dsa_cp.py:_build_local_token_metadata uses local_start=tp_rank*ceil(N/TP), intersecting request query intervals with that token block | Actual num_input_tokens, local_start/end, metadata implementation/branch and query order |
| DSA CP head redistribution | _restore_tp_head_layout views [L,TP,H,D], permutes to [TP,L,H,D], sends head chunks through all_to_all_single, receives source-rank token blocks | Actual TP group order, L/head dimensions and skip_all_to_all branch |
| Attention output projection | Normal wo_b uses TP head reduction/scatter; full-weight branch skips all-to-all and calls quant_method.apply without the normal TP reduction | full_gather_wo_a_enabled, attn_state, concrete linear implementation and mmrs_fusion branch |
| MoE prepare/finalize | PrepareAndFinalizeWithAllGather chooses EP or DP preparation; hidden and router logits gather with corresponding layout, optional PCP padding/gather; finalization performs corresponding reduce/scatter/trim | Concrete prepare class, enable_sp/pass, actual DP/PCP sizes, pad and metadata; AllGather dispatcher alone does not prove prepare class |
| Expert sorting | TokenDispatcherWithAllGather uses npu_moe_init_routing and carries expanded_row_idx to npu_moe_token_unpermute | Native inverse-map contract and concrete dispatched implementation; route capture occurs before expert sorting |
| Final output | Model MTP buffer and handoff gather dim0 then unpad; adapter selects target_logits_indices | Actual selected indices, output shape, group order, same graph entry/cycle association |

BaseDeviceCommunicator implements dim0 all_gather as group-rank concatenation and reduce_scatter as contiguous group-rank blocks. NPU communicator inherits these implementations. Async all_gather uses the same rank-concatenation contract. Record group membership **in collective rank order**; identical member sets are insufficient.

### DSA CP algebra: why the transpose can preserve token identity

For the frozen N=96, TP=8 case, let L=12 only after verifying the active branch's padded length. A local query row on source rank r has label 12r+u, 0<=u<12. The send buffer's leading axis selects destination head block d; each block contains those same 12 token rows. All-to-all concatenates source-rank blocks at destination d. Its output therefore has labels 0..95 in order, with head block d. Normal wo_b reduces matching rows across heads and scatters back to the local contiguous token block.

The full-weight skip-all-to-all branch instead keeps local token rows and bypasses the normal scatter. This is a different proof branch, not evidence of a reorder. The Python predicate is TP>1, enable_dsa_cp_with_o_proj_tp and attention state outside DecodeOnly/SpecDecoding; capture its actual value and output-projection path.

This argument assumes the local attention output follows its query-row order, consistent head dimensions, and matching collective group order. Those premises need evidence. Native TND sparse attention, npu_transpose_batchmatmul, HC pre/post, fused mm-reduce-scatter and token unpermute need pinned operator contracts or targeted semantic checks. A plausible operator name does not complete the proof.

## 2. Minimal first stage: branch certificate without extra device collectives

Collect once at graph capture, then bind to every selected replay:

1. Full source/extension provenance: source SHAs for the complete active call chain, package/build identifiers for native/custom ops, compile options, registered implementation classes and effective custom-op dispatch. Read the live container's mounted paths; host copies alone are not sufficient provenance for a future run.
2. Per rank and per layer 0..42: concrete attention, linear, MoE prepare/finalize and dispatcher classes; input/output shapes and strides; FlashComm1, sequence-parallel MoE, SP/pass, DSA CP, attention state, full-weight/skip-A2A, mmrs-fusion and fused-op branches; TP/EP/DP/PCP group ordered global ranks, rank-in-group, padded/live lengths and DP prefix metadata. Enumerate all layers even if a common descriptor/hash can compress them.
3. Graph certificate: graph/cache key, capture generation, static input/output/route-buffer association and source descriptor hash. Record whether each custom op is opaque or decomposed by compilation and the relevant compiled graph/operator sequence. A Python hook that runs only during capture cannot claim a per-replay observation without this binding.
4. Selected replay certificate: rank, cohort/request-slot identities, cycle, graph key/generation, active/padded slot mask, actual positions/input IDs, actual target_logits_indices and the existing route-buffer snapshot generation. Copy changing tensors at the same selected replay boundary; do not pair capture-time values with replay-time routes.
5. Offline compose row maps, starting at the real Target input and ending at each pre-dispatch router and selected output. Prove by induction that each row-preserving operation keeps its incoming map, and each layout operation applies its actual map. Check residual branches agree before addition. Verify query labels, not KV labels, across attention.

A verified certificate can eliminate redundant device labels for static chunk/gather maps. It cannot establish an undocumented native behavior. If native semantics or graph dispatch remain unresolved, label the result a conditional source proof and proceed to the targeted fallback below.

## 3. Required fallback sites and identity rules

Seed candidate_row_id **once**, at the actual Target input: t=8*slot+offset for this shape, joined to cohort, request ID and cycle. Save position and token value as attributes, not as unique identity: token values and positions can repeat. Padding identity is -1. If input rows differ from this order, seed the real request/offset mapping rather than forcing the formula.

Carry or reconstruct identity from the **same real indices, slice bounds, permutation metadata and group order used by the data transform**. Fresh arange(96) at each router is invalid. A parallel guessed map can reproduce the assumption without testing it.

| Site when its proof is unresolved | Minimum added evidence |
|---|---|
| Input embedding/local partition | Input labels and actual split/gather capacity/block boundaries; label vector at first local hidden input |
| DSA local metadata/query | Local query labels with local_start/end and actual request query boundaries; sparse-attention output/query-order contract |
| DSA head all-to-all/output projection | Input/output label maps using the actual head/token reshape and collective source order; branch descriptor. Feature head chunks of an output token must agree on token label |
| MoE preparation | Labels before prepare and at the **exact hidden/router-logit rows consumed by W4A8 select_experts**, for all 43 layers, paired with the same topk_ids buffer |
| Expert permutation/combine | Actual expanded_row_idx and needed native routing metadata; verify the inverse mapping restores the incoming labels before the next layer. Capture post-finalize labels if the inverse/collective chain is not proven |
| Final hidden gather/index selection | Output labels after gather/unpad and after the actual target_logits_indices selection |

Labels follow query identity through attention, not the mixture of KV content. Labels do not undergo GEMM, layernorm, HC arithmetic, or sum-reduction. At reduce-scatter, first prove that all contributions for a row represent the same identity; then select the output block. Summing integer labels is incorrect. Expert replication makes multiple expert copies of one token: propagate identity through actual routing indices and verify inverse placement, not arithmetic expert aggregation.

Prefer deriving collective label maps offline from the real partition metadata and all8 descriptors. If this cannot close a branch, device shadow labels must travel through the actual relevant index transformation, with explicit dependencies on its metadata. A separate shadow collective tests a convention but does not prove that hidden data took the same path; link it to the real graph operation and actual buffers. Do not add a label collective solely to restate a known rank-order rule.

A native op with neither trustworthy ordering contract nor exposed row map needs a focused semantic test against a reference or an operator-level identity trace. Simply bypassing that op for labels does not validate its implementation. Synthetic token-dependent query inputs can test an attention ordering contract; raw token IDs injected as ordinary hidden values change the model and are not an acceptable production route comparison.

## 4. Lowest-perturbation execution and validation gate

This document authorizes no execution. For the later approved run:

- Reuse the clean workload and selected cycles64/65. One complete 12-request cohort across all8 ranks is the minimal identity diagnostic; it is not automatically a replacement for Run403's five-cohort attribution. To produce a comparable retained-route dataset, retain exact60 POSTs, c12, 48+12 clients and all five cohorts.
- Preallocate small integer/metadata snapshot buffers. Static maps and branches can be saved once per graph entry; only changing replay values need device copies. Do not add .item(), CPU tensor reads, synchronize(), barriers, or new waits inside the serving/graph hot path.
- Snapshot immediately on the producing stream or using an existing proven dependency after replay; protect snapshot lifetime until export. Export after the measured cohort or at an existing safe completion boundary. Do not silently force graph serialization to make the trace convenient.
- Verify all8 source/build hashes, one shared workload trajectory, FULL graph path, zero unexpected post-handoff ModelRunner/oracle calls, correct graph generation and route pointer/snapshot lifetime. Capture actual Runtime request IDs and slot mappings. Exact60 POSTs, max12 concurrent client requests and server outstanding requests <=12 are separate gates for the full run; reject extra clients or leftover requests.
- Baseline A/A establishes normal variability, then an instrumented diagnostic B with identical fixed inputs/config. Compare successful 1024-token outputs, acceptance/count trajectories, input/route IDs and group-list parity. If nondeterminism prevents exact trajectory parity, report it and compare independently aligned records; never assert equivalence from shape alone. Compare throughput/cycle timing to A/A variability to quantify diagnostic overhead, without promoting B timing to Current.
- Avoid requesting a full benchmark merely to verify static source composition. First run CPU/offline map checks on the recorded descriptor: padding case, nontrivial group-rank order, both DSA branches, inverse routing indices and residual joins. A deliberate common row permutation must be detected or correctly reindexed; a validator that still blesses slot-major in that case is tautological.

Disqualify a closure claim for missing layers/ranks, ambiguous graph binding, stale buffers, unrecorded dynamic gather/index, unknown native ordering contract, duplicate/lost non-padding labels at an identity-preserving boundary, misaligned hidden/router-logit rows, or wrong residual identity. A measured nonidentity permutation is not automatically a model bug: retain its map and recompute row attribution by labels. It does disqualify the unconditional slot-major assumption.

## 5. Deliverable and permitted Bound promotion

Deliver source/build manifest, per-layer branch/graph certificate, composed row maps, unresolved-contract list, raw changing identity metadata, an independent validator and retained unions recomputed by token identity. Report one of:

- **Source-and-runtime identity closed**, listing the trusted native contracts explicitly; or
- **Instrumented identity closed** for named ranks/layers/cycles and the validated propagation chain; or
- **Conditional layout only**, with exact missing boundaries.

Even successful closure upgrades only sampled retained token-to-router attribution for this executed algorithm. It does not prove that retained expert storage bytes are compulsory HBM traffic, that rejected rows can be removed online, that all Draft body work is removable, or that a reduced schedule is executable. All finite bound endpoints remain null.

## Inspected source SHA256 snapshot

This is a read-only host-source snapshot for this design, not a future live-runtime attestation.

```
11dd3e983dc1d628bf42a95581f9617861471b1f658739db44c20df886147247 vllm-ascend/vllm_ascend/models/deepseek_v4.py
ab93899f54889999cd767b0db01334ad46f20e29fca0d96ce0fee31538fe2a52 vllm-ascend/vllm_ascend/ops/register_custom_ops.py
a4a8517abc3a4f8494f27cc4cc3a876264ef8fea8cfc6de9abdbaf75d42c3b24 vllm-ascend/vllm_ascend/ops/vocab_parallel_embedding.py
3abeb11a3d82f3919d67f4f7c5ab404faa476c2eb8455d2eca777dc1712811fc vllm-ascend/vllm_ascend/ops/dsa.py
599f85b8dd4c13f72aa458e21ae7fa13e18548a645fd03b8a534a6b951e7ee02 vllm-ascend/vllm_ascend/ops/linear_op.py
27cbbf92a02f16301aad2a35091e258b8459dc77744bd8294f2f11a4c126004e vllm-ascend/vllm_ascend/attention/context_parallel/dsa_cp.py
c467d0ed0a0d41525aa90008aafbfd26bdbb718b0e29db7de2bed9abe09cabdd vllm-ascend/vllm_ascend/ops/fused_moe/prepare_finalize.py
80c7061b59d4593f55ce5c18d601bdc7b1831bfe9c268c5bc8129d733d09c215 vllm-ascend/vllm_ascend/ops/fused_moe/token_dispatcher.py
b24da9cc65bbf7e2dbf6bfaee403435d88bbe125a9bd589e319988fc7be143a6 vllm-ascend/vllm_ascend/ops/fused_moe/fused_moe.py
9febce077840c33eeab9bc86c2b06629c2a90dc219b3cbee71656255ad82ba8d vllm-ascend/vllm_ascend/distributed/utils.py
f06d1a1a8d92e6ab39ebf7931d3eb3d6688d212606b1c409d7bed8eed5ba1fc2 vllm/vllm/model_executor/models/utils.py
c4fafc71bbb3a7652ecdf425bf116e3a9ad203ba44baf005cbab5db88a8b8221 vllm/vllm/distributed/device_communicators/base_device_communicator.py
```

Confidence: high in the explicit chunk/gather/A2A algebra and limitations; medium in coverage of the full native/compiled runtime chain, which is precisely the unresolved Run407 gate.
