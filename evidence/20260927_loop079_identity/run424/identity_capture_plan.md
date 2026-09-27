# Run424 — minimal all43 Target row-identity certificate

Read-only design review, 2026-09-27. Only this document is written. No borrowed
source changes, service interaction, framework import or NPU operation occurred.
Sources inspected on host; future loaded-process/mounted-source attestation is
still required. This review builds on Run403 capture, Run407 and Run420.

## Verdict and stop point

One frozen diagnostic can close the **actual branch, graph, buffer, query and
logit-index certificate** with capture-time Host metadata plus small selected-
cycle snapshots. That is the minimum useful next measurement. It cannot, by
itself, prove every opaque native operator preserves query order. Prepare a
native-contract checklist before execution and report `CONDITIONAL_NATIVE`
where a link is not established. Do not rename the branch certificate an
unconditional all43 identity proof.

The first promotion gate is: for every selected rank/cohort/cycle and all43
routers, one composed map from actual request-generation/slot/query-offset
labels reaches the exact W4A8 topk_ids rows, with residual/head contributions
consistent, and the final logits/acceptance indexing correctly related to that
map. Stop at the first unsupported branch, missing graph binding or native
contract; save its exact boundary and choose one targeted semantic test. Do not
add43 independent arange vectors or redundant label collectives.

## What Run403 actually contains

Inspected `run403/capture/rank0_cohort4.json`, cycle64: one replay entry with
`BatchDescriptor(num_tokens=96, num_reqs=12, uniform=True, has_lora=False,
num_active_loras=0)` and43 named Target layers. Route/group pointers and
process-local graph entry identity are recorded. Target positions/input IDs,
active mask and raw acceptance counts are present. **target_logits_indices,
query_start_loc, request IDs and per-layer branch/group maps are absent.**
The file says cohort-local slot only. Identical routes/histograms across ranks
cannot distinguish a common row permutation. Configured initial_output_counts
are zero; Run420 explains why this does not certify external pre-handoff zero.

Do not retroactively assign current source/build or a newly observed branch to
old Run403 processes. Reuse its capture mechanism, not an inferred historical
row map. Keep Run405 retained-union numbers conditional until joined evidence
exists. External delivery/remaining-output proof is a separate Run420 ledger
join even if the row certificate succeeds.

## Current source facts that constrain instrumentation

Framework root: `/data/wio/vllm_ascend_26/framework/vllm-ascend/vllm_ascend`.

- `compilation/acl_graph.py:135–151`: actual cache lookup is wrapper-local
  `concrete_aclgraph_entries[forward_context.batch_descriptor]`; entry is created
  only on a missing descriptor. Capture executes `self.runnable` inside
  `torch.npu.graph` at189; replay invokes `entry.aclgraph.replay()` at263.
  `runtime_mode` mismatch may bypass the wrapper. Capture generation, wrapper
  identity and concrete entry identity are necessary, in addition to repr(key).
- The existing entry `input_addresses` at163 and debug replay comparison at242
  inspect **positional args only**. Direct Target calls input_ids/positions as
  keyword arguments. An empty positional-address list is not an input-binding
  certificate: explicitly record relevant kwargs and Runtime pointers.
- `runtime/target_adapter.py:76` performs
  `sample_hidden = hidden_states[state.target_logits_indices]` immediately before
  compute_logits. ModelRunner's handoff construction at2314 uses arange(96), but
  its real state tensor must be captured. Its value is not proved by shape or
  constructor inspection. The generic SpecDecode target_logits_indices near1409
  is a different tensor/path; do not instrument that name by text alone.
- `ops/vocab_parallel_embedding.py:165–246` has two different paths. Origin ends
  in maybe_pad_and_reduce. embed_tp all-gathers per-rank capacity-sized input
  blocks through static buffers, performs vocab-sharded embedding, reduce-
  scatters and returns each rank's live prefix. Capture forward_type, capacity,
  actual input distribution and embedding comm-group order. These maps are not
  interchangeable merely because both return96-shaped tensors.
- `ops/register_custom_ops.py:22–104`: residual chunking depends on actual x and
  residual row counts. Gather depends on FlashComm/label/is_ep_comm, dp_metadata
  presence and enable_sp_by_pass; pad/reduce has corresponding TP or EP branches.
  DP metadata presence must be recorded even under DP1. Record actual prefix
  lengths/padded_length; equal total rows alone is insufficient.
- `models/deepseek_v4.py:377,471,505` separately controls explicit sequence-
  parallel MoE chunk/gather via use_sequence_parallel_moe. Do not infer that
  value from FlashComm1.
- `attention/context_parallel/dsa_cp.py:1326` chooses full-weight/skip-A2A iff
  TP>1, enable_dsa_cp_with_o_proj_tp, and attention state is outside DecodeOnly/
  SpecDecoding. `_restore_tp_head_layout:1645–1675` otherwise views local rows as
  [L,TP,H,D], permutes to [TP,L,H,D] and all-to-alls over actual TP group. The
  normal and full-weight paths have different row maps and wo_b reduction paths.
- `runtime/target_metadata.py` is the product updater on the frozen native-
  metadata path. It constructs local query prefixes by intersecting the global
  query interval with [tp_rank*ceil(N/TP), end), then updates actual CPGroupBinding
  buffers. Generic DSA builder capture-time values do not establish selected-
  replay geometry. Capture the product updater's actual bindings/commit path.
- `ops/fused_moe/prepare_finalize.py:363` chooses EP preparation when enable_sp
  or enable_sp_by_pass; otherwise DP preparation. EP preparation separately
  gathers hidden rows and router logits; PCP>1 adds another pad/gather. Record
  prepare/finalize classes and actual branch, not just AllGather dispatcher.
- Route IDs originate in W4A8 select_experts (w4a8.py:539). Expert dispatch then
  permutes rows; AllGather token_dispatcher carries expanded_row_idx to native
  unpermute. Correct router attribution at layer0 does not prove layer1 if this
  inverse-map link remains opaque. HC pre/post is another native identity link.

## Minimum Host metadata: once per concrete captured graph

Reuse Run403's graph_begin/end and replay tracking, extended as follows. These
are Host descriptions of tensor metadata; no tensor.item/cpu/tolist is allowed
inside capture or selected execution.

1. Provenance: run/rank/process-start ID, mounted file SHA and loaded module
   path/version, build IDs/SHA of invoked custom/native libraries, compilation
   mode/options and callable/registered implementation identities. Pin only the
   active chain; do not dump the whole environment.
2. Graph key: wrapper ID + concrete entry ID + monotonic capture-generation ID,
   structured BatchDescriptor fields and runtime_mode, graph object ID, runnable
   implementation/cache artifact digest when available. Record concrete capture
   operation inventory or explicitly mark compiled/native dispatch unresolved.
3. Input/output binding: input_ids and positions kwargs, hidden output, logits
   selection tensor, and each route/expanded-index buffer: pointer, storage base/
   byte interval, offset, shape, stride, dtype and device. Record alias edges.
   Capture-time snapshots cannot be inferred from Python object retention alone.
4. One ordered descriptor per layer0..42 (shared descriptors may be interned,
   but all43 layer references must exist): actual embedding/attention/linear/
   prepare/finalize/dispatcher class and selected method; FlashComm1, enable_sp,
   enable_sp_by_pass, explicit SP-MoE, DP metadata presence, is_ep_comm/label,
   live/padded lengths, PCP branch, attention state, CP mode, full-weight/skip-A2A,
   output projection/mmrs fusion branch; row-relevant before/after tensor views.
5. Ordered global rank lists and rank-in-group for **each actual collective
   handle**, including embedding comm_group, TP, EP, DP and PCP. Equal rank sets
   are insufficient. Record local row start/end, token/head axes and head counts.
6. Residual/HC joins: identify incoming branch buffer/map IDs and actual first-
   dimension sizes. At each merge the two branches must denote the same query
   labels before combining feature channels. Use no arithmetic on labels.
7. Request handoff: actual req_ids in slot order plus request-generation identity,
   run/cohort and Scheduler transaction join key, from ModelRunner's existing
   `_extreme_req_ids`/Runtime report path. All8 agree on order. The row identity
   seed is this tuple, not a token value, position or freshly invented request ID.

Group descriptors are collected once per group; per-layer pointer/branch rows
reference them. Static shape/layout descriptors do not require device copies.
A layer hook that is compiled away or sees a fake tensor does not count as an
actual graph-capture observation. Missing all43 coverage is a hard stop.

## Minimum selected-cycle device delta to Run403

Use the same cycles64/65, producer stream and existing completion boundaries,
export after the cohort. Prefer preallocated snapshots. Existing IDs/positions,
active masks, topk IDs, group counts and acceptance records are retained.

| New values | Why / exact timing |
|---|---|
| actual `state.target_logits_indices` (96 int64) | Snapshot at the adapter's actual indexing boundary, associated with its input/output buffer bindings and current graph generation. Capturing only after a later reuse is invalid. |
| actual `state.target_query_start_loc` (13 integers), target_seq_lens (12), slot_mapping (96) | Snapshot after selected Target metadata prepare/commit and before forward. Query prefixes identify slot boundaries; seq/slot values support request ownership joins. Slot mapping alone is not a query identity proof. |
| actual bound CP local_query_start_loc, input_positions, start_pos, local_seq_lens | One snapshot per distinct binding, deduplicated by storage interval; do not duplicate per layer when storage aliases. Read the buffers consumed by this graph, after the product updater commit, and record local_start/end/tokens_per_rank as Host metadata. |
| native expanded_row_idx only at unresolved expert inverse links | Optional conditional extension: retain graph-owned pointers, stage actual indices for all distinct affected layers immediately after selected replay. Actual shape/dtype/encoding must be pinned, not presumed. This allows testing routing inversion but cannot replace its native contract. |

The mandatory extra general tensors are roughly1–2KB/rank/cycle before distinct
CP bindings (logit indices alone768B); no hidden-state/logit/KV bulk dump is
needed. If expanded-row indices are [96,6] int32,43 layers add99,072B/rank/cycle,
but this is a conditional size calculation, not an observed ABI. Do not add that
payload until its decoding/validator is specified. No new label collective is
required for static ordered gather/chunk maps.

At selected entry freeze `(run, rank, cohort, entry cycle)`. Record the replayed
entry/generation and descriptor digest **before replay**; after replay snapshot
only route buffers registered to those exact entries. A graph recapture requires
a new generation and a fresh descriptor; never select by registration order.
Keep strong references until staging completes and preserve same-stream ordering
before the next replay. If a producer uses another stream, use only an existing
proven join; do not add a diagnostic wait to make this certificate convenient.

## Offline composition and decisive tests

Seed once from actual query boundaries and request order: label=(request
 generation, slot, candidate offset), with position/input token as attributes.
For actual eight-row slots this is t=8*slot+offset; padding carries -1. For N96/
TP8, local12-row blocks cut across requests: rank1 row0 is global12, slot1 offset4.
Do not substitute local request ordinal for global row identity.

Compose maps using the recorded ordered ranks, actual prefixes/slices, pads,
head transpose and actual gather/index metadata. At reductions verify equal row
labels across all contributions, then select the destination block; never sum
labels. For each of43 exact W4A8 rows pair the composed label with its same-buffer
expert IDs. Continue through actual routing inverse and finalize to the next
layer. Finally apply **actual target_logits_indices** to the final hidden map
and prove its relationship to acceptance's [B,8] view and prediction offsets.
A nonidentity map is not automatically a model bug, but it forbids assuming an
unconditional retained prefix; remapping requires the complete acceptance link.

Required CPU negative cases before service: nontrivial group-rank order;
DP1-with-dp_metadata versus absent metadata; unequal DP prefixes with equal total;
nonzero padding; rank block splitting a request; both DSA A2A/full-weight branches;
common row permutation shared across all ranks; logits-index permutation; stale
capture generation; mismatched hidden/router-logit gather; and unequal residual
branch identities. A validator that blesses the common permutation as slot-major
has merely repeated the assumption.

## Native/compiled contracts still outside this minimal capture

The following need independently pinned operator contracts or targeted tests:

- F.embedding/vocab masking, quant GEMM and fused mm-reduce-scatter preserve the
  specified token axis while combining features/shards.
- Native HC pre/post, transpose-batchmatmul and partial rotary transform feature
  dimensions without silently permuting the query axis.
- Sparse/TND attention returns outputs in the actual query order described by
  supplied CP/QLI/SAS metadata; capturing metadata bytes alone does not validate
  native interpretation or async update binding.
- npu_moe_init_routing's expanded_row_idx encoding, global/local expert filters
  and npu_moe_token_unpermute restore each original query row, including padding.
- HCCL AllGather/ReduceScatter/AllToAll respect the recorded handle's ordered
  group-rank concatenation and match source/destination element layouts.
- The replayed compiled artifact uses those native implementations and correct
  capture/update buffers, with no unobserved fusion/rewrite. A Python source SHA,
  descriptor repr or pointer match alone cannot prove the compiled operation list.

If one contract is missing, retain a boundary-specific conditional proof. Use a
small operator semantic test/reference or actual exposed permutation metadata
for that boundary next. Shadow labels routed through independently guessed maps
cannot validate opaque computation; injecting labels as model features changes
routes and is not an identity test.

## One frozen execution and final gate

A single instrumented 48 warmup +12 diagnostic execution preserves comparability
with Run403: exactly60 POSTs, c12, five cohorts ×8 ranks, selected64/65, all1024
outputs, actual FULL replay, no post-handoff oracle use, clean source restoration
and request-generation coverage. Reuse clean lifecycle gates. Existing A/A
controls contextualize instrumentation; diagnostic timing is never Formal Current.
Do not demand identical tokens across nondeterministic independent executions as
an identity proof; require internally joined same-run maps/counts/routes.

Report three separate outcomes:

1. `BRANCH_BINDING_CLOSED` only when every active all43 descriptor, ordered group,
   graph generation and selected changing index/CP snapshot is covered.
2. `ROW_IDENTITY_CLOSED_UNDER_LISTED_NATIVE_CONTRACTS` only after the complete map
   and named trusted native contracts pass; otherwise `CONDITIONAL_NATIVE` with
   the first failing boundary. No silent native assumptions.
3. `EXTERNAL_RETAINED_ATTRIBUTION_CLOSED` only after joining Run420's authoritative
   request/output ledger. Internal initial_output_counts=0 is not sufficient.

Even outcome3 proves only sampled attribution for this executed algorithm. It
does not make packed weight sets compulsory HBM bytes, remove rejected rows
causally, establish attainable capacity, or produce any finite Algorithm,
Hardware, Scheduling or Product ceiling.

Performance-knowledge search `row identity FlashComm graph` found R14's warning
against false identity from equal total row counts and R35's concrete separate
hidden/router EP gather path. These are historical mechanisms, not runtime
certificates for current DP1×TP8. Current source and the new joined records decide.
MD'
## Current host-source SHA256 (not loaded-runtime attestation)

```text
c46d60318098077064c0ff57eae98bc937dcf943bda531918388be4e9128acd5  /data/wio/Inference_Foundry/runtime/target_adapter.py
940a9861551455e64d8163b4c09fec91ca87ac7880ff8f67adecf8435d91a16a  /data/wio/Inference_Foundry/runtime/target_metadata.py
2053dafbd46638d0da23e14b96c59ea01995ec2f0441d01928f1c9484c7b4ed5  /data/wio/Inference_Foundry/bootstrap/vllm_target_handoff.py
6396ca409d633ee61c4433a762982cf901b893f252c864c0ff05ac17bdbdb3f6  /data/wio/vllm_ascend_26/framework/vllm-ascend/vllm_ascend/compilation/acl_graph.py
004dbd0d5b1a5c3f9533fa1d12327bd0ae421fe24bb2674544b2c4a25d74aaba  /data/wio/vllm_ascend_26/framework/vllm-ascend/vllm_ascend/worker/model_runner_v1.py
a4a8517abc3a4f8494f27cc4cc3a876264ef8fea8cfc6de9abdbaf75d42c3b24  /data/wio/vllm_ascend_26/framework/vllm-ascend/vllm_ascend/ops/vocab_parallel_embedding.py
ab93899f54889999cd767b0db01334ad46f20e29fca0d96ce0fee31538fe2a52  /data/wio/vllm_ascend_26/framework/vllm-ascend/vllm_ascend/ops/register_custom_ops.py
c467d0ed0a0d41525aa90008aafbfd26bdbb718b0e29db7de2bed9abe09cabdd  /data/wio/vllm_ascend_26/framework/vllm-ascend/vllm_ascend/ops/fused_moe/prepare_finalize.py
80c7061b59d4593f55ce5c18d601bdc7b1831bfe9c268c5bc8129d733d09c215  /data/wio/vllm_ascend_26/framework/vllm-ascend/vllm_ascend/ops/fused_moe/token_dispatcher.py
1cda64ddb13a88f2fe86ccdb59b6555864148e3994022631915720114d7fb2e5  /data/wio/vllm_ascend_26/framework/vllm-ascend/vllm_ascend/quantization/methods/w4a8.py
11dd3e983dc1d628bf42a95581f9617861471b1f658739db44c20df886147247  /data/wio/vllm_ascend_26/framework/vllm-ascend/vllm_ascend/models/deepseek_v4.py
27cbbf92a02f16301aad2a35091e258b8459dc77744bd8294f2f11a4c126004e  /data/wio/vllm_ascend_26/framework/vllm-ascend/vllm_ascend/attention/context_parallel/dsa_cp.py
```
