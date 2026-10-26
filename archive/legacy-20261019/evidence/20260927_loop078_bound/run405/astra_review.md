# Astra independent Bound review — Runs403/404/405

Verdict: ACCEPT raw route/ownership observations and reducer arithmetic. Retained-set results must be labeled **conditional on the slot-major candidate-row layout at every captured router**. Run404 does not independently prove that end-to-end layout. No numeric Algorithm/Resource latency floor, compulsory HBM bound, executable Scheduling Bound, or finite Product ceiling can be promoted.

## Independent checks

- 40 capture files +40 Runtime files: five cohorts ×eight ranks, cycles64/65. Cohort cycle totals are281,287,303,318,296; useful counts are44/29,35/42,59/45,48/49,58/49.
- All selected Target matrices have96 rows ×top6 across43 layers; Draft has84 rows ×top6 across3 layers, Q=7 and sample_from_anchor=true.
- Independently recomputed replicated all8 route equality, top6 uniqueness/range, contiguous EP ownership and each local32-entry group_list against the global route histogram. Recomputed retained unions, histogram envelope, packed payload and GEMM-equivalent arithmetic: PASS.
- Added an independent check absent from Run404: every selected Target position equals initial_positions[slot] + sum(previous masked counts[slot]) + row_offset. PASS on both cycles and all cohort reference records.
- Runtime reports pass with FULL Target and zero post-handoff oracle/ModelRunner calls. Six borrowed sources match original SHA after restoration, and before/after manifests agree.
- Exactly60 server POSTs;48+12 client requests succeed with1024 output tokens. Client timestamp sweep maximum concurrency12; sampled server Running+Waiting maximum12. There is no Run389-style extra-client evidence.
- Runtime reports contain external req_ids; capture JSON does not. They can be joined through validated rank/cohort/trajectory, but Run404 currently reports cohort-local slot identity only. Do not claim a direct client-index attribution, because client JSON does not retain SSE IDs.

## What the row-mapping evidence proves — and does not

Source inspection supports the intended mapping:
- Runtime constructs slot-major [last token, seven draft tokens] rows and contiguous positions.
- The installed handoff creates target_logits_indices=arange(96); the adapter indexes hidden states with that tensor. This is source evidence, not a captured runtime invariant.
- W4A8 captures topk_ids immediately after select_experts and before expert dispatch. Within this call, rows align with the input hidden_states/router_logits.
- TokenDispatcherWithAllGather performs expert sorting using npu_moe_init_routing; combine uses its expanded_row_idx through npu_moe_token_unpermute to restore input order. Thus expert sorting itself is not evidence of a persistent candidate permutation.
- Decoder operations and feature-axis reshapes normally preserve the leading query-row identity.

However, the actual server enables FlashComm1 and DSA CP. The inspected model has sequence-parallel chunk/gather branches; MoE prepare/finalize uses maybe_all_gather_and_maybe_unpad / maybe_pad_and_reduce; attention follows a custom DSA path. The capture does not record the actual per-layer branch flags, group ordering and row-map composition across these operations. I did not establish a complete source-and-runtime proof of all native/custom layout transformations for all43 layers. No permutation bug was observed; the proof is incomplete.

Run404 checks row count, contiguous request positions, cross-rank equality and route histograms. Those invariants all survive a common route-row permutation on every rank. Checking endpoint positions separately from the router matrix does not join those identities. Accordingly, “actual_retained” is the exact union for the selected array indices **given** the assumed slot-major mapping, not an unqualified measured token-identity union.

Minimum closure: record the actual target_logits_indices, Target input query boundaries and row identity (cohort,slot,offset,position), and capture a per-layer candidate_row_id vector at the exact pre-dispatch router input. Propagate that vector through the actual chunk/gather/padding/unpadding/query-layout maps, with source pins, runtime branch values and collective rank ordering. A fresh arange(96) attached independently at every router is tautological and does not close the gap. A source proof with captured runtime branch/maps can replace redundant labels where native query-order contracts are established. Pin the additional model, prepare/finalize, dispatcher, attention and handoff files used in that proof.

## Run404 validator limitations

valid=true is a structural route-capture gate, not an end-to-end row-label, numerical model-equivalence or performance certificate.

It verifies Draft64 output equals the next Target's seven candidate IDs in active slots; it does not independently recompute greedy acceptance from Target logits, compare the candidate IDs against per-layer hidden rows, or verify all Draft body/head numerical results. The Draft metadata position/sample-index checks validate declared query geometry, not every internal routing-to-output dependency. Draft65 output is not subjected to a following-cycle handoff check.

The current records do pass the additional position-prefix check above. The validator could enforce that explicitly, plus runtime request-ID/trajectory joins, complete output types/shapes and helper-source provenance. Capture helper, validator and reducer should have recorded hashes alongside the six patched source hashes for reproducibility.

Draft source pointers can be reused between layers; this is compatible with eager snapshots cloned immediately on the producing stream. Target has43 distinct route/group buffers in inspected records and snapshots after the selected replay. Pointer uniqueness is useful lifetime evidence but not token identity proof.

## Reducer math and interpretation

For each layer, with histogram h_e and U selected rows, lower_unique finds the smallest k whose k largest min(h_e,U) capacities cover6U; upper=min(active_experts,6U). These are valid histogram-only necessary envelopes, not attainable choices or a coupled cross-layer schedule. The selected-prefix union lies within them in every case.

The constants are internally consistent:
- Standard GEMM equivalent per routed expert-token pair:2×(4096×4096+2048×4096)=50,331,648 operations.
- Raw packed W4 payload per expert/layer pair:(4096×4096+2048×4096)/2=12,582,912 bytes.

Under those dimensions/formats and the conditional row mapping, Target retained-set cardinality is3354–4629 expert/layer pairs, corresponding to42.203–58.246 GB acrossTP8 and376.581–766.148 GFLOP for29–59 selected rows. Current96-row Target routed arithmetic is1246.614 GFLOP. These are useful partial-work descriptors; neither scales/metadata nor all dense, attention, compression, KV, routing, communication, sampling and Host work are included.

Draft84×6×3 gives76.101 GFLOP only if its expert dimensions match; selected packed payload is2.995–3.498 GB only if its packed format matches. The capture does not contain actual weight dimensions/strides/dtypes/storage sizes to discharge those assumptions. The same provenance should accompany the Target constant. Storage-set bytes count an expert/layer payload once; they do not count actual HBM transactions, cache hits, rereads or residency.

## Draft and causality caution

The three captured Draft routed layers describe the84-query body, not the complete Draft algorithm. Context KV projection, dense operators, noncausal attention, Markov feedback/head, sampling and state operations remain outside this numerator. The source supports sample-from-anchor input geometry and potentially noncausal body attention; Run404 does not capture the actual per-layer causal flag. Keep the noncausal warning, but establish active flags before making a universal dependency statement.

count65−1 gives accepted Draft64 token attribution under the current verifier and clipping contract. It does not identify removable body rows. One accepted output can depend on other body-query rows through attention; removing them can change outputs, routes and acceptance. Target retained prefixes likewise use future acceptance knowledge: a reduced ordinary causal computation may be a useful relaxation under fixed layout/algorithm assumptions, but this does not supply an online method that knows the rejection boundary in advance.

## Bound update

Algorithm/Resource uncertainty shrinks in observed route multiplicity, ownership, candidate/query cardinality and conditional expert-set attribution on ten sampled cycles. The earlier histogram envelope becomes a measured array-index union conditional on layout. Hardware traffic/capacity and compulsory work remain unresolved. Scheduling uncertainty does not numerically shrink: no causal executable reduced-row schedule or mixed resource-constrained DAG is supplied. Diagnostic snapshots perturb execution, and no Run403 timing should calibrate formal Current.

Keep all latency-floor and finite Product-ceiling fields null. Promote the observations only with the layout and expert-format conditions attached. Highest-value next closure is row-identity propagation plus actual Target/Draft weight-format metadata; then address whether any dependency-preserving execution can exploit the conditional work reduction.

Confidence: high for raw parity, clean-run gates and reducer arithmetic; conditional/medium for retained token attribution; insufficient for complete Draft work, compulsory HBM, achievable scheduling or a numeric Resource latency bound.

Read-only audit; only this review file was written.
