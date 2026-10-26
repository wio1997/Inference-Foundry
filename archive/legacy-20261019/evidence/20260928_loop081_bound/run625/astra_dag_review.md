# Run625 independent Astra High DAG review

Initial artifacts: generator `ad474668bf505c6225cb5619a38fffcd81b3d2662bd772fc46427655f7ead190`; JSON `c1c4d7df650bc1faff51e3c479305d69c18b44947153955b6401b3773d3c1f6c`.

**Initial verdict: useful untimed partial template, corrections required before Scheduling use.** No service/NPU work. All node costs and strict endpoints are null; no numerical Bound was incorrectly emitted.

## Independent reproduction

During concurrent Sol revisions, the first read/compute reproduction (excluding final writes) matched the then-current saved object:50 nodes/72 edges,45 value_required,9 partial_value_required,12 conditional_reader,6 current_python_order. These intermediate counts must not be attributed to the original artifact pins above; the final pinned reproduction below is authoritative. Both declared topologies are acyclic. In-memory negative checks reject duplicate node, absent source endpoint, self-edge, and a Target65→Target64 required backedge. Independently exercised the same four-endpoint null predicate with1.0 in each field: all rejected. These are structural/schema checks, not dependency correctness proofs.

## Required corrections communicated to Sol

1. **Missing current value edge:** state_advance_c→draft_prepare_c. advance_state writes masked num_sampled and last_sampled_tokens; proposer then consumes these through host count launch and next_token_ids. Source: fixed_decode.py331–360, extreme_decode.py449–480, handoff331/380. This is stronger than mere current Python order. A future fused implementation can produce those values directly, so do not interpret it as requiring this exact separate state-update kernel.
2. **Freshness conflation:** context_scatter→reader admission currently requires semantic freshness. Remove this from the current writer/reader-generation gate. Current last-writer provenance can pass for semantically reused values; proving fresh unavoidable work is a separate Resource certificate.
3. **Missing current issue bridge:** context_scatter45_c→first query execution_c, labeled current_python_order only. Current first-pass code completes the three-layer context-precompute issue sequence before model query forward. It is not a global algorithmic barrier.
4. **False serialization risk:** query_layer currently feeds both query scatter and reader, but may mean Q projection or a whole layer. Split layer input→Q path and layer input→KV path, or forbid any timed interpretation until split. Source has independent Q/KV production and an optional multistream implementation. Same-layer output projection/residual/MoE and head remain omitted, so reader→next layer is only a partial contribution edge.
5. **Coarsening warning:** FULL Target→combined aux/logits does not prove every auxiliary value waits for complete Target Graph execution. Before a timed architecture bound, separate produced values/readiness points and internal resource/dependency graph. Composite duration can be valid for a current measured schedule, not automatically a lower-duration node under arbitrary scheduling.
6. **Unresolved is not absent:** explicitly retain cross-cycle retained-KV read/write edges, Host count-copy63→64/64→65, target metadata prepare/commit, Draft commit, predecessor/successor and all8 HCCL joins as unknown. State/proposal64→Target65 is correct as a coarse source value relation, but does not complete the cross-cycle DAG. Isolated product_arrival is only a placeholder, not a product boundary.
7. **Required value does not mean required physical node:** edges describe value use conditional on executing the represented operations. They do not prove every node/materialization occurs in every legal runtime. Fusion, direct forwarding, cache reuse and versioned storage remain allowed; current storage hazards must not be promoted to algorithmic serialization.
8. **Reference provenance:**1483us is rounded from Run616 census1482.78us;23.13us is query→reader native start spacing. Pin/read the census itself. Neither number is a node duration, transfer to this W0, or removable gap.

## What is supported

Per-layer context projections use the same context hidden/positions and separate layer weights; no preceding context projection output is an input to the next. Serial loop issue edges belong in current_python_order. This establishes value independence at this source granularity, not simultaneous feasibility under Cube/HBM/stream contention.

Keeping both context and query writer→reader edges conditional is correct until actual ABI eligible row, last-writer generation, overwrite/lifetime and execution coverage are joined in the same acquisition. No witness is not an independence proof. A proven current cache dependency still needs separation into true producer-value dependency versus avoidable storage/materialization ordering.

Source checks additional to Run623 pins: runtime/extreme_decode.py SHA `eb4b5142cc475a51e06c6eec8238e9cfeb918b568e8a5f27b0a4bc7d26963499`; runtime/fixed_decode.py SHA `7d74f4cbba380ff0a9296cc06d86e0cb9c9c921b6213b7965700831a29a6bafa`; bootstrap/vllm_dspark_handoff.py SHA `fe7039361894acd70666f72c3089efd635404f1fe9cbc685db21272d946a317e`.

**Strict endpoints and Current-to-Bound numerical gap remain null. Formal Current571.681 tok/s is unchanged.** Revisions may pass as an untimed partial template; this review does not approve a live observer.

## Final revision recheck — supersedes initial verdict

**SCOPED PASS as an untimed, incomplete source-dependency template.** Final generator SHA256 `ba579d8acbe1ebf1c5f52b845bc943579bcdc1ba6271a70555a342f89b173479`; final JSON `f984932c05eb2ce9493fd95968ebcb60d94fdac0e670443954b50041efac4533`.

Independently reproduced saved JSON byte-for-byte without output writes. Final64 nodes/88 edges:61 value_required,9 partial_value_required,12 conditional_reader,6 current_python_order. Both topologies are acyclic; duplicate/missing/self/backedge and four flat null negative checks reject. All four per-node cost/lower-duration fields and strict endpoints remain null.

The revision adds state_advance→prepare; splits Q and KV from shared layer input; adds the current context→query issue bridge; removes Resource freshness from current writer-reader admission; pins and reads exact census values; labels Target as a producer scope rather than completion barrier; separates Target logits and auxiliary-hidden readiness nodes; and separates conditional current value use from compulsory physical operations. Cross-cycle cache/Host/metadata unknowns are explicit. These address the review findings; no remaining correction is required for the stated partial untimed scope.

Before assigning times, the generic Target producer scope and draft_prepare composite still need actual per-output readiness/internal refinement. Same-layer post-attention work, production waits, retained state and global resource constraints remain omitted. Acyclicity or value independence does not demonstrate feasible overlap or an architectural critical path. This graph maps evidence obligations; it is not a lower-bound certificate or executable schedule.

### Metadata follow-up challenge

Sol's proposed metadata readiness node addresses a real omitted input dependency, but a single global draft_metadata_ready completion barrier would be too coarse. Prefer separate context_slot_args→context_scatter, query_slot_args→query_scatter, and branch-specific reader_metadata_args→reader value-use edges. Actual source calls require these operands; their content-generation/validity and production readiness are distinct obligations. The context projection consumes hidden/positions, not scatter slot indices: split the current draft_context_input description to avoid making WKV wait for slot construction. Do not attach every metadata node unconditionally to completion of the entire draft_prepare composite; portions may derive from entry state, accepted state or prefetch and can be prepared independently. Their upstream dependencies remain explicitly unresolved until expanded. This is acceptable as an omitted obligation in the current untimed partial scope, but must be represented before any complete Scheduling/time inference.

## Final metadata revision — authoritative final verdict

**SCOPED PASS, source-only untimed partial DAG.** This final revision supersedes the earlier artifact identities and topology counts.

- Generator SHA256: `87fdcaf3c8293a23c0c11f09aefc7d4a471dce25f4369f5824f1123c304a4fdc`.
- Output SHA256: `c2b80720c0501a50fb605335d69a5d3657d6afe5b5f9a66e349046d007ac5b74`.
- Independently reproduced output bytes exactly without output writes: **82 nodes/106 edges** =79 value_required +9 partial_value_required +12 conditional_reader +6 current_python_order.
- All18 context-slot/query-slot/reader-metadata operand nodes have the expected one downstream value-use edge and no fabricated upstream completion edge. Context slot operands do not gate context projection. Readiness ancestry remains null, and the scope explicitly permits entry-state/accepted-state/prior-cycle origins without asserting prefetch legality.
- Both topological checks pass. Duplicate/missing/self-edge negatives reject; a metadata-consumer→its slot-operand backedge rejects as a required cycle. All per-node costs/lower durations remain null; strict endpoints remain null.

No remaining false-necessary edge correction is required for this explicitly conditional current-operation template. The independent context projection relation, conditional row reader edges and cycle64 proposal/state→Target65 relations retain the previously reviewed scope.

**Unbound-node interpretation is mandatory:** metadata nodes appear as roots in a topological traversal because their ancestry is unknown; that does not mean their values exist at time zero, are freely precomputable, or cost zero. Any later scheduler/critical-path evaluator must reject a complete/feasible/numerical claim until these roots receive actual source/value-generation and readiness constraints (or are explicitly credited as proven entry values). Do not silently substitute zero for null. Similarly, absence of inter-layer metadata edges is not proof of separate allocations or independent production.

The combined reader-metadata placeholder still needs branch-specific expansion before timing; Target producer scope, Draft preparation, same-layer tails, cross-cycle cache/Host/count-copy and all8 resources remain incomplete. This revision therefore improves the evidence-obligation map without promoting any Scheduling/Hardware/Product Bound or live readiness. Formal Current571.681 tok/s remains unchanged.
