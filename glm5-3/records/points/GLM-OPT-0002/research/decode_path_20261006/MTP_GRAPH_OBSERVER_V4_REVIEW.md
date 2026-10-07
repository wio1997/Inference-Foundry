# Independent observer v4 review

2026-10-07. Offline source-only. Inspected complete observer-v3 addition to production candidate, exact v3→v4 diff, v4 CPU_result, actual ForwardContext/_ExtraForwardContextProxy, proposer dummy/runtime callsites, ACLGraphEntry/wrapper and CUDAGraphMode. No service/device operation or frozen Run270 edit.

## Fixed failure

Run270 raw `/private/tmp/glm270-candidate-first-error.raw` directly identifies `_EXTRA_CTX.num_actual_tokens`. Actual proxy's whitelist399–458 rejects it, and base ForwardContext has no such field. The setter argument is local preparation data for MC2 mask; moving the read to base context would also be wrong.

V4 removes that access. Its discriminator (`sampling_metadata` present/non-None, no prefill, num_input_tokens2, batch1, diagnostic root set) matches actual callsites: dummy_run `_runnable` kwargs omit sampling_metadata; `_propose` includes it. Later fallback inside `_run_merged_draft` obtains runner sampling metadata, but that happens after the observer entry and does not mutate its kwargs. Consequently dummy/profile/capture calls remain unobserved even when the runner itself has sampling metadata. This is a valid **real proposer call** discriminator for this implementation, not proof of graph eligibility or replay.

V4 CPU_result e6c7fe64... reports actual context/proxy AST, rejection of old attribute, four dummy/profile/capture cases, prefill exclusion, real mmap/filesystem and 16 lifecycle cases. This addresses the specific mock namespace defect; it is not device correctness.

## Remaining concrete observer edge case

Observer unconditionally obtains `next(iter(steps[0].values()))` and reads `.num_decodes`, `.dcp_context.slot_mapping`, etc., even when `_glm_k1_graph_contract` returns None. Production intentionally permits native eager fallback for an unsupported contract. An empty metadata map raises StopIteration; a first None/indexer or other metadata value raises AttributeError; an unsupported value with no dcp_context raises as well. Thus the observer can turn a lawful production fallback into diagnostic failure. This is a reproducible source counterexample, not evidence that current one-layer SFA metadata actually has this shape.

Minimum hardening: select a verified AscendSFADCPMetadata value with non-None dcp_context before detailed tensor reads; if none exists, record a compact unsupported-metadata/fallback row rather than dereference it. Keep signature-null and production result visible. Add cases for empty map, first None followed by SFA, and non-SFA/no-DCP metadata. Alternatively explicitly freeze observation to the verified exact metadata shape and document that the observer is incapable of validating general fallback; do not claim its tests establish fallback transparency. The first approach avoids another preventable observer-only abort.

## Other API/lifecycle checks

- Base context defines batch_descriptor and cudagraph_runtime_mode. Descriptor may be None in NONE execution; dictionary `.get(None)` is safe here. Observer does not itself dereference descriptor.uniform. Runtime enum strings are `FULL`/`NONE`, so current FULL string test agrees with the actual enum; exact enum comparison would be clearer but is not the observed failure.
- ACLGraphEntry actually has aclgraph/output. Captured greedy K1 output is a tensor [B,1]; production verifies shape and returns an owned clone. Observer data_ptr/tolist calls are therefore supported on that admitted path.
- runner.use_async_scheduling exists in inherited initialization; load/runtime timing occurs after construction. Load witness does not access the late ctor field that caused Run268. TP rank lookup matches TP16/DCP16/DP1 scope.
- `actual_replay` is inferred from had_graph + matching nonempty signature + FULL + mode; it is source-path evidence. Do not present it alone as direct device execution proof. Successful result D2H establishes completed output access, but not numerical agreement with eager.
- Mode0 excludes graph dispatch per call while dummy capture remains enabled by production use_cuda_graph; this intentionally allows pre-captured mode1 comparison, rather than proving mode0 allocated no graphs. Root absent defaults mode1 and disables observation as designed.
- mmap/file and immutable-witness assertions are intentional diagnostics. Changing selector between `_propose` selection and observer mode-read could produce inconsistent labels; controller must change it only at quiescent request boundaries. No current violation established.
- Input/metadata/output CPU snapshots synchronize and perturb execution. Only the first two observed rows per transition incur them, but those rows cannot be performance samples. A real-call discriminator can observe eager fallback and first-time capture too; `had_graph`/matches/selected mode must remain in the record to distinguish them.

## Recommendation

The v4 discriminator fixes the demonstrated missing-field defect without introducing a fabricated replacement context field. No other nonexistent context API was found. Harden the concrete metadata fallback dereference before treating the observer as transparent to all candidate fallback paths. Production candidate remains unchanged. Recovery and any later bounded correctness decision stay with the sole controller/Root; this review is neither a Run authorization nor a performance conclusion.
