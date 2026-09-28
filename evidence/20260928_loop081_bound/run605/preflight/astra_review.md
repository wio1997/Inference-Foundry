# Run605 independent Astra preflight — final delta review

Verdict: **SCOPED LIVE-READY. No remaining identified live blocker in the exact final snapshot below.** Supersedes the initial NOT LIVE-READY review and intermediate snapshots. This is a guarded new-W₀ dispatch/geometry/Host-marker acquisition, not a complete readiness certificate, formal TPS run or numerical Bound. No service/NPU workload or installed-source write was performed by this review.

## Final identities

- patch:66ce3f327c315dcffb13838997277b1e3c6b01241f9b295fbe88bda38e4c4b8c
- validator:4f6502c517e7aff06114ed0428b9521a985be52a3ba3637ff6b0219143f56d62
- selftest:ef7a66720e9e38b314adcdec47eb267e37cb90fb8a1879ea9532b69b6bae1899
- hook selftest:23d93e43fc3bf44d304d8d259bb1115b36e7a295093ca182f96d8f773583f1f6
- wrapper:00a49997639cc6fc0566e37a32c5eaa9d90d88b4e9fd04a0852088f5dcb19f76
- patch_check.json:d23caf39a5793c450641ed9271792aad715f4407a47232c2439cb117130e6467
- selftest.json:40d4c75a1b41e620e8978cc77eabcc942248eccea691c38b48afc7d379ccc490

## Independent verification

Rebuilt prepared() against restored installed sources, compiled all generated code, matched every saved patch manifest field and confirmed9 keys/9 unique resolved paths. A pure in-memory install/restore simulation passed the revised unique-path/composed-patch design; no installed source was touched.

Re-ran final validator selftest into a temporary file: exit0, exact saved-JSON equality, positive plus15 negatives; the positive suite also exercises legal NONE→FULL-wrapper fallthrough. Re-ran generated-AST hook selftest: exit0, ordinary DSpark/context recording, no recording/event allocation for fixed Runtime shim, actual Graph entry modes FULL/NONE and ContextVar reset. bash-n final wrapper passed; its gate requires16 result entries and matching validator/selftest hashes, and executes the hook selftest before install.

The independent earlier astra_hook_mock.json additionally exercised extracted complete Graph fallthrough/capture/replay methods and runner ContextVar propagation/reset including exception propagation on the pre-final patch. Its recorded patch SHA is historical, not the final identity. For the final newly changed fallthrough body, independently executed the generated complete __call__ with a NONE incoming mode and FULL wrapper: exact runnable return object preserved and entry≤fallthrough-enter≤host-return recorded. These are CPU Python mocks with fake event/Graph objects, **not** real Torch/Dynamo/NPU capture validation.

## Closed blockers and delta findings

1. Duplicate draft source is removed. The inherited draft key is reused; product.patch('draft', src) preserves Run597 witnesses before the new observer. Unique-path gate prevents the prior install-race failure. Imported product patch and hook test are included in SCRIPTS.
2. Ordinary owner guards require runner.input_batch and _p602_marks. FixedDP1 shim lacks these and is skipped, preventing the previous first-Runtime-Draft AttributeError. Hooks remain on the actual ordinary set_inputs_first_pass/context method call; they do not depend on a compiled-model callback executing on every replay.
3. Output markers are written to the independent returns/ directory created by the wrapper. Run602's strict product/ file set is preserved. Run605 requires exactly32 return files and checks rank/cohort/run/request identity, post-serve-sync chronology and records raw hashes.
4. Graph incoming mode is recorded separately from wrapper mode. NONE→FULL-wrapper fallthrough is valid; capture/replay requires incoming==wrapper, and a FULL parent needs actual capture/replay evidence rather than any arbitrary Graph row. This repairs the intermediate assertion that would reject legitimate ordinary prefill/mixed calls.
5. Complete Graph Host spans now lie within their parent Target Host span. Existing sync flag/order and branch chronology are checked. Fallthrough records a real post-runnable Host timestamp, not just its entry. No new synchronization was inserted.
6. Actual attention summary must be nonempty and have an attention state. Depth/group overflow fails rather than silently truncating. all8 validates stable forward/Draft/context/Graph semantic signatures, including request IDs and geometry; it excludes per-rank pointer/event IDs and wall times. This is deliberately fail-closed: any legitimate rank-specific capture difference must be reviewed explicitly rather than silently averaged.

## Hook/metadata plausibility and remaining scope

The actual CP metadata types contain DSA in their class names; their integer counters and enum attn_state fit the helper. Existing metadata dictionaries/lists are expected within its depth bound. It deduplicates shared Python metadata objects and records a limited summary, **not** complete layer/group→storage ownership or all compiled attention dispatches. The validator accepts fewer than16 summary rows; unexpected valid structure at that limit will reject the acquisition rather than establish false coverage. No static claim of complete layer identity is made.

The outer _model_forward wrapper obtains the active forward context, and its canonical import references the patched ACLGraph module ContextVar. CPU tests corroborate same-thread propagation/reset. The actual runtime Graph records must still demonstrate it on all8; ContextVar does not automatically propagate into arbitrary execution threads. Capturing and replaying remain distinct branches. No GPU/NPU data-dependent Host read is added by metadata traversal.

The first_target and first_draft fields are now explicitly entry-event presence markers. The first Target marker occurs after Runtime input/metadata preparation and before target.execute. The first Draft marker precedes proposer refresh/prepare/new context-store work. Target does not consume Draft-context KV, and the Draft entry is not the precise first read of ordinary context. Their event differences are downstream marker intervals, not proof of writer readiness, consumer waiting or exposed critical path. A context method-end marker also does not individually prove all layer stores performed nonempty writes.

Events are dynamically allocated, current-stream records add device commands, metadata traversal adds Host work, and output-marker JSON writes are synchronous. The controller does not provide an observer OFF/ON timing gate. This is acceptable for the scoped acquisition; it precludes uninstrumented wall-time, pure busy-time, removed-gap or speedup claims. Cross-device timing is not established by per-device event differences.

## Output marker ordering

The source constructs self._extreme_serving_output, then opens the separate return file, timestamps model_runner_output_built_ns and writes it, then execute_model returns None. Therefore the timestamp proves the output object had been constructed by that point; it is after file open and before file-write completion. It is **not** the exact constructor end, actual sample_tokens return, scheduler consumption, socket delivery or client receipt. Product token joins and the independently recorded later scheduler/API/client boundaries remain necessary.

## Controller and admission boundary

Warmup48 is validated before arm; measured48/c12/1024 uses the unchanged frozen configuration and inherited basis collector. Source patch/check/compile, hash-pinned base/selftests, guarded owned stop and restore remain required. The acquisition may be a different W₀ from Run602 or Run99; do not force71 ordinary pairs or equality of their times.

The final Product/dispatch admissions precede cleanup. Adopt only after exact9 cleanup exits are all0, all source/script bytes match restored originals, all8 semantic gates and96-client/basis/Runtime/Product joins pass. A runtime/Dynamo failure should preserve artifacts and revert through the existing controller; mock success is not permission to ignore a live failure.

Next action: the guarded diagnostic may run once under these final identities. After it completes, independently verify branch coverage, actual metadata cardinality, event stream domains, output/clock joins, cleanup and observer scope. Resource freshness/compulsory work and strict Resource/Scheduling/Product endpoints remain unproven; no finite Bound or formal571.681tok/s update follows from preflight.
