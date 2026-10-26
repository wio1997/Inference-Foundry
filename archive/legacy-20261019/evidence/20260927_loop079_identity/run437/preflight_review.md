# Run437 preflight review — Astra

2026-09-27. Read-only review of four new scripts, source_check.json and relevant current adapter/metadata/ACLGraph source. Compiled generated patches in memory and verified all six original/patched SHA pairs match source_check. No source install, service or NPU operation.

## Verdict: HOLD before install/start

Capture can collect useful diagnostics, but cleanup is unsafe and validator can accept data that does not prove the requested row identity. P1 items below should be resolved before running under an all43 identity objective. `source_check` proves insertion anchors/hash/compilation only; it is not a runtime or semantic preflight.

## P1 — stop failure is ignored before restoring borrowed sources

`run_loop079_row_capture_run437.sh:cleanup` uses stop `|| true`, immediately restores if manifest exists, and ignores restore failure with another `|| true`. This can restore files while a live worker/service still executes patched code. A successful benchmark may exit0 even when cleanup fails. `source_after.sha256` is written but never compared to the before manifest. Before install, a failed health check also does not prove workers are stopped: a starting, unhealthy or orphaned server may remain.

Required minimum: use the previously reviewed stop-verification approach, skip /proc state Z zombies, independently verify device process/memory release as an operational gate, and refuse restore if stop or live-worker verification fails. Preserve the original run exit code and report explicit stop/verify/restore/SHA-match codes; any cleanup failure must make final status fail and leave a clear recovery instruction. Trap TERM/INT as appropriate. Confirm stopped state before install too. Do not reuse the old shell pattern that swallowed errors.

## P1 — accepted arbitrary logits permutation is ignored by retained-row attribution

Validator accepts any permutation of0..95 for target_logits_indices, then computes retained expert union using `ids[s*8+j]`. Current Target adapter explicitly selects `hidden_states[state.target_logits_indices]` before logits. A nonidentity permutation therefore changes which hidden/route row produces each verifier row, while current validator still attributes to identity row order. It will accept this malformed attribution.

For the frozen supported layout, require exact identity `list(range(96))` on every rank and exact query prefix `[8*s for s in range(13)]`; alternatively explicitly compose logits→hidden row mapping and validate request/position semantics before calculating retained sets. Also validate cross-rank equality of logits_indices/query_start_loc/seq_lens/slot_mapping; the current cross-rank field list omits all four and only rank0 has their shape/value checks. Initial per-slot positions and count history should bind the snapshot position progression or remain explicitly unproved.

## P1 — graph metadata is not joined to the state whose values are snapshotted

Graph capture/replay metadata equality is useful. But offline checks never require mode FULL, expected input_ids/positions kwargs, correct shapes/dtypes, or pointer equality with the runtime state snapshot sources. `after_target` stores state values without their storage descriptors. Thus a replay binding with empty kwargs/irrelevant tensors, matching capture/replay metadata, can satisfy the current checks while snapshots describe unrelated buffers. A collection of piecewise graph entries containing43 layers could also pass the declared FULL objective.

Record original state tensor descriptors (data/storage pointer, offset, shape/stride/dtype/device) alongside the clones. Require the supported actual FULL entry count/identity and join input_ids/positions at the real invocation nesting to those state descriptors. Fail on missing/opaque expected fields; tensor_tree's depth cutoff and omission of non-tensor values must not silently count as complete graph argument binding. Bind capture generation/wrapper or ensure entry lifetime uniqueness rather than relying solely on a potentially recycled Python id. Native graph freshness from Run433 is not a substitute for this join.

## P1 — CP snapshots have no semantic validator or layer-binding proof

Validator checks only that each CP pointer appears as a key in cp_binding_values. Empty refs, malformed values, wrong query prefixes, stale positions/local_seq_lens and arbitrary ratios can all pass. Recording cp rank/local_start/end alone does not prove the local query map.

For each supported CP binding, validate exact field set, shape/dtype and values against `FixedTargetMetadataUpdater` formulas: local query prefix from clipped global request segments, input_positions from Target positions, start/local_seq_lens from actual metadata update. Record group identity and layer→binding use; current index/ratio/ref list is not an all43 binding map. `seq_lens` and native metadata/selected branch association are also absent. Either implement a source-specific validator and binding capture or name this a raw snapshot artifact and explicitly leave all43 row composition unproved.

## P1 — no all43 branch/row-composition evidence despite generic valid=True

The43 exact ordinal set, replicated route matrices and EP histogram parity validate routing cardinality/expert ownership. They do not prove token row order through embedding, FlashComm, DSA skip-A2A/restore, sequence-parallel prepare/finalize and native transforms. No per-layer branch identifiers or ordered TP/DP/PCP group maps are recorded. ep_ranks is captured but not checked. Run424 explicitly required these bindings; Run432/433 only cover sampled native MoE ABI.

Current validator returns `valid=True` and retained sets without an explicit `ROW_IDENTITY_CONDITIONAL_NATIVE` scope or missing-contract ledger. Avoid labeling that an all43 row certificate. Minimum safe diagnostic output: separate ROUTE_GROUP_PARITY_VALID / GRAPH_INPUT_BINDING_VALID / CP_VALUE_MAP_VALID from unresolved ROW_COMPOSITION and EXTERNAL_RETENTION; emit retained unions only as conditional on the named row map. If the run objective is the full row certificate, collect and validate the missing branch map before claiming completion.

## Additional preflight gaps

- Runner finishes after POST count60; it does not invoke the offline validator, require five complete eight-rank cohorts, or validate benchmark responses/output lengths. A partial diagnostic with one cohort can pass validator.main. Add expected cohort/rank/sample coverage and independently checked frozen request gate before reporting success.
- No new empty output-directory enforcement: stale cohort files can be mixed with a new process (export numbers by existing glob count). Fail on preexisting capture/runtime/results unless explicitly resumed with process/run identities. Add run UUID/worker PID/capture generation to every record.
- Fail-closed import preflight should run in the actual container working directory (`from scripts import loop079_row_capture`) without NPU computation; source compile alone does not resolve module availability.
- Post-Target clones assume the producing graph's streams have completed or joined the current stream. Confirm the exact production FULL synchronization branch and record that source/branch; the helper's module docstring is not a stream-order proof. Preserve selected-cycle-only device cloning and cohort-end D2H.
- Native quantized scale/GMM and graph all-slot contracts stay conditional. Snapshotting expert_map and count parity do not close them or external request publication. The export correctly says external request IDs are unavailable; retain that restriction.

## Checks performed

In-memory patch generation, compile and SHA comparison succeeded for graph/w4a8/comm/draft/runtime/serving (exit0). No semantic test or device execution performed. The identified validator counterexamples follow directly from accepted predicates: arbitrary logits permutation, empty CP refs, equal non-FULL capture/replay bindings. They should become CPU negative fixtures in the revised validator.

## Reviewed script hashes

- `loop079_row_capture.py`: `fe47ce5986d57de62d8e9028c321c304ccf5347c47d2a3b45b02da8725a6dc24`
- `loop079_row_capture_patch.py`: `6edb6247c1de807a4ea229ac17fa68b99e0ce4bf2beec5e26fa357028c47e103`
- `loop079_row_capture_validate.py`: `3df3271a9ecffe8e4ac5b895996b93fadf9ad7b1bb6310512aeae8b4687f90cc`
- `run_loop079_row_capture_run437.sh`: `c7fcc7c40d09b040ebce7587a12749caf6184526bdc4c0b6e7fec17c120eab78`
