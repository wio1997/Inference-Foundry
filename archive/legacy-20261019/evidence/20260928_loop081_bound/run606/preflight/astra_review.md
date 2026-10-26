# Astra independent Run606 final preflight

Verdict: **SCOPED LIVE-READY** for one guarded diagnostic acquisition. No remaining source/preflight blocker identified in the inspected version. Dispatch evidence still requires complete live admission.

## Exact inspected identities

| File | SHA256 |
|---|---|
| loop081_dispatch_patch_run606.py | c47717e04c99a1a9dfa4350be11c510c5cf3d52253b25194dbf2bc6f2097e8ee |
| loop081_dispatch_validate_run606.py | 9f9f8321312febfa4cb03d30e1ee8b4d659b43dab1ba172573fc7875a5c71b91 |
| loop081_dispatch_selftest_run606.py | 0c0be36bae7187a74a66d88f440650059f570ffabd9067010be7d91935671754 |
| loop081_dispatch_hook_selftest_run606.py | 43edce0baade9b2b2efa6e198a411b020e810f7ab7b5f1f1cddb86bbe6208350 |
| run_loop081_dispatch_run606.sh | cf347a98b251bb044161cf456e525fed6ac7837e313d754f898b6c9b94f7cc4e |
| preflight/patch_check.json | 5ab2a78425c050d75518c5975d557bd3d395db77de340adbb005bb96d0f0c7da |
| preflight/selftest.json | 71ddda383c7169cfdae43a6de684743354a88dc7b1ee7494619e5be9d86ae833 |

## Independent delta checks

1. Regenerated prepared() against restored installation, verified **9 keys / 9 unique paths**, all original and patched SHA values equal patch_check, and compiled all nine generated sources. The first audit command compared the manifest to the JSON including its action field and failed an audit assertion; correcting that comparison by excluding only action produced the PASS stated here. It was not a patch/source error.
2. Read the generated complete nested proposal closure, not only the insertion snippet. It recomputes the Run606 arm gate locally. Canonical dspark ContextVar is set to the actual outer runner only around the unchanged self.propose_draft_token_ids arguments/assignment, then reset in finally before optional CPU-copy work.
3. Executed that **exact generated closure AST** with CPU mocks: unarmed owner=None; armed ordinary invocation with persistent shim; previous outer context restored; original proposal exception propagates and owner resets. All passed. Return/assignment and copy ordering remained unchanged.
4. Independently reran the shipped patched-hook AST mock: DSpark and dflash read the same canonical module ContextVar; a shim-backed proposer produces one ordinary input/context record when owner is bound and none with owner=None. Its event allocation count confirms no context event allocation on the Runtime no-owner path.
5. Verified the selftest artifact's validator/selftest SHA bindings and positive + 15 corruption rejection results; bash -n passed. Existing Graph gates, metadata bounds, raw Product SHA join and exact 32 return-file admission are retained.
6. Controller uses a fresh Run606 output path, ownership lock, warm48 validated before arm, measured48/c12/1024, 96 HTTP POST gate, Basis -> Product -> dispatch admission, owned stop, verified stop before restore, nine-source/script before-after checks and failure-preserving exit status. It executes the hook test before install.

## Why the ownership fix is appropriate

The observation owner and proposer.runner serve different purposes. The latter remains the existing fixed-DP1 execution shim. The ContextVar carries only Host observation ownership through the synchronous Python call chain. The dflash import occurs at method execution, avoiding a new module-import circular dependency. No draft IDs, positions, KV tensors, acceptance, cycle controls or execution stream selection are modified by this delta. No new synchronization is introduced.

Coverage must still be proved live: the corresponding DSpark/context Python hooks must execute once per ordinary call. Failure of that coverage, source identity, all8 semantic equality, finite event intervals or output/basis admission invalidates the acquisition; do not fill missing records from another run. ContextVar does not propagate automatically to arbitrary worker threads; current inspected call path is synchronous, and ordinal coverage remains the fail-closed protection.

## Scope limits and posterior requirements

- This is a **new diagnostic W0**, not Run99 or Run605 same-state timing. Preserve Run605 invalid evidence.
- Current-stream context-store events and Runtime entry markers do not prove side-stream completion or the exact first KV read. The first Target marker is not a Draft-context consumer.
- Host return/Graph dispatch/current-stream elapsed times are current observations, not compulsory work, removable latency, critical-path bounds or a perturbation-controlled speedup.
- After fixing the silent skip, Run606 now actually records ordinary context events; that device instrumentation cost was absent in failed Run605. Any quantitative timing/performance conclusion still needs an appropriate perturbation control. Acquisition identity can be admitted without such a speedup claim.
- No complete compulsory compute/traffic certificate, fresh-work minimality, exact-board capacity ceiling or finite Resource/Scheduling/Product Bound is supplied. Strict endpoints remain null.
