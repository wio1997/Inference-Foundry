# Run508 — independent source-only CLI and cleanup preflight

2026-09-27. **PASS / GO for one fresh controlled Run508 diagnostic with the exact files below.** The direct-script import failure is fixed and the actual controller invocation pattern is independently exercised. No remaining launch blocker was found in this bounded review. Actual Run508 must still pass its complete acquisition and cleanup gates.

**Run507 remains INVALID.** Its original `run_exit=1`, `final_gate_exit=1` and `final_exit=1` are unchanged. Stop, stop verification, restore, source SHA and SHA comparison are all0. Posthoc structural parsing does not retroactively make its controller run clean.

## Frozen files

| File | SHA256 |
| --- | --- |
| loop079_target_frontier_validate_run507.py | `a55df975f7995f42fdefc260ea24be4009c1e77693d5bada53f07ca9ea3dc976` |
| run_loop079_target_frontier_run508_b.sh | `52afaa7b9614beca55f4c1dd2caa1d4ee23e7264846831911799027cb052ff39` |
| loop079_target_frontier_update_validate_run507.py | `a5682e8fab06a1f46895592b4270975c6b34e9eb1a1d0ce22db9c2c1373ea481` |
| loop079_target_frontier_run507.py | `6656e6dd4acff146559f249594c889aeac779576a62d0856766f51668849f12c` |
| loop079_target_frontier_patch_run507.py | `bd9a45e57153462ee0c1de60f797ed0823b35c23558cfb174e865fdc5fc01ab5` |

The helper, patcher and update validator are unchanged from the preceding source review. The new controller differs from Run507 only in fresh output path, log run name and associated freshness-error labels. Its `EXTREME_RUN507_*` helper interface is intentionally retained; a new UUID is still generated. Existing output/log rejection, exact48+12/exact60/all8, source hashes and cleanup gates remain intact.

## Reproduced entry-point defect and fix

The prior preflight's in-process validator fixture supplied a synthetic `scripts` package, so it did not exercise the absolute-path direct-script import used by the controller. This review reproduces the old validator's `ModuleNotFoundError: No module named 'scripts'` in a fresh subprocess. The earlier preflight missed that execution-mode defect.

The repaired validator imports the package path when available and falls back to its sibling module only when the missing module is exactly `scripts`; unrelated nested missing dependencies are re-raised. Both the production absolute-path invocation and package `-m` invocation now pass.

## Independent checks: 22 PASS

`cpu_review_results.json` and `astra_audit.py` record/reproduce the checks. `reviewed_sources/` preserves the five audited files. Subprocesses used `python3 /data/wio/Inference_Foundry/scripts/...`, working directory `/tmp`, and no `PYTHONPATH`.

- The fixed absolute-path root validator accepts the immutable Run507 raw structure as40 update rows with source-derived zip counts `[0]`. The standalone absolute-path update validator successfully creates its summary. Both outputs were written under this Run508 preflight directory with explicit posthoc-invalid-Run507 names; no original Run507 validation file was replaced.
- Absolute-path `--final-only` against actual Run507 rejects. Its failed local validation and exit1 lifecycle are preserved.
- In a disposable directory explicitly named `SYNTHETIC_ONLY`, copied records receive relocated dump paths and synthetic successful cleanup status solely to exercise final admission. The controller-order sequence—root validation, update-summary generation, final-only—passes through actual CLI subprocesses.
- Final-only rejects an incorrect inferred record count, missing captured snapshot, wrong generation, fabricated device-completion claim, missing update summary and modified summary. These are real subprocess exit-code checks, not only imported function calls.
- The actual Run508 cleanup function passes the seven established mocked success/failure paths. Additional mocks prove that an original update-stage exit7 remains exit7 and a final-only return9 changes an otherwise successful controller exit to1.
- Controller shell syntax passes. Every original Run507 file and every reviewed source hash remains unchanged after testing.

Synthetic successful lifecycle bytes existed only in the disposable fixture. They are neither a Run507 repair nor evidence about an actual clean acquisition.

## Evidence limits

The posthoc Run507 rows distinguish nonzero attention keys from empty main GraphParams lists and zero source-derived zipped iterations. This is useful diagnosis of an invalid acquisition, not an admitted replacement for fresh Run508 evidence. A future clean zero-count result would narrowly close selected-call update-loop work; it would not establish graph-wide no-work, typed output writers, native event identity/reset generation or model-terminal→caller R1 completion.

The preceding Run507 source review's source-inference/stable-list assumptions remain. No timing, Run487 cross-run transfer, compulsory-work/capacity certificate or finite Bound follows. This review did not start/query any service, execute NPU work, install patches, modify shared scripts or change TaskCtl/model. Only preflight evidence and temporary CPU fixtures were written.
