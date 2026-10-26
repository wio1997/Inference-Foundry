# Astra independent Run605 posterior review

Verdict: **dispatch acquisition INVALID / partial observations retained**. Do not rewrite the raw packet or create a passing dispatch admission.

## Root cause

The restored `bootstrap/vllm_dspark_handoff.py` (SHA fe7039361894acd70666f72c3089efd635404f1fe9cbc685db21272d946a317e) assigns the borrowed proposer `self.proposer.runner = _DP1RunnerShim(inputs)` in DirectDSparkHandoff.__init__, without restoring the previous runner. The shim has an input_batch containing the LoRA map, but lacks ordinary req_ids and _p602_marks. The same proposer is subsequently used by ordinary model-runner proposals. Both Run605 Draft/context observers require _p602_marks on proposer.runner, so after warmup handoff they skip ordinary calls as well as Runtime calls.

This is an observer ownership/lifecycle defect, not evidence of zero Draft/context work. The previous preflight mocks distinguished ordinary runner from shim but omitted ordinary invocation after the proposer retained its shim. The scoped LIVE-READY verdict did not detect this lifecycle case.

## Raw posterior audit

- Product admission status product_identity_pass; all **281** raw SHA entries independently rehashed successfully.
- All 32 rank/cohort workers: pairs per rank/cohort **9,13,10,12**, total **44**. forward/graphs each cover every ordinary pair; draft_inputs/context_stores both empty. First validator failure is `Draft input ordinal list`.
- Independently checked the surviving fields without synthesizing missing records: Target/proposal/forward ordinal identity and uniqueness, actual metadata presence and shape, FULL coverage, Graph parent/mode/Host containment and replay synchronization chronology, all8 forward/Graph semantic equality, exact return identity and timestamp after serve_synced. These partial checks passed.
- Rank0 observed branches: **30 NONE/fallthrough, 14 FULL/replay**; all replay records take the existing synchronization. These are current dispatch occurrences, not compulsory work or critical-path lower bounds.
- source_before/source_after and scripts_before/scripts_after snapshots are byte-identical. Run606 original-source preflight additionally revalidated the installed nine exact original SHA values.
- Actual cleanup_status: **run_exit=1 and final_exit=1**; stop_exit, stop_verify_exit, restore_exit, source_sha_exit, source_compare_exit, script_sha_exit, script_compare_exit are all **0**. Thus seven cleanup/identity operations succeeded while acquisition failed. “Nine values all zero” is incorrect.

No additional failure was found in observable Target/Graph/return fields. Draft geometry, context ownership, their all8 consistency and finite event intervals remain **unobserved**, not passed. Basis/Product success does not repair dispatch admission. No bound endpoints or formal TPS change follows.

## Run606 minimal remedy

Bind the actual outer ordinary runner in a canonical dspark_proposer ContextVar only around the nested ordinary self.propose_draft_token_ids invocation, reset in finally, and have DSpark/dflash observers use that owner. Keep the real proposer.runner untouched. Runtime invocation occurs outside that dynamic scope and sees None. A required regression is warmup handoff -> ordinary invocation with proposer.runner still a shim; also test unarmed, exception reset and restoration of an existing outer context.

This report adds no service/NPU execution and preserves all Run605 raw evidence.
