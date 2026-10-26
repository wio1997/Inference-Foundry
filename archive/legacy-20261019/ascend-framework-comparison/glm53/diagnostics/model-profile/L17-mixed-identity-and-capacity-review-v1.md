Primary design constraint for the D-only lifecycle/capacity drafts.

The retained P runtime is always the accepted L12C native candidate. Its variant, stage roots, optional extension and native mapping/MTP guards must not be selected from a new D rollback/candidate selector. The existing ff5a9234 mixed identity helper hardcodes old D stages and uses one top-level selected_variant for P; it cannot be reused unchanged for the new mixed runtime.

Keep explicit per-role selection in any future new capacity/identity record (for example selected_P_variant=candidate, selected_D_variant=candidate or rollback). Preserve P34/P35 actual CIDs/names/run roots, source manifests and private extension SHA verbatim. Derive only D36/D41 stage, NoCopy/audit expectation and run roots from the separately selected D variant and actual native READY CIDs. Actual model-start/capture/client launchers require a new independently reviewed mixed identity renderer. No manual or guessed D service IDs/CIDs are allowed.

If startup has not executed EagerRunner.load_batch yet on every registered SDK worker, report first-use coverage as pending. The existing bounded functional matrix may supply first real use after core startup/source/config/layout registration succeeds; the missing proof cannot be filled from CPU tests. All32 native branch evidence plus actual bs2 draft coverage, count/semantic/cache/Graph/MTP/SDMA gates are required before formal performance qualification.

This note changes no runtime, launch authorization or model budget.
