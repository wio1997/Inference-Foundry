# Run507 — independent source-only preflight

2026-09-27. **PASS / GO for one controlled Run507 diagnostic using the exact final sources below.** No remaining launch blocker was found after the SHA and final-admission corrections. This is permission to proceed from the preflight's technical perspective, not a result certificate: actual all8/exact60 admission, source restoration and cleanup must still pass. No finite Bound, typed output writer, native event correlation or terminal→caller completion is established by this preflight.

The review performed read-only inspection, CPU AST/function mocks, synthetic record validation and mocked shell cleanup. It did not launch/query the service, execute NPU work, install a patch, or modify shared sources, scripts, model or TaskCtl. Only this preflight directory and temporary fixtures were written.

## Frozen review identity

`reviewed_v2/` contains the exact audited five files. `cpu_review_results_v2.json` records **35 passing checks**, including the expected final negatives. `astra_audit_v2.py` reproduces them. The earlier `reviewed_v1/` and `cpu_review_results_v1.json` preserve the independently reproduced final-admission failure before correction.

| File | SHA256 |
| --- | --- |
| loop079_target_frontier_run507.py | `6656e6dd4acff146559f249594c889aeac779576a62d0856766f51668849f12c` |
| loop079_target_frontier_patch_run507.py | `bd9a45e57153462ee0c1de60f797ed0823b35c23558cfb174e865fdc5fc01ab5` |
| loop079_target_frontier_update_validate_run507.py | `a5682e8fab06a1f46895592b4270975c6b34e9eb1a1d0ce22db9c2c1373ea481` |
| loop079_target_frontier_validate_run507.py | `7afc2877f0978c2c012c34a1e31fb5f5e2057a5bc9fb9114342ebc7ea23b302c` |
| run_loop079_target_frontier_run507_b.sh | `69d587209a3b795a937d7b4ca636ed0a2eab830d9b6360eb6a6838d29c2150ec` |

All six actual installed originals match the patcher's pins. Independently applying each patch in memory and compiling it produces exactly the new validator's six-source manifest. Actual patched graph SHA is `6d2ee39b2980a8554717524d04d5ddb5672633cc0bf107eac3daee39d3441199`, matching both helper and validator. Full original/patched hashes are in the CPU results. Installed MLA source matches `67b4479cca16fd164b33b91b2fa4f48e95f2474720e484abdf3b3897a8ea1545`.

## GraphParams and selected context

The snapshot reads `acl_graph.get_graph_params`, the same imported main-model function used by the pinned MLA capture and update source. Its installed `_select_graph_params` chooses the LoRA variant from the active forward context. The capture hook runs after successful ordinary capture but before the outer forward context exits. `capture_family` restricts it to main FULL96, 12 requests, no LoRA, matching model/batch/input ownership. Consequently the selected main non-LoRA pool is the correct one for this path; the helper does not accidentally read draft params. CPU execution of the exact installed selector confirms main/no-LoRA and alternative LoRA selection.

The selected actual update is inside the original handoff's forward context and Target scope. The original model call returns after `replay_pre`/`replay_post`, and the after-update branch then invokes the existing backend getter exactly once and the resolved callable exactly once. Thus this accepted path already has `D['replay']`; a before-update/missing-replay fixture rejects. The new hook rejects draft context, wrong shape, missing metadata, changed capture generation and changed GraphParams/element identities. It records the same replay entry, graph, capture generation and selected observation ordinal.

The snapshots contain GraphParams ID and ordered element IDs/counts of the three lists. They are Python object-pool identity observations, not native event/handle IDs. The ordinary global GraphParams lists retain these objects in this frozen path. The snapshot does not itself add a separate strong-reference lifetime ledger, itemwise key→layer mapping or a general concurrent-mutation proof. Do not extend its same-generation interpretation to replaced pools, arbitrary concurrent mutation, recapture or another process.

## Zip-to-record inference

The exact installed MLA method was AST-extracted and executed with CPU-only stream/FIA/event doubles. Zero keys returns before the loop; nonzero keys with an empty list executes zero iterations; unequal lengths execute the minimum; ordinary positive cases produce begin→FIA→end→record once per zipped item. All four cases pass. No installed torch/NPU module was imported for this source execution.

For the pinned main branch, stable ordinary lists/metadata and a successful method return, the source has no loop skip/early exit and supports `min(key_count, params_count, handles_count, events_count)` successful record calls. The hook's end runs only after the actual callable returns. An exception leaves the selected acquisition incomplete and cannot produce a successful record certificate.

Nevertheless `zip_iteration_count` and `source_inferred_successful_event_records` are **source-derived counts from pre-call snapshots**, not per-iteration native observations. This matches the narrower Run506 plan. It does not measure duration or prove completion of record/FIA/device work. A positive count still needs native event/handle correlation and reset/replay-generation semantics to determine which in-flight wait it releases. A zero count closes only this selected call's update-loop work, not graph work generally. The fields `device_event_completion` and `native_event_id` correctly remain `unobserved`.

## Admission and cleanup

The final controller uses a fresh Run507 output directory, dedicated log name and new UUID, and retains exact48+12 clients, exact60 successful HTTP requests, 40 capture/Runtime records, all8 graph dumps, raw SHA/size/task/stream checks and same-acquisition ownership gates. No new raw stream getter, diagnostic synchronize or event wait was added; the existing ordinary synchronization and drain remain.

The final Run507 validator retains Run494's base graph/source/receiver/output checks with the newly recomputed hashes. `validate_root` now invokes the update validator. `final_admit` recomputes the raw update ledger after cleanup and requires exact equality with `update_validation.json`.

Using temporary copies of Run502 records enriched with explicitly synthetic Run507 fields, positive base/update/final admission passes. Negatives reject wrong capture generation, a different selected GraphParams owner, invalid object IDs, a fabricated native completion claim and wrong inferred record count. Final admission also rejects a missing new field, absent update summary and a modified update summary. These fixtures exercise schema and control flow; their synthetic values are not acquisition evidence.

The Run507 controller's actual cleanup functions were extracted and executed with every external command mocked. Success and six stop/process/HBM/memory/restore/hash failure paths behave as required; restoration is skipped when stop is unproved. An independent original exit7 fixture retains final exit7 even when mocked cleanup and final-validator commands return zero. Shell syntax passes. Six source restore/compare gates remain required.

## Corrections required and incorporated

1. Initially the helper and reused Run494 validator retained the old graph/six-source patched hashes. Those would reject Run507. The final helper and independent Run507 validator now match the actual in-memory patch products, and the controller uses the new validator.
2. The first updated validator's final gate did not rerun the new update validation. An independently mutated cohort1 inferred count was rejected by the update validator but accepted by final admission. The final version closes this hole by validating the new fields from `validate_root` and comparing the recomputed update summary in `final_admit`. The same negative now rejects.

## Remaining evidence boundary

Run502/503 remains a valid post-drain structural diagnostic; Run506 remains a source-only design. This acquisition can add main-branch Python GraphParams/event/handle identity and source-derived update counts. It does not close ReduceMean mixed-core ABI, four exact last writers, overlapping writes, native event/reset generations, model-terminal→caller R1 continuation or cycle64 dynamic dump correspondence. Stream102 configuration and Python IDs must not be equated with exported stream/event suffixes. Run487 time is not transferred, summed or converted into a floor. Actual resource contention, necessary durations, formal-window work/capacity and repeated formal correctness/performance remain separate gates.
