# Run501 — independent Run502/v4 source-only preflight

2026-09-27. **PASS: one controlled Run502 diagnostic acquisition may launch using the reviewed frozen files.** The v4 schema repair is minimal, merged bytes match stage, Run502 uses fresh acquisition paths, and the Run499 identity/source/lifecycle gates remain intact. This verdict does not admit Run494's failed acquisition or pre-admit Run502 results.

No service launch/query, NPU query/workload, patch installation, shared-source/script change, TaskCtl update or model update was performed. Only source/artifact reads, CPU tests and Run501 evidence writes occurred.

## Frozen change and controller check

All four v4 staged-file SHA256 values match both the stage manifest and their live script bytes. Relative to saved reviewed v3, the only production parser changes are the exact `MEMCPY_ASYNC` token added to helper `graph_task_metadata` and offline `native_tasks`. Test changes are the representative common-fields-only memcpy task and missing/null/string/bool/duplicate identity negatives, plus associated fixture counts. The patcher, cleanup test and original Run494 controller are byte-identical to v3.

Run502 controller SHA256 is `d28c78acf15265d9a376bef5b22d24942903c29583dfa3845b12c610b0bf5673`. Independent exact-text comparison shows only the OUT path, RUN_TS and fresh-path error text differ from the reviewed Run494 controller. OUT is `evidence/20260927_loop079_identity/run502/b_candidate`; RUN_TS is `LOOP079-RUN502-DUMP-B`. Both the candidate directory and resulting server-log path were absent at review time. Fresh-directory/log refusal remains active.

The controller still creates a new `uuid.uuid4()` and passes it through `EXTREME_RUN494_RUN_ID`, with new capture/dump/runtime paths. Keeping the Run494 helper module and environment interface is correct: the acquisition identity is the newly generated UUID and Run502 paths, not the helper filename. No old raw or metadata is copied into the new acquisition.

## Verification results

- Runtime CPU suites and offline positive/adversarial tests pass, including MEMCPY_ASYNC common-field acceptance and its malformed identity rejection. Controller `bash -n` passes.
- Independent Run499 attacks still reject missing/invalid/duplicate/native-mixed task identity, unsupported kinds, missing kernel fields, wrong stream/model metadata, nonhex backend hash, changed selected implementation, same outer callable with substituted backend closure, and the same debug function bound to another graph. Missing/wrong dump owner IDs reject.
- Actual generated backend execution counts one original getter and one update call. No post-drain resolver call is added. A throwing update leaves return_count0 and cannot obtain a successful dump certificate.
- All six live restored sources independently match original pins; in-memory regenerated patched bytes match the full validator manifest. Serving patched SHA remains `b970ea3efef1c5febe62ac906bfb23146713bd52403cc5fac95d29760f4cd947`; graph patched SHA remains `91a9f202c5d455dd92a4129113c1830e30073259ebd8c18a3b49775d848fd8c7`.
- Both **live v4 parsers**, without in-memory code modification, pass each of Run494's eight retained raw dumps: 5,412 tasks and43 MEMCPY_ASYNC tasks per rank. Every raw SHA still matches Run500's recorded hash. These are parser regression inputs only; their failed acquisition status is unchanged.
- Cleanup's success and six failure cases pass both the stock harness and the harness explicitly pointed at the **new Run502 controller**. All external operations in that harness are mocked. Proven stop before restore, nonzero failure propagation and final source/admission checks remain intact.
- All eight reviewed script hashes were recorded in `reviewed_scripts.sha256` and rechecked successfully after execution.

## Scope retained for the controlled acquisition

Entry/capture generation, exact Runtime Target model/batch/input/output ownership, selected actual backend, debug method function/source and receiver identity, four hidden/aux leaves, `after` update, UUID, task/model/stream metadata and raw byte hashes remain required. Selected hooks add no raw stream getter, source-file read or synchronization; dump/source inspection remains after ordinary counts_cpu drain. Metadata continues to label post-drain state after later replays, not frozen cycle64 dynamic parameters.

Run502 still needs exact48+12 clients with1024 outputs, exact60 HTTP200, all8 ranks × five cohort records, Runtime trajectory/request joins, eight successful dump/meta joins, proven stop, six-source restore and final admission. Future unsupported task types and any other failure remain fail-closed. The controller must run those gates; this source review does not replace them.

Run494 remains invalid. Run502, even if admitted, will supply diagnostic provenance for subsequent output-last-writer, kernel argument and event/child-stream join analysis. It does not establish compulsory work, effective private-stream completion, timing savings, formal Current or a finite Bound endpoint.

Evidence: `stage_checks.json`, `astra_stage_checks.py`, `independent_results.json`, `astra_independent_checks.py`, `runtime_cpu.log`, `validator_cpu.json`, `cleanup_cpu.log`, `source_check.json`, `reviewed_scripts.sha256`.
