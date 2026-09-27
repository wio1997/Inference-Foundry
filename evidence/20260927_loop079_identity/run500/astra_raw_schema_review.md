# Run500 — independent invalid Run494 raw graph schema review

2026-09-27. **PASS for the narrow proposed schema repair: add exactly `MEMCPY_ASYNC` as a supported non-kernel task in both parsers.** That change is sufficient to pass every existing task-schema requirement on all eight retained raw dumps. I agree with merging the narrowly staged v4 whitelist/test change, then independently reviewing its frozen hashes and using a fresh output/log identity for reacquisition. **Run494 B remains invalid and cannot be promoted.** This is a source/CPU schema decision, not retrospective acquisition admission or a launch review of a not-yet-frozen v4.

Only raw evidence and source were read, and CPU checks plus Run500 evidence were written. No service, NPU query/workload, patch installation, shared-script/source edit, TaskCtl update or model update occurred.

## Independent raw verification

`raw_schema_checks.json` records each original dump's exact SHA256 and size. Files range from 3,652,239 to 3,652,280 bytes. All eight independently parse as 5,412 tasks, one native model45, with stream0/1/99 counts 1,040/3,985/387. Task IDs are unique by `(Model Id, Stream Id, Task Id)`, nonnegative exact integers and contiguous from zero within each observed stream. Task names, args mappings, types and all currently required kernel argument fields pass.

The task-type distribution is identical on all ranks:

| Type | Per rank |
| --- | ---: |
| KERNEL_AIVEC | 1,656 |
| KERNEL_AICORE | 235 |
| KERNEL_MIX_AIC | 492 |
| KERNEL_MIX_AIV | 174 |
| EVENT_RECORD | 1,038 |
| EVENT_WAIT | 735 |
| EVENT_RESET | 1,038 |
| MEMCPY_ASYNC | 43 |
| NOTIFY_RECORD | 1 |

Every `MEMCPY_ASYNC` occurs on stream1, has name `MEMCPY_ASYNC`, and exposes only the four common args: Model Id, Stream Id, Task Id and Task Type. The exporter supplies no source/destination address, transfer size or direction for these tasks. They should therefore pass the existing non-kernel common-field contract; inventing mandatory kernel or transfer fields would reject the actual installed format.

I compiled the helper and offline validator in memory with only the single additional whitelist token and ran both functions on each raw file. Both return identical task/model/stream metadata on all eight files. A common-only MEMCPY_ASYNC CPU fixture passes both; missing Task Id, a duplicate task identity and the unrecognized similar name `MEMCPY_ASYNC_X` still reject. No shared script was modified.

## Minimal v4 change

1. Add `'MEMCPY_ASYNC'` to the `supported` sets in helper `graph_task_metadata()` and offline `native_tasks()`. Preserve exact common field/type/identity checks and every other gate. No kernel-source patch or six-source manifest digest change is needed for this schema-only adjustment.
2. Add a positive common-fields-only MEMCPY_ASYNC fixture and missing/invalid task identity and unknown-kind negatives in both parser tests. Verify both parsers on these retained raw files without constructing missing acquisition metadata or marking this acquisition valid.
3. Freeze the new helper/validator/test hashes. Use a new controller output directory and server log identity for any new acquisition; the existing fresh-path refusal must remain. Keep Run494 B's raw, failed status and client evidence intact.

No further task-schema blocker appears in these eight files after that exact token is added. A future capture could still contain a different kind or data shape; keep unknowns fail-closed. This finding does not guarantee later all8, metadata, HTTP, Runtime, source-restoration or final admission will pass.

## Event observations and interpretation

Every rank has 1,038 distinct event-record IDs, 735 distinct event-wait IDs and 1,038 distinct reset IDs, with no repeated ID within each type. Every wait suffix has exactly one record suffix in the same raw dump. Reset IDs equal record IDs. There are 303 record IDs with no wait in the dump; that is an observed scope boundary, not evidence that those records are unnecessary or that a missing dependency exists. Do not require every recorded event to have a captured wait as a new admission gate.

Matching name suffixes yields candidate record-stream → wait-stream pairs: 1→99:172, 99→1:43, 0→1:260 and 1→0:260, identical across ranks. Taking increasing Task Id within each stream as a candidate sequence, and same-suffix record→wait as candidate cross-stream edges, yields an acyclic structural graph in which all5,412 listed tasks can reach the one terminal NOTIFY_RECORD. This is a conditional check of the exported structure, not an execution-time or required-output completion certificate. It does not certify reset/replay event generations, an external event producer, the Python replay stream's identity, kernel output arguments, the four actual last writers or the after-update consumer relationship. JSON list position is not global task order; exporter `ts`/`dur` is not observed latency.

MEMCPY_ASYNC identifies a task kind only. These raw rows cannot establish physical bytes, mandatory communication, HBM work or a capacity/latency lower bound.

## Why Run494 B stays invalid; next-admission risks

All32 warmup capture rows pass the existing detailed v3 validator; warmup48 has exactly48 requests with1024 outputs and no recorded errors. Bench12 contains truncated outputs of1–11 tokens, so the exact60×1024 contract fails regardless of any client error flag. There are zero graph meta files and no successful cohort5 capture certificates. Final admission is false. The saved cleanup reports run_exit1/final_exit1, with stop, stop verification, restore, SHA and SHA comparison all0; before/after source SHA files are byte-identical.

Source ordering establishes that the runtime's selected entry/model/batch/output ownership, retained actual backend and bound debug method checks occur before raw export; the present files were written before the schema rejection. However, successful post-schema metadata and cohort5 records were never persisted. Do not reconstruct them from warmup or another process, and do not use raw graph/model IDs as a cross-run join. Missing cohort5 source/callable/owner/hash joins and request completeness require a fresh acquisition.

The 32 warmup records and complete raw schema scan reveal no additional immediate schema/shape/branch mismatch. Export/storage failure, all8 post-dump row completion, and the whole-cohort client/final gates remain untested on a successful diagnostic. Preserve the v3 graph receiver check and post-drain-state label. The exact cycle64 dynamic task parameters are still not frozen by a later dump.

Evidence: `astra_raw_checks.py`, `raw_schema_checks.json`; original files remain in `run494/b_candidate/graph_dump/`. The supported-format repair advances diagnostics only; all finite Bound endpoints and formal Current remain unchanged.
