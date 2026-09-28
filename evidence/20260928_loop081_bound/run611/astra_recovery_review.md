# Run611 independent Astra High recovery review

## Verdict

**SCOPED PASS for preservation and offline parsing of the eight Run610 raw sessions.** Host-window containment is **conditional**, not a clock-error certificate. This does not repair Run610's controller outcome or admit a formal/undisturbed performance result.

Final recovery script SHA256: **1fa5f346b140b0500802fd50c953b9bc65311e4269e5e78943bc19edcf2f84d4**.
Final raw_recovery_admission.json SHA256: **9903001902801471e005601bf331028a4d02a8b04cb8689f11a5d81c645bd802**.
Independent rerun to a temporary output exited0 and reproduced the final admission byte-for-byte. No service, profiling session, or NPU workload was started; installed sources were not edited.

## Original failure is retained

Run610 has run_exit=1 and final_exit=1. The seven stop/restore/source/script checks are0. Basis, Product and dispatch admissions passed; the original raw validator raised its rank0 clock-containment error. Its later all_file.complete assumption would also reject these Level0 files, which instead carry per-payload and end .done markers. These facts justify a separate recovery record, not changing Run610 to PASS.

My Run610 preflight accepted synthetic clock/file fixtures that did not establish the installed native clock domain or actual Level0 completion convention. That earlier confidence was insufficient for the raw validator. This posterior correction supersedes that part of the preflight; the preflight and failed controller artifacts remain historical evidence.

## Clock-domain correction

The initial Run611 revision incorrectly interpreted the earlier numeric start_info value as “predates capture” and compared it to end_info. Both claims are withdrawn.

Installed profiler_interface.start_trace obtains start_info.start_monotonic through torch_npu._C._profiler._get_monotonic(); _dump_profiler_info obtains end_info.MonotonicTimeEnd through Python time.monotonic_ns(). An independent Host-only three-sample probe found the native call inside the CLOCK_MONOTONIC_RAW brackets3/3 and inside Python CLOCK_MONOTONIC brackets0/3. The installed header also explicitly implements CLOCK_MONOTONIC_RAW. The roughly23-second numerical difference is therefore not evidence of an early capture. Do not sort/subtract those two domains.

Probe, method, samples and native/source hashes: astra_host_clock_witness.json, SHA256 **fabf5ca8968316529864dc28cff7eef9e7c39851db3c675bd06083772a14b42d**. Probe exit0; it establishes the current installed API domain, not a historical RAW-to-MONOTONIC calibration for Run610.

Final code retains the raw start value with an explicit domain name and no cross-domain comparison. It maps Python wall markers using client beginning/end dual-clock anchors plus rank-local end_info wall/monotonic anchors, with an explicitly assumed1ms margin. Client endpoint offsets differ by3720ns; that difference is neither a global error bound nor proof of no intervening wall adjustment. Rank3's local offset falls3370ns below the client endpoint minimum, illustrating sequential read latency and the inadequacy of the original narrow endpoint envelope. Same time namespace permits comparison of the relevant clock domains; it does not prove the conversion error bound.

The revised containment is a useful consistency check under the stated offset/no-large-excursion assumption. It must remain conditional. In particular, present-time RAW/monotonic readings cannot retrospectively provide a precise cross-domain calibration.

## Independent provenance and completeness checks

- Rehashed all604 raw files:56,783,476 bytes; all matched. Rechecked underlying Basis/Product/dispatch raw source hashes and admission joins.
- Rehashed all nine restored source files and all21 controller-pinned scripts/data paths against before/after manifests: no drift. Original invalid validator, failure log, cleanup and client-admission pins matched.
- Eight distinct rank sessions/windows bind current run tag, cohort5, cycles64–66, corresponding Basis/Runtime/Product evidence, exact Level0/ACL_AICORE_NONE, and device_N directory identity.
- Final script checks nonempty host/device payloads, nonempty FRAMEWORK/torch.op_range and torch.op_mark, and .done filesize equality for194 host/device data payloads plus16 end_info files:210 markers.
- Independently extended the filesize check to **all282** .done markers, including metadata/start/log/sample files; every corresponding payload exists and length matches. No empty data payload or missing data .done was found.
- Independently tested completed_payload on a positive fixture and truncated payload, wrong declared length and missing marker negatives; all behaved correctly.

A .done size declaration certifies the producer's recorded file length, not that no native records were dropped or every task was captured. SHA preserves bytes, not trace semantics. Whole-session/parser integrity, task correlation and missing-event diagnostics still require offline parser review.

## Admissible next use and remaining limits

Proceed with offline parsing of copies while preserving all604 source hashes. Native device/process/stream and Graph identities, CPU launch correlation, actual task membership, cross-rank HCCL matches, and missing/dropped records must be checked before reconstructing a critical path.

Host cycle64 start can include cycle63 device carry-in. Profiler.stop forces a cycle66-end device fence; start/stop and record_function instrumentation also add overhead. The three-cycle control window is not proof of three isolated complete native cycles. No slot-label overlap, scope nesting, entry marker or occupancy measurement by itself proves a necessary dependency, available overlap, compulsory traffic or resource-capacity bound.

Use this acquisition's actual acceptance/count/output ledger; it is not Run606 or Run99 W0. No acceptance/cycle optimization or algorithm change is supported. Current formal remains571.681 tok/s; strict Resource, Scheduling and Product endpoints remain null.
