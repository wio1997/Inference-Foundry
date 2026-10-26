# Run408 — Astra GO review of Run401 count-marker experiment

## Verdict

**GO for B as a bounded, instrumented local-ordering diagnostic, only after the pre-B gates below are positively verified. While A0 remains running or its closure gate is unavailable, immediate B launch is NO-GO.**

No source-level need to change the original copy or dependency path was found. This is not approval to promote original-schedule safety, a universal happens-before edge, or finite Bound endpoints. A0/A1 currently lack per-cycle acceptance trajectories; their aggregate agreement cannot discharge the analyzer's original-schedule extrapolation gate.

Review was read-only except this requested Markdown file. No service/NPU operation or source mutation. Inspected the five Run401 scripts, source_check_final/source_check_sol, validation, timing_preflight, runner and arm gate. Source snapshots and scripts are pinned below.

## Verified evidence

- The six current borrowed sources match the original SHA entries in source_check_sol.json. source_check_sol.json equals source_check_final.json.
- Independently generated each patch in memory, compiled it, and matched every generated SHA against the reviewed manifest. Existing synchronize, wait_stream, wait_event and non_blocking=True counts are unchanged. These count checks supplement, rather than replace, the call-site inspection.
- validation.json records successful CPU tests and helper import. Its npu_preflight_executed=false describes the earlier validation, not the later timing report.
- timing_preflight.json covers 8 devices x32 trials =256 records with passed=true. Current preflight script SHA matches producer_script_sha256. Maximum direct-versus-anchor discrepancy is 7.45058e-8 ms; maximum adjacent-event interval is 0.1003400013 ms. Reported direct uncertainty=0.2006800026 ms, anchor uncertainty=0.4013600051 ms, local marker allowance=0.6003240026 ms.
- These are empirical allowances from finite tests, not hardware error bounds. The Python Event wrapper documents timing and completion behavior but delegates to native code; it is not an independent proof of all backend semantics.

## Marker and copy semantics

Original path remains: current-stream wait on copy stream -> original pinned-destination nonblocking copy -> original production event. Normal next-cycle commit still synchronizes that same production event before reading the CPU destination. No diagnostic wait is inserted between the D2H reader and the next writer.

- R_BEGIN64 is on the actual copy stream immediately before the unchanged copy.
- R_DONE64 is after the original production event. It therefore conservatively follows copy completion. The original synchronize can return before R_DONE executes; this is expected and does not make CPU consumption invalid.
- W_PRE65 is before the full unchanged where/to/copy_ expression, after the preceding accepted-token write. W_POST follows its closing parenthesis. W_PRE brackets more than the physical copy kernel, but is safely earlier than the overwrite on that stream.
- With a valid same-device timer, R_DONE64 < W_PRE65 establishes a conservative observed read-before-overwrite ordering in B. A negative/overlapping margin is inconclusive: R_DONE is deliberately late and W_PRE deliberately early. It is not evidence of a race.
- CPU host markers bracket the existing production-event synchronize, first numeric count add, first distinct seq-mirror add, all mirror updates, and next copy launch. True first numeric count consumption is the add, not the earlier view slice.
- Cohort-end event synchronize/anchor operations occur after the ordinary count_history CPU drain. No new explicit hot-path synchronization appears in the generated patch. Lazy event initialization and Python marker/storage bookkeeping can still perturb scheduling; absence of explicit waits is not a zero-overhead guarantee.
- Actual stream IDs, source/destination pointers, frozen entry64/65, production-event identity and generation tracking are useful gates. Per-event storage dictionaries reuse the start snapshot with pointer checks; they are not independent full live storage/shape captures. Pinned source code and an unchanged state tensor make this adequate for this narrow diagnostic, not a general alias verifier.
- New hooks run on all cycles with sparse selected work. The preflight's four enqueue allowance does not cover all possible wrapper/bookkeeping effects. Do not treat margin greater than that allowance as mathematical original-schedule proof.

## Host consumer coverage

The first matched seq_lens_cpu clone is an actual numeric memory read, with a pointer join to the recorded mirror. A subsequent max/item may also be observed if it is the first matched site. This can establish one linked consumer, not every consumer across all attention backends.

The helper stops after the first downstream_pre. Consequently, when the clone is captured first, its derived_ptr is retained but the later derived-clone consumer is not traced. Do not describe this as a demonstrated full clone-to-DSA lineage or proof of the ultimate semantic consumer. A dsa_cp/native/other backend that bypasses these sites can leave downstream absent; that must stay INCONCLUSIVE, not be filled in from source intent.

Parking is intentionally excluded. The helper flags selected-cycle parking, and the analyzer rejects that branch. No conclusion here closes the separate conditional parking-generation path.

## Hard gates before launching B

These are operational requirements; the current arm runner does not enforce them all automatically.

1. A0 completed its exact48+12 workload and loop078_count_arm_gate reports PASS. Its sources must match the reviewed **original** SHA manifest, not merely match each other before/after.
2. A0 shutdown completed; all worker/launcher processes are gone, endpoint unavailable and all8 devices have the recorded idle state. A failing /health request alone does not prove absence of an initializing, stuck or unhealthy service. Verify no orphan clients and no second launch in flight.
3. Use a fresh B evidence directory and patch state directory. Record the seven script SHAs below, the exact preflight report hash and original/installed six-source manifests. expected-check protects generated source hashes but does not pin the imported helper's contents.
4. Current container torch/torch_npu versions match the preflight stack; the isolated all8 timing report passes and has the verified producer hash. No preflight rerun while A0 or another service is using devices.
5. Effective service environment excludes other event/route captures, profiler/diagnose conditions and unintended inherited settings. The helper rejects two named competing capture variables and known runtime diagnostic flags, but the runner's unset variables do not clear arbitrary container-inherited environment. Verify frozen model/config/lifecycle provenance.
6. B launches only after those gates. After B, stop successfully and verify process/device release **before** source restoration; verify every restored SHA. Then launch clean A1.

Runner hazard: cleanup suppresses stop errors and invokes restore even if stop failed; it also suppresses restore errors. Therefore a successful shell exit or source_after file is not sufficient lifecycle evidence. The coordinator must verify the stop actually succeeded before permitting source restoration. This review does not authorize stopping or restoring an active A0.

## Analyzer and arm-gate interpretation

The arm gate provides useful exact60 POST, successful 48+12 requests,1024 outputs, client concurrency,40 reports/all8 parity, FULL graph, zero oracle/ModelRunner calls, mirror-exact, restored-source and idle-HBM checks. Limits:

- Before/after source equality does not independently prove the expected original version.
- Runtime wall is a sum of per-cohort maximum local rank durations, not a globally synchronized makespan.
- Client concurrency is checked within each file; sequential invocation supplies the between-file ordering premise. Runtime request-set cardinality and report identity still need association with the server/client ledger.
- Server POST count is provided by a file; it does not check server Running+Waiting peak or prove absence of another non-benchmark client by itself. Inspect clean process/request provenance.
- B arm gate checks capture count/run_id; the separate analyzer is required for event contents, layer-independent consumer metadata and timing consistency.
- acceptance_sha256=null honestly states that A0/A1 expose aggregate cycles/window means, not complete per-cycle acceptance. Never synthesize a trajectory hash from cycle totals or means.

The analyzer deliberately preserves local B results if control_gates fails. Therefore top-level ACCEPTED is only its scoped local result, not a clean frozen-workload certificate. In particular, a false/missing external HTTP/lifecycle/source control may be caught by control_gates while local ACCEPTED remains possible. For publication as frozen c12 evidence, independently require **B arm gate PASS plus the external provenance checks**, and state control_gate_passed/original_schedule_extrapolation_supported separately.

With the currently designed A0/A1 artifacts, control_gate_passed and original_schedule_extrapolation_supported must remain false because per-cycle acceptance is unavailable. This is not a reason to discard valid B event observations. Even if a future exact-trajectory A/A gate passes, the two-control min/max wall envelope and empirical marker allowance are supporting evidence only, not a bound on all local schedule perturbations.

No cross-rank clock subtraction, causal all8 makespan, removable duration, compulsory HBM claim, or finite Product ceiling follows. Retain null finite endpoints.

## Script SHA256 at review

```
cca1ac32d846b5df89ff414e27b9f693c5770d264df04807d94235f9f69bab9d loop078_count_markers.py
463d3b2654332b1cf9cf2f74ad188c77fc85554dc01036de0f7c69e51c61c4b9 loop078_count_markers_patch.py
a624384cfbbf9b88e2501164d008fac88f274b741dee166fc7a25029f2a09aa0 loop078_count_markers_analyze.py
5d5b29fa2565760a394faa967ce6ac29338ceab63bd22c86bffa281ed2d8ed50 loop078_count_markers_preflight.py
824e9504b496a4a5d9c968aaa76e1cf893536aadc496a08216aa4db986ec728d loop078_count_markers_selftest.py
1998585b863cbae844604eb489d811deb3f5fee44aeb6a2194b441f6192f2247 run_loop078_count_marker_arm.sh
d35d91deedcc52356a0d5420b72e6fd3f02de178facafef250fe19fbb6ed8e8d loop078_count_arm_gate.py
```

Confidence: high in inspected patch placement and unchanged explicit dependency/copy semantics; moderate in generalization beyond these selected B observations. Pre-B service closure and effective environment are pending operational gates, not facts certified by this read-only review.
