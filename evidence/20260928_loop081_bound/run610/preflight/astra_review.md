# Run610 independent Astra High preflight

## Verdict

**SCOPED LIVE-READY** for one guarded fixed-algorithm instrumented Current acquisition. This supersedes the initial NOT-READY assessment: the profiler fence is now explicitly admitted as observer behavior, and an all8 raw-profile admission gate plus CPU corruption tests were added. No service or NPU workload was started by this review; installed model/runtime sources were not changed.

The authorization scope is acquisition, followed by source/raw/cleanup and native-trace review. It is not a formal E2E result, an unperturbed execution-time measurement, a complete typed dependency packet, or a finite Resource/Scheduling/Product endpoint.

## Exact reviewed identity

| File | SHA256 |
|---|---|
| scripts/loop081_profile_patch_run610.py | 74f86669895718feb07fd750003a1b89bace55ae6e5da5731d33f261c57992d2 |
| scripts/loop081_profile_validate_run610.py | 4fb534060d93e83a7c35cb141028d199e3f644d03fb891f748360f882e2833bc |
| scripts/loop081_profile_selftest_run610.py | 5cf43de8dbafa9bcd4955ccc88adb6aa30d9708715145a1691f5de96da2b1223 |
| scripts/run_loop081_profile_run610.sh | 70f9509a26bb9ad7ac3e86883d13903d8dfe639273b0fae86924182172ec13af |
| run610/preflight/patch_check.json | 39669c41b03cdb178814eabcc5284e72d24837e8642df7d92fe9625031de0586 |

The independent prepared() invocation reproduced patch_check exactly, verified nine unique paths against original SHA, and compiled all nine generated sources. Installed torch_npu profiler source pins: profiler.py 1a04902572383b74a401449e73929b2c4ce728d6c81bb59341876b03c7a82168; profiler_interface.py 8183b6470d552c33907784252f4009050aa7066955371c460232bc60b8e0af65; experimental_config.py ce05c3d9abaa79329e2a08b59014fefdbd5a3c7ceae1e0266c5ab688d764fb3b.

## Independent checks and execution safety

- bash -n passed. Final new validator CPU fixture independently rerun: positive passed; missing rank, missing session, Basis SHA corruption, Host clock violation, wrong level, and missing device raw were rejected. Exit 0. These are synthetic parser/provenance tests, not native profiler success evidence.
- Actual generated cohort hook was executed with mocked objects: warmup unarmed cohorts select no profile, armed cohort5 selects its directory, later cohorts do not. Extracted existing profiler begin/end methods were CPU-mocked: cycle63 does not start, cycle64 starts, and stop occurs after completing cycle66 when cycle_index is67.
- Profile hook is after served-cohort admission and before Runtime.run. The ordinary proposer owner ContextVar fix and Run606 hooks are inherited unchanged. Warmup is completed and validated before measured arm creation. 48 warmup +48 measured, c12,1024, exact96 POST gates remain.
- New controller explicitly freezes SYNC_TARGET=0 and MEMORY_ACCESS=0. Final validator requires exactly Level0 and ACL_AICORE_NONE; the earlier negative-only “not Level1” test was insufficient and is superseded.
- Fresh output directory, lock, prestart idle checks, exact RUN_TS-owned stop, bounded release check, nine-source backup/restore with SHA/race checks, before/after script/source comparison, and final nine-field cleanup status remain. Restore is gated on proven stopped state; failed acquisition does not become success merely because cleanup succeeds.
- New selftest runs before source installation. Basis/Product/dispatch validators run for the current Run610 tag, followed by profiler admission before parsing. No source restoration occurs while live service is unproven stopped.

## Observer and evidence boundaries

1. Installed profiler_interface.stop_trace explicitly calls torch.npu.synchronize() for NPU activity. Runtime calls profiler.stop at cycle66 end. Thus this run introduces a device fence even with SYNC_TARGET=0; “no new hot-path sync” would be false. Start/stop/finalization and CPU scope overhead also exist. PROFILE_SCOPES=1 applies to all Runtime cohorts, including warmup, not only the selected window.
2. Start at Host cycle64 does not drain prior work: cycle63 device tail may enter the raw session. Stop at66 drains queued work. Window first_cycle64/count3 is an instrumentation control interval, not proof of exactly three complete native cycles. Never select the first three Graphs or split by Host midpoints as cycle attribution.
3. Final admission binds eight rank windows, one raw session per rank, activity/Level0 configuration, nonempty host/device raw with device completion marker, every raw-file SHA, and same-run cohort5 Basis/Runtime/Product hashes. Profiler monotonic interval must be inside same worker's Runtime Host interval. Window wall timestamps are not subtracted from monotonic timestamps.
4. This proves raw acquisition provenance and Host containment only. Native device/process/stream identity, Graph generation/replay, CPU launch-to-device correlation, carry-in/out tasks, HCCL matching and critical-path edges still need post-acquisition parsing. Rank labels alone are not a native engine dependency certificate.
5. Existing wrappers do not add typed KV storage generations or native last-writer/first-reader labels. Run610 is a native-trace substrate for the Run609 design, not completion of all its typed-node requirements. Draft65→Target66 closure remains conditional on identifying the actual first consuming work; entry/Host markers are not automatically that read.
6. Use this run's actual Basis counts/positions/Host parks/outputs and observed Graph/shape. Prior Run606 selected sums47/59 and label overlap49 are hypothesis priors, not admission thresholds or same-W0 identities. Profiler perturbation may affect Host timing; independent diagnostic W0 must not be equated to Run99 or Run606.
7. Active/logical/physical rows, current traffic and measured task durations are not compulsory work or resource-capacity upper certificates. Native occupancy is not exposed critical-path cost. All strict endpoints remain null; formal Current remains571.681 tok/s.

## Post-acquisition rejection gates

Any failed Basis/Product/dispatch/profile identity gate, missing/partial rank trace, parse/correlation failure, source/script mismatch, or nonzero required cleanup action blocks promotion of the affected evidence. Preserve raw artifacts and actual exit status. A successful acquisition permits an instrumented Current timeline with explicit unresolved edges; it does not by itself admit a saving, attainable reschedule, or formal Product Bound.
