# Run401 implementation review — CPU/source-only, not installed

Prepared scripts:

- `scripts/loop078_count_markers.py`: sparse device/Host hooks; event objects and
  fixed record slots allocated at cohort start, cycle selection frozen at step
  entry. Original source tensor, production event and stream dependencies remain.
- `scripts/loop078_count_markers_patch.py`: six-source reversible patch;
  read-only `check`, hash-pinned `install`, hash-verified `restore`.
- `scripts/loop078_count_markers_analyze.py`: same-device margin, Host ordering,
  storage/generation and first numeric consumer checks. Local instrumented
  observation is separate from A/A-based original-schedule extrapolation.
- `scripts/loop078_count_markers_preflight.py`: isolated NPU known-order two-stream
  timing and D2H host-visibility calibration. **Created, not executed.**
- `scripts/loop078_count_markers_selftest.py`: synthetic CPU-only analyzer and
  generated-source tests. Never imports torch or executes the NPU preflight.

## Patch sites and assertions

Six sources: runtime/extreme_decode.py (step-entry tag), runtime/fixed_decode.py
(exact overwrite bracket), bootstrap/vllm_dspark_handoff.py (copy, original sync,
first count add, mirror updates, caller reason), runtime/fixed_serving.py
(preallocation, true progress tolist consumer, end export), framework
spec_decode/utils.py (first mirror clone), attention/dsa_v1.py (first max/item).
The current adapter uses public execute -> _execute; normal commit is patched
in `_execute`, parking/validation in their own methods.

`R_DONE64` is recorded after the existing production completion event, so that
existing synchronize may return before this conservative timing marker executes.
`W_PRE65` is directly before the unchanged full num_sampled.copy_(where(...))
expression, after accepted_tokens.copy_; W_POST follows its closing parenthesis.
Each selected event records actual current stream, device, generation and source
storage identity. CPU timestamps bracket the existing sync and actual numeric
add (not the preceding slice), first distinct seq mirror update, all mirror
updates, next launch, progress read and first matched downstream numeric read.
All new synchronization lives in cohort-end export, after ordinary history drain.

A downstream read is observed only when its pointer aliases a pre-recorded CPU
mirror or the explicitly tracked clone output. Unmatched aliases/other backend
paths leave downstream INCONCLUSIVE. Utils clone and DSA max/item are source
pinned candidate sites; actual backend coverage remains a runtime gate.

The helper's eager-Draft field `adapter.proposer.use_cuda_graph` is defined and
forced False by installed dspark_proposer.py; source stream/event/destination
fields are allocated by DirectDSparkHandoff. New helper imports only standard
library until an enabled cohort starts. Container-context helper import passed.

## Exact next commands (Sol owns lifecycle; none run here)

On the remote host, while service is stopped, first source-check:

```sh
cd /data/wio/Inference_Foundry
python3 scripts/loop078_count_markers_patch.py check --record evidence/20260927_loop078_bound/run401/source_check.json
```

With all eight devices free, isolated timing calibration inside the existing
container (this performs NPU work and must precede workload timing):

```sh
docker exec vllm-ascend26-dsv4f-w4a8 bash -lc 'cd /data/wio/Inference_Foundry && python3 scripts/loop078_count_markers_preflight.py --devices 0,1,2,3,4,5,6,7 --trials 32 --output evidence/20260927_loop078_bound/run401/timing_preflight.json'
```

The preflight uses known-order waits only in its isolated test; it confirms
same-device cross-stream elapsed-time consistency and exact CPU destination
contents after the original completion event. It exercises unique lazily
initialized events, records package/API source, and derives empirical timing
uncertainty and local marker allowance from worst observed discrepancies,
adjacent-event intervals and host enqueue durations. These finite-sample values
are conservative empirical allowances, **not hard bounds or a universal API
contract**. Inspect event implementation/documentation in the emitted report
before interpreting runtime margins; inability to justify semantics blocks the
gate even if a numerical test passes. No runtime preflight report is fabricated.

Run clean A0 first under the frozen warmup48 + diagnostic12 lifecycle. For B,
stop service, then install only the exact reviewed generated patch:

```sh
python3 scripts/loop078_count_markers_patch.py install --expected-check evidence/20260927_loop078_bound/run401/source_check.json --state-dir /tmp/extreme_run401_originals --record evidence/20260927_loop078_bound/run401/install.json
```

B service environment adds `EXTREME_COUNT_MARKER_DIR=<absolute capture dir>` and
`EXTREME_COUNT_MARKER_RUN_ID=<unique B run id>`. Disable other route/event capture,
profiler and diagnostic conditions. Capture env must persist through five
cohorts; helper labels first four warmup and fifth diagnostic and analyzer checks
exact coverage. Use clean Run394 lifecycle/request ledger controls, not an ad-hoc
extra health/benchmark request. A0/A1 leave these variables unset and sources clean.

After B service is stopped:

```sh
python3 scripts/loop078_count_markers_patch.py restore --state-dir /tmp/extreme_run401_originals --record evidence/20260927_loop078_bound/run401/restore.json
```

Run clean A1. Populate `manifest.template.json` with measured control results,
real ledger hashes, and paths/SHA256 of preflight and supporting evidence.

```sh
python3 scripts/loop078_count_markers_analyze.py <capture-dir> --manifest <completed-manifest.json> --output <analysis.json>
```

## Limitations and interpretation

- No patch install, service launch, NPU operation or timing result occurred here.
- External API/request/client IDs, clean lifecycle, FULL Target and no oracle
  calls are external ledger gates; hooks only preserve cohort-local slot identity.
- A0/B/A1 acceptance mismatch or B wall outside the A/A envelope disables original
  schedule extrapolation, while preserving per-rank instrumented local results.
  Original-schedule extrapolation is empirical, never a new happens-before edge.
- Negative/uncertainty-overlapping margin is INCONCLUSIVE, not a demonstrated race.
  Explicit captured generation/storage error or external B mirror correctness
  failure is REJECTED. Missing artifacts/coverage is INCONCLUSIVE.
- End anchor comparisons carry a separate empirically measured anchor uncertainty;
  direct nearby cross-stream margin is preferred when supported.
- Local/global clock types are never subtracted. No cross-rank makespan or
  removable-cost inference; Algorithm/Hardware floors and Product ceiling null.
- Event lazy initialization and marker calls can alter slack; the isolated
  preflight and A/A controls do not turn a finite-sample allowance into a proof.
- Runtime state is single-process, sequential-cohort only. Conditional parking
  is explicitly excluded; a separate parking-generation experiment is needed.

Validation: generated six sources compile; unchanged compound-copy statement and
counts of existing synchronize/wait/copy calls verified; synthetic pass and
corruption tests pass; all five new scripts compile; container helper import
exit 0. See validation.json. Borrowed source hashes unchanged after checks.
