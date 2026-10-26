# Run544 — independent posthoc admission of Run542 Host ledger

**PASS for posthoc Host-lineage diagnostic evidence. Run542's original controller outcome remains FAIL.** Corrected offline reduction can admit the retained raw data without rerunning the workload or changing the original cleanup/status records. This is not a formal E2E performance result, a semantic model correctness certificate, or a finite Bound.

## Failure cause and minimal correction

Run542 completed96 HTTP requests and both client phases, then its original server reducer failed on `measured: Runner did not report FULL target graph`. The saved original validator SHA256 is `88e5997e1e4edcab840e1b792eb8aa6c3dbf5907d62f9b81f0b363c5de595dee`, matching the original before/after script manifests. I independently executed those exact original bytes against the same retained data and reproduced exit1 at the same check.

The actual source defines `CUDAGraphMode.__str__` as `return self.name` in `framework/vllm/vllm/config/compilation.py:99`; the Runner records `str(_target_inputs.aclgraph_runtime_mode)`. All64 actual Runner records report `FULL`, the expected serialization. The validator and synthetic fixture incorrectly expected enum repr `CUDAGraphMode.FULL`. My prior independent preflight accepted that fixture and missed this source-level formatting distinction; its readiness PASS did not prevent this avoidable acquisition-admission failure.

The corrected validator SHA256 is `52454f39a1fdf45c2e52981ef22a7b7b1e3c299338b049141c286a1a685b6473`. Exact textual comparison confirms the only validator change replaces that literal with `FULL` and applies it to both phases. This strengthens warmup coverage while correcting measured serialization. The selftest changes only corresponding FULL/NONE fixture strings. It does not relax token, sequence, identity, phase, footer, count, graph-mode or cleanup requirements.

## Original acquisition cleanup

Original `cleanup_status.txt` retains run_exit=1, admission_exit=1 and final_exit=1. Stop, stopped-state verification, restoration, source SHA, script SHA and both comparisons all recorded0. The original source_before/source_after files are byte-identical; install/restore cover the same five sources, restored=true and helper_sha_drift=false. I independently checked all five current source hashes against those restored originals. The original script before/after manifests are also byte-identical: the reducer edit occurred after acquisition, not during it. Original stop evidence records all8 cards at3434–3445MB and successful idle verification.

Run542 must not be relabelled as a clean controller success and its failed final admission must not be overwritten. Run543/544 are separate offline diagnostic admission records with their own validator hashes.

## Independent replay and completeness

`run544/audit.py` independently replayed the original failure, then the corrected server reducer. Corrected reduction exited0 and is byte-identical to `run543/server_admission_replayed.json`. I separately reran the client validator in the acquisition's existing container environment; exit0 and output byte-identical to original `run542/client_admission.json`. An initial Host-only client replay lacked aiohttp and failed before validation; it is recorded separately and is not data evidence.

The accepted server reduction contains22120 substantive records from10 processes,96 unique requests across warmup48/measured48,96 exact client/server SSE sequences and64 all8 rank/cohort records. Complete data/footer SHA segments, process sequences, request/generation joins, phase marker identities, all8 Runtime parity, clipped Scheduler admissions and Output/API raw-count chains pass. `input_hashes.json` pins137 inputs including original/corrected validators, ledger data/footers, client artifacts, reports and cleanup/source manifests.

| Phase | Prebulk Scheduler G | Runtime incoming | Bulk admitted | Rank0 cohort cycles |
|---|---:|---:|---:|---|
| Warmup |485|49,152|48,667|282,321,298,304|
| Measured |401|49,152|48,751|289,299,322,308|

These satisfy the observed output accounting: ordinary prior admissions plus clipped bulk admissions produce49,152 external raw consumed tokens per phase. They describe this instrumented acquisition; the Runtime incoming count is not an external useful-token count and the cycles are not an attainable Scheduling floor.

## Bound admission boundary

Admit exact request/phase/Host raw-token ownership and API representation receipt as posthoc evidence, with the original acquisition failure visible. Do not infer text is an injective raw-ID encoding or that parser-suppressed control tokens were externally published. Fresh necessary Target work, exact prefill/seed/KV device-ready, typed collective completion, mixed-resource service capacity, controller cross-process clock identity and Host-progress/count-history alignment remain unresolved. The phase marker and sequential successful warmup barrier establish the controller's phase protocol; unsupported cross-clock duration comparisons remain excluded.

No workload rerun is needed to fix this schema mismatch. Use the retained ledger to select the next Bound-calibration question. Formal Current remains571.681tok/s; no finite Algorithm/Resource, Hardware, Scheduling or Product endpoint follows from this admission.
