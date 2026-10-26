# Run437 final preflight review v3 — Astra

2026-09-27. Final runner source review, mocked stop-verifier tests, bash syntax check, in-memory generated-patch/helper compilation and read-only source/output-path checks. No real docker probe, service command, NPU query/computation or patch installation was executed here.

## Disposition

**Static preflight PASS: the reviewed runner may proceed to the planned real diagnostic, conditional on its actual startup idle-resource/process/import/source checks passing.** No remaining demonstrated P1 service-safety or unconditional certificate-promotion defect was found in this review. This is permission to collect the scoped diagnostic, not acceptance of future results or an all43 row certificate.

The v2 cleanup P1 is fixed: both the /proc docker probe and npu-smi invocation now explicitly propagate failure via `|| return $?`. Under cleanup's `set +e`, either failure prevents successful verify_stopped status, skips restore, and contributes to a nonzero final exit.

## Pure CPU shell tests

`cleanup_v3_mock.json` records tests on the extracted current verify_stopped function. curl/docker/npu-smi/python3 were replaced entirely with shell functions; no external device/process probe ran.

| Case | Injected failure | verify_stopped exit | Result |
|---|---:|---:|---|
|docker fails; HBM path would succeed|1|1|PASS|
|npu-smi fails; final parser would succeed|7|7|PASS|
|all stop checks pass|0|0|PASS|

The failing probes return early; a successful later parser cannot mask them. `bash -n` exit0. Six current original and generated patched SHA pairs still match source_check_v2.json and compile in memory; all three Python scripts compile (exit0). Acquisition output paths were checked read-only and are fresh at review time.

## Safety gates retained in final runner

- Refuses preexisting acquisition outputs and a healthy server; checks live /proc processes while skipping state Z zombies, orphan benchmark processes, Host memory and all-eight-card idle device state before installing.
- Performs actual-container namespace import, source check and before SHA snapshot before patch install.
- Traps EXIT/INT/TERM; reports run/stop/verification/restore/SHA/final codes. Stop failure or unproved stop refuses restore. Restore failure/source mismatch cannot produce an overall successful exit.
- Requires60 POSTs, correct48+12 response/output lengths,40 capture and40 runtime files, then runs offline validation.

Actual health/process/HBM/import gates were **not executed by this review**. They remain mandatory runner gates at launch; freshness and reviewed hashes must still hold then. A stop failure must be handled from cleanup_status.txt and preserved patch state; do not manually bypass the refusal while workers may be live.

## Evidence scope after acquisition

Keep `diagnostic_valid`, `all43_row_identity_certificate=False`, all43 CONDITIONAL_NATIVE and external_retention OPEN. Route/group parity, selected FULL kwargs→state storage and five CP value-prefix maps are the intended checks. Native row composition, per-layer branch/layer→CP binding, graph all-slot semantics and external request attribution remain open as described in v2. A passing diagnostic cannot promote W_minus, C_plus, Scheduling or Product automatically.

Review final validation/client/cleanup artifacts after run completion before using conditional retained unions. No formal performance or overhead conclusion follows from this instrumented run.

## Final reviewed hashes

- `loop079_row_capture.py`: `632d87dfbb1b84109eb0740c5bdca8446fa41ec075b2c55453faec5ad12de377`
- `loop079_row_capture_patch.py`: `6edb6247c1de807a4ea229ac17fa68b99e0ce4bf2beec5e26fa357028c47e103`
- `loop079_row_capture_validate.py`: `0d6f67f1ed879cbbf6419774402069a2f43893a702f84fe366fd24dac7181381`
- `run_loop079_row_capture_run437.sh`: `ab845a613ce44f61c165d2a6fd5b060885101e83723cb843478bf81b38f24589`
