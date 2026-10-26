# Run529 — source-only formal Host ledger patch

2026-09-27. No borrowed/project source patch was installed. No service or NPU
execution was launched. Formal Current and all Bound endpoints are unchanged.

## Deliverables

- `scripts/loop079_formal_ledger_server.py`: per-process JSONL sidecar, raw
  Scheduler before/after G and immutable input copy, Runner rank/cohort/slot
  handoff, fixed-serving Host progress and existing post-drain token/count
  histories, OutputProcessor request/queue raw IDs, and every API serialized
  SSE yield plus raw input/parser outcome. Each record carries run/process
  sequence, monotonic clock, and explicit phase marker identity.
- `scripts/loop079_formal_ledger_patch.py`: SHA-pinned five-source reversible
  generator. `check` compiles all patched sources and helper without mutation.
  `install` and `restore` require `--offline-confirmed` and a guarded controller.
  Backups of all five sources and the manifest are fsynced before first source
  mutation. Each source install/restore uses a same-directory temporary file,
  fsync, atomic replace, and directory fsync; it never truncates a source in
  place. Restore accepts a mixture of original and expected patched bytes
  after a partial install, and refuses any third hash. Restore requires the
  manifest's source key set to equal the five pinned sources exactly. A helper
  SHA drift after service shutdown is recorded but cannot block restoration
  when source paths, original backups, and patched/current SHA all verify.
- `scripts/loop079_formal_ledger_phase.py`: atomic `warmup` then `measured`
  marker with append-only transition history. Measured transition requires the
  preceding warmup marker and SHA of a completed client report. The controller
  must verify the barrier and no in-flight requests; cohort ordinal does not
  establish phase.
- `scripts/loop079_formal_ledger_selftest.py`: source-only CPU fixture.

The source manifest in `patch_check.json` covers exactly:

1. `/data/wio/vllm_ascend_26/framework/vllm-ascend/vllm_ascend/patch/platform/patch_kv_delivery_preemption.py`
2. `/data/wio/vllm_ascend_26/framework/vllm-ascend/vllm_ascend/worker/model_runner_v1.py`
3. `/data/wio/Inference_Foundry/runtime/fixed_serving.py`
4. `/data/wio/vllm_ascend_26/framework/vllm/vllm/v1/engine/output_processor.py`
5. `/data/wio/vllm_ascend_26/framework/vllm/vllm/entrypoints/openai/chat_completion/serving.py`

Final script SHA256: helper `5c381122b9b1781bad2733c2f4b1e296ff82e929baf2de247225085983f3df20`,
patch generator `3dd36594271405bfb852327ea0e0200d9ddabda0eab4454d7241794fab31f64f`,
phase control `e75debc5b33a45195dedab6120bf5c9e04e05694217ccd9edf26a1c3ecb45de8`,
selftest `d03d31ccb77217ddaa461ebd1aa60e3d5d8237edca299baa29a985f642046ff2`.

## Source-only results

- `python3 scripts/loop079_formal_ledger_patch.py check --record evidence/20260927_loop079_identity/run529/patch_check.json`: **exit 0**, five exact original SHA pins, five patched-source `compile()` checks, helper `compile()` check. Final check record SHA256 `db0692b35d83d3f264cb3b0d9d6aa84f9fc1d805101004737b500b24271e3bfe`.
- Host `python3 -m scripts.loop079_formal_ledger_selftest`: **exit 0**, log `selftest_host.log` SHA256 `9be1e1912c0e6cdb23e34a0744110d2a6adee2df622e1ecc417c50418f1c03f3`.
- Target container `docker exec -w /data/wio/Inference_Foundry vllm-ascend26-dsv4f-w4a8 python -m scripts.loop079_formal_ledger_selftest`: **exit 0**, log `selftest_container.log` SHA256 `5fa609743b84d3700aa682b954c66d2195a0679705fc540d8dd148c37282eae0`. Test covers exact anchor replacement, original SHA pins, syntax, explicit offline flag denial, fixture-only installation, injected interruption after a partial temporary write leaving original SHA intact, injected failure after first source replacement followed by mixed-hash restore despite a changed helper SHA, rejection of missing and extra manifest source keys without source mutation, idempotent install/restore, immutable incoming token copy, clipped admitted prefix, exact DONE payload, warmup→measured markers, and ledger truncation rejection. Fixture paths were temporary, not the five live source paths.
- Target container `from scripts import loop079_formal_ledger_server`: **exit 0** with repository root on `sys.path`.
- Five live source SHA after checks remain respectively `5ba894b9...`, `004dbd0d...`, `137f1cca...`, `ee103512...`, `860e27d5...`, matching preflight. None contains the marker.

## Controller interface

Set `EXTREME_FORMAL_LEDGER_DIR`, `EXTREME_FORMAL_LEDGER_RUN_ID`, and
`EXTREME_FORMAL_LEDGER_PHASE_FILE` in all server processes. Call phase
`warmup --run-id ID --marker PATH --transitions PATH` before admission.
After externally verifying 48/48 warmup and no in-flight request, call phase
`measured` with the same arguments plus `--completed-client-report PATH`.
Run the frozen measured48 without changing the request body.

All events emit into `pid<PID>.jsonl` with `event,run_id,pid,seq,monotonic_ns,
phase,phase_generation,phase_marker_sha256,phase_run_id_matches`; sequence is
per process. `clock_origin` also records boot ID/time namespace/resolution.
Terminal flush occurs at bulk Scheduler append, Runtime drain, Runner done,
Output finish, and API DONE. Post-drain `token_history/count_history` are copied
only from CPU tensors already present in serving; no per-cycle D2H was added.
Runner `cohort` matches its existing 1-based `rankN_cohortM.json` index; phase
is never inferred from this number.
Each flush writes a separate `pid<PID>.flush.jsonl` footer after fsyncing the
JSONL blob, with reason, committed record count, first/last sequence, byte
offsets, and SHA256 of the exact blob; the footer and directory are fsynced
before return. `verify_flushes()` rejects missing, truncated, changed, or
uncommitted ledger bytes. The API `api_consume.output_request_id` is the
external RequestOutput ID; `api_request_id` is the API response ID. It makes
no claim to contain the EngineCore internal ID. That ID must be joined via
`output_add_request`/`output_receive` and exact IDs, not positional timing.

## Remaining gates and limits

This is a **narrow Host lineage subset**, not a Scheduling Bound certificate.
Cache lookup/admission ranges, exact prefill/device-ready and seed/device-ready
events, selected fresh semantic Target rows, actual first Target submission,
native typed terminal completion, and all8 collective-ready edges remain
unknown. `phase` is an explicit controller marker, but admission requires
client and server IDs plus complete 96-request joins; the reducer must reject
unknown phase or a phase transition while earlier requests remain active.
API token consumption does not imply raw-ID publication when parser suppresses
a delta. The serialized SSE payload can be compared byte-for-byte with the
client; it does not make text an injective token-ID map.
The per-event phase marker read and JSONL/fsync logging perturb Host execution;
this remains a lineage diagnostic and its measured duration is not a formal
performance result.

Before any live diagnostic, independently check the exact five source SHA,
actual container invocation/import path, owner classes, service idle state,
phase barrier, all8 ranks, startup/benchmark/validator exits, and cleanup.
`--offline-confirmed` is an assertion by the guarded controller, not a process
discovery mechanism. Any failed live install requires stop/idle verification
before invoking restore with the saved state directory. Do not promote a
partial or invalid diagnostic to formal evidence.
