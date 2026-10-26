# Run535 — independent final Host-ledger live-readiness review

2026-09-27. **FAIL for live acquisition on the reviewed snapshot.** The Run529 atomic-file fix and API field correction are substantively correct; remaining controller and barrier defects must be fixed before launching. No live source mutation, service startup or NPU execution was performed by this review. Current Formal E2E stays 571.681 tok/s; no Bound endpoint is promoted.

## Reviewed snapshot

- Server helper SHA256 `5c381122b9b1781bad2733c2f4b1e296ff82e929baf2de247225085983f3df20`.
- Patcher `6cd41503eb8a7958e6069df1469437a95feb3d700ef90deefaf3c15cd408a022`.
- Phase writer `e75debc5b33a45195dedab6120bf5c9e04e05694217ccd9edf26a1c3ecb45de8`.
- Selftest `4e4324a86f22164cb9e903e4d00bf22920b59735770ac5dc286845e058b3e7da`.
- Revised remote barrier `30aa157773de3154c47d764ef5eb061c40d28651379f53504132a86030f96e79`.
- Local controller `/Users/wio/work/.remote_edit/run_loop079_formal_ledger_run533.sh`, SHA256 `fcbb74d693ba603432a290dab8139a9aa0e3c1a3ead752e4f9d6ae46c0cb0fa6`; not yet installed remotely.

The initial barrier used `glob("pid*.jsonl")` then rejected the new footer filenames. The revised barrier above fixes this and checks footer/data file-set equality and `verify_flushes`. Findings below refer to remaining defects in that revision.

## Required executable corrections

1. **Controller stop verification must propagate every failed probe.** `cleanup` runs with `set +e`. `verify_stopped` runs its process probe, then npu-smi, then a Python HBM check without explicit failure returns. If the process probe fails (including live VLLM process detection or docker failure), a successful HBM check replaces that failure with function exit 0 and permits source restore. Add `|| return` to each probe and use this same gate before installation. Process count <=10 and health failure are insufficient to prove absence of an unhealthy VLLM worker. Preserve failed probe status in cleanup evidence.

2. **Barrier must use actual API identity fields.** `api_consume` now correctly emits `output_request_id` and `api_request_id`, with no `external_req_id`. The revised barrier still includes it in the generic external_req_id check, therefore all normal acquisitions fail. Require its two external fields equal one another and belong to the exact client response set. Keep true internal mapping through output_add/output_receive/output_queue.

3. **Complete the barrier's terminal and identity gates.** Require Runner target_graph_mode FULL and exact Host mirror; terminal Scheduler bulk stopped=true, stale=false, resumable=false, g_after=max_tokens=1024 and delta arithmetic, with admitted prefix; match output_queue internal/external/output_request_id against the same bijection. Validate footer PID/run identity in addition to segments/sequence/digest. Keep same ordered12 IDs for handoff/drain/done on all8 and exact48 union. Full Runtime history/raw parity and ordered SSE reconstruction may remain the final reducer's responsibility.

4. **Verified restore must survive a separately changed helper.** The patcher still rejects helper SHA drift before restoring known, verified source backups. A helper change should invalidate acquisition, not prevent recovery. Restore also needs exactly the SOURCES key set (as install already requires). Retain refusal of third-party source hashes. The same-directory temporary write/fsync/replace and durable original backups already fix the torn-source issue.

5. **Controller output isolation and lifecycle.** Reject nonempty/existing run output or use an atomically created dedicated run directory and controller lock before any overwrite; the present mkdir -p and early probe writes can overwrite evidence from an earlier attempt before marker/state checks fail. Install INT/TERM handling that exits through cleanup. Pin collector/helper/validator/controller inputs before starting and preserve explicit stage exits. Final admission must run after stop, idle verification, restore and five-file hash comparison, with all mandatory exits zero; calling final_admit inside the body before EXIT cleanup cannot establish this.

6. **Finish and independently exercise the final reducer.** At review time the controller references server_validate and final_admit scripts that do not yet exist. Require exact96 unique IDs/48 per phase, two phase-marker identities and a valid warmup barrier, all8×8 cohort identities, ordered raw Runtime→Scheduler→Output/API accounting, byte-identical server/client SSE payloads after one framing removal, valid complete flush segments and process origins, and no unknown/stale/resumed request gap. Validate container imports in the actual launch environment. Missing or rejected data stays diagnostic evidence and does not silently become a finite Bound.

## Independent source/CPU verification

Host `python3 -m scripts.loop079_formal_ledger_selftest` exited0 on the reviewed package, covering five temporary source copies, interrupted temporary write, mixed partial-install restore, idempotence, phase and ledger truncation. Read-only patch check exited0 and reproduced manifest SHA256 `db0692b35d83d3f264cb3b0d9d6aa84f9fc1d805101004737b500b24271e3bfe`; all five actual sources still match original SHA pins. No source was installed. A target-container bash login invocation from the real repository working directory successfully imported the helper; the current trailing-empty PYTHONPATH component resolves the repository. This does not by itself prove every spawned process preserves that environment; the controller should retain explicit repository import availability.

The flush implementation now fsyncs data, writes a sidecar containing committed records, first/last sequence, offsets, SHA and reason, then fsyncs sidecar/directory. This supports a complete-record reducer. It does not make in-memory records externally committed, and completeness still needs all expected terminal events. The atomic patch implementation preserves complete old/new bytes on interruption and source modes. Narrow source-hook design passes with the recovery caveats above.

## Bound interpretation

Even a passing96-request acquisition establishes this instrumented Host lineage only. Per-event phase-file reads and JSON/fsync alter timings. It supplies neither fresh necessary Target multiplicity nor prefill/seed/KV device-ready, typed collective completion, mixed-resource service capacity, or a Product lower-bound DAG. Those remain separate evidence questions. Use accepted joins to choose the next discriminating Scheduling/Resource measurement; do not transfer its diagnostic TPS into the formal571.681 result or manufacture a finite overall ceiling.
