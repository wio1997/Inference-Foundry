# Run536 — two-phase formal Host ledger reducer, source-only

2026-09-27. No service, patch installation, NPU run, or formal E2E was performed. Current remains 571.681 tok/s; every finite Bound endpoint remains unadmitted.

## Artifacts and verdict

- `scripts/loop079_formal_ledger_server_validate.py` SHA256 `88e5997e1e4edcab840e1b792eb8aa6c3dbf5907d62f9b81f0b363c5de595dee`.
- `scripts/loop079_formal_ledger_server_validate_selftest.py` SHA256 `74c72204b3d8a7977b04b9bd6c04582c2a4d28110a21994936c606389885f66c`.
- Host and target-container `python -m scripts.loop079_formal_ledger_server_validate_selftest`: exit 0; one full 96-request synthetic positive and ten deliberate negatives, with eight distinct rank PIDs, 64 rank/cohorts, and a G=1 clipped terminal bulk case. Logs `selftest_host.log` and `selftest_container.log`, both SHA256 `c0330db7e4d0ef6bea2595504ee0c823943c66e49ec4debe206166d179defc42`.
- Host `py_compile` of both files: exit 0.
- Verdict: **source-only reducer ready for independent preflight**, not yet live diagnostic admission.

## Admission logic

The reducer checks every PID's fsynced flush footer against exact committed byte ranges, SHA256, record counts, contiguous sequence, process/run identity, and common boot/time namespace. It rejects incomplete footers, truncated or uncommitted ledger bytes. It requires 48+48 client IDs from the separately admitted client report; the warmup report's exact request index and hash bound into the measured transition; both phase marker hashes and ordered generations; and all8 rank/cohort handoff, Host cycle, post-drain, and Runner coverage (warmup cohort 1–4, measured cohort 5–8). Each rank/cohort's Host records must stay in one PID, and eight ranks must have distinct PIDs.

It replays every accepted-token/count history to the clipped Runtime retained 1024 IDs, checks per-cycle Host event coverage and final progress threshold, and checks all8 Runtime token/count parity. Measured Runner reports must use `CUDAGraphMode.FULL`. For each request, it replays *all* Scheduler appends with G continuity, stable `generation_object_id`, and raw admitted prefix, requiring non-stale, non-resumable, zero stale tokens and a single stopped terminal bulk. The terminal bulk's full incoming list must equal Runtime retained 1024 IDs. If G preexists, its admitted prefix is only the first `1024-G` Runtime retained IDs. The formal output is ordinary admitted IDs followed by that clipped bulk prefix. Output receive and queue raw IDs, API consumed raw IDs and cumulative count must equal that reconstructed formal sequence. Every server serialized SSE yield must equal the exact client's captured SSE payload bytes, including a single terminal DONE.

The synthetic positive deliberately uses an ordinary token 99 before the Runtime's repeated token 7, so treating all 1024 Runtime IDs as formal output would fail. Negatives corrupt a ledger byte, footer framing, SSE bytes, warmup transition hash, API raw IDs, API cumulative, Scheduler G, Runtime retained IDs, Scheduler generation identity, and measured Runner FULL mode; all reject.

## Interface

`python3 scripts/loop079_formal_ledger_server_validate.py --ledger-dir ... --client-report ... --warmup-report ... --warmup-dir ... --measured-dir ... --phase-transitions ... --run-id ... --output ...`

Success writes `server_two_phase_admitted` with per-phase request IDs and summaries. Failure exits nonzero and does not write an admission report. A guarded controller must require this exact status and perform its own stop/restore/source SHA checks before final admission.

## Scope limits

This is Host lineage and observed output admission. It does **not** establish a fresh required Target row, cache/KV residency, prefill/seed device readiness, typed collective completion, device critical path, resource overlap, attainable Scheduling Bound, or Product E2E TPS. `serving_host_cycle` is checked for membership, sequence, shape, parked-state monotonicity and final progress threshold; its progress-to-count equality is not admitted as q/R calibration. Controller timestamp is not compared with Host/container monotonic timestamps because the phase marker does not record the controller's boot/time namespace identity. The transition is admitted by validated warmup report hash, phase hashes, per-process order, and complete request joins, not an unsupported cross-process clock inequality. Instrumentation perturbs Host execution.
