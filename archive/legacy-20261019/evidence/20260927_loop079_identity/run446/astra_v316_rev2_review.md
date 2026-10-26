# Run446 — V3.16 rev2 independent follow-up

2026-09-27. Follow-up to Run443 on corrected Run444, generator extreme_bound_calibration_v3_16_rev2.py and Run445. Read-only source/log/manifest review plus stdlib-only mutations in fresh temporary copies. No service, NPU execution or runtime-source modification.

## Verdict

**ACCEPT the reviewed rev2 artifact as an all-open, scope-preserving evidence ledger.** The blocking Run441 payload error is corrected using a separately identified Run444. The known endpoint guard holes are closed, direct PRIOR lineage is included, stale next_measurement.specific_gate is replaced, and the saved output retains 19 false proof-node certifications with all numerical Bound endpoints null. Formal Current remains Run99 median 571.681 tok/s.

This accepts the evidence ledger and observed isolated service point. It does not certify a finite Bound, a production communication completion contract or a removable E2E interval.

## Corrected sample

Run444 CLI size 24,821,760 is the aggregate receive size at TP8. The installed source divides its element count by eight, giving exactly 3,102,720 input bytes/rank, matching the current [96,16160] BF16 terminal input. Three logs independently report 235.63/261.26/238.76 microseconds and success; median is 238.76 microseconds.

The source was independently checked in the actual test container vllm-ascend26-dsv4f-w4a8. The installed binary hash b276a969e778ab2d627643225fc9a29b45878b016999079de9ae705a73997b9d matches the manifest. Installed source hashes match the saved source. The initial Run443 container discrepancy is resolved: dsv4ab is a distinct legacy container and its /tmp executable is unrelated.

The timing description now matches the implementation: root-rank ACL event loop duration divided by 30, which may include submission gaps; final-buffer checking follows an extra untimed collective. Bandwidth is aggregate output bytes divided by this interval. This remains an isolated HCCL Test communicator observation without model contention, producer arrivals, layout materialization, argmax consumer or original torch backend join. No lower latency floor, C_plus or Product saving follows.

The fields named three_processes and independent_process_logs should be read as three independent eight-rank MPI jobs. This legacy naming is imprecise but the manifest explicitly identifies eight ranks and does not affect any number or endpoint.

## Generator and final negative controls

The final baseline regenerates Run445 **byte-for-byte**. The final parser validates rank/dtype/size/iteration/check settings from the saved command manifest, verifies saved-source content hash, requires one successful result row per log, rejects failure/error lines, and checks positive finite metrics. Bandwidth consistency correctly permits the 0.01-microsecond and 0.00001-GB/s print-rounding intervals.

All of the following final mutations were rejected:

- Each registered numeric endpoint key placed under proof_dag: numeric_tps, numeric_s, latency_floor_s, finite_tps_upper_bound, maximum_capacity_bound and unresolved_gap_tps.
- Zero reported duration.
- A root success row accompanied by another rank's failure line.
- Bandwidth inconsistent with bytes/time.
- Changed manifest rank count.
- Modified saved source without matching source hash.

The earlier rev2 checks also rejected duplicate cohort identities, the old smaller CLI payload and the existing platform maximum_capacity_bound mutation. The original DAG controls rejected dangling dependencies, cycles, node-scope mismatch and unexpected certification; the DAG validator and these safeguards are retained. No tests imported a device/framework package.

PRIOR Run435, Run444 source/logs/manifest and other directly consumed inputs are now recorded. The new specific_gate first binds actual installed HCCL stream-join semantics, then requests the bounded original-path slice with controls. Run437's sample cannot certify all43 composition; Run444's sample cannot certify capacity or scheduling duration lower bounds. Revised prose names Run437/444.

## Remaining scope boundary

The ledger still relies on trusted human-reviewed evidence for actual execution and physical truth. A manifest is an attestation, not a replayed or cryptographically signed execution record; the script checks selected fields rather than proving every command string or source-build relationship. Run437's summary is not independently recomputed by this generator. These are explicit evidence limits, not current numerical promotions.

Whole-tree checking now covers all identified endpoint keys in the current schema. Future endpoint aliases require schema/registry updates; a key-name guard cannot identify arbitrary newly invented numeric claims. Future numeric promotion still requires a separate typed W/C or necessary-path checker with units, output/window/class joins, rank aggregation, conservative uncertainty and immutable evidence identity. Preserve nulls until then.

## Reviewed SHA256

- Final rev2 generator: fd29efb4a0dedd055e8fac5e91ae261bf19ef3a2d92848f45535c813a798e2e2
- Regenerated Run445 JSON: 2e61c9809f4788f35b2597ac86335ed80546b0712249fa2d49ff145209fd58c3
- Run444 command manifest: ac1bbc3c7593afbd6a7bebef15c66e75c7aad226cfa556254c4758eb188c2f8d
- Installed/saved AllGather source: 8a5339c7310160c8028dc4d481b9a44a9326979124c8ccf7b5b31732b9015d15
