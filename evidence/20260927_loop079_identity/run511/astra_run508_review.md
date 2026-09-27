# Run511 — independent Bound-first review of Run508

2026-09-27. **PASS, with the promotion scope below.** Run508 is a fresh admitted diagnostic. Its selected `AscendMLAImpl.update_graph_params` calls have a source-inferred empty zip loop on all 40 selected rank/cohort observations. **No finite Bound or performance promotion follows.** Formal Current remains Run99 median **571.681 tok/s** for DeepSeek V4 Flash W4A8, 8×910B3, DP1×TP8, DSpark7, frozen 48×32K→1024 c12. Algorithm/Resource, Hardware/Resource, Scheduling/Execution and Product E2E finite endpoints remain **null**.

## Independent acquisition checks

`astra_audit.py` reruns the reviewed root validator and final admission on original remote paths, then independently checks raw identities, source reconstruction, request arithmetic and graph structure. `independent_results.json` records PASS. `input_sha256_manifest.json` freezes **172 inputs**, all rehashed unchanged at audit end, including every Run508 acquisition/preflight file, the actual server log, reviewed scripts, six restored sources, the installed MLA source and reference records.

- UUID `98ed90e0-b51b-4034-a84f-4626207762c4`; exact 8 ranks × 5 cohorts, 40 capture files and 40 Runtime files. Selected cycle is64. Cohort cycles are293/300/312/324/305. Within each rank, PID, entry, graph, generation1, Target model/batch, complete argument descriptors and four-output owner tree are stable across all five observations; selected ordinals are1–5. Identity comparisons stay within process/rank/acquisition.
- All eight cohort5 dump/meta pairs join the selected graph, capture generation, model, output-owner SHA and debug-dump receiver. Raw SHA, bytes and task counts match. Each raw dump's PID matches that rank's capture PID. The actual selected backend/implementation/update callable agrees across cohorts and with its post-drain source certificate.
- The five preflight-reviewed scripts equal current scripts byte-for-byte. All six archived original sources match their recorded original SHA and current restored sources. Independently applying the reviewed pure patch function to each archived original reproduces its full recorded patched SHA. No patch was installed by this review.
- The client files independently establish48+12 requests, all error-free and exactly1024 output tokens. Combined client intervals peak at12 concurrent requests; warmup ends before diagnostic begins. Runtime reports pass FULL/host-mirror/ownership checks, match all-rank acceptance trajectories and staged counts, and contain60 distinct request IDs. Client records do not contain those Runtime IDs; this is the existing acquisition/cohort join, not a newly observed per-request token-text lineage or complete semantic oracle.
- Actual server log contains60 chat POSTs, allHTTP200. Eight ERROR-prefixed lines at12:16:10 report `AsyncLLM output_handler`/`EngineDeadError` after `Parent process exited` and shutdown cleanup. The context is preserved in `shutdown_log_excerpt.txt`; the log is not error-free. All eight cleanup status fields, including final gate, are0. Saved stop probes show no live VLLM/client process and HBM3435–3446MiB/card. Source before/after files are identical. This certifies recorded shutdown and source restoration; no service was restarted or queried in this review.

## Selected MLA update: precise new evidence

Installed `mla_v1.py` SHA is `67b4479cca16fd164b33b91b2fa4f48e95f2474720e484abdf3b3897a8ea1545`; a matching copy is archived here. Source lines968–1097 and the reviewed patch show that the helper observes the already-resolved actual callable immediately before invocation and records its successful return. It does not substitute a different backend or infer the selected callable by resolving it later.

Every selected observation has:

| Item | Value |
| --- | --- |
| Actual method | `AscendMLAImpl.update_graph_params` |
| Branch / shape | main Target, `after`,96 tokens; draft metadata argument isNone |
| Attention-key count |170 |
| Captured and selected main GraphParams identity | equal, tied to that rank's selected entry/generation |
| `attn_params[96]` / `handles[96]` / `events[96]` |0 /0 /0, with empty object-ID arrays |
| Source-derived zip iterations |0 |
| Successful method returns |1 |
| Source-inferred `event.record` calls |0 |
| Native event ID / device completion |unobserved |
| Configured update stream |logical102 on the same rank/device |

The saved evidence gives **key count**, not the170 individual key names or ordering. Empty lists make the count sufficient for this zero-iteration conclusion. There is no nonempty event/handle object whose native identity or reset generation has been certified.

The source's zero-key early return does **not** explain this result:170 keys cause entry into `with torch.npu.stream(update_stream)`, followed by an empty four-way zip. Under the reviewed stable-list/source assumptions, that selected loop executes no `graph_task_update_begin`, FIA `.out`, `graph_task_update_end` or `event.record`. Counts are source-derived from immediate pre-call snapshots and successful return, not independently traced native API counters. The helper leaves the original call and stream-context entry/exit intact.

**Promote only the selected-call fact:** in this acquisition, those40 calls do not contribute the loop-body private update/FIA/event-record dependency previously left unresolved by Run502/503. Do not claim all work on stream102 is absent, all private work is absent, all graph waits are satisfied, or the method costs zero time. Do not transfer the result to another replay generation, shape, future call, historical trajectory or source version. Run507 remains INVALID despite its posthoc parseable records.

## Remaining Scheduling and Bound obligations

Each graph again has5,412 tasks on internal streams0/1/99 with counts1,040/3,985/387. Matching exported record/wait suffixes and per-stream task order gives an acyclic **conditional** DAG in which all tasks reach internal stream1/task3984 NOTIFY_RECORD. The audit independently recomputed this structure without using exporter `ts`/`dur` as timings.

Zero selected MLA-loop work removes one narrowly scoped candidate external dependency; it does not supply typed four-output last-writer certificates, an exhaustive write/alias inventory, native event reset-generation semantics, or the model-terminal→caller continuation contract. Post-drain dumps follow later replays and are not cycle64 dynamic-parameter snapshots. Numeric similarity of internal/caller stream IDs does not prove their identity. R1 therefore retains its conditional completion meaning.

No interval from Run487 or another run is imported. Run508's instrumented diagnostic TPS or event intervals establish neither passive Current exposure, an all-rank makespan, removable time, necessary latency, a legal attainable schedule nor Product improvement. The necessary-work witness and matching genuine aggregate capacity upper certificate remain separate open Resource gates. The next Scheduling work should target typed output producers and the native model terminal→caller join; repeating stream timing for the already-empty selected MLA loop has no demonstrated value.

This review performed CPU-only/read-only analysis and wrote only Run511 evidence. It did not start a service, execute an NPU workload, alter a hot path, update TaskCtl or change the Bound model.
