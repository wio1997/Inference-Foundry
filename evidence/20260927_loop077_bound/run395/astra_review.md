# Astra independent review — Run394 / Run395

Verdict: ACCEPT as clean original-path sparse current-stream diagnostic evidence. INCONCLUSIVE for a resource floor, attainable scheduling floor, all8 critical path, or finite Product TPS ceiling. Keep finite bound endpoints null and formal Current at its separately accepted value.

## Independently verified gates

- Recomputed raw event differences, all reported stage/paired summary statistics, and Runtime-clipped useful output: PASS.
- Exactly 40 event files and 40 Runtime files: five cohorts, eight ranks, cycles 64/65, 80 rank-cycles and 40 adjacent-cycle pairs. All 80 nested marker orders pass.
- Cohort cycle totals: 306, 314, 317, 309, 292. All ranks agree on cohort request IDs and acceptance histories. There are 60 unique Runtime request IDs; every cohort starts with zero prior output and produces 12 × 1024 tokens.
- All Runtime reports pass, FULL Target graph, host-mirror exact, and zero post-handoff oracle/ModelRunner calls.
- Both clients pass: 48 + 12 requests, exact 1024 output tokens, no errors. Independent client interval sweep gives maximum concurrency 12. The Run394 server log contains exactly 60 successful POSTs; its sampled Running+Waiting maximum is 12. This removes the Run389 orphan-client contamination concern for this run; access count alone is not a universal isolation proof.
- Source-before/source-after hashes match for all three borrowed sources. Stop log reports per-card HBM back near 3.4 GB.
- Client output files do not preserve SSE request IDs, so direct client-index-to-Runtime-ID attribution remains unavailable. The exact aggregate cardinality is established.

## Numerical observations that may enter V3.7

All numbers are attained, instrumented current-path spans, not lower bounds.

| Quantity | Range / median (ms) |
| --- | --- |
| Same-rank begin64 → begin65 | 54.009–62.602 / 56.548 |
| Within-step begin → draft_commit | 54.007–62.584 |
| Pre-Target marker → Target marker | 46.083–55.052 / 49.465 |
| state_advance → proposer | 6.260–8.692 / 6.397 |
| Target64 end → next pre-Target marker | 7.465–16.654 / 7.648 |
| draft_commit64 → begin65 | 0.001–0.355 / 0.002 |

Same-cycle useful outputs for the five cohort pairs are 56/41, 65/43, 39/40, 54/53, and 36/44 tokens. These couple observed original-path timing to actual retained work on ten sampled cohort-cycles. Eight ranks are correlated observations, not 80 independent workload samples.

Stage variation must not be read as changing pure compute cost. For example, cohort1 cycle64 prepare+metadata+Target is 47.247–47.260 ms across ranks. Waiting/submission delay can move between the metadata and Target intervals. Neither summing stage medians nor taking the maximum rank duration constructs an aligned all8 makespan.

## Marker, stream and clock assessment

The repaired patch freezes Runtime selection before cycle_index increments, so both sampled Runtime groups are complete; DSpark samples the same incoming cycles. The installed torch_npu Event.record defaults to current_stream. Source ordering and raw nested order agree: state_advance → DSpark.begin → DSpark.model → Runtime.proposer → draft_commit.

Current-stream intervals include queued work and exposed submission/wait gaps. The installed FULL Graph wrapper conditionally synchronizes the current stream before replay; these timings do not isolate native graph kernel service. Target includes the binding and logits work. The cohort loop's history staging and host scheduling occur between draft_commit and the next begin, so begin64→begin65 is the better observed cadence than begin→draft_commit alone.

The host-copy side stream waits on the current stream, copies counts, and records an event. The following DSpark call synchronizes that prior event on the CPU before updating mirrors. Thus the host_mirror marker is not a side-stream completion marker for the current copy. A near-zero current-stream interval does not imply zero DMA/host cost. Downstream output correctness supports functional dependency behavior; it does not independently timestamp every Graph/HCCL branch or prove a complete all-stream endpoint. Stream IDs, per-branch joins and resource attribution are absent.

The anchor is now bracketed correctly before record and after synchronize, conditional on a stable host clock. Width is 0.220–0.545 ms. Selected events precede the anchor by 13.170–14.522 s; device/host rate agreement over that lag is uncalibrated. Do not treat reconstructed midpoints as exact cross-rank timestamps. Subtracting long elapsed values also exposes approximately microsecond quantization; tiny reported gaps should not be overinterpreted.

## Which uncertainty shrinks

- Algorithm/Resource: same-trajectory retained output and executed cycle cardinality are now measured under clean c12. No new compulsory FLOPs, unique necessary bytes, acceptance-invariant work, or algorithmic latency floor is proved.
- Hardware: original mixed-path stage envelopes are observed without the prior full profiler. No AIC/AIV/HBM/link capacity, cache-residency attribution, physical HCCL service floor, or isolated communication exposure is established.
- Scheduling/Execution: uncertainty about actual phase placement, observed local cycle cadence, and the size of sampled intercycle exposure is reduced. All8 arrival/joins, mixed-resource contention, legal refill/prefill/seed overlap, and a resource-constrained critical path remain open.
- Product: no new formal E2E calibration or finite ceiling. The 48-request warmup plus 12-request diagnostic is not a replacement for the accepted formal repeat protocol.

## Next measurement

Highest value: a clean same-path adjacent-cycle measurement that records existing producer/completion/consumer dependencies on all eight ranks, especially count-copy completion and next DSpark consumer, plus Graph/HCCL completion joins. Add recorded stream IDs, explicit cycle/request identities, and close-in clock calibration with stated uncertainty; avoid adding hot-path barriers that manufacture serialization. Measure sparse-event overhead with an uninstrumented control before using these as unperturbed stage costs. Then combine the joined DAG with separately justified mandatory work and mixed capacity; current measured costs alone cannot prove a floor.

Confidence: high for gates, arithmetic and source-level marker interpretation; medium for representativeness of the ten cohort-cycles; insufficient for cross-rank critical-path attribution or any finite bound.

Evidence read: run394/events/*.json, run394/runtime/*.json, warmup48.json, bench.json, server log, source SHA files, stop.log, run395/analysis.json, event patch and Runtime/proposer sources, installed torch_npu streams.py and vllm_ascend compilation/acl_graph.py. Only this review file was written.
