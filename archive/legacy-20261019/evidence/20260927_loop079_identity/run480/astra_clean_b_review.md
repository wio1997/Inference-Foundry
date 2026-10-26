# Run480 — independent review of Run477 clean terminal B2

2026-09-27. **PASS for post-cleanup diagnostic admission, with the scope limitations below.** The capture is usable as no-getter, instrumented, same-device terminal-path evidence. It does not establish an uninstrumented Current interval, numerical Scheduling lower bound, removable E2E time or new formal TPS. Current Formal remains571.681 tok/s.

Evidence root: `run472/b_clean2`; UUID `cf05ab77-b821-4273-9000-9434e3a0e6d0`. Review script and independently recomputed results are preserved in `run480/independent_check.py` and `independent_checks.json`. Only evidence was written; no source/service/NPU action was performed.

## Admission and provenance

I reran the offline validator and final admission directly on raw files; both pass. I separately verified every input SHA in Run477 reducer, recomputed all overall min/median/max values from40 raw event records and matched the reducer to1e-12ms. Review evidence hashes cover80 capture/Runtime records and acquisition metadata. Six source-before/source-after lines are byte-identical; the actual current six source files also match those original hashes. Install, patch manifest and restore records identify the same six source anchors reviewed in Run474. All eight cleanup fields, including final_gate_exit, are0. The stored stop proof has no live VLLM or benchmark client, HBM3435–3447MiB/card and MemAvailable989240688KiB. These are acquisition-time saved probes, not a new live idle claim.

There are exactly40 slices: ranks0–7 × cohorts1–5, one selected cycle64 each; each rank retains one worker PID across cohorts. All streams are direct logical ID0, device_index=rank, device_type20, tensor device npu:rank. Setup and all five markers agree; no raw handle identity is present. Source-only Run474 establishes that this helper uses the direct fields and that indirect current-stream/capture-state inspection avoids the former getter drain. The capture itself is not a trace of every native queue operation; no claim that all event/Python overhead disappeared follows.

Every record has the normal logits branch, TP group0–7/HCCL/NPUCommunicator, one ordinary native all_gather_into_tensor, no active dispatch modes/coalescing/compile/capture, and the pinned wrapper/C10D/loaded torch_npu hashes. Marker order is P/J/G/C0/C1 with distinct events; Host metadata is monotone and native entry/return is between P/J Host records. Host metadata is not a device completion clock.

All40 native outputs are BF16[768,16160],24,821,760 bytes, and layout returns are BF16[96,129280] in a different24,821,760-byte allocation. Layout→gather→Target→C0 retains storage; C1 is int64[12,8]. Observed contiguous16160-token vocab shards cover0–129280 without gaps. This retains the structural materialization finding; it does not prove physical HBM traffic or that a replacement representation avoids all equivalent communication/work.

## Correctness scope, workload and execution

Server log contains exactly60 chat POSTs, all200. Warmup48 and diagnostic12 each report all requests successful and1024 output tokens, c12, with maximum observed client concurrency12. Across Runtime ledgers there are60 distinct request IDs, consistent across ranks. All40 Runtime records pass Host-mirror equality, generated counts[1024]×12, requested/actual FULL Target graph, handoff before forward, zero subsequent oracle target calls and zero model-runner cycles. Accepted count matrices and cycle counts agree across8 ranks. I independently summed every slot: accepted totals equal staged counts, staged minus useful equals reported overshoot, and all slots are still active at selected cycle64.

These gates establish output-length/count/ownership/Host-mirror correctness for this diagnostic. The saved HTTP records do not include token IDs/text for a fresh semantic oracle comparison; do not describe this run as an independent new full-token oracle proof. The reviewed patch preserves mathematical operations, and existing frozen correctness obligations remain applicable. Actual HTTP input usage is32851–32853 tokens for the named32K dataset, identical to the Run439 usage range; retain the existing dataset/chat-serialization contract rather than silently substituting32768 actual runtime tokens in a work model.

| Cohort | Cycles | Useful output tokens | Staged tokens | Overshoot | Useful tokens/cycle | Selected cycle64 accepted tokens |
|---|---:|---:|---:|---:|---:|---:|
|1|341|12288|12380|92|36.035|41|
|2|315|12288|12364|76|39.010|59|
|3|305|12288|12368|80|40.289|42|
|4|307|12288|12372|84|40.026|49|
|5|293|12288|12367|79|41.939|42|

Do not multiply useful counts by8: ranks duplicate the same TP trajectory. Five cohorts together contain61440 useful tokens over1561 cohort-cycles (39.359 useful tokens/cycle), not an algorithmic maximum. The four warmup cohorts alone contain49152 useful tokens over1268 cycles. Tail zeros/parking and overshoot matter; a cycle64 sample is not representative of the complete refill/seed/KV/state trajectory. These matrices are Current trajectory evidence, not a replacement for the prior FIFO/fixed-duration scheduling construction.

No ERROR, Traceback, OUT_OF_SCOPE, RuntimeError or AssertionError was found in the server log. There are60 warning lines, preserved in `server_warning_lines.txt`. Normalized comparison with Run439 shows one newly occurring warning class: all8 ranks invalidate a code-changed compile cache and recompile before serving. Existing missing optional Quarot alignment, configuration normalization and disabled compilation warnings recur; they are not newly discovered diagnostic failures. Compilation ends before the successful acquisition. The benchmark12 summary577.550 tok/s is an instrumented12-request diagnostic, not the frozen repeated48-request formal score.

## Independently recomputed timing

Units below are microseconds; raw Event.elapsed_time fields are milliseconds. Quartiles use inclusive interpolation. All samples are one selected same-device chain;40 samples are clustered into five8-rank collective cohorts and must not be treated as40 independent experiments.

| Interval | Min | Q1 | Median | Q3 | Max | MAD |
|---|---:|---:|---:|---:|---:|---:|
|P→J|173.92|178.495|182.19|198.590|206.72|5.19|
|J→G|27.90|29.890|30.86|33.905|38.14|1.27|
|G→C0|0.14|0.160|0.16|0.250|0.42|0.00|
|C0→C1|51.96|52.535|52.80|53.345|56.02|0.50|
|P→C1|256.44|263.380|267.36|281.695|296.32|6.32|

P→C1 is the sum of the four adjacent intervals of each same-device chain, then summarized across chains. It is not the sum of independently chosen medians/minima or different-rank observations. The native J completion interpretation remains conditional on the already pinned ordinary C10D/HCCL Work-wait/current-stream join path. P→J includes native collective exposure and preceding original enqueue/layout/Python work; it is not isolated link latency. J→G includes original layout work; C0→C1 is the instrumented argmax envelope.

Cohort P→J medians are178.75,203.86,195.34,177.00,181.23µs; P→C1 medians263.79,288.26,280.81,261.10,265.98µs. Thus the principal variation follows shared cohort conditions. Rank P→J medians span178.52–184.64µs. Rank5 has relatively large J→G median36.20µs, rank1 has34.20µs, versus29.46–31.42µs for others; five observations/rank do not identify a persistent hardware/topology defect. No cross-device common-clock alignment exists, so these distributions are not an8-rank makespan.

The interval values lie on an approximately20ns numerical grid (maximum floating representation residual below0.000365 tick). This is observed timestamp granularity, not calibrated accuracy or EventRecord overhead. G→C0 at0.14–0.42µs is near an unresolved marker/queue-resolution floor. Its median0.16µs cannot establish either zero real Host work or a compulsory160ns scheduling edge; Host descriptor/gate work can be hidden behind queued device work. There was no matched event-overhead calibration or A0/A1 perturbation control.

## Historical comparison: descriptive only

| Instrumented interval median | Run463 queue-draining B | Run477 no-getter B2 |
|---|---:|---:|
|P→J|200.40µs|182.19µs|
|J→G|30.96µs|30.86µs|
|G→C0|0.16µs|0.16µs|
|C0→C1|53.07µs|52.80µs|
|P→C1|284.24µs|267.36µs|

These are unmatched runs. Old cohort cycles308/320/299/299/298 differ from new341/315/305/307/293; every complete accepted-count trajectory differs. The probes also differ by removing nine possible Host task-queue drains per selected slice. Therefore the18.21µs P→J median difference is not measured getter overhead, a causal latency improvement or E2E saving. Similar J→G/argmax scales do not retrospectively rehabilitate old passive-timing claims. Preserve old raw evidence under its queue-draining intervention label.

## What enters the Bound model

1. **Current observed execution:** admit the above distributions in an explicit instrumented-local-cost column with scope, clustering and measurement limitations. Admit observed stream/group/shape/lineage and actual FULL/Runtime ownership. This reduces uncertainty about the corrected terminal sequence and confirms the old traffic/materialization hypothesis remains structurally present.
2. **Scheduling DAG:** admit observed terminal ordering and native completion dependencies as properties of this implementation. Model local-logit production→native all-gather/join→layout→argmax with current shapes. Do not elevate those operator/collective/materialization boundaries into mathematical necessities. They can change under alternative ownership/layout/reduction designs.
3. **Scheduling-aware Bound:** no numeric lower bound or saving ceiling follows. Missing evidence includes uninstrumented exposure, earlier Target/hidden-output producers, original waits, cross-rank arrival/critical-path placement, possible overlap and resource contention under an alternative schedule. Minima over40 observations are not attainable schedule limits. Multiplying0.267ms by cycle counts would be an unsupported full-trajectory saving estimate.
4. **Resource/Hardware and Product:** this adds no compulsory HBM/link bytes or attainable compute/HBM/HCCL rate. Allocation bytes are not measured traffic, and aggregate collective message shapes are not topology wire bytes. It does not close prefill/seed/KV/state/serving or algorithmic useful-cycle floors. No revised Product ceiling or reduction of the formal571.681→true-limit gap is quantitatively justified yet.

## Most informative next measurement

Proceed with the separately scoped **Target FULL replay frontier** after Run469 conditions are implemented and independently preflighted, rather than another terminal kernel optimization or broad repeat of clean terminal timing. Its explicit uncertainty is how much Current cycle critical path lies in the original pre-replay wait/Host submission, aggregate FULL replay, hidden/aux gather and selected-hidden construction, and which edges are genuine producer dependencies versus current scheduling choices.

Use T, existing-only S0/S1 Host timestamps, R0/R1 on the actual replay stream, per-output hidden/aux gather H and U after selection. Preserve immutable IDs, setup outside capture, exact graph entry/generation and loaded native-library provenance. R1 can certify captured producer completion only when successful native capture/rejoin semantics and actual required output-tree membership are established; otherwise label it a conditional replay-stream envelope. Do not add missing waits to make the certificate easy. Keep terminal markers out of that selected cycle. Bind samples to active shapes/useful-count trajectory, and predeclare the missing bound component each result will resolve.

A0/A1 matching and marker calibration become necessary before promoting frontier cost to uninstrumented Current exposure or translating an intervention into saving estimates; they are not a reason to continue measuring the small terminal seam indefinitely. Use the frontier to update the global dependency model, then choose the largest unresolved resource/scheduling term. Ultimately replay timing still needs compulsory work/traffic, attainable hardware rates and feasible overlap to become a Scheduling/Product Bound.
