# Run589 independent final review

2026-09-28. **SCOPED PASS for the same-acquisition labeled partial MatMul → event-ordered native RS → labeled copy identities, and the selected measured-cohort request lineage.** This is structural Scheduling evidence, not a typed last-writer, fixed-trajectory equivalence, timing, Resource lower bound or Product ceiling certificate.

Reviewer: Astra delegated Bound reviewer. No service or NPU computation was started, no source was changed. CPU-only readers, hashing, raw-client replay, validator replay and independent graph/Runtime checks were used. Only this review file was written.

## Independent checks

- Recomputed all **98 hashes** in final_admission.json: 64 Runtime, 24 capture/Graph/meta and 10 provenance inputs. Final admission SHA256: `7c27ba704b75f4e6b8bcb6f768d629f44ec3655d5f5a0ee097c129c5e55065b7`.
- Recomputed the current five source and fourteen script/sidecar/dataset manifest entries; both before/after manifests match and current bytes still match. Restore reports no helper drift. Cleanup reports all ten statuses zero. Final stop verification records no live owned service and eight idle HBM values 3444/3442/3443/3436/3441/3435/3446/3436 MiB. These are the recorded post-run cleanup facts; this review did not start another device workload.
- Replayed both 48-request raw-client admissions in memory, including frozen dataset/body hashes, SSE payload hashes, DONE/usage/1024-token and actual c12 checks. Reconstructed request_index and both summaries exactly match the saved client admission, with 96 distinct response IDs.
- Independently checked every Runtime rank/cohort, 1024 counts, pass/host-mirror/FULL flags, all-rank request order and client phase/dataset membership. All8 cycle counts agree within each cohort: **303, 306, 292, 320, 316, 310, 312, 318**.
- Replayed all8 select_rank reducers and matched final capture rows. In addition to current validator gates, checked each graph meta start/end/cycle count against Runtime: cohort5 start0/end316/cycles316; capture relative cycle0 and Runtime cycle0; selected replay ordinal1226 through post-cohort replay count1541 reconcile exactly to316 replays. The same twelve measured requests are present in capture, graph meta, all8 Runtime and client records. The Run583 missing request/cohort association is closed for this acquisition.

## Native structural result

Each same-run selected FULL96 dump has **5412 unique (Model, Stream, Task) tuples**, all native Model45. Both labels preserve rank and capture_serial5 and match their source capture strings exactly. All8 use:

`stream1/task40 MatMul → stream1/task41 record(P) → stream0/task12 wait(P) → stream0/task13 aiv_reduce_scatter_bfloat16_t → stream0/task15 record(R) → stream1/task42 wait(R) → stream1/task43 MEMCPY_ASYNC → stream1/task44 HcPost candidate`.

P/R by rank0–7 are `4221/4232, 4326/4329, 3241/3237, 4397/4399, 3264/3185, 4318/4320, 4329/4331, 4265/4272`. The partial tag is on task40 and the copy tag on task43. The RS retains original `group_name_3`; it was not overwritten by a diagnostic label. The loaded capture branch is SequenceRowParallelOp with AscendUnquantizedLinearMethod, ordinary FlashComm1 RS, no OTP/MMRS and pad0.

As a separate raw-word corroboration on all8, MatMul task40 argument word2 equals the capture partial pointer; RS task13 words32/33 equal partial/reduced pointers; HcPost task44 word0 equals attention destination. These checks support consistency of the same-run association. **Word positions are not independently certified semantic ABI roles**, and the HcPost typed input remains unproved. In particular, no copy src/dst/count fields exist in the native exporter; its source/destination/98304-byte label is supplied by the capture hook. Partial geometry is BF16 [96,4096] (786432 logical bytes), reduced/destination geometry BF16 [12,4096] (98304 logical bytes).

## Scope and validator limits

No false join was found within the admitted scope. The graph is dumped after cohort5 synchronization, after316 cycles. Its argument words are **not a frozen cycle0 parameter snapshot**. Request linkage certifies which Graph was submitted for the selected first Target; the dump certifies the retained same-generation static graph after the cohort. Neither substitutes for a per-cycle device trace.

The current validator does not itself require graph-meta start/end/cycles and replay-count delta to equal the selected Runtime record. This review checked those raw equalities and found them true; future acquisitions should enforce them automatically. Native pointer-word corroboration above is also supplemental, not a substitute for a typed ABI decoder.

Task count5412 and the two correct tag locations do not prove complete uninstrumented graph equivalence or marker noninterference. There is no matched A0/A1 trajectory control. Event/task order is a fact of this captured implementation, not proof that every such boundary is mathematically unavoidable. Buffer reuse, all overlapping writers, compiled HC-post input lowering, actual HCCL communicator/message semantics, device readiness/completion timestamps and cross-rank arrival timing remain unclosed.

## Bound promotion

**Promote:** one same-acquisition Current Scheduling subgraph with source-labeled native producer and copy endpoints, event-ordered RS identity, preserved communication group label, and selected measured request/Graph-generation lineage. The earlier uncertainty about whether Python capture aliases correspond to the intended native MatMul/copy tasks is materially reduced.

**Do not promote:** typed final writer or HcPost consumer; a measured or necessary slice duration; removable Host/communication wait; legal overlap amount; a43-layer sum; compulsory FLOPs/HBM/link traffic; certified exact-board capacity C+/B; fixed-W0 equality to Run99; or a Product gain/ceiling. Logical operand byte counts are not physical traffic or compulsory traffic. Formal Current remains **571.681 tok/s**, and finite strict Resource/Hardware, Scheduling/Execution and Product endpoints and numeric Current→credible-limit distance remain unknown.

Next resolve the same-generation typed consumer/alias lifetime from source/object evidence and the new labeled endpoints before another timing acquisition. Once that identity is sufficient, acquire a bounded same-run native task/readiness window with a matched perturbation control. Read actual device times from a profiler/event source; never use debug_dump synthetic ts/dur. Continue compulsory-work and matching capacity work independently.

Primary evidence: `final_admission.json`, `live/b/capture/`, `live/b/runtime/`, `live/b/{warmup_client,measured_client,client_admission.json}`, `live/{cleanup_status.txt,restore.json,source_before.sha256,source_after.sha256,scripts_before.sha256,scripts_after.sha256,final_stop_verify.log}`.
