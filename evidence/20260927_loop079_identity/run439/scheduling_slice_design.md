# Run439 — minimum original-path logits collective scheduling slice

Date: 2026-09-27. **Design only; no implementation or execution.** Frozen DeepSeek V4 W4A8, 8×910B3, DP1TP8, DSpark7, warm-cache 48×32K→1024 c12. Current formal remains 571.681 tok/s. Run437 is active; this review read saved evidence/source and writes only this document.

## Exact question and first unresolved edge

Measure one terminal Target edge:

local vocabulary-shard logits ready → logits AllGather output complete and joined to the consumer stream → gathered-layout materialization → greedy verification argmax.

The first unresolved edge is **native HCCL completion → calling/consumer-stream readiness at the return from dist.all_gather_into_tensor**, not the Python call-return timestamp itself. Run391 maps this current path to Target HCCL ordinal264, BF16 input [96,16160] and gathered logits [96,129280]. Use ordinal264 only as a prior; actual scoped call, shape, group and storage identity must identify the new event.

This is smaller than instrumenting all Graph collectives or the entire Target→Draft chain. FixedTargetAdapter.execute calls binding.forward first, then compute_logits separately (target_adapter.py:64–77); the latter supplies FixedGreedyAcceptance.argmax (fixed_acceptance.py:27). Verify at runtime that this selected logits path is eager/outside capture and is not redirected into a new graph or alternate backend.

Run395 has only broad current-stream Target envelopes. Run400 lacks internal Graph/HCCL completion edges. Run410 established observed count-copy read-before-overwrite, but Run412 reports missing actual downstream Draft/DSA consumers and unmatched A/A controls. Those facts do not close the selected logits join. This experiment deliberately stops at argmax; it does not reopen count-copy, Draft feedback, parking or a full all8 critical path.

## Preflight gates before spending a service run

1. After Run437 completes and verified cleanup/restoration, pin baseline sources and actual loaded torch/torch_npu/CANN/HCCL libraries. Shared source read during Run437 is reference only; check restored originals before preparing patches.
2. Confirm instantiated normal logits branch: lmhead_tp_enable=false, enable_reduce_sample=false, use_all_gather=true, TP group has exactly the intended eight ranks, logits_as_input=false. Record actual soft_cap, scale, head dtype, org_vocab_size, communicator type and custom-op dispatch. A different branch is OUT_OF_SCOPE; do not adapt silently.
3. Prove the actual dispatch reaches DeviceCommunicatorBase.all_gather. NPUCommunicator inherits that method in the reviewed source; GroupCoordinator may call torch.ops.vllm.all_gather before reaching it. Record dispatch/group identity, not just class defaults.
4. **Mandatory completion gate:** establish from the matching installed native backend contract/source, or a correlated native trace of the existing record/wait dependencies, that the ordinary synchronous collective establishes producer-stream→HCCL and HCCL→calling-stream ordering. Python source only shows dist.all_gather_into_tensor(..., group=...) without async_op=True. It does not expose the native HCCL join; Run391 found no installed ProcessGroupHCCL.cpp. API return must not be assumed to mean device completion.
5. If gate4 cannot be closed read-only, do not launch a full service diagnostic claiming HCCL exposure. The bounded next prerequisite is an idle-resource, exact-installed-backend sentinel/trace check: shard-distinct data, ordinary collective, immediate consumer on the original stream, and trace correlation showing actual completion→consumer wait. Finite correct sentinel outputs alone are not proof of backend stream semantics. This prerequisite is separately scheduled after Run437, never concurrent with it.
6. Reuse the Run401 event-API review and timing preflight as a starting point, not a transferable numeric error bound. Validate fresh preallocated timing events and same-device elapsed-time semantics for the installed build. No cross-rank clock calibration is needed for the primary local endpoint.

If the completion gate remains open, a later capture may be labeled only API/current-stream envelope; J below cannot be called HCCL-complete and the intended join experiment is INCONCLUSIVE.

## Exact hook sites and minimal payload

Freeze selection at ExtremeDecodeRuntime.step entry, before cycle_index can increment. Capture c=64 in each eligible cohort; do not collect c65 merely to duplicate this single-cycle edge. Existing 48 warmup +12 diagnostic gives five cohorts ×8 ranks =40 selected slices. Tag the four warmup cohorts and measured diagnostic cohort separately. No parking at this cycle is an asserted runtime condition, not an assumption transferred from an old run.

Preallocate five unique timing-enabled events per selected slice outside timed execution. All events record on the **actual stream submitting the relevant operations**. Do not create a diagnostic side stream, add waits/barriers, change async_op, clone tensors, poll completion, or copy tensors to Host in the selected path.

| Event | Exact source anchor | Meaning and limitation |
|---|---|---|
| P | AscendLogitsProcessor._get_logits_normal, immediately after logits = self._apply_head(...) at vocab_parallel_embedding.py:337, before _gather_logits | Local shard producer boundary on its submitting stream; not a common-clock global arrival |
| J | DeviceCommunicatorBase.all_gather immediately after dist.all_gather_into_tensor(...) at base_device_communicator.py:213, before reshape/movedim | Completion boundary on the calling stream **only after gate4**; includes the existing backend join |
| G | _get_logits_normal immediately after logits = self._gather_logits(logits), before vocabulary slicing | Completed gather-layout boundary on that stream |
| C0 | FixedGreedyAcceptance.execute immediately before target.logits.argmax(dim=-1), fixed_acceptance.py:27 | Consumer-ready/enqueue boundary after prior stream work; not exact kernel start |
| C1 | Immediately after the unchanged argmax(...).view(...) expression, before greedy_accept | Consumer completion boundary including its submission gap and argmax/view work |

The J→G split matters: base_device_communicator.py:215–221 reshapes, movedim and reshapes rank-major output; the final reshape may materialize data. Charging this whole segment to wire communication would be incorrect.

A per-thread scoped tag set only for this Runtime Target compute_logits invocation selects the native hook, avoiding all other all-gathers. Each tag must encounter exactly one native call and one consumer on each rank. Record run UUID, worker PID, cohort/request IDs, cycle/generation, actual stream handles/device, group unique identity and ordered members, communicator/custom-op branch, tensor data/storage pointers, offsets, shapes, strides and dtypes. Bind local logits to native input and native output through the gather-layout result and vocabulary slice to target.logits. Pointer changes are expected across a materializing reshape; record the returned tensor relationships rather than requiring pointer equality everywhere.

Preserve current operation order and original expressions. Host monotonic timestamps around the existing native call may diagnose submission overhead, but are not GPU/NPU completion times. Append to preallocated memory; export only after the existing cohort drain and synchronization. Events must not be reused before export.

## Clock, completion and interpretation

Primary quantities are same-device elapsed intervals P→J, J→G, G→C0 and C0→C1. Record/verify stream identity at all five sites. A stream switch without an existing demonstrated dependency is a failure, not an invitation to add a wait.

P→J is a current local collective-path envelope: it may contain rank-arrival waiting, allocation/submission gaps, backend scheduling and transport. It is not isolated HCCL wire service. J→G covers layout completion and related gaps. G→C0 covers postprocessing/adapter and consumer submission exposure; it is not automatically Host-only idle. C0→C1 is the current consumer envelope.

Do not reconstruct cross-rank device timestamps from cohort-end Host anchors. Run395's anchor lag and uncalibrated rate agreement invalidate that shortcut. Report eight correlated local intervals per cohort; their maximum is an observed slowest local interval, not an aligned all8 makespan. All8 group/cycle identity verifies participation, not synchronized clocks or latest-producer arrival. Quantifying cross-rank arrival skew would require a separate clock/correlation certificate and is out of scope.

Use empirical timer/marker uncertainty from the preflight; if an interval is comparable to that uncertainty, classify it unresolved. Do not subtract a measured average overhead from individual events to manufacture precise original costs. The new markers can delay local arrival and change the collective critical rank.

This design can establish **Current instrumented dependency/exposure**, with empirical evidence about overhead if controls pass. It cannot establish a strict Scheduling latency floor: observed durations are not unavoidable service minima, and full-logits materialization is not proved algorithmically necessary. No Resource/Product TPS ceiling or attainable saving follows. Even a large measured P→J does not prove that removing the collective would save that amount; a legal intervention and formal E2E would be required.

## A0–B–A1 and contamination controls

Controller owns lifecycle. Freeze one baseline and exact arm manifest before A0; no service/NPU action is performed by this design.

- Each arm uses the same frozen source/configuration, container image, caches/warmup protocol, affinity, request dataset and client, with only B's reviewed sparse markers enabled. A0/A1 use restored originals; other capture/profiler/self-replay/sync flags are off. Existing uninstrumented Runtime count histories and branch reports are retained consistently across all arms.
- A0, B and A1 each run exactly48 warmup +12 diagnostic requests,1024 outputs each,c12. Distinguish client measurement wall from warmup and Runtime cohort sums. These are diagnostics, not formal repeated48-request throughput.
- Before every launch verify no live/orphan benchmark, health waiter, service or worker tree; health-down alone is insufficient. Gate container process census and Host memory/pinned-memory pressure in addition to all8 idle-device release. Bind one service UUID, client UUID and output directory per arm. Enforce empty output paths.
- Freeze a resource-clean starting policy. If any arm needs a container reset, allocator recovery, library change or different cleanup procedure after A0, invalidate that A/B/A series. Preserve its valid local observations and restart a fresh whole series from the same reset policy; do not reuse Run417-style reset A1 as a matched control.
- Require exact60 POSTs, client interval sweep max12, matching request ledger, five complete eight-rank Runtime cohorts, FULL Target, zero post-handoff oracle calls, exact output lengths and Host mirror gates. Access-count equality alone cannot exclude external load.
- Compare per-cohort cycle counts and available per-cycle accepted/useful counts, including c64 active shapes, across controls and B. Do not require long generated-token hashes against an unrelated run. Missing histories or differing workload trajectory blocks causal overhead subtraction/extrapolation; retain local instrumented observations.
- Predeclare the comparison fields and tolerance/noise policy from controls before inspecting B. One A0/A1 pair cannot establish a statistical hard overhead bound; B inside their envelope is only “no detected material overhead,” not proof of zero disturbance. Clear drift, acceptance mismatch or source/configuration mismatch yields INCONCLUSIVE for uninstrumented extrapolation.
- Stop/verify/restore and before/after SHA gates must fail closed. Never restore sources while service stop is unproved. No swallowed cleanup error; per-arm final status includes client, source, process, Host-memory and device-release gates.

## Acceptance and bounded stop

ACCEPT only the selected local joined path if branch/dispatch/storage identity, native completion semantics, all40 slices, correctness and timing validity pass. Report controls separately. Missing native wait provenance or a changed stream permits at most an API envelope; missing ranks/generations, extra collective calls, capturing the path into a graph, or altered source order makes the intended certificate INCONCLUSIVE.

Stop after this one slice is classified. A small/noise-limited result deprioritizes this slice; a material result supports designing a distinct legal intervention. Do not expand automatically to all265 collectives, Draft/KV, or a full DAG. Any failure should identify the first missing field/edge rather than launch another broad capture.

## Evidence and source references

Read: Run395 independent review; Run400 ledger/review; Run401 design, implementation review and preflight source; Run410 gate; Run412 findings; Run391 ordered HCCL source map. No new timer, source-patch or device test was run.

Targeted knowledge queries “logits allgather” and “communication overlap” used sources commit db3beef223e0b5acd81ccccb444600c1b22aac8a. Returned related MoE coalescing/DSA overlap work, not a certificate for this terminal DP1TP8 logits→argmax join. Historical mechanisms/timings are not transferred.

Reviewed source SHA256 (repin after Run437 restoration):
- runtime/target_adapter.py: c46d60318098077064c0ff57eae98bc937dcf943bda531918388be4e9128acd5
- runtime/fixed_acceptance.py: 824289bae3bd31676e3a697cc59b4169a345c70e10063c5660763e3e3557c6ae
- bootstrap/vllm_target_handoff.py: 2053dafbd46638d0da23e14b96c59ea01995ec2f0441d01928f1c9484c7b4ed5
- ASCEND/ops/vocab_parallel_embedding.py: a4a8517abc3a4f8494f27cc4cc3a876264ef8fea8cfc6de9abdbaf75d42c3b24
- VLLM/model_executor/layers/logits_processor.py: b2b17669a923a57d34c0a0b94902c4e7742f88a73e57ca9b881c84207fceaded
- VLLM/distributed/device_communicators/base_device_communicator.py: c4fafc71bbb3a7652ecdf425bf116e3a9ad203ba44baf005cbab5db88a8b8221
- ASCEND/distributed/device_communicators/npu_communicator.py: e91429f5cfba0b8af002b7c781db795d4826d7d96e4f08a9684664eb6d7ed275
- VLLM/distributed/parallel_state.py: ecfa5eeda697e9982591c93eca835fcc53463267c1c22110700819efcecee74f

ASCEND=/data/wio/vllm_ascend_26/framework/vllm-ascend/vllm_ascend; VLLM=/data/wio/vllm_ascend_26/framework/vllm/vllm. Line numbers are review-time locators; exact anchors/hash checks govern implementation.
