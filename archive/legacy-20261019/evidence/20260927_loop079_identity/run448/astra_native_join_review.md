# Run448 — independent review of Run447 native logits join

2026-09-27. **ACCEPT as a conditional source certificate, with explicit bypass gates below.** No service change, NPU query/computation, framework import or patch operation was performed. Installed package files were read through a stdlib-only container command; public commit source was fetched directly. Only this review is written.

## Independently verified

Installed torch_npu/version.py declares 2.10.0.post4 and git_version 5dd8ef3f9b375b5ae4a83538d5785754148c3302. Recomputed libtorch_npu.so and installed distributed_c10d.py hashes exactly match Run447. Independently fetched ProcessGroupHCCL.cpp also matches Run447's ba393be1672c7ce84dbbe9e9705ad9164a70af52a1883d487931ab1e212278ab. These pin observed artifacts and declared source provenance; package metadata is not a reproducible-build proof or a record of libraries loaded by a particular historical worker.

At that [exact commit](https://github.com/Ascend/pytorch/blob/5dd8ef3f9b375b5ae4a83538d5785754148c3302/torch_npu/csrc/distributed/ProcessGroupHCCL.cpp), the ordinary path supports the claimed chain:

- syncStreams:239–250 records on caller-current and makes HCCL wait.
- _allgather_base:5895–5949 dispatches through collective and enqueues HcclAllGather on its supplied HCCL stream.
- collective:4104 and4249–4271 preserves producer ordering, submits work, then records its end event.
- WorkHCCL::wait:1143–1152 calls synchronizeInternal;1055–1067 makes the current stream at wait time depend on the end event.

Capture-status handling in this collective does not remove these dependencies; it changes other handling. Blocking-wait configuration can additionally block the Host. Neither Work.wait's return value nor a completed Future is a device timestamp or an independent correctness certificate.

Same-commit [NPUEvent.cpp:115–156](https://github.com/Ascend/pytorch/blob/5dd8ef3f9b375b5ae4a83538d5785754148c3302/torch_npu/csrc/core/npu/NPUEvent.cpp) uses queued record/wait operations; per-stream-queue/capture handling can add Host-side effects. elapsed_time itself drains/synchronizes:159–181, so evaluate it only after the cohort, as Run439 specifies.

Current vLLM source hashes still match Run439. The ordinary torch.ops.vllm.all_gather implementation forwards to GroupCoordinator._all_gather_out_place (parallel_state.py:160–167,686–689), then inherited DeviceCommunicatorBase.all_gather. NPUCommunicator does not override all_gather. The base call uses dist.all_gather_into_tensor without async_op, then performs layout reshapes. Custom-op wrapping alone therefore does not invalidate the source chain.

## Challenge: Run447 omits concrete Python bypass conditions

The exact installed distributed_c10d.py has two relevant branches before its ordinary _allgather_base/work.wait path:

1. **4084–4092: has_torch_function** redirects to an override/mode, potentially functional collectives. Plain-looking callsites alone do not exclude this dispatch.
2. **4116–4123: group in _world.pg_coalesce_state** queues a collective representation and returns immediately. A J event after this call would not identify completion of the deferred collective.

The usual path is4125–4132, with async_op=False and non-null Work. Also require real membership/backend; the not-in-group branch returns without a collective. Do not equate a None public API result with proof that Work.wait ran: both the normal synchronous path and coalescing can return None.

Graph/FakeTensor/compiled execution is a separate scope issue. The fake custom-op implementation and graph replay do not execute the ordinary Python hooks per replay. Existing FULL Target forward does not itself disqualify the later eager logits call; Run439 must establish that this particular logits invocation is outside capture/replay/compiled replacement. A captured native dependency may be correct while the proposed sparse Python timestamps are invalid for its replay generation.

## Exact remaining Run439 dynamic gate

For each selected cycle/rank, capture a small branch manifest under the scoped logits tag:

- Actual normal AscendLogitsProcessor branch; lmhead_tp=false, reduce_sample=false, use_all_gather=true. Record head dtype, logits shape, scale/soft-cap and actual loaded function identities.
- Exactly one intended native call; real eight-rank TP membership and ordered group identity; concrete HCCL backend. Bind source/library fingerprints to these workers at startup.
- No tensor/function/dispatch override or functional/compiled replacement; no active Python coalescing context; async_op=False. Require observed entry/exit through the original ordinary API path, not just its public return type. Any unhandled wrapper/backend branch is OUT_OF_SCOPE.
- The actual current stream at local producer, collective entry/return, J, layout completion and argmax consumer is the same device/stream. The native source joins to the stream current at wait, not an arbitrary later consumer stream. A changed stream requires its own existing dependency proof; never insert a diagnostic wait to repair it.
- No capture or replay substitution at the selected logits call; exactly one matched event generation per scoped invocation. Preserve input→native output→layout/slice→argmax storage lineage and expected shape/dtype. Keep Run439's all-rank coverage, correctness, timer and A0–B–A1 gates.

Under these dynamic conditions and accepted installed-source provenance, Run447 closes Run439's **conditional native ordering prerequisite**. Another generic sentinel run or full native trace is not mandatory merely to rediscover the same ordering. If an override, coalescing, different backend or graph branch occurs, stop and audit that branch; sparse event numbers cannot compensate for missing dispatch evidence.

J is then a same-stream timestamp after the existing native join, not the exact HCCL end time. P→J remains a current envelope including arrival/submission/backend delays; J→G may include layout materialization. No cross-rank aligned makespan, wire service time, compulsory work, strict latency floor, removable saving or numeric Bound is promoted.

## Identity

- Reviewed Run447 Markdown SHA256: 02ef9c856e38005fbac0fe12f2e98fe172be44f8eb02f47801dae88ebd901107
- Installed version.py: f11e02e3acade31a81657136784872a1d70e2f232cab0ef19d3a605aec47424d
- Installed libtorch_npu.so: 83fb9a0eb249aef6bca7f8463047fcb4d3062cc4849f17ddf31a5e050c838842
- Installed distributed_c10d.py: 31155344233ae77360a630bbeacf1b43cfa18f1923cf2103cd5b3142db39ec5c
- Current target_adapter, fixed_acceptance, parallel_state, base_device_communicator, npu_communicator and Ascend vocab_parallel_embedding hashes were rechecked against Run439: all unchanged.

Installed package root: /usr/local/python3.12.13/lib/python3.12/site-packages. No original Run99/Run437 dynamic branch or loaded-library fact is inferred from this filesystem review.
