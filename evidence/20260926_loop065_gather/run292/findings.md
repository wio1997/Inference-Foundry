# Run292 delayed hidden gather device screen

The instance-bound B launcher exited 0. Server logs show exactly 60 POSTs and maximum 12 in flight. Warmup48 and bench12 all returned exactly 1024 tokens. All eight ranks captured the selected Target layer2 delayed path with BF16 local `[12,4096]`, gathered `[96,4096]`, pad0. All 40 rank-cohort Runtime reports passed FULL Graph/host-mirror checks, five cohorts per rank. Cleanup restored the borrowed source to the pinned SHA and left NPUs idle.

The latest exported profiler directory per rank contains two sampled cycles. Across all 16 rank-cycle samples, the layer2 hidden AllGather device interval overlaps main-stream local Q compute. Median HCCL task duration is 14.88 us and median actual device interval overlap is 13.75 us. The gather ends before its downstream WKV use; Astra independently matched Model45/Task53 to Q Tasks143–148 and measured the main-stream join wait at 0–0.22 us. This is a real execution scheduling relaxation, not merely asynchronous host submission.

The instrumented 531.622/550.371 warmup/bench tok/s are diagnostic only. This B profile cannot be compared with unprofiled Run291. Matched immediate profile Run293, same-prestate typed numerical parity, original A0 control, complete-cycle and repeated formal E2E remain required before a KEEP or TPS/bound promotion. Resource/Hardware and Product numeric bounds remain unknown.

Evidence: server/launcher/runtime logs, `profile/rank*_ascend_pt/ASCEND_PROFILER_OUTPUT/trace_view.json` (latest per rank), `timeline.json`, independent Astra screen.
