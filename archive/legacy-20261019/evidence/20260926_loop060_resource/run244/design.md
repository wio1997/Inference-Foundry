# Run244 design: graph-path target MemoryAccess calibration

Existing Run107 gives 15 valid synchronized target windows, 43 GMM1, 43 GMM2, 236 quant matmul, 62 compressor, 126 scatter and 260 HCCL tasks/window, but no HBM counters. Run148/150 Level1 MemoryAccess measured one-card GMM under synthetic inputs, leaving transfer to real eight-rank graph uncertain. Run237 eager HCCL is path-incompatible and excluded.

## Frozen diagnostic

- Use the product fixed DeepSeek V4 W4A8 DP1TP8 DSpark7 service with static KV max, target FULL graph and the same 12x1024 c12 client as Run107. Complete a legal 48-request warmup first. Correctness gate: 48/48 warmup, 12/12 diagnostic exact 1024, eight rank runtime records pass.
- In `runtime/extreme_decode.py`, add an opt-in `EXTREME_RUNTIME_CYCLE_PROFILE_MEMORY_ACCESS=1` that configures `torch_npu.profiler._ExperimentalConfig(profiler_level=ProfilerLevel.Level1, aic_metrics=AiCMetrics.MemoryAccess)`. Default profiler behavior and normal serving stay identical.
- Capture only target cycles 64–65, eight ranks, `EXTREME_RUNTIME_CYCLE_PROFILE_SYNC_TARGET=1`, as Run107 did. Graph target must remain FULL. Keep both raw trace and kernel_details per rank; no formal E2E throughput claim under profiler/synchronization.
- Match kernel_details `Start Time(us)` to trace_view `extreme::target` CPU scopes on the same clock, then check expected family counts before computing per-family counter bytes. HCCL tasks do not carry AICore memory counters; report them separately. Reject a rank-cycle if contaminated or counts differ. Preserve *per-window* family bytes, task sums, overlap and rank spread; never add task time to E2E.
- Success for model calibration: at least 12 valid rank-cycle target windows with nonzero read/write counters for GMM, quant matmul, compressor, sparse attention and cache scatter, plus an explicit graph-vs-one-card GMM comparison. If counter export lacks these fields, record invalid and keep UNKNOWN. Complete stop and verify idle, with no borrowed framework source edits.

## Predicted discrimination

If real graph GMM read/active-packed ratio stays near Run148/150 (~1.06–1.08), GMM is a narrow traffic gap and the next capacity work should focus on the non-GMM DSA/quant chain. If GMM read multiples or bandwidth differ materially, revise the weight floor first. If non-GMM HBM bytes explain most 30ms compute remainder at attainable shape-specific bandwidth, prioritize dataflow/bytes; if they do not, prioritize dependencies, launch, utilization and possible layer fusion. No measured kernel family bytes alone imply an achievable E2E saving.

Resources: one eight-card service load and two profiled target cycles, estimated profiler output under 100MB plus warmup/diagnostic requests. Abort and record failure on OOM, missing counters, runtime correctness failure or source state mismatch; restore service idle. Avoid repeat Run93/Loop034 official E2E.
