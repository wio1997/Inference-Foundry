# Run337 — original-path unprofiled warm-prefill Host progress

## Frozen diagnostic and gates

The original Extreme 8×910B3 DP1×TP8 DSpark7 FULL Graph service ran 48×32K→1024 c12 warmup, then one measured 12-request cohort. Client gates were 48/48 and 12/12 outputs of exactly 1024 tokens. All eight Runtime cohort reports passed in FULL Graph mode. The first measured residual-prefill `_model_forward` had `num_tokens_padded=88` on every rank. No CANN profiler or additional `torch.npu.synchronize()` was used. The runner exited 0, restored borrowed ModelRunner source SHA256 `004dbd0d5b1a5c3f9533fa1d12327bd0ae421fe24bb2674544b2c4a25d74aaba`, and stopped service (HBM returned to about 3.4 GB/card). The one-cohort 612.514 diagnostic tok/s is not a formal performance result.

## Observation

The wrapper captured 221 Host collective calls/rank: 87 ReduceScatter, 43 AllToAll, 91 AllGather. Their kind, input bytes and dtype sequences matched on all eight ranks. Relative to Run333's 264 CANN HCCL tasks/rank, 43 AllGather calls were outside the wrapped Python APIs; the capture is a complete *wrapper scope*, not a complete HCCL trace. Wrapper input bytes are API tensor sizes, not physical link bytes.

The eight Host forward walls were 322.394–364.099 ms; each thread's CPU/wall ratio was 0.9994–0.9998. Forward entry spread was 2.940 ms and return spread 43.668 ms. Among the 221 captured calls, Host entry spread rose from 3.684 ms at the first call to 43.646 ms at the last (median 23.749 ms); latest entry came from rank 6 for 134 calls and rank 5 for 87. AllGather/ReduceScatter/AllToAll showed comparable accumulated spread. Unlike profiler Run333, rank 4 was not the persistent late submitter. The slowdown pattern therefore survives without CANN profiling, while the late-rank identity and magnitude are sensitive to execution conditions.

The rank5-minus-rank2 Host entry gap grew by 41.465 ms over 221 events. Of that observed growth, adjacent ReduceScatter→AllToAll intervals contributed 21.248 ms across 43 repetitions; AllGather→next ReduceScatter contributed 9.871 ms. Rank6-minus-rank2 shows the same broad pattern (20.686 ms and 7.700 ms). This is a difference of Host entry timestamps, not a source-proven per-operator cost or exposed Product saving (`transition_skew.json`). The 43 AG calls absent from the wrapper likely follow the current `distributed/utils.py::all_gather_async` direct `dist.all_gather_into_tensor` path, so a complete Host/HCCL DAG still needs API or flow alignment.

This is evidence of rank-dependent Host execution/submission progress during a legal warm residual-prefill. Host call entry is not an input-ready device event. The approximately 41 ms growth in rank spread is **not** a measured removable Product latency; it can be hidden, shifted or amplified by downstream device/stream work. The large 322–364 ms Host wall is also not an attainable Graph latency floor. Actual device completion, all 264 collective producer arrivals, DSpark seed-ready and Product E2E exposure remain open.

## Historical prior and next decision

Pinned history `db3beef` R14 shows that a Host prepare/finalize pool did not turn into old DP2 service gain; current DP1 uses a different direct branch. R20's hidden AllGather/local-Q prefill overlap already exists in current `dsa_cp.py` and its cold TP4 result cannot be applied to the TP8 local11/global88 residual shape. R26 shows local event-schedule gain can fail at service boundary. Current Run326 routed48 full-MoE private Graph was slower at decode96, so it is a caution for any Graph candidate, not a verdict about warm prefill88. These priors favor a same-shape, complete-endpoint test before any live architecture change.

Candidate next gate, subject to independent Bound review: original warmed padded88 full-MoE eager A / original-operations Graph B / eager A2 on identical inputs and route/state, all eight ranks, with numerical and complete-endpoint timing gates. It must report Host submission, device latest completion and A/A2 drift; a Host-only saving does not qualify. No Hardware/Resource, Scheduling-aware attainable, or Product E2E numeric bound changes. Formal Current remains 571.681 tok/s.

Evidence: `analysis.json`, `marks/rank{0..7}.json`, `warmup48.json`, `measured12.json`, eight `runtime/rank*_cohort5.json`, `patch_restore.json`, `stop.log`. Analyzer: `scripts/loop074_run337_host_collective_analyze.py`.
