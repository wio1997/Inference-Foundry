# Run394 clean sparse original-path capture

Run394 repeated the original fixed serving path after Run393 invalidated the overlapping Run389/390 clients. Preflight found no orphan health waiter or bench client. The reversible patch captured current-stream NPU events at Target/acceptance/state/DSpark boundaries only on cycles 64 and 65. It copied the existing acceptance history at cohort end; no Level1 profiler or per-cycle D2H was used.

Execution exit 0. Warmup 48 and diagnostic 12 requests each returned exactly 1024 output tokens with no errors. The single APIServer logged exactly 60 successful POSTs. Five cohorts produced 40 event and 40 Runtime reports across eight ranks; Runtime FULL Graph pass on all reports. Bench diagnostic output TPS was 555.925 and is not a new formal performance result. Borrowed-source SHA files match before/after and the service stopped.

Run395 validates 80 rank-cycles and 40 same-rank cycle64 to 65 pairs. Current-stream cycle64 begin to cycle65 begin median is 56.548 ms, with observed Target-label stage median 49.465 ms and proposer-label stage median 6.397 ms across selected cycles. Same-rank Target-end to next pre-Target marker median is 7.648 ms. The Host anchor record/synchronize brackets span 0.220–0.545 ms.

These are sparse-instrumented Current path observations under clean c12, not an uninstrumented Product wall interval, isolated operator service, critical-path saving, Hardware floor or attainable Scheduling floor. Internal Target Graph and proposer side streams, all-rank clock/collective joins, marker A/A overhead, legal prefill/seed/parking and acceptance trajectory remain unresolved. Run396 V3.7 records conditional observations and leaves finite TPS upper endpoints null.
