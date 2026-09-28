# Run610 — INVALID scoped profile controller

The guarded warm48/measured48 c12/1024 diagnostic completed all 96 client requests. All8 Basis, Product and dispatch identity admissions passed. The client rate of 555.1125 tok/s is profiler perturbed and is **not** a formal E2E result.

The original controller returned exit1 because its validator compared `profiler_info.start_info.start_monotonic` with Python `time.monotonic_ns` Host marks. Independent Host-only testing of the installed `torch_npu._C._profiler._get_monotonic()` established that the start value uses `CLOCK_MONOTONIC_RAW`; `end_info.MonotonicTimeEnd` uses Python `CLOCK_MONOTONIC`. Those values cannot be ordered or subtracted directly. The original validator also expected `all_file.complete`, which this Level0 format does not emit. Preserve this failed validator and its log unchanged.

The run and final controller exits are 1. All seven stop/verify/restore/source/script cleanup checks are 0; source and scripts before/after manifests match, the service is stopped and all8 NPUs idle. The eight raw profiler sessions remain under `run610/live/b/profile/cohort5/`. They are handled only by separate Run611 offline admission. Neither this run nor its diagnostic TPS changes any Bound endpoint.

Evidence: `live/b/profile_validate.log`, `live/cleanup_status.txt`, `live/b/client_admission.json`, three upstream admissions, and Run611 clock witness/recovery review.
