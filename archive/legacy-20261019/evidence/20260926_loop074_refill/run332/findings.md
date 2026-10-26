# Run332 — original warmed prefix work and all-rank request mapping

## Protocol and validity

The original Extreme serving path ran 48×32K→1024 c12 warmup, then 48×32K→1024 c12 diagnostic measurement on 8×910B3 DP1×TP8 DSpark7. No admission, Target, Draft, KV or serving schedule changed. The measurement window used passive Host scheduler fields and buffered mark output at each Runtime handoff. Both client passes completed 48/48×1024; 64/64 eight-rank Runtime cohort reports passed, including all four measured cohorts in FULL Graph. The runner exited 0, stopped the service and restored the borrowed ModelRunner source to SHA256 `004dbd0d5b1a5c3f9533fa1d12327bd0ae421fe24bb2674544b2c4a25d74aaba`.

## New evidence

- All eight ranks emitted the same 42 semantic marks per rank. Exactly 48 new measured requests mapped through client stream ID prefix → internal scheduler request ID → one of four Runtime cohorts and its slot. Each cohort's 12 slots matched client order (`0–11`, `12–23`, `24–35`, `36–47`). This mapping applies to Run332, not retrospectively to Run287.
- Every new request arrived at the worker with `num_computed_tokens=32768` from the existing prefix cache. Prompt lengths were 32,851 (46 requests), 32,852 (1) and 32,853 (1), leaving **83/84/85 actual prompt tokens**, respectively. The all-rank scheduler marks show those exact prompt residual counts were scheduled before speculative decode. Thus the warmed protocol does not require 32K new prompt computation per refill. This is one legal warmed trajectory; cache residency under a future continuous scheduler remains to be proved.
- Four measured cohorts took **2.150, 2.433, 2.400 and 2.102 s** from their earliest request execute entry to the latest rank's `build_extreme_runtime` return. These are Host-clock preparation envelopes including multiple scheduled calls, not isolated prefill device occupancy or DSpark seed-ready endpoints. Per-request scheduled decode/spec steps are kept separate in `analysis.json`; the 8/16/24-token calls in prior Run239 cannot all be called prompt prefill.
- Diagnostic client output was 48/48 and 623.357 tok/s in this one instrumented pass. It is **not** a new formal Current or a causal performance improvement. Formal Current remains Run99 median 571.681 tok/s.

## Dual-Bound decision

Run331's b=2–4 zero-cost conditional schedule retains 54–65 target cycles against the Run287 four-cohort trajectory, with 9–18 refill batches. Run332 removes the major uncertainty about warm-prefix *quantity* but not the complete incremental cost. Present evidence suggests examining batch preparation and potential overlap, not starting a live reset solely from the cycle model. A microbatch's actual per-call Host/device/collective/seed cost, its effect on continuing 11 slots, Graph reuse, KV ownership and acceptance trajectory remain unknown.

Next use a short original-path all8 device and Host capture across representative natural 83-token residual prefill shapes through the first DSpark seed consumer. Preserve existing stream and HCCL dependencies; do not substitute a default-stream event or Host forward return for seed readiness. Compare the measured resource occupancy and preparation episodes with the existing four cohort preparations before a one-microbatch live semantics test. No numeric Hardware/Resource, Scheduling-aware attainable or Product E2E Bound is updated.

Primary artifacts: `analysis.json`, `marks/rank*.jsonl`, `measured48.json`, `warmup48.json`, `runtime/rank*_cohort*.json`, `patch_install.json`, `patch_restore.json`.
