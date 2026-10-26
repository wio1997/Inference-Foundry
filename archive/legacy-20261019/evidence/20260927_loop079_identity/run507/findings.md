# Run507 invalid diagnostic

The guarded 48 warmup + 12 diagnostic workload completed its HTTP and capture phases, but the controller returned exit 1. The validator was invoked by absolute script path and imported `scripts.loop079_target_frontier_update_validate_run507`; in that invocation `scripts` was not on `sys.path`, causing `ModuleNotFoundError`. Final admission rejected the run. Stop, stop verification, restoration of all six borrowed sources, and source SHA comparison each returned 0. The service is stopped.

Post hoc structural validation after fixing the import path does **not** change Run507 status. Its observed MLA update counts are inadmissible for Bound promotion. The fix and final-only negative controls were independently preflighted for a fresh Run508 acquisition. No formal E2E TPS, correctness, or finite Bound claim follows from Run507.

Evidence: `b_candidate/cleanup_status.txt`, `b_candidate/final_admission.json`, `b_candidate/local_validation.log`, `../run508/preflight/astra_review.md`.
