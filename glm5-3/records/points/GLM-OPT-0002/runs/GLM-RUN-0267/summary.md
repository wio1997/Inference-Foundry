# GLM-RUN-0267 — H10 NOT_APPLICABLE

Original frozen controller FAILED06:22:46Z at baseline applicability witness; mode1 never enabled, no candidate NPU correctness or A/B/A/B. All16 workers actually use async_scheduling=true; H10's synchronous PP1 path is ineligible. Baseline short8, natural complete23/EOS and terminal short8 match golden IDs/usage and all16 KV transfer. These three requests are correctness evidence, not gain or formal93% workload.

Separate zero-request APPLICABILITY-CLOSURE-20261007 completed06:28:07Z/phase0 with no reload/source mutation and same workers. Preserves original failure/source/spec. Retained H6/H5all16, H10/H9/H8/H4off, pure KV/API repairs, health200/idle. Independent reduce_early_mtp_not_applicable_267.py passes raw SSE/metrics/native/mode/identity/closure checks; applicability_reduced.json is authoritative derived result. The earlier reduce_early_mtp_267.py expects a completed21-request comparison and was not run or relaxed to fake a pass.

CPU source/guard proofs remain valid for the scoped synchronous branch, not current runtime gain. Avoidable applicability assumption acknowledged; future resets establish effective scheduling/queue mode before a reload. GoalReview172/independent async-source review continue code research on H6/H5. FinalCurrent=None; complete API/formal SLA/cumulative-stack stability remain open.
