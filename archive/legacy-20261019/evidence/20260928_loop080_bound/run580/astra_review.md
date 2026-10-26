# Astra High independent Run580 and V3.36 review

Scoped PASS. Eight copied Level0 traces each contain two Model33 GMM Graph replays of 86 native tasks and two Model32 HCCL Graph replays of 265 AIV collective tasks. Task IDs, streams and timestamps join `task_time.csv` to true task events in `trace_view.json`; the separate all-reduce barriers and OnesLike setup are excluded. All 534 original raw files match their copied inputs and remain immutable. Same-ordinal timing and rank-local spans in `timeline_attribution.json` were independently recomputed.

Under concurrent replay, HCCL's first native task starts 0.37–0.42 ms after GMM's first. The exported task intervals overlap 12.12–12.61 ms. The first 84/86 matched HCCL task-time medians rise from serial 13.34–14.36 µs to concurrent 144.73–146.40 µs; the matched tail stays near 13–14 µs. One HCCL task/rank crosses the GMM end. Task intervals are not physical AIV occupancy, HBM/link traffic or proof of which rank waited. A single profiled replay per arm is mechanism evidence only; repeated unprofiled events are the service-time evidence.

V3.36 generator SHA `b8c1d737ea513710e64f16555f4dc6a564b8d1a6e42c4cb2a2e1f550af010fb0` and output SHA `d3a5c5ad9c2f7cca7ae9864627118c77b1fc120134fa748b2f9d83fb933e4f71` replay exactly. Four finite-endpoint injections and four provenance-class SHA negatives are rejected; the timeline pins the eight bench rows it reads. It does not promote any strict Resource, Scheduling or Product endpoint.

Next prioritize one production fixed-W₀ producer-ready → collective submit/completion → consumer-ready slice to establish a legal overlap window. Add Level1 PMU or rank-arrival probes only if the slice makes that resource distinction decision-critical.
