# Run658 — proposer Host-call dispersion versus current-stream duration

Read-only reuse of Run656 ON full48 packets, pinned to Run657's 32 input SHA256 values. The same 714 steady, unparked adjacent cycles are selected. Independent direct recomputation from raw packets agreed on both range medians and latest-rank counts. No service or NPU work ran.

Across each cycle's eight ranks, proposer Host-call duration max-minus-min is median **4.549 ms** (p95 8.465 ms). The corresponding current-stream Event proposer duration max-minus-min is median **0.058 ms** (p95 0.746 ms). The rank with latest proposer-after Host submit has the longest recorded Event duration in 142/714 cycles, or is within 5 us of it in 200/714.

This disfavors the simple explanation that all Host completion skew comes from one rank consistently spending longer inside its current-stream proposer Event stage. It does **not** establish that Host overhead is exposed or removable: Event stage includes queue/wait, other streams and collectives are unobserved here, and per-rank Event origins cannot be subtracted across ranks. Run657's Host skew and this duration comparison are same W0 and may overlap with rank arrival/HCCL peer-wait observations; their numbers are nonadditive.

Decision: before a Runtime change, use a selected all8 same-W0 ready→issue→native/collective packet to distinguish delayed Host enqueue from necessary producer completion and resource contention. Dynamically select the latest rank. Do not retry the rejected metadata-overlap mode or infer a Product TPS ceiling.
