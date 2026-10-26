# Run487–490 — Target FULL replay/hidden frontier

2026-09-27. Frozen DeepSeek V4 Flash W4A8, eight 910B3, DP1×TP8, DSpark7. Source-only Run484 repair cleared independent Run486 preflight. Run487 then completed one clean 48-request warmup plus 12-request c12 diagnostic, all requests exactly 1024 output tokens; this is **not** a repeated formal performance run.

Run487 controller exited0, with exact60 HTTP200 POSTs, 40 capture and40 Runtime reports across ranks0–7/cohorts1–5, all selected cycle64 FULL graphs, 12-slot staged/accepted/overshoot count equality, and post-stop source restoration. Its local and final validators passed. Run488 independently reproduced all86 reducer input hashes, all event statistics, source pins and cleanup; its result is PASS **only for instrumented conditional Current evidence**. HTTP length/Host-mirror/count correctness does not add a fresh semantic token-oracle comparison.

| Same-device selected interval | Minimum | Median | Maximum |
| --- | ---: | ---: | ---: |
| T→R0: Target entry to replay start |0.500960 ms|0.782230 ms|1.144820 ms|
| R0→R1: FULL replay stream |45.250641 ms|45.986019 ms|48.147678 ms|
| R1→H: replay return to gathered hidden/aux |0.059200 ms|0.090500 ms|0.142400 ms|
| H→U: gathered output to sampled hidden |0.025480 ms|0.026770 ms|0.033880 ms|
| T→U: adjacent sum on each chain |46.372820 ms|46.904029 ms|48.830900 ms|

Each rank's selected entry is the startup captured FULL96 graph bound to the same Runtime Target model/batch and input/position storage. All40 records have one `after` graph-update call, configured stream102 with **unresolved private-stream relation**, and four ordered ordinary BF16 `[12,4096]` TP8 hidden/aux AllGathers producing `[96,4096]` leaves. R1 is a caller-stream replay marker. It is downstream of required output producers **only if** those producers are members of the captured graph with valid child-stream joins. Neither output last-writer membership nor effective update-stream work was proved by Run487. The existing pre-replay Host sync cannot be subtracted from T→R0 as removable overhead.

The five cohorts contain 294/337/294/300/298 Runtime cycles, with 12,288 useful output tokens each. They are five correlated trajectories, not40 independent formal repeats. There is no common clock across ranks, matched A0–B–A1 probe control, passive original-path cost, compulsory traffic/work, resource-capacity upper certificate or legal replacement schedule. Run394's Target label and Run477's terminal seam have different boundaries and trajectories, so they are not additive or matched differences.

Run489 V3.21 admits the exact reviewed local observations and explicit dependency conditions while keeping all finite Algorithm/Resource, Hardware/Resource, Scheduling/Execution and Product E2E endpoints null. Run490 Astra High independently accepted the byte-identical model rebuild, 19 unresolved proof nodes and all-null endpoints. Formal Current remains Run99 median **571.681 tok/s**; the Current→credible-limit distance remains numerically unresolved.

Next Bound gate: in a new same-process acquisition, dump the exact selected graph after ordinary drain and bind all four output leaves to actual last writers and event/child-stream joins. Bind the effective graph-update backend and whether its after call enqueues work. Align source/config/shape/branch to Run487; never compare process-local IDs or addresses across runs. Only then decompose replay resource/arrival constraints and test an attainable schedule. In parallel, the strict Resource path still needs one fresh retained in-window work witness and an authoritative matching 910B3 C-plus certificate.

Evidence: `run484/b_candidate/` raw+final admission, `run487/intervals.json`, `run488/astra_frontier_review.md`, `run489/bound_calibration_v3_21.json`, `run490/astra_v3_21_review.md`.
