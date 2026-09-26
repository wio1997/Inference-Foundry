# Loop069 live owner16 integration gate

Scope: frozen DP1×TP8, c12, Target FULL Graph, layer2 c4 producer only; AllGather and local query retained. These three 12×1024 cohorts are diagnostics, not the 48×32K→1024 formal E2E.

| Run | Path | Client | 8-rank Runtime | Cycles | Latest-rank wall | Latest-rank ms/cycle |
| --- | --- | --- | --- | ---: | ---: | ---: |
| 310 | original | 12/12×1024 | 8/8 FULL | 315 | 17.8116 s | 56.5449 |
| 311 | layer2 owner16 | 12/12×1024 | 8/8 FULL | 306 | 17.3837 s | 56.8096 |
| 312 | original repeat | 12/12×1024 | 8/8 FULL | 290 | 16.4265 s | 56.6430 |

Run311 records all eight owner producer-complete captures with Graph key `(96,12,uniform,non-LoRA,0)`, ordinary runner dispatch geometry, Extreme handoff qsl `[0,8,…,96]`, and Runtime replay dispatch. This establishes that the specialized Graph was constructed and used during one complete cohort. The source patch was SHA-guarded and all four files were restored; service stopped and cards were idle.

Across separate services, Run310 versus Run311 differs in all 12 reasoning hashes and one nonempty content hash. Run310 versus **unchanged original** Run312 also differs in all 12 reasoning hashes and two content hashes. Thus cross-service content equality is unstable here. This does not excuse a candidate semantic error; it makes the present content comparison nonidentifying. Acceptance trajectories and cycle counts differ. Run311's lower total wall and higher client TPS are not attributable to the owner intervention; its latest-rank wall/cycle is above both original diagnostics. No product performance gain is established.

The Run311 script exited 1 after completing the client and Runtime because `ROOT` was not exported into the container for offline comparison. The comparison and structural checks were performed afterwards. `ROOT` export and cleanup propagation are fixed for future runs. This harness error does not invalidate the observed 12-client and 8-rank completion, but Run311 remains **correctness inconclusive**.

Next: same-prestate private Graph with metadata generation *inside* capture. Compare original full and owner producer→QLI→Sparse plus typed owner writes, with A/A control and synthetic `start_pos` shifts of −1/−4 to expose stale Graph bindings. Synthetic shifts are metadata sensitivity tests, not real next cycles. If they pass, test real Runtime cycle metadata and persistent cache behavior before any all-layer or formal E2E promotion. Historical R13 is a host metadata scheduling prior from old DP2×TP4; it does not decide this Graph correctness issue.

Current formal 571.681 tok/s remains unchanged. Resource/Hardware, Scheduling-aware and Product E2E numerical bounds remain unknown.
