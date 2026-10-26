# Run667 — metadata Graph formal E2E: reject promotion

Frozen Run99 bench.py protocol, 48×32K→1024 c12, warmup48 plus three measured 48-request repeats per arm. Both arms: 48/48 successes on each repeat, 192 server POSTs, 128/128 FULL Target Graph rank-cohort rows pass. ON metadata Graph capture/replays are active. Controller exit0, stop/NPU idle and exact source/script SHA restoration pass.

| repeat | OFF tok/s | ON tok/s | OFF cycles | ON cycles | OFF ms/cycle | ON ms/cycle | OFF client−runtime s | ON client−runtime s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 578.117 | 571.982 | 1194 | 1199 | 56.538 | 55.293 | 17.514 | 19.636 |
| 2 | 577.073 | 567.938 | 1212 | 1209 | 56.861 | 55.311 | 16.259 | 19.673 |
| 3 | 634.358 | 542.738 | 1166 | 1256 | 57.147 | 55.491 | 10.850 | 20.867 |

OFF median **578.117 tok/s**; ON median **567.938 tok/s**, **−1.761%** versus contemporary control. ON saves about 1.2–1.6 ms per fixed-serving cycle, but client-to-runtime residual is higher in all three repeats. This residual includes prefill, scheduler/HTTP integration, and cohort boundaries; the current evidence does not isolate its cause. ON repeat3 also required 1256 cycles versus OFF 1166. Run666 diagnostic +2.9% is therefore not a formal gain. Keep formal Current Run99 571.681 tok/s and do not install the candidate by default.

Next performance question: isolate where the 3–10 s residual change arises with stage-aligned per-cohort timing or existing service evidence before another candidate. Prioritize prefill/scheduler handoff and output-drain timing; retain the proven 1.2–1.6 ms/cycle metadata subpath saving only as a possible component of a broader structural change.
