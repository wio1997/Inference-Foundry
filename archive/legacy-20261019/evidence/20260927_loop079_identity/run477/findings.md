# Run477 — no-getter terminal B acquisition

2026-09-27. Frozen DeepSeek V4 Flash W4A8, 8×910B3, DP1×TP8, DSpark7. This is a diagnostic 48-request warmup followed by one 12-request/c12/1024 request cohort; it is not a formal E2E performance repeat.

The Run472 no-getter helper passed Run474 independent source review and Run476 controller recheck. The live B2 controller exited 0. All 60 HTTP requests completed with exactly 1024 output tokens and no request errors. Admission found five cohorts × eight ranks = 40 selected cycle64 records, the ordinary BF16 `[96,16160]` TP8 terminal-logits AllGather, FULL Target graph, exact Runtime output/count gates, and no competing profiler. `final_admission.json` is valid after stop, idle-NPU verification, six-file restoration and byte-identical source SHA. The service is stopped.

The versioned reducer gives the following same-device, *instrumented* local intervals over 40 correlated rank/cohort slices:

| Edge | min / median / max (ms) |
| --- | ---: |
| P→J | 0.17392 / 0.18219 / 0.20672 |
| J→G | 0.02790 / 0.03086 / 0.03814 |
| G→C0 | 0.00014 / 0.00016 / 0.00042 |
| C0→C1 | 0.05196 / 0.05280 / 0.05602 |
| P→C1 adjacent sum | 0.25644 / 0.26736 / 0.29632 |

These are not 40 independent workload repeats or an aligned all-rank makespan. G→C0 is below the earlier calibrated useful precision of this event method. Distinct native gather output and final layout storages are each 24,821,760 bytes; this is current materialization, not compulsory traffic. The current input is 3,102,720 bytes/rank.

Run463's queue-drained diagnostic had P→C1 median 0.28424 ms and P→J median 0.20040 ms. Run477's lower medians are consistent with the removed raw stream getter, but the acquisitions are unmatched and differ in trajectory/system state, so neither the difference nor either absolute interval can be assigned to uninstrumented Current or removable E2E time. The selected event/Python calls still perturb submission. No finite Algorithm/Resource, Hardware/Resource, Scheduling/Execution or Product E2E Bound changes; formal Current stays 571.681 tok/s. Next evidence should bind the larger Target replay/hidden-output frontier separately and, if terminal cost becomes decision-critical, run matched unchanged-contract A0–B–A1 controls.

Artifacts: `run472/b_clean2/` raw acquisition and admission, `run477/intervals.json` reproducible reduction. Run480 independently accepted diagnostic admission and recomputed all statistics. Its review confirms that the saved HTTP records establish length/count and Host-mirror parity, while a new semantic token-oracle comparison was not collected.
