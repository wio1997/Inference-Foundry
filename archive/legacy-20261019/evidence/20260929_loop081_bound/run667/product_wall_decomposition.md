# Run667 Product wall reconstruction

The c12 client admits four serial groups of 12. The cohort join below follows their order and validates 12 Runtime request IDs and 12×1024 output per group. `bench.py` did not save response IDs, so the join is order-based, not an exact ID join. Runtime rows have duration but no absolute start/end stamp.

| repeat | arm | client s | sum start→last first s | fixed Runtime s | remaining after last first s | cycles | useful tok/cycle |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | off_a | 85.021 | 16.957 | 67.507 | 0.565 | 1194 | 41.166 |
| 1 | on | 85.933 | 18.901 | 66.296 | 0.745 | 1199 | 40.994 |
| 2 | off_a | 85.175 | 15.792 | 68.915 | 0.476 | 1212 | 40.554 |
| 2 | on | 86.545 | 18.916 | 66.871 | 0.765 | 1209 | 40.655 |
| 3 | off_a | 77.483 | 10.383 | 66.633 | 0.487 | 1166 | 42.154 |
| 3 | on | 90.563 | 20.118 | 69.696 | 0.773 | 1256 | 39.134 |

The larger ON client−Runtime residual in Run667 is nearly all in the start→last-first-token side of the cohort. The remaining post-first term is 0.48–0.77 s per four cohorts. ON repeat3 has 1256 vs OFF 1166 cycles, lowering useful tokens/cycle from 42.15 to 39.13 and adding ~3.1 s of Runtime wall independently of the ~10 s pre-first residual difference. This decomposition locates the wall region but cannot separate prefill, seed, scheduler handoff and per-cohort Graph capture, or establish causality. Run666 OFF/ON/OFF had ON pre-first shorter than both controls, so a direct Graph-causes-residual claim is unsupported. Run668 collects only missing absolute stage boundaries.
