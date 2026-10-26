# Run339 — same-topology HCCL Test service calibration

Installed CANN Toolkit 9.1.0 and driver 26.0.rc1 on eight 910B3 were checked before using the bundled HCCL Test. The test ran under serving HCCL settings. Its bundled source was built unchanged; the first link exited 2 because the Makefile omitted Open MPI C++ linkage, and a second link with explicit `-lmpi_cxx` exited 0. The infinite `-i 0` / wrong-size AllGather pilot was excluded. Five finite cases each ran in three independent eight-rank processes, five warmups plus 30 measured device-only iterations per process, result check success and exit 0. This is a resource-service calibration, not a serving benchmark.

| Collective/payload | API input B/rank | count in padded88 forward | independent min/median/max device µs |
|---|---:|---:|---:|
| ag_hidden_bf16 | 90,112 | 90 | 10.97/11.03/11.07 |
| ag_router_fp32 | 11,264 | 43 | 4.87/4.95/5.00 |
| ag_extra_bf16 | 360,448 | 1 | 31.81/32.10/32.20 |
| rs_bf16 | 720,896 | 87 | 11.77/11.81/11.92 |
| a2a_bf16 | 720,896 | 43 | 12.43/12.43/12.51 |

The hidden/DSA AllGather count 90 includes 43 source/Run333 inferred calls outside the Run337 Host wrapper. Run337 directly captured 221 of the 264 expected calls per rank. For AllGather, HCCL Test `-b` denotes total output bytes, while ReduceScatter uses total input and AllToAll input/output; `design.md` records the installed source mapping. The tool-reported algorithm bandwidth is not physical link traffic.

These medians demonstrate an attainable isolated service point for the actual sizes and dtype on this topology. They do not supply a strict hardware latency floor or a product-like 264-collective chain capacity: algorithm selection, buffers, producer arrival, Graph launch, compute/HBM contention and overlap may differ. Three process averages are not a tail-latency distribution. Multiplying medians by counts or subtracting them from Run333/337 waits would misattribute dependencies. Run337 confirms a no-profiler Host arrival skew, but has no aligned device completion or seed-ready endpoint. Resource/Hardware, Scheduling-aware and Product numeric ceilings remain unknown; Current Formal remains 571.681 tok/s.

Astra High revised its priority after checking current Run225/227: the executed `_moe_forward_shared` precompile boundary and four-layer first88 Graph were already measured, while the independent all43 Graph bank failed memory allocation at layer17. Repeating a single unchanged-work layer is lower information value than measuring natural b=2–4 refill preparation. Run338 only missed the wrong postcompile hook. A separate exact mixed-collective chain could later calibrate resource contention if the original path reveals it as the dominant uncertainty.

Evidence: `design.md`, `analysis.json`, each case log in base/repeat2/repeat3, `build.log`, `build_retry.log`, and `scripts/loop074_run339_analyze.py`.
