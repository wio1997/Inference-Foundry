# Reset168 — one profiling-OFF CPU-clock attribution

H4 formal promotion INCONCLUSIVE/PARKED, stock mode0/source restored. No second patch or parameter candidate. Existing Run249 establishes within-model host lateness under profilerON; Run251 short patch signal does not establish the residual largest profOFF cause.

Decisive missing fact: per-rank main-thread **actual CPU execution time** versus D request wall when torch/NPU profiler is OFF. The ON wall scopes cannot distinguish profiler cost or host scheduling/blocking from actual eager producer work. Host perf binary is absent. Linux schedstat runtime counter is available; sched_schedstats=0, so runnable-wait time must not be used.640 CPUs and unchanged resident D16 are available.

One minimal diagnostic Run252: existing unique controller, P249/D251 untouched, original disk source/selector0 verified, one same2334prompt/8token native P→KV→D fixture. Before D POST and after D response, read owned workers' main schedstat CPU runtime and process CPU ticks; poll main counters every20ms, snapshot all threads before/after for attribution. No Torch/NPU profile, model reload, layout/config change, scan or source hook. Counters measure CPU use, not a formal benchmark gain. Observer's own CPU cost is recorded. No OS sysctl/perf installation.

Decision table: main execution close to D window on slow ranks -> residual eager host work remains a profiling-OFF critical-path candidate; low main CPU with large wall -> do not extrapolate ON producer gaps, inspect background runtime/device dependencies instead; uneven main/background CPU -> correlate actual producing threads before selecting another patch. No amount of CPU stat alone proves a Python function or removable budget.20ms polling and100Hz aggregate ticks set resolution limits. A full-thread CPU snapshot can include heartbeat work outside the exact D interval; report its wider boundaries.

No performance KEEP or Current change. No further NPU request after this one unless its evidence changes the code decision and a new bounded diagnostic is justified.
