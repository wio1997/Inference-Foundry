# Run284 matched immediate control

The SHA-guarded borrowed DSA CP patch was restored after the run. All 60 client requests returned exactly 1024 output tokens. All five 12-request cohorts produced 40/40 passing eight-rank FULL Graph reports. The profiled 12-request client rate (519.374 tok/s) is diagnostic only.

The same profiler configuration as Run282 captured two cycles on each of eight ranks. The delayed-join overlap path reached the selected layer's next AllToAll completion sooner than the immediate-join path in 16/16 rank-cycle comparisons. Paired median delta was -39.875 us (range -64.0 to -31.25 us). The difference of group medians is -40.875 us, a separate statistic. Selected-layer eight-rank envelope differences were -49.25 and -39.0 us. These are cross-run, non-identical states; ranks are coupled by collectives. Run282 concurrent Rotary inflated about 59 us versus immediate, but its local join endpoint remained earlier.

Immediate already reorders the original A0 QLI→main Compressor path to main Compressor→join→QLI, so this comparison isolates delayed versus immediate join, not candidate versus accepted A0. Full-cycle runtime was 60.903 versus 60.988 ms/cycle in distinct acceptance trajectories; no E2E improvement is established. The next gate is one-layer same-state A0/overlap/A0 numerical and post-cache comparison, then original A0 versus all-layer FULL Graph and formal E2E if correctness passes. No KEEP decision.

Sources: `matched_profile.json`, `astra_matched_scope_review.md`, Run282 profiler exports, launcher/runtime/restore logs.
