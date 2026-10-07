# Run283 — H13 scoped device correctness PASS

Fixed four standard PD requests completed: A short 2334/8, B short 2334/8, B natural 58/23 EOS, retained A short 2334/8. Independent reduction of original SSE, exact IDs, native transfer and raw per-rank witnesses passes. All16 actual target A/B capture and replay retain equal capture input addresses, distinct graph pools and owned outputs, exact bytes/strides/ND format, masks and input nonmutation. No recovery was used.

H6/H5 and target FULL remain on, H11 off; terminal H13 mode0/observe0 transition3, all16 P/D healthy/idle. P249 unchanged; D root3189247/start312624732 belongs to this new epoch. Later Run284 uses these same workers; use its terminal guard/control snapshots for latest mode state.

This is scoped device and PD correctness, not full API, long/concurrent workload, SLA, production two-pool memory acceptance or performance gain. Original Run281/282 failures are retained and not relabelled by this PASS. See correctness_reduced.json and independent_correctness_reduction.py. The production local prepare candidate is unchanged; the dual-bank selector is a research comparison mechanism. Current remains None.
