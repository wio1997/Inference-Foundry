# Run279 — corrected cached RoPE correctness

COMPLETED 2026-10-07T11:54:10.266864Z; spec `85e349950d66d99290e2f6c59a25f02e31dbfccb8b80e139ead5b7d01d96cd83`, 74 source pins. Four fixed actual standard-PD requests, no recovery. Parent Run277 remains FAILED; Run278 remains parked/not started.

[Independent raw reduction](correctness_reduced.json) PASS: old short2334/8, dense short2334/8, dense natural58/23/EOS, retained old short2334/8. Exact token IDs/usage/finish/SSE and all16 native transfer checked. All16 cached-table stride128→64, BF16/ND format2, stable paired-table lifetime and unchanged persistent graph output pointers established. Actual target and ordinary MTP both use cached tables; no noncached-role quota.

H6/H5 and target FULL retained. H11 off, terminal H12 mode0/transition3. Terminal all16 healthy/idle, no recovery. This establishes scoped device correctness, not timing gain, long/concurrent/fullAPI or formal80K/600/93% SLA. Matched performance is separately decided by Run280.

Raw fetched once; `FETCHED_TERMINAL_INPUTS.json` records exact input hashes. `reduce_correctness_raw.py` is the independent reducer used once. Original controller/raw/source facts are preserved.
