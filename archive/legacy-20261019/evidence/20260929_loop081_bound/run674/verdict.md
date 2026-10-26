# Run674 — KEEP initial prefix-hash memo

Formal Current: **676.8246305810562 output tok/s** on frozen48×32K→1024,c12, same model/hardware/DSpark7. Input-token cache ON, renderer workers4, initial prefix-hash memo ON. Metadata Graph OFF.

Single resident service OFF_A / ON / OFF_B, each warm48 +3formal48. Medians578.152305 /676.824631 /628.846654 tok/s. ON beats both brackets (+17.067%/+7.630%); OFF drift forbids assigning the full first contrast to caching. Every ON repeat beats every OFF_B repeat. ON residual8.184/6.473/5.431s vs OFF_B11.068/11.103/13.121s. ON cycles1200/1181/1171 vsOFF_B1197/1197/1206: trajectory gains are separate, not Framework wall.

All576 requests complete1024, frozen input-token counts;384 Runtime records across48cohorts×8ranks, FULL graph, host mirror exact, no oracle calls, all8 cycles/acceptance/staged/overshoot parity. Prefix memo CPU semantic gate checks exact hashes, continuation/fallback/typed key/seed change/mutation/LRU; actual block size2 confirmed. ON144 measured hits and no new misses. OFF_B retains resident memo but bypasses lookup/insert, counters unchanged. All9 cleanup statuses0; current source and helper SHA match original; independent npu-smi shows8idle and no processes.

Hash scoped measured times ON .598/.362/.289s vsOFF_B4.552/4.585/6.718s support causality, but these times are not additive Product savings. This is repeated fixed-workload cache specialization, not a cold/unseen-prompt claim. Numerical Framework ceiling and maximum removable gap remain unidentified.

Next: reuse existing metadata Graph arithmetic in one full-stack OFF/ON/OFF test with both caches ON. New question is whether proven cycle saving now survives Product boundary; no standalone Graph microtuning. If flat/worse, move to publication/successor-arrival structural gate.
