# Run575 independent Astra High review

**PASS for the offline selected Graph task-name → packaged object → installed CANN9.1 tiling-key association.**

Astra independently replayed final script SHA `4ca6bb704c56ca3f69bcb6a174d2d2f3c868c84a840bb19e9d6d4945c0bd6251` and obtained byte-identical output SHA `f77dba1188b0bb66456880871b85c2114b1402b453b41c621b5ad7bc9273a437`. It recomputed 344 tasks = 8 ranks × 43, all selected object prefix `7d53…`, with per-rank key counts 2×2, 514×20, 1026×21. It rejected wrong key/object/rank, duplicate task ID, Graph SHA, BF16/FP16 object, KV layout, q/output dtype and Run574 pin mutations. The final hardening pins the installed CANN header byte hash; a one-newline header change is rejected.

The installed CANN encoder appends BOOL1 and UINT4 fields in declaration order and maps UINT values to their declared-list indices. The decoded selected keys are `FLASH_DECODE=0`, Query `TND`, KV `PA_ND`, and `SWA`/`CFA`/`SCFA`. This association does **not** prove actual loaded device object bytes, source→object compiler provenance, dynamic query prefix/head metadata, exact layer ordinal or first-post-parking all-layer semantic row identity. It does not move a finite Bound endpoint.
