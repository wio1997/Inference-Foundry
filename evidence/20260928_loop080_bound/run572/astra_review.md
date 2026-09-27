# Run572 independent Astra High review

Initial version **FAIL** despite byte-exact reproduction: uncalibrated cross-rank clock coordinates were labeled all8 overlap; `COMMUNICATION` lumped 265 physical HCCL `AivKernel` rows with 265 `hcom_*` pseudo envelopes per window; invalid/NaN durations and boundary carry-in were not gated. Astra required rank-local scope, split event classes, integer/Decimal time arithmetic and boundary/negative checks.

Final rank-local revision **PASS**. Astra independently rebuilt the output byte for byte, rejected NaN/Infinity, unsupported precision and unknown-core negatives, and verified all16 windows have the 265+265 split with zero carry-in/cross-end. Cross-rank physical overlap and all finite strict Bound endpoints remain null. The zero pseudo-envelope-only interval means only that exported pseudo intervals are covered by exported AI task intervals; it does not establish hidden communication or removable time.

A final additional positive gate requires `aiv_time(us)>0` and AIV main-memory read counter `>0` for each HCCL `AivKernel`; it passed and did not change the output SHA256 `503adcc54c3a5ce8969c06d55e30b82ca5ed4187214a4f83e6b00a22c2953b26`. Final script SHA256 `163d6660fdba19a518c867576770c1a0b75b43d0c0d155439d99dee21095a6c3`.
