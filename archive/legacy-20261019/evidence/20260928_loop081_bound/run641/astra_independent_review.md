# Astra High independent Run641 review

Verdict: KEEP as a fixed-observed-cost, fixed-stream-affinity **conditional mathematical relaxation only**. It is not a legal whole-Graph schedule or Product Framework bound.

Astra independently checked raw SHA, 24×5,412 Model45 task identities, triple static key occurrence, within-stream physical nonoverlap and all24 results. Each Graph has260 HCOM-named tasks (RS87, AG130, A2A43), all stream0. HCOM intervals have zero intersection with stream1 physical work and full coverage by stream1 waits. Current serialization coverage is established; removable waits, peer-wait-free HCCL service and legal overlap are not.

The restricted lower makespan is `max(stream physical sum)=36.905–38.083ms`; Current45.102–72.209ms. Difference median15.819ms is conditional maximum relaxation, not obtainable saving. Correlation r≈0.999688 between difference and HCOM duration is substantially algebraic: difference−HCOM equals stream99 physical sum plus span−all physical sum, around2.398ms median. Do not call it independent causal support. Run579/580 GMM/HCCL contention also precludes cost-free whole-chain overlap.

Astra requested replay SHA, rank/ordinal bijection and static-key guards. Sol added them after review; output JSON SHA remained `3473482f2216794ea4dabd851f46cb1b1025f8ec67fd01f585f05d86d6bf0a3d`, current script SHA `f67a6e2758d37ca1a408581a98e178c69ab376f7711153d15c89bfd408304754`. Largest next uncertainty: real producer→collective→first-consumer edges, HCCL active service versus peer wait, and mixed-resource cost at the only legal independent cut.
