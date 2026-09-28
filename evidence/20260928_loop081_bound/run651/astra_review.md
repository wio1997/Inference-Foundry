# Astra High independent Run651 review

**SCOPED PASS** for within-Run638-ON all8 Host-submit skew only. The reviewer independently checked all 32 packet SHA, 96 rank-cycle groups, unique per-label event generation and 12 all8 stage groups. Recomputed medians: Target before4.940ms, forward return1.602ms; paired skew change−3.397ms and reduction11/12. Proposer before1.835ms, after4.704ms; paired increase+2.772ms in12/12. Eight proposer-after→successor-Target-before pairs all increase (median+0.492ms), with the same latest rank in7/8. Latest proposer-after rank counts: rank1 seven, rank2 two, rank6 two, rank4 one; no ties.

`host_submit_ns` precedes Event.record and denotes Host arrival at the marker site. It does not certify device issue/completion, producer readiness or collective completion. Skew is dispersion, not stage duration or removable wall. Carry is temporal association, not causal proof. Same time namespace supports conditional Host comparison but does not replace complete boot/clock admission. The evidence is only Run638 ON early cycles63–65; cross-arm fixed-W₀ failed, so no observer-neutral or Run606/611/99 timing transfer. The next packet should cover all8 DSpark→commit/metadata→next Target and select the latest rank dynamically.

Independent reported SHA prefixes: script `d2af5602`, JSON `62931b8c`, recovery `d608f8ad`. Read-only; no service/NPU work.
