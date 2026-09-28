# Run651 — within-W₀ all8 Host-submit skew regeneration

Read-only replay of Run638's admitted ON-arm 32 event packets (all8 × four measured cohorts), SHA-pinned to `arm_scoped_recovery.json`. For each cohort, cycles63–65 provide 12 all8 Host-submit comparisons at the same named stage. No new service, Event, profiler or source edit. All records share the same reported Linux time namespace; millisecond cross-rank Host offset is conditionally interpretable, but no independent error bound or native producer completion is certified.

| Host-submit stage | Median all8 spread | Range | Latest rank |
|---|---:|---:|---|
| Target before call | 4.940 ms | 2.392–7.808 ms | ranks1/2/4/6 across12 samples; rank7 never latest |
| Target forward return | 1.602 ms | 0.488–3.742 ms | ranks1/2/4/5/6 |
| Proposer before call | 1.835 ms | 0.765–3.994 ms | ranks1/2/4/5/6 |
| Proposer after call | 4.704 ms | 2.095–7.369 ms | ranks1/2/4/6; rank7 never latest |
| Draft commit after | 4.719 ms | 2.094–7.382 ms | ranks1/2/4/6 |

The **paired** Target-before→forward-return spread change is negative in11/12, median−3.397ms. The paired proposer-before→proposer-after change is positive in12/12, median+2.772ms. In eight adjacent cycle transitions, the rank latest at proposer return is also latest at next Target before call in7/8; next-Target spread exceeds prior proposer-return spread in8/8, median0.492ms. Thus, in this ON W₀'s sampled window, all8 Host issue converges over Target and diverges again across the proposer/Host region, with much of that skew persisting to the next Target. Run611's rank7 latest entry is one W₀/observer instance, **not** a stable rank assignment; a follow-up should sample the actual latest rank dynamically on all8.

These are Host marker-submit spreads, not HCCL active service, device completion, intrinsic DSpark kernel latency or available scheduling saving. A Target collective may cause convergence, but this packet alone does not isolate why. Proposer-return skew can include necessary state/DSpark work, resource contention, Host issue and waits. The Run638 cross-arm fixed-W₀ check failed, so no OFF→ON/Run606/Run99 timing transfer or Product TPS follows. The next fixed-primitive packet should bind actual value versions/producer streams and separate legal predecessor time from issue overhead, with representative shape/branch and observer controls. Framework-only whole-Product interval remains unidentified; Formal Current571.681tok/s.

Astra High independently rehashed all32 packets and recomputed96 rank-cycle marker groups and every paired statistic. Its verdict is **SCOPED PASS** for Host-submit dispersion only; packet Host time namespace alone is not a full boot/clock identity proof. Review: `astra_review.md`.
