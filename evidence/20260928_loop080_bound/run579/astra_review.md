# Astra High independent Run579 Bound review

Scoped PASS. The final admission JSON SHA is `64233da44fcb4051c6cc067ed21c7e37bd0d2389155e4c5e9d46ba7f9911e45e`; reducer and post-stop final replay byte-for-byte. Twenty-seven service, 64 Runtime and nine post-stop provenance items match. Eight ranks pass fresh GMM and all265 HCCL output validation, plus event-order checks and negative admission tests.

The slowest rank median joint times are GMM 10.266 ms, HCCL 4.180 ms, serial 14.402 ms and concurrent 16.205 ms. Concurrent is 12.52% slower than serial in this protocol; every rank's minimum concurrent sample exceeds its maximum serial sample. Both balanced timing blocks agree. Concurrent GMM branch completion is 12.836–13.250 ms and HCCL branch completion is 16.075–16.204 ms.

The result rejects simple whole-chain ideal overlap for the independently ready Run579 fixture. Branch completion includes wait and scheduling, so the exact cause is not yet isolated among late HCCL launch, Graph/internal stream interaction, resource contention and rank arrival. The fresh correctness pass takes 78–84 ms because it includes validation kernels; it is not service latency. No production dependency legality, strict C⁺/B, finite Resource/Scheduling/Product endpoint or formal TPS is established.

V3.35 generator SHA `81be76a4f71b84553fa43739c0ce1e3c156b7c404719b52cf373c15557c410d7` and output SHA `410763b5a05e5cc06af9d2d303e60481a29ab67f64a36ba2c4a04c4e773bea63` passed independent byte replay, seven input-SHA mutation negatives and four finite-endpoint injection negatives. The next high-value measurement is a short-window all8 serial/concurrent task timeline, interpreted against Run579's unprofiled timing. Then capture the legal production producer/consumer DAG.
