# Run670 — renderer workers1/4 formal E2E

Frozen warm48 +3×48, c12/1024, identical dataset/model/sampling/DP1TP8/DSpark7 and original Runtime source. Metadata Graph disabled in both arms. Actual serving process argv verified1/4 workers.192 HTTP posts per arm, every output1024,128 unique all8×16 FULL Runtime rows per arm pass. All nine cleanup/restore/source/script exits0; service stopped. No model source changes.

| Arm | Repeat TPS | Median TPS | cycles | client minus Runtime s |
|---|---|---:|---|---|
| workers1 |577.397 /549.169 /551.503|551.503|1169 /1207 /1177|12.483 /13.164 /14.848|
| workers4 |564.335 /567.011 /548.038|564.335|1229 /1189 /1210|12.017 /11.492 /13.579|

Contemporary median change+2.327%. Residual decreases in all three ordinal comparisons by.467/1.672/1.270s, consistent with partial preparation improvement, substantially less than the isolated CPU queue saving. This does not by itself locate the saved interval inside API/Core/prefill. Cycle differences+60/-18/+33 are trajectory changes and cannot be counted as Framework overhead. Runtime normalized ms/cycle is62.142/63.246/63.106 for worker1 and61.091/63.241/62.900 for worker4; do not transplant Run667's~56ms cadence or attribute the entire Runtime change to renderer concurrency.

Disposition: no Current promotion. The positive contemporary median is small relative to repeat/trajectory variation and candidate median remains below achieved Run99 571.681. Keep worker4 as an experimental preparation option for the next matched candidate; do not silently change default serving. Numerical Bound and maximal removable gap remain unknown. The next stronger same-semantics intervention is bounded memoization of repeated input tokenization, independently CPU-gated by Run671 and requiring loaded integration plus formal E2E. No model outputs are cached.
