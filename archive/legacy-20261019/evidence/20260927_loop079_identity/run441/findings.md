# Run441 — smaller-payload HCCL Test control, corrected interpretation

The server was stopped and all eight 910B3 devices had about 3.44 GiB baseline HBM before the test. The installed CANN 9.1 `tools/hccl_test` binary (`all_gather_test` SHA256 `b276a969e778ab2d627643225fc9a29b45878b016999079de9ae705a73997b9d`) supports `bfp16` in its actual help output, despite the installed README's shorter dtype list. Open MPI is 4.1.2. No framework source or service was changed.

Three independent eight-rank HCCL Test processes ran CLI `data_size=3,102,720` with 10 warmup and 30 measured iterations, result checking enabled and `onlydevicetime=0`. All reported `success`:

| Process | Reported average time (µs) | Reported algorithm bandwidth (GB/s) |
|---|---:|---:|
| 1 | 221.77 | 13.99096 |
| 2 | 179.45 | 17.29036 |
| 3 | 205.78 | 15.07814 |

**Correction from independent Run443 review:** the installed HCCL Test AllGather interprets `data_size` as aggregate receive bytes and divides the element count by eight ranks when allocating each input. Run441 therefore exercised **387,840 input bytes/rank**, not the Target logits `[96,16160]` input of 3,102,720 bytes/rank. Median is 205.78 µs; observed range is 179.45–221.77 µs. Reported algorithm bandwidth uses the tool's aggregate-output-byte convention, not physical per-link traffic. The test remains a correct isolated smaller-message observation. Earlier labeling as the exact terminal logits payload is superseded by Run444; do not use these timings for that message, Current exposure, compulsory communication or a Product ceiling.
