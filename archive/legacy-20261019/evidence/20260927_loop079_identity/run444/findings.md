# Run444 — corrected exact terminal logits payload HCCL Test

Run391's current Target terminal logits AllGather has BF16 input `[96,16160]`, or **3,102,720 bytes/rank**. Installed CANN9.1 HCCL Test interprets `data_size` for AllGather as aggregate receive size and divides its element count by the eight participating ranks. The matching CLI `-b/-e` size is therefore **24,821,760 bytes**, not the Run441 value. This interpretation is source-backed by `HcclOpBaseAllgatherTest::init_malloc_Ksize_by_data` in `/usr/local/Ascend/cann-9.1.0/tools/hccl_test/opbase_test/hccl_allgather_rootinfo_test.cc` and independently caught in Run443.

With the service stopped and all eight devices idle, three independent MPI processes ran the installed `all_gather_test` with eight ranks, bfp16, 10 warmup and 30 measured iterations, correctness checking enabled and `onlydevicetime=0`. All three reported success:

| Process | Tool average time (µs) | Tool algorithm bandwidth (GB/s) |
|---|---:|---:|
| 1 | 235.63 | 105.34091 |
| 2 | 261.26 | 95.00667 |
| 3 | 238.76 | 103.95910 |

Median is **238.76 µs**, observed range **235.63–261.26 µs**. Installed test source records ACL events on each rank's test stream around all 30 collective enqueues and prints the root rank's elapsed interval divided by 30. That interval can include submission gaps; it is neither a rank-mean nor a pure kernel-duration sum. Correctness checks one additional collective after the timed loop. The reported algorithm bandwidth is aggregate output bytes divided by that tool interval, not physical link bytes/s. Its isolated communicator, no model contention and lack of output layout/acceptance consumer prevent direct transfer to Current exposure. This is an attained isolated same-payload service observation only. Strict Resource, Scheduling and Product finite endpoints remain null; the Run439 native join and original-path timing gate are still needed.
