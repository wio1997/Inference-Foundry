# Loop061: graph HCCL payload versus attainable link capacity

Frozen product: DeepSeek V4 Flash W4A8, 8x910B3, DP1xTP8, DSpark7,
48x32K→1024 c12, warm-cache. Formal Extreme is still 571.681 tok/s
median. Stock 543.655 tok/s is only a baseline.

## Environment and measurement applicability

Host npu-smi/driver 26.0.rc1, 8x910B3 with all 28 pairwise topologies
reported HCCS. Serving container CANN 9.1.0, torch-npu 2.10.0.post4.
The CANN 9.1.0 bundled HCCL Test source was built inside the dedicated
container; its Makefile required a local temporary-copy addition of
-lmpi_cxx for Ubuntu OpenMPI 4.1.2. The installed toolkit was not edited.
See environment.txt and Run251. The user-linked CANN 8.3/8.5/9.0 pages
inform measurement method; the actual 9.1.0 source and README determine
flags here.

## Run250: exact target graph trace payload

Run246's 80 exported target graph rank-cycle windows all have 265 HCCL
AivKernel events and the same name/size signature:

| Event | Count/cycle | Reported size per event |
| --- | ---: | ---: |
| allgatherAivKernel | 43 | 12,288 B |
| allgatherAivKernel | 90 | 98,304 B |
| allgatherAivKernel | 1 | 393,216 B |
| allgatherAivKernel | 1 | 3,102,720 B |
| reduce_scatterAivKernel | 87 | 98,304 B |
| alltoallAivKernel | 43 | 98,304 B |

The sum of the *reported operation data sizes* is 25,651,200 B per
rank-cycle, not network traffic. The 98,304/12,288 values agree
numerically with the Run152 BF16[49,152]/FP32[3,072] adjacent-gather
sizes. This cross-check does not identify rank-unique send volume or
full-output volume. Every exported link/transport type is INVALID_TYPE;
communication.json transit-size fields are zero. We cannot derive
physical HCCS bytes, per-collective link bandwidth or necessary
collective latency from this trace. The 265 event count is a current
implementation count, not a proof that all 265 are mathematically
compulsory under a redesigned TP schedule.

## Runs251–254: official HCCL Test reference capacity

Run252 attempted -i 0. This version repeats a fixed size indefinitely;
the exact mpirun process was interrupted. It has an invalid terminal
benchmark result and no value is used in the bound. All 8 NPUs returned
idle.

Run253 used positive integer -i 1024 and finite 50-iteration tests,
except its initial 96 KiB parser preflight (-i 1K) failed and was
corrected. Every accepted case exited 0, reported the requested
data_size, and passed check_result:

| Operation | data_size | normal aveg_time |
| --- | ---: | ---: |
| AllGather FP32 | 12,288 B | 142.32 us |
| AllGather BF16 | 98,304 B | 159.32 us |
| AllGather BF16 | 393,216 B | 151.04 us |
| AllGather BF16 | 3,102,720 B | 161.12 us |
| ReduceScatter BF16 | 98,304 B | 114.36 us |
| AllToAll BF16 | 98,304 B | 211.07 us |

Run254 paired -t0 and -t1 on the same 98,304 B BF16 payload, with
HCCL_BUFFSIZE=256 MB, 50 iterations, 20 warmups and all-rank checks:

| Operation | -t0 aveg_time | -t1 device-only aveg_time |
| --- | ---: | ---: |
| AllGather | 167.40 us | 40.03 us |
| ReduceScatter | 102.23 us | 39.74 us |
| AllToAll | 205.65 us | 62.17 us |

These are tool-specific arithmetic averages, not confidence intervals.
The official HCCL Test describes data_size as participating data per
NPU and alg_bandwidth as data_size / time. Neither is a physical wire
counter. The -t1 value excludes tool Host dispatch and launch under its
documented conditions. It does not reproduce Extreme's captured Graph,
interleaved compute, AIV expansion, rank arrival or source/destination
dependencies. In particular, neither -t0 nor -t1 may replace Run152
Graph task durations or be multiplied by 265 to infer exposed E2E time.
Run254 exit 0, check success, 8/8 NPUs idle.

## Independent Astra High Bound Review (Run255)

Read-only Astra High review of V0 and Runs247–249 found no defensible
hardware-attainable TPS interval. V0 exactly replays the formal samples,
but Engineering 581–607 and Aggressive 616–682 tok/s use assumed
exposed millisecond savings. They remain sensitivity scenarios, without
attainability confidence. It would be misleading to call their ratio
to current throughput a percentage of hardware ceiling. The
9.244GB/8.462GB GMM ratio mixes different route/cycle samples and its
numerator includes more than packed weights; it only weakens a claim of
large gross weight reread. Attainable isolated bandwidth is not a
strict physical capacity upper bound, and max(resource times) is only
an optimistic relaxation until mixed-resource kernels and stream
dependencies are modeled.

The reviewer recomputed three Compressor classes' unique two-matrix
BF16 weight footprints as 0.3523/0.0881/0.1678 GB per target cycle,
0.6082 GB in total, versus 1.5309 GB *reported read counters*.
The difference includes activation, state, metadata, workspace and
potential repeat reads; it is not established waste. Nominal two-matrix
multiply work is about 58.385 GFLOP/cycle. The highest-value next
mechanism investigation is a source-level read census and same-state
Graph-path intervention, starting with the c4 small-product shape only
if source/dataflow evidence shows a removable cost. Full numerical,
state/KV and serving correctness followed by repeated formal E2E
remain mandatory for KEEP.

## Community reference

The newly opened vLLM-Ascend Q3 MRV2 PD profile RFC targets W8A8,
32 logical NPUs, heterogeneous PD parallelism and DSpark5; its target
Decode Graph remains FULL_DECODE_ONLY and draft remains eager. It is a
source of design patterns and regression gates, not comparable
performance data for this W4A8 TP8 DSpark7 mixed-service contract:
https://github.com/vllm-project/vllm-ascend/issues/17453

Official references:
- https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910/devaids/hccltool/HCCLpertest_16_0005.html
- https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/devaids/hccltool/HCCLpertest_16_0006.html
- https://www.hiascend.com/document/detail/zh/canncommercial/601/devtools/auxiliarydevtool/atlasprofiling_16_0121.html

## Decision

PIVOT. The trace payload and standalone capacity are measured, but
wire bytes, graph-path attainable communication service, necessary TP
message sizes and critical-path exposure remain unproven. Do not call
the existing scenario a hardware bound. Next investigate exact-shape
non-GMM compulsory work versus actual reads and choose a
correctness-preserving Graph-path intervention; expand to execution
architecture if no meaningful local exposed saving is found.
