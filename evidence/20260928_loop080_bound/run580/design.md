# Run580 — explain Run579 mixed Graph slowdown with a short task timeline

## Bound question

Run579's correctly joined, independent-ready 43-layer production-weight GMM Graph and current-order full265 TP8 HCCL Graph took 16.205 ms concurrent versus 14.402 ms serial at the slowest rank median. The result falsifies naive whole-chain ideal overlap for that protocol, but rank-local branch events cannot separate delayed HCCL launch, interleaved GMM service inflation, internal HCCL stream scheduling or collective rank-arrival wait. This is a Scheduling/Resource **mechanism** experiment, not a Product overlap experiment. DSpark7 acceptance, cycle count and output semantics remain frozen.

## Protocol

Repeat the exact Run579 warm48+measured48/c12/1024 service and terminal mixed fixture with a new run tag. Before profiler starts, persist all8 balanced 20-sample four-arm unprofiled device-event results in rank-specific checkpoint files. The profiler may change timing; these pre-profile samples are the same-run service reference and are not formal E2E.

After the checkpoint, use CANN9.1 Level0 CPU+NPU profiling, one warmup serial step and two active steps: serial then concurrent. Disable automatic analysis (`analyse_flag=False`) so Run578's online parser failure cannot erase the unprofiled checkpoint. Mark the two Graph replay Host submission scopes separately and retain full rank-local raw trace with source, script and profile configuration provenance. Stage all8 rank synchronization before each profiled step; the barrier may appear in the profile and must be excluded when classifying 265-chain tasks. Profiled event time is mechanism context only.

Post-stop final admission requires original source restoration, all8 idle, 96 POSTs, 64 Runtime records, fresh dual-Graph correctness, event order and immutable raw inventory hashes. **Copy each rank's raw profile tree** to `run580/offline_parse/input` and run `torch_npu.profiler.profiler.analyse(..., export_type='text')` only on the copy. The original raw tree must remain byte-for-byte unchanged. Parsed task attribution must verify separate complete GMM and HCCL replays and distinguish barrier tasks using Graph identity, task sequence, stream and Host markers; active window duration alone is insufficient.

## Allowed inference

A complete, attributable task timeline can indicate whether the current whole-chain HCCL Graph begins late, whether GMM task spans expand under concurrence and whether collective tasks expose waits. Level0 task spans are not physical HBM/AIV occupancy; cross-rank clocks are not aligned. Even an explained Run579 slowdown does not rule out shorter dependency-preserving segments or establish production DAG legality. No strict Resource/Hardware, Scheduling/Execution or Product E2E endpoint is promoted by this diagnostic alone.
