# Run580: copied Level0 task timeline of mixed GMM × HCCL service

## Admission

The frozen warm48+measured48/c12/1024 service completed with 96 POSTs, 64 all-rank Runtime records, fresh dual-Graph output checks, eight idle NPUs and exact source restoration. The terminal diagnostic saved all8 **unprofiled** four-arm checkpoint files before profiling. Its slowest rank median joint times were GMM 10.312 ms, HCCL 4.282 ms, serial 14.573 ms and concurrent 16.242 ms. These agree in direction with Run579; neither terminal measurement is formal E2E TPS.

Level0 CPU+NPU profiling recorded one warmup serial step, then one active serial and one active concurrent step per rank. Automatic online analysis was disabled. The post-stop final admission pins 534 raw profiler files. All eight raw trees were copied and parsed only in `offline_parse/input`; original raw SHA checks still pass. The copied trace and task CSV are pinned in `timeline_attribution.json`.

## Task attribution and timing

Each rank's active trace has two full Model33 GMM Graph replays (86 alternating GMM1/GMM2 native tasks each) and two full Model32 HCCL Graph replays (265 ordered AIV collective tasks each). CSV and trace agree on task ID, stream and timestamp. Two separate all-reduce barriers and their setup tasks are excluded. The HCCL call sequence matches the SHA-pinned Run378 RS87/AG135/A2A43 ledger. Graph submit Host markers fall within their active step.

| Rank-local range | Serial | Concurrent |
|---|---:|---:|
| GMM native task envelope | 9.72–10.25 ms | 12.53–12.99 ms |
| HCCL native task envelope | 4.07–4.44 ms | 15.10–15.35 ms |
| HCCL first task after GMM first | 9.73–10.26 ms | 0.37–0.42 ms |
| GMM/HCCL task-envelope overlap | 0 | 12.12–12.61 ms |

For the same first 84 or 86 HCCL ordinals whose concurrent tasks start before the GMM envelope ends, serial task-time medians are 13.34–14.36 μs, whereas concurrent medians are 144.73–146.40 μs. After the GMM envelope ends, the remaining matched HCCL ordinals are near 13–14 μs in both arms. Exactly one HCCL task per rank crosses the GMM end boundary; prefix task-time sums must not be interpreted as pure overlap-time integrals. The whole HCCL Graph is **not** delayed until after GMM; the observed slowdown arises while their task envelopes coexist.

These are Level0 task scheduling intervals. They do not measure active AIV cycles, physical HBM or network bytes, or rank wait attribution. Without PMU and legal production edge data, the result cannot apportion slowdown among resource competition, Graph/internal stream scheduling and HCCL synchronization. Cross-rank timestamp subtraction is not used. Only one profiled replay per arm supports the mechanism; the unprofiled repeated arms support the service-time comparison.

## Bound effect and next measurement

Run580 strengthens Run579's rejection of simple whole-chain ideal overlap and rules out late **initial** HCCL Graph launch as the principal explanation for this fixture. It does not rule out shorter or differently staged legal overlap, establish production Target/DSpark/KV dependencies, or supply a finite Resource/Hardware, Scheduling/Execution or Product E2E endpoint. Formal Current remains Run99 median **571.681 tok/s**; the numeric distance to a credible whole-system limit remains unknown.

The next Bound priority is the actual frozen-W₀ production dependency DAG: producer-ready, collective submit/completion, consumer-ready, Target, DSpark, KV and Host edges. In parallel, compulsory W-minus/traffic and attainable cumulative C⁺/B still need matched-shape hardware evidence. Any new schedule should be tested only after the DAG proves which boundaries can legally move.
