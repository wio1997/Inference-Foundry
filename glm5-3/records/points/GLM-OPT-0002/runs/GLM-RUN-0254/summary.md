# Run254 — minimal profiler-OFF user CPU instruction diagnostic

Actual controller/phase completed with exit0; final_status.json reports completed. Historical manifest.json PLANNED is preserved unchanged. This file is a derived execution summary, not a replacement raw record.

One stock-semantic P249/D253 standard PD request, 2334 prompt/8 output tokens, same IDs/chunks as Run249, all 2334 external KV hits, finish=length. TPOT 256.793022ms, D wall2.22954614s, PD wall3.155732745s; these are diagnostic timings, not complete E2E Product Gain or formal SLA acceptance. No NPU profiler, parameter change, source activation, kernel change or service reload.

16 D main-thread CPU-clock callchains: 3223 samples, zero lost, enabled window2.241002328s, reader CPU38.419030ms. Fixed10,000,000ns user CPU period. Original17,277,931-byte raw remains at cpu_samples_raw.json on167, SHA25663be953de17ebefce743738bae3b6d7780a84373d52a147d936f480c8e70d803. Before/after guards match protected owners/source, but their schedstat time window includes reader setup and P request and must not be divided by the enabled sampling interval.

[Derived attribution](../../research/decode_path_20261006/CPU_INSTRUCTION_DECISION.md) explains the exact MC2 capability-query path and the limits of inclusive wrapper samples. Source/CPU evidence alone does not establish model correctness or E2E saving. 本阶段没有新增代码级性能 KEEP。
