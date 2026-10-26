# Run579 — fixed-work GMM × TP8 collective resource and completion probe

## Bound question

Run577 measures 43-layer real-weight GMM Graph service without other work; Run578 supplies current Graph MemoryAccess counters but no paired time. Run382/384/387 measure isolated current-order HCCL Graph service with synthetic buffers. The unresolved Engineering/Scheduling resource question is whether **the same independent GMM and HCCL work can execute concurrently on the eight 910B3 ranks**, or AIV/HBM/HCCL scheduling contention forces serial-like joint completion. This tests resource compatibility, not whether the production Target DAG legally exposes every collective concurrently with those GMMs.

Historical R20 measured 157ms timeline overlap but only ~59ms request-level TTFT gain in A2/EP8 prefill; R27 exposed EP/MoE RS dependencies. These are condition-specific priors, not current TP8 verdicts. The current collective inventory is Run378/391: 265 ordered TP8 calls (RS87, AG135, A2A43) and source/ABI-counted tensor geometry, **not compulsory communication work or physical link bytes**. Use the full chain; 43×98,304B AG would miss RS/A2A, router FP32 and tail payloads.

## Stage 1: device completion gate

`loop080_mixed_event_gate_run579.py` captures one BF16 TP8 AG `[12,4096]→[96,4096]` on a private NPU stream. After eager warmup and sync, three generations change inputs and poison outputs. A separate join stream waits the branch event, then checks every rank block on device and records an end event; only that end is synchronized before reading mismatch. The controller requires fresh output, one run tag, all8 rank JSON, script SHA, no preexisting service and final all8 idle. This is a narrow AG event/consumer gate, not a 265-chain or GMM concurrency certificate.

## Stage 2: planned terminal mixed service

After guarded warm48+measured48/c12/1024, borrow production-resident W4A8 Target weights read-only as in Run577, use Run576 SHA-pinned per-rank groups and private nonzero activation. Capture one private 43×(GMM1→GMM2) Graph. Separately capture the exact Run378 265-call TP8 HCCL chain with **per-call distinct input/output buffers** and matching collective order. Inputs are ready before the diagnostic; no model continuation follows it.

Before timing, validate eager/Graph GMM routed rows; verify every collective output for a fresh generation, with poisoned buffers and a join-stream device consumer after the communication branch event. Then run identical-work arms on all8 ranks:

1. GMM Graph alone;
2. HCCL Graph alone;
3. explicit GMM→HCCL serial dependency;
4. two private streams launched from common start and joined after both branch completions.

Use local device event intervals from the common start/end envelope and independent Host spans; no cross-rank timestamp subtraction. Repeat balanced A/B arms in the same load, record per-rank sample distributions and output checks. Compare **measured serial versus measured dual-stream joint completion**, never Run577+Run382 numbers. Keep no profiler in timing arms. Source patch, request/Runtime admission, all8 barrier, stop/restore and bounded scratch-memory gates follow Run577; do not touch algorithm, acceptance or Product output.

## Interpretation

A positive joint reduction establishes attainable mixed-resource overlap for **already-ready independent branches** under this fixture. A null/negative result constrains this schedule and may reveal resource contention. Neither result alone proves which collectives can move across Target layers or DSpark/Target cycles. The next Scheduling step binds producer arrival, native HCCL completion, consumer readiness and state/KV dependencies on one fixed W₀ production trajectory. Formal necessary work/traffic W-minus and exact-board cumulative compute/HBM/HCCL C⁺/B remain separate gates. All strict finite Resource, Scheduling and Product endpoints stay null until those certificates exist.
