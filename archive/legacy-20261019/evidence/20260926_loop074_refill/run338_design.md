# Run338 design — unchanged full-MoE Graph on real warmed residual prefill

## Hypothesis and prior

Run337 no-profiler all8 padded88 prefill found cumulative Host collective-entry spread from 3.68 to 43.65 ms with Host CPU/wall about 1. A Graph replay of the *same original complete MoE* could reduce eager submission on this critical path. This is a Scheduling/Execution experiment, not a claim that the 43.65 ms is entirely removable.

Pinned historical source `db3beef`: R14 old DP2 Host prepare/finalize fusion saved only 0.40% period because its fast path did not cover steady decode and device/communication hid part of Host saving. R20's TP4 cold-prefill hidden AG/local-Q overlap already exists in current source; it is not this candidate. R26 local event schedule improved isolation but not old service. Current Run326 tested routed48 compression *between two Graphs at decode96* and was slower at the complete endpoint; its Graph/eager numerical drift and all-rank endpoint methods constrain this test, but it did not test original eager versus original Graph at prefill88.

## Frozen test

After 48×32K→1024 c12 warmup, instrument only the first measured padded88 original `_model_forward`. At layer4 MoE input, assert all8 real local `[11,4096]`, global padded88, FP/BF16 actual dtype, `FULL_DECODE_ONLY` prefill eager context, TP8/DP1/ALLGATHER/FlashComm1. Record actual `context.num_tokens`, `pad_size` and dtype. Take the identical live hidden tensor and original `module.forward`. Warm original eager, capture unchanged original complete MoE once, then 3 warmup plus 10 measured eager A / Graph B / eager A2 triplets. Keep all original hidden/logits gathers, routing, shared/routed work and final ReduceScatter. Final serving forward uses original eager output; private Graph result is never substituted.

Each arm has all8 barrier and device sync *outside* its timer, then Host submission timer, NPU stream start/end Events, and Host completion after end Event synchronization. Report all8 max-rank stream endpoint, max-rank Host submit and rendezvous Host completion. Event span includes Host enqueue gaps and must not be labeled intrinsic device compute. Capture cost, shape/context key and memory effect belong to Product accounting if the mechanism passes.

## Decision gates

- Runner exit0, 48/48 warmup and12/12 measured exactly1024, all8 Runtime FULL Graph, all8 fixture complete, borrowed ModelRunner SHA restored, service stopped.
- A/A2 and A/B all-rank output finite and shape equal; live input unchanged. Record max_abs, RMS, signed mean in every pair. Since original Graph/eager showed numerical drift in Run326, A/B product correctness is not inferred from local output alone; any live candidate would need frozen semantics and state/acceptance checks.
- B must improve all8 latest complete endpoint beyond A/A2 drift with stable paired direction. Host submission alone is insufficient. A failed or inconclusive private gate stops this Graph implementation without ruling out other execution architectures.
- No one-layer gain multiplied by 43 or promoted directly to Product E2E. Only a passing private gate can proceed to original-path warm prefill/seed-ready integration and then repeated formal 48×32K→1024 c12 E2E.

Hardware/Resource, Scheduling-aware attainable and Product numeric bounds remain UNKNOWN. Formal Current is 571.681 tok/s.
