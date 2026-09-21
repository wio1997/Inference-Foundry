# DeepSeek Extreme P0 Specialized Runtime

## Frozen product

- Model and weights: DeepSeek V4 Flash W4A8
- Hardware: one node, 8×Ascend 910B3
- Parallelism: DP1×TP8
- Speculation: DSpark7 with all three trained draft layers
- Primary KPI: mixed 48×32K→1K c12 output TPS, accepted baseline 543.65 tok/s
- Correctness oracle and operator source: current vLLM/vLLM-Ascend stack

## Runtime boundary

The runtime preserves model semantics, quantized weights, TP8 communication, KV updates, DSpark proposal, target verification, sampling, prefix-cache behavior required by the frozen workload, and the minimal serving ingress/egress needed to run it. No framework abstraction is retained without evidence that it is necessary on the critical path.

## Target execution chain

1. A fixed admission ring for the frozen concurrency envelope, with stable request slots.
2. Preallocated device-resident token, position, sequence, block-table, slot-mapping, RNG and acceptance state.
3. Static prefill plans for the frozen 32K prompt shape and prefix-cache layout.
4. A fixed-shape DSpark7 proposer step using all three trained layers.
5. A fixed-shape target verification/decode step.
6. Device-side acceptance, state advance and next-step preparation wherever operator support permits.
7. Segmented graph or persistent replay around graph-safe compute and collectives; explicit narrow boundaries around unavoidable mutable KV or host-visible output.
8. Batched output draining outside the critical device execution chain.

## Removal candidates

- General request and scheduler objects per decode step
- Dynamic batch and dynamic shape dispatch inside the frozen c12/max16 envelope
- Repeated attention/KV metadata builders
- Backend selection and compatibility branches
- Per-step Python object construction and orchestration
- Redundant CPU mirrors, scalar reads and Host↔Device synchronization
- Generic KV allocation/eviction policy after fixed slot ownership is established

Each removal candidate requires an exact contract, a minimal implementation, correctness evidence and an E2E result. Inclusive profiler time is not a savings estimate.

## Build phases

### Phase A — semantic contract extraction

Record the exact tensors, mutations, collectives and ordering for one warm pure-decode cycle: target hidden-state handoff, DSpark inputs, three-layer proposer, draft tokens/logits, target verification, acceptance, KV writes and sequence advance. Classify every input as constant, per-request device state, per-step derived state or host-visible output.

### Phase B — standalone fixed-cycle replay

Instantiate real weights and Ascend operators while bypassing scheduler/request objects. Replay one fixed batch cycle from preallocated buffers and compare draft logits/tokens, accepted tokens, target outputs and KV/state mutations against the oracle.

### Phase C — graph and device-resident state

Capture graph-safe segments, retain mutable state in fixed-address buffers, eliminate repeated metadata construction and reduce host launch/synchronization boundaries. Benchmark cost per advanced token before service integration.

### Phase D — fixed serving runtime

Add the minimum admission, prefix plan, slot lifecycle and output drain required for the frozen mixed workload. Compare full correctness, TTFT, TPOT and Output TPS against the accepted vLLM-Ascend baseline.

## Promotion gates

- Exact or tolerance-defined tensor/state parity at every extracted boundary
- Deterministic request completion and output correctness under frozen seeds/settings
- No performance KEEP from microbenchmarks alone
- Full E2E improvement must exceed the measured 4.3% baseline noise and repeat
- New achievable-gap entries require causal evidence that time is removable or overlap can increase

## Loop028 checkpoint and paused boundary

The first executable specialized-runtime segment is now proven at c12: once proposer inputs and metadata are materialized, the exact real-weight three-layer DSpark7 closure can run again on TP8 and reproduce all draft tokens on 8/8 ranks. Most observed proposer inputs already have fixed process-local addresses; dedicated stable buffers are still required for target token IDs and target positions.

Phase B remains incomplete for the whole decode cycle. The next implementation boundary is target verification → acceptance → device-resident sequence/KV advance → next proposer inputs, with explicit parity for accepted tokens and all mutated state. Only after that boundary is exact should segmented graph, persistent replay or wider fused regions receive an E2E performance gate.

Work is paused after Loop028 by user request.
