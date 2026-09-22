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

## Parallel build tracks

These are concurrent workstreams, not serial gates. Contract extraction is
performed only where the standalone implementation needs a semantic answer.
The runtime should become executable as early as possible and then absorb more
of the real decode DAG behind parity checks.

### Runtime-owned decode loop

Build a fixed-buffer, continuous multi-cycle entry for c12. Weight/operator
bootstrap may initially come from the oracle process, but no SchedulerOutput,
request object, dynamic batch selection or generic ModelRunner dispatch belongs
inside the cycle.

### Semantic migration

Move target verification, greedy acceptance, sequence advance, target/draft KV
and recurrent-state mutations into runtime-owned buffers as needed by the live
implementation. Compare consecutive outputs and touched state against the
oracle; do not wait for unrelated serving or prefill contracts.

### Execution restructuring

Continuously evaluate fixed replay, segmented/full graph, persistent execution,
communication/compute overlap and device-resident control. No mechanism is
mandatory; the real DAG and E2E result decide.

### Fusion and kernels

Search across existing operator boundaries for fusion regions and SuperKernels,
especially verification/acceptance/state advance/next-input preparation,
Markov7, layer-local quant/norm/MoE communication, and target-to-proposer hidden
handoff. Local kernels are promoted only when they serve the runtime DAG.

### Minimal serving shell

Admission/reset, frozen 32K prefix plans, slot lifecycle and output draining are
added around the working decode loop. They do not block early decode-runtime
bring-up.

## Promotion gates

- Exact or tolerance-defined tensor/state parity at every extracted boundary
- Deterministic request completion and output correctness under frozen seeds/settings
- No performance KEEP from microbenchmarks alone
- Full E2E improvement must exceed the measured 4.3% baseline noise and repeat
- New achievable-gap entries require causal evidence that time is removable or overlap can increase

## Loop028 boundary

The first executable specialized-runtime segment is now proven at c12: once proposer inputs and metadata are materialized, the exact real-weight three-layer DSpark7 closure can run again on TP8 and reproduce all draft tokens on 8/8 ranks. Most observed proposer inputs already have fixed process-local addresses; dedicated stable buffers are still required for target token IDs and target positions.

The proposer region is reusable evidence, not the architecture boundary.

## Loop029 live objective

Run the first continuous multi-cycle decode path through `runtime/` using fixed
c12 buffers. The implementation, semantic extraction and fusion-region search
advance together. The immediate migration order is whichever dependency is
needed to make the next real cycle executable: target forward, greedy
verification/acceptance, KV/recurrent state advance and DSpark proposal.

## Loop029 achieved boundary — 2026-09-22

The first real-weight boundary is executable. One-time bootstrap supplies the
loaded model/operator objects, TP8/EP groups, 67 physical KV/DSA cache tensors,
fixed c12 buffers and DSpark7. `ExtremeDecodeRuntime` then takes control before
the generic target forward and owns eight continuous cycles. The verified run
retains no `ModelRunner`, performs no oracle target call after handoff, advances
state exactly and finishes with identical state on all eight ranks.

This is the new implementation base, not a vLLM hook optimization. The remaining
borrowed pieces are bootstrap-only model/operator construction, prebuilt
attention metadata and CPU-backed DSpark common-state refresh. The next boundary
move is to replace those refreshes with fixed/device-resident runtime state,
then profile the owned DAG and evaluate graph/replay, persistence and cross-op
fusion. Eager direct target execution is current fact, not a commitment against
later graph execution.
