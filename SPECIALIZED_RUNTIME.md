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

## Loop031–032 profiled boundary — 2026-09-22

Eight-rank runtime-only tracing now covers target, TP/EP communication,
acceptance, state advance, DSpark common refresh, all three trained proposer
layers, and host launch/synchronization regions. The first profiling-driven
runtime change removes the three per-cycle device-to-host common-state pulls.
Extreme Runtime keeps the fixed bootstrap CPU metadata, transfers only the 12
acceptance counts on a side stream, and consumes that transfer one target cycle
later immediately before its next DSpark consumer.

The matched traces reduce `dspark_refresh_common` median from 12.483 ms to
0.366 ms, proposer median from 66.825 ms to 52.170 ms, and the profiled cycle
median from 536.705 ms to 529.211 ms. A 64-cycle real-weight burn-in passes on
all eight ranks with identical final state and exact post-window CPU/device
mirror parity. Its 38.27 tok/s internal decode-window rate is not an E2E claim:
the current bootstrap harness does not yet implement fixed 48-request refill or
user-visible output draining.

The dominant measured gap is now target verification: 469.789 ms host median
and 465.056 ms device-union median for the fixed 96-token target region. The
current handoff explicitly uses eager `CUDAGraphMode.NONE` with
`skip_compiled=True`. The next structural experiment is therefore a
runtime-owned fixed target replay/graph boundary, followed by the fixed refill
and output sink required for a comparable 48×32K→1024 c12 E2E A/B against
543.65 tok/s Stock.

## Loop033 target replay boundary — 2026-09-22

The fixed 96-token target region now reuses the dispatcher-captured decode
graph from Extreme-owned buffers and metadata. The continuous cycle still
retains no `ModelRunner` or `Scheduler`; the bootstrap lends the compiled graph
and graph-metadata updater once, just as it lends weights and operators.

A matched, unprofiled 64-cycle A/B preserves exact device state, Host mirrors
and rank parity on all eight ranks. Median cycle wall falls from 433.587 ms to
53.286 ms (-87.71%), while the internal decode-window rate rises from 38.271 to
294.400 accepted tok/s (+669.2%). This promotes target replay into the runtime
architecture. It is still not the formal product E2E number: the remaining
boundary is the fixed 48-request admission/refill and output-drain shell needed
to execute the frozen 48×32K→1024 c12 protocol and compare directly with Stock
543.65 tok/s.

## Loop034 fixed serving shell — completed 2026-09-23

The first serving boundary keeps an admitted 12-request cohort inside Extreme
Runtime until every slot has produced its frozen 1024-token limit. Accepted
tokens are staged in fixed device history buffers and copied to Host once at
cohort completion. A lagged 12-count Host progress mirror, already required by
DSpark common state, terminates the loop without synchronizing the proposer on
every cycle. The scheduler is entered only between completed cohorts, not on
the decode hot path.

The one-time bootstrap now reserves 1024 KV/DSA lookahead tokens per admitted
request before handoff, so long decode does not address unallocated block-table
entries. A bulk completion frame lets the existing HTTP/output control plane
publish the completed cohort; its normal one-step speculative accounting is
explicitly bypassed for this marked frame. CPU tests cover exact per-slot
trimming, different acceptance counts, one-cycle-lag completion and output
transport serialization.

The NPU E2E gate passed. Warmup plus three `48×32K→1024, c12` measurements all
completed 48/48 requests at exactly 1024 tokens. Sixteen cohorts across the
four workloads produced 128 passing rank records, with exact Host mirrors and
zero post-handoff ModelRunner cycles. Formal median output TPS is
`217.342 tok/s`; TTFT p50 is `1942.120 ms` and TPOT p50 `53.298 ms`. Against
Stock `543.655 tok/s`, the current runtime is `60.022%` slower.

The bulk serving boundary is therefore promoted for correctness but not for
performance. Rank-0 cohort wall median is `53.373 s` for roughly 1025 cycles,
which localizes the dominant gap to sustained target/proposer/communication
execution rather than Scheduler or HTTP publication. The next runtime loop
must profile that complete chain and select a structural intervention from the
measured critical path.
