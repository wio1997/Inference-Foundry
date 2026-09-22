# 执行日志

- `2026-09-21T13:23:51Z` Loop 已冻结。下一步：Define the runtime-owned fixed state ABI and implement the first direct multi-cycle harness around existing target/proposer operators, adding only the missing verification/acceptance/KV-state semantics needed to make cycle t+1 consume cycle t output.

- `2026-09-21T13:29:36Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run-20260921T132936Z`（test）。

- `2026-09-21T13:29:48Z` Run `run-20260921T132936Z` 记录为 `pass`；正确性为 `not-applicable`。Dedicated fixed-c12 runtime ABI executed 8 causally connected cycles with deterministic test operators; fixed target buffers and device tensor state advanced without SchedulerOutput, request objects, InputBatch or ModelRunner. This validates runtime structure only, not real-model parity.

- `2026-09-21T14:03:50Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run-20260921T140350Z`（test）。

- `2026-09-21T14:04:04Z` Run `run-20260921T140350Z` 记录为 `error`；正确性为 `invalid`。First real-weight shadow wiring used .gpu on the already-device positions tensor and killed all workers before a comparison; no semantic or performance conclusion.

- `2026-09-21T14:04:04Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run-20260921T140404Z`（test）。

- `2026-09-21T14:04:16Z` Run `run-20260921T140404Z` 记录为 `pass`；正确性为 `pass`。On all 8 TP ranks, standalone acceptance/state advance predicted 19 consecutive next-cycle target input/position/query/seq/slot tensors exactly and 23 consecutive core input/position/query/seq transitions exactly. One slot-mapping-only mismatch occurred at cycle20 and recovered at cycles21-23; later mismatches belong to a new request batch because the diagnostic shadow intentionally had no admission reset. Both real c12 requests completed 12/12. This validates the steady-state state ABI, not standalone target/KV execution.

- `2026-09-21T14:20:36Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run-20260921T142036Z`（test）。

- `2026-09-21T14:20:46Z` Run `run-20260921T142036Z` 记录为 `error`；正确性为 `invalid`。Standalone greedy acceptance reached device execution but aclnnInplaceScatter rejected int64 target argmax into an int32 output buffer. No accepted-token parity result. Explicit cast fixed the adapter and an 8-cycle NPU semantic test now passes.

- `2026-09-21T14:34:22Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run-20260921T143422Z`（test）。

- `2026-09-21T14:34:29Z` Run `run-20260921T143422Z` 记录为 `pass`；正确性为 `pass`。Real 8x910B3 c12 run passed 12/12. Across 8 TP ranks, standalone next-target input prediction was exact for 28/28 comparisons per rank (224 total), including all five ABI fields. Independent fixed greedy acceptance matched oracle accepted tokens and counts for 29/29 comparisons per rank (232 total). Target forward, DSpark proposal, and KV/recurrent storage remain oracle-owned; diagnostic throughput is invalid for performance.

- `2026-09-21T14:34:38Z` 暂存知识变化 `loop029-fixed-state-acceptance-parity`：For frozen DeepSeek V4 Flash W4A8 c12 decode, fixed device tensors reproduce next target input ABI and temperature-0 greedy acceptance exactly across consecutive cycles on all 8 TP ranks; target, DSpark, and KV execution are not yet runtime-owned.

- `2026-09-21T15:14:06Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run-20260921T151406Z`（test）。

- `2026-09-21T15:14:15Z` Run `run-20260921T151406Z` 记录为 `pass`；正确性为 `not-applicable`。Fixed target adapter ABI passed CPU and Ascend NPU semantic tests: two direct calls reused stable input and position buffer addresses, returned fixed 96-token hidden/logit tensors, and exposed a cache-fingerprint hook. This is structure validation only; live DeepSeek weights, attention context, and physical KV cache are not yet bound.

- `2026-09-21T15:19:48Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run-20260921T151948Z`（test）。

- `2026-09-21T15:19:56Z` Run `run-20260921T151948Z` 记录为 `pass`；正确性为 `not-applicable`。Product-owned ExtremeDecodeRuntime drove 8 continuous cycles on CPU and Ascend NPU with fixed state and explicit prepare-target, target, acceptance, state-advance, proposer ordering. Target and proposer each ran 8 times. The standalone process imported no vLLM modules. Operators were deterministic test bindings, so this proves control-plane independence, not real-weight correctness.

- `2026-09-21T15:22:44Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run-20260921T152244Z`（test）。

- `2026-09-21T15:22:52Z` Run `run-20260921T152244Z` 记录为 `pass`；正确性为 `not-applicable`。Ascend NPU standalone driver retained direct ownership of fixed cache tensor references, checked address/shape/stride stability, and cloned selected physical elements for exact parity fingerprints across its 8-cycle run. Synthetic caches only; live DSA cache handoff remains pending.

- `2026-09-22T02:48:02Z` 为用例 `mixed_32k_1024_c12` 创建 Run `real-target-handoff-20260922`（test）。

- `2026-09-22T03:03:57Z` Run `real-target-handoff-20260922` 记录为 `fail`；正确性为 `fail`。8/8 ranks bound 67 real cache tensors and completed request, but direct replay diverged from graph oracle: hidden max_abs 0.7919921875 and logits max_abs 3.5 identically on all ranks; forward-context parity and double-write effect must be isolated.

- `2026-09-22T03:09:05Z` 为用例 `mixed_32k_1024_c12` 创建 Run `real-target-context-parity-20260922`（test）。

- `2026-09-22T03:15:01Z` Run `real-target-context-parity-20260922` 记录为 `invalid`；正确性为 `invalid`。Stopped during startup: graph outputs may alias static buffers, so the v2 probe required snapshots before subsequent direct calls.

- `2026-09-22T03:18:03Z` 为用例 `mixed_32k_1024_c12` 创建 Run `real-target-snapshot-parity-20260922`（test）。

- `2026-09-22T03:32:26Z` Run `real-target-snapshot-parity-20260922` 记录为 `fail`；正确性为 `fail`。Independent snapshots prove replay from an already-mutated live cache is invalid: FULL-context oracle/direct hidden max_abs 0.505859375, logits 3.5, and direct/direct hidden 0.62939453125 identically on 8/8 ranks. Target writes are non-idempotent; validation must restore touched DSA/KV slots to the same pre-state.

- `2026-09-22T03:36:20Z` 为用例 `mixed_32k_1024_c12` 创建 Run `real-target-transactional-cache-20260922`（test）。

- `2026-09-22T03:53:59Z` Run `real-target-transactional-cache-20260922` 记录为 `error`；正确性为 `invalid`。All 8 ranks reached real cache snapshot; non-contiguous [block,32,...] cache views rejected slot flattening before target replay. No parity result; fixed by stride-preserving block/offset indexing.

- `2026-09-22T03:54:20Z` 为用例 `mixed_32k_1024_c12` 创建 Run `real-extreme-runtime-8cycle-20260922`（test）。

- `2026-09-22T04:09:04Z` Run `real-extreme-runtime-8cycle-20260922` 记录为 `error`；正确性为 `invalid`。8/8 ranks built the Extreme Runtime and reached its first direct target call; graph metadata update exposed an implicit thread-local current_vllm_config dependency. Explicit target/DSpark config contexts added; no continuous-cycle correctness result yet.

- `2026-09-22T04:09:17Z` 为用例 `mixed_32k_1024_c12` 创建 Run `real-extreme-runtime-config-context-20260922`（test）。
