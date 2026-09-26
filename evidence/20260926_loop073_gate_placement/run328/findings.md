# Run328 — full-active replicated gate placement, private complete MoE Graph

## Context and prior

- Frozen DeepSeek V4 W4A8, 8×910B3 DP1×TP8 DSpark7, 48×32K→1024 c12. Formal Current remains Run99 median 571.681 tok/s. This run uses a **12×1024 eager diagnostic carrier**, not the formal FULL Graph Product protocol.
- Pinned history `db3beef/R11` proposed the gate/gather exchange but did not implement it; `R35` coalesced two gathers without moving gate and saw Host enqueue savings that did not reliably reach E2E. PK-016 separates those older conditions from current DP1×TP8.
- Current source: layer4 gate is replicated `4096→256`, FP32 weights, no bias/quantization; current ALLGATHER prepare gathers hidden and then logits. B reuses the hidden gather, computes global96 gate, removes logits gather, keeps original selector, routed/shared experts and final communication. Conditional arithmetic exchange is +176.16 MFLOP/rank/layer and removal of one 12,288 B local-input logits collective; physical wire bytes remain unknown.

## Test and provenance

- All 8 loaded gate weights were compared exactly to rank0 before capture. Same real full-active layer4 state at cycle64, local12/global96, mask 12/12 active. SHA-guarded private A/A_repeat/B/A2 full-MoE Graph, independent original A2 capture, 3 warmup and 10 measured triplets.
- B starts shared TP input gather only after B routed hidden gather, then shared gate-up waits for global gate event. This preserves a coherent same-group HCCL order but is one scheduling choice, not every possible gate placement.
- 12/12 client requests returned exactly 1024 output tokens; 8/8 Runtime reports passed. Runner exit0, service stopped, borrowed bootstrap restored to SHA `f644bd14ac1cb9c8365ba2464abbff989d7f2c4c2d58279a8e18a5bce918716f`. Private Graph capture/replay and final sync passed all ranks.

## Numerical and complete-endpoint result

- Gate weights exactly equal across all ranks; global96 B logits versus A max absolute error **1.38283×10⁻⁵** (RMS max 1.98587×10⁻⁶); independent A2 logits are bit exact to A. Routing expert IDs are exact for B versus A on all measured rank-triplets. Seven of eight rank outputs stayed within the predeclared A/A_repeat/A2 max/RMS/signed-mean diagnostic envelope. Rank6 exceeded only signed-mean absolute envelope (2.22338×10⁻⁶ B versus 2.11472×10⁻⁶ control); max absolute and RMS stayed within. This is a numerical screen, **not frozen correctness**.

| Complete layer4 MoE endpoint, 10-triplet median | A | A_repeat | B | A2 |
|---|---:|---:|---:|---:|
| Max-rank NPU event ms | 0.54025 | 0.53167 | **0.55143** | 0.53960 |
| Rank-rendezvous Host ms | 0.86468 | 0.85080 | **0.86666** | 0.85176 |

- B was strictly faster than all three original controls in **0/10** device and **0/10** Host triplets. Paired max-rank B−mean(A,A2) ranged **−27.09 to +67.66 µs**, median **+7.795 µs**, positive in 7/10; the signal is variable and does not show repeatable benefit. The 80.894 tok/s carrier result is eager diagnostic throughput and must not be compared to formal FULL Graph throughput.

## Decision and bound effect

**Do not integrate this B schedule into live Target:** the complete one-layer endpoint has no stable gain, and frozen correctness has not been proven. The per-token linear exchange is mathematically valid under the tested replicated-weight and row-independent assumptions; FP32 execution changes tiny logits values without changing observed top-k IDs. This run does not rule out other overlap/fusion implementations of the exchange, and does not establish a general HCCL or gate capacity floor. No numeric Hardware/Resource, Scheduling-aware or Product E2E bound update; formal Current stays 571.681 tok/s. Reassess Current→Bound gap outside this narrow MoE branch before selecting the next experiment.

Primary artifacts: `fixture_check.json`, `fixture/rank*_cohort1.json`, `runtime/rank*_cohort1.json`, `bench12.json`, `patch.json`, `restore.log`, `stop.log`.
