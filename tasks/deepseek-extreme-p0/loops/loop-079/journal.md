# 执行日志

- `2026-09-27T04:41:50Z` Loop 已冻结。下一步：Read-only source/Graph branch audit and exact minimal instrumentation plan; reuse Run403 route evidence and Run407 design before any service launch.

- `2026-09-27T04:45:16Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run419`（design-check）。

- `2026-09-27T04:45:16Z` Run `run419` 记录为 `pass`；正确性为 `not-applicable`。Source-pinned Run403 audit confirms internal zero initial output counts in 40/40 rank-cohort records, FULL Graph handoff before current model forward, and 60 Runtime request IDs. External pre-handoff published p_i remains unproved because Scheduler Request and API SSE publication are not joined to route slots. Conditional five-cohort DS7 cardinality640 is not Product bound.

- `2026-09-27T04:50:13Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run420`（review）。

- `2026-09-27T04:50:13Z` Run `run420` 记录为 `pass`；正确性为 `not-applicable`。Independent review accepts only configured/internal zero. Scheduler bulk append may trim 1024 to remaining cap when prior generated outputs exist; Run403 has no zero-output predicate. Minimum Host-only authoritative pre/post bulk ledger per request, plus resumable reset and class/provenance gate, can prove p_i=0 without SSE only if all pre-counts are zero. Row identity still needs active branch/group/Graph binding and targeted native contract checks.

- `2026-09-27T04:53:49Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run421`（benchmark）。

- `2026-09-27T05:08:55Z` Run `run421` 记录为 `pass`；正确性为 `pass`。Run421 diagnostic exact60 POST, 48+12 exact1024 c12, 40 all8 FULL Runtime reports, source restored and NPU idle. Scheduler bulk ledger 60/60 joined: prior generated counts 1–33, total584; Runtime incoming61440, admitted60856, clipped584. This is not formal TPS or external pre-handoff publication proof.

- `2026-09-27T05:09:25Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run422`（review）。

- `2026-09-27T05:10:55Z` Run `run422` 记录为 `pass`；正确性为 `not-applicable`。Independent raw/source review confirms 60 positive terminal pre-bulk Scheduler generated counts totaling584 and exact clipping; neither handoff-generated g_H nor handoff-published p_H is thereby known. Run421-only, no finite Bound promotion.

- `2026-09-27T05:11:03Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run423`（review）。

- `2026-09-27T05:11:03Z` Run `run423` 记录为 `pass`；正确性为 `not-applicable`。A single matching compulsory W-minus and genuine C-plus is enough for a loose strict bound; actual 910B3 SKU/BF16 peak and universal work necessity are not certified, so numeric strict ceiling remains null.

- `2026-09-27T05:11:10Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run424`（design-check）。

- `2026-09-27T05:11:10Z` Run `run424` 记录为 `pass`；正确性为 `not-applicable`。Identified kwargs graph-address coverage, actual Runtime CP geometry, logits indices and request identity requirements; unresolved native row order remains conditional, no row identity promoted.

- `2026-09-27T05:12:21Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run425`（simulation）。

- `2026-09-27T05:12:21Z` Run `run425` 记录为 `pass`；正确性为 `not-applicable`。V3.12 records Run421 terminal 584 generated tokens clipped from 61440 Runtime output, preserves unknown handoff generated/published counts and null finite Product ceiling, and adds strict W-minus/C-plus sufficiency gate.

- `2026-09-27T05:14:24Z` 暂存知识变化 `PK-042`：Run421 clean frozen diagnostic observed 584 Scheduler generated tokens before terminal bulk, causing 584 of 61440 Runtime IDs to be clipped; handoff-time generated and API-published counts remain unknown.

- `2026-09-27T05:23:44Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run426`（design-check）。

- `2026-09-27T05:23:44Z` Run `run426` 记录为 `pass`；正确性为 `not-applicable`。Design separates device-completed, Scheduler-committed, OutputProcessor-received, API-consumed and SSE-yielded states with clock/ordinal gates; raw token publication cannot be inferred from parser consumption alone.

- `2026-09-27T05:23:44Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run427`（benchmark）。

- `2026-09-27T05:34:47Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run428`（review）。

- `2026-09-27T05:38:39Z` Run `run427` 记录为 `pass`；正确性为 `pass`。Clean frozen 48+12 c12 diagnostic, exact60 POST and 40 all8 FULL Runtime reports, four-source SHA restore and idle NPUs. Host timeline digest/clock gate: terminal Scheduler prior796; global H_probe committed interval[795,796], API raw consumed conservative[674,796], generated-output yield event count[213,268]. No formal timing/ceiling promotion.

- `2026-09-27T05:43:20Z` Run `run428` 记录为 `pass`；正确性为 `not-applicable`。Independent Astra accepted Run427 scoped Host ledger: G(H_probe) [795,796], API raw consumed [674,796], Chat generated-output yields [213,268], at least 50 requests/213 nonempty reasoning yields; no finite bound or client receipt promotion

- `2026-09-27T05:44:06Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run429`（review）。

- `2026-09-27T05:44:06Z` Run `run429` 记录为 `pass`；正确性为 `not-applicable`。V3.13 promoted only reviewed Run427 Host cutoff intervals; all finite Bound endpoints remain null and formal Current remains 571.681

- `2026-09-27T05:45:42Z` 暂存知识变化 `PK-043`：Run427 Host diagnostic proves cohort H_probe Scheduler G=[795,796], Chat raw consumed A=[674,796], and at least 213 nonempty reasoning generator yields from at least 50 requests, but no device completion, literal raw-token publication or finite TPS Bound

- `2026-09-27T05:46:48Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run431`（review）。

- `2026-09-27T05:47:12Z` Run `run431` 记录为 `pass`；正确性为 `not-applicable`。Board/OEM/clock/HBM identity pinned; exact matching BF16/W4A8 genuine maximum C_plus remains unbound; no numeric ceiling

- `2026-09-27T05:49:49Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run430`（review）。

- `2026-09-27T05:49:49Z` Run `run430` 记录为 `pass`；正确性为 `not-applicable`。Installed CANN9.1/torch_npu native source reviewed, DSA CPU map 4608 PASS; all43 identity CONDITIONAL_NATIVE, first gap partial-EP V3 index/abs/masked unpermute

- `2026-09-27T05:52:43Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run432`（test）。

- `2026-09-27T05:53:42Z` Run `run432` 记录为 `pass`；正确性为 `pass`。Isolated single-910B3 eager CANN9.1 partial-EP quant_mode1 routing→abs→masked unpermute semantic sentinel passed 4x2 and actual96x6 rank0/nonzero EP ranges; no Graph/all43 promotion

- `2026-09-27T05:55:48Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run433`（test）。

- `2026-09-27T05:57:31Z` Run `run433` 记录为 `pass`；正确性为 `pass`。Single-device 96x6 quant1 partial-EP Graph replay three generations passed; fixed-address input changed routes 85→69→85 and returned SHA, scoped native ABI only

- `2026-09-27T05:59:07Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run434`（review）。

- `2026-09-27T05:59:07Z` Run `run434` 记录为 `pass`；正确性为 `not-applicable`。V3.14 adds explicit evidence-status proof-obligation DAG; sampled MoE ABI/graph refresh promoted narrowly; strict resource and achievable Product endpoints remain null

- `2026-09-27T06:02:25Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run435`（review）。

- `2026-09-27T06:02:25Z` Run `run435` 记录为 `pass`；正确性为 `not-applicable`。V3.15 separates strict W/C ceiling, scheduling latency relaxation and attainable schedule gates with 15-node validated proof dependency graph; no numeric endpoint promoted

- `2026-09-27T06:02:47Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run436`（test）。

- `2026-09-27T06:02:47Z` Run `run436` 记录为 `pass`；正确性为 `pass`。Validated V3.15 actual 17-node proof DAG: dangling, cycle and scope mutations reject; one W alternative without C does not promote strict ceiling; Run435 summary node count corrected here

- `2026-09-27T06:11:12Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run437`（benchmark）。

- `2026-09-27T06:21:20Z` 暂存知识变化 `PK-044`：On installed CANN9.1/torch_npu2.10, isolated 910B3 quant_mode1 partial-EP MoE routing sampled eager 4x2/96x6 and fixed-storage graph 96x6 across three generations preserve token and expert-segment row mapping with masked unpermute; this does not certify the production all43 FULL Graph or a compulsory retained-route numerator.

- `2026-09-27T06:24:26Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run438`（review）。

- `2026-09-27T06:24:27Z` Run `run438` 记录为 `pass`；正确性为 `not-applicable`。Independent review prioritizes original-path mixed-service seam for useful scheduling range; one fresh required BF16 projection plus exact-board C_plus for loose strict ceiling. All numeric endpoints stay null; Run437 cannot promote all43.

- `2026-09-27T06:34:29Z` Run `run437` 记录为 `pass`；正确性为 `pass`。Clean 48+12 c12 diagnostic, exact60 successful POST and 40 all8 FULL capture/runtime files; explicit input/position storage and CP prefix values validated. Cleanup/source SHA gates pass. all43 row identity remains conditional; no formal TPS or numeric Bound promotion.

- `2026-09-27T06:34:30Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run439`（design-check）。

- `2026-09-27T06:34:31Z` Run `run439` 记录为 `pass`；正确性为 `not-applicable`。Selected terminal logits AllGather→argmax original-path slice; native completion join is prerequisite, A0-B-A1 controls frozen; design only and no timing claim.

- `2026-09-27T06:34:31Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run440`（review）。

- `2026-09-27T06:34:32Z` Run `run440` 记录为 `pass`；正确性为 `not-applicable`。Astra accepted Run437 scoped FULL input/CP diagnostic and reproduced validator; pointer/CP mutations reject, common row permutation passes, so all43 and numeric Bound remain open.

- `2026-09-27T06:34:33Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run441`（benchmark）。

- `2026-09-27T06:34:33Z` Run `run441` 记录为 `pass`；正确性为 `pass`。Three exact-payload all8 BF16 HCCL Test isolated host-inclusive averages 221.77/179.45/205.78us, all correctness success; attained isolated point only, no current native join, compulsory traffic or Product ceiling.

- `2026-09-27T06:45:56Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run442`（simulation）。

- `2026-09-27T06:45:57Z` Run `run442` 记录为 `invalid`；正确性为 `invalid`。Initial V3.16 Run442 used Run441 CLI aggregate bytes as per-rank input and described wrong message shape; superseded by corrected Run444/445. No numeric endpoint had been promoted.

- `2026-09-27T06:45:57Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run444`（benchmark）。

- `2026-09-27T06:45:58Z` Run `run444` 记录为 `pass`；正确性为 `pass`。Corrected installed AllGather CLI size 24821760B gives 3102720B/rank input; three root-rank ACL event-loop averages 235.63/261.26/238.76us, all success. Isolated attainable point only; no Bound endpoint.

- `2026-09-27T06:45:59Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run445`（simulation）。

- `2026-09-27T06:49:19Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run443`（review）。

- `2026-09-27T06:49:20Z` Run `run443` 记录为 `pass`；正确性为 `not-applicable`。Astra found Run441 CLI data_size is aggregate AllGather output, eightfold smaller per-rank input than claimed; initial Run442 invalidated, numeric guard gaps identified.

- `2026-09-27T06:49:20Z` Run `run445` 记录为 `pass`；正确性为 `not-applicable`。Corrected V3.16 rev2 uses Run444 exact 3102720B/rank isolated HCCL observation, adds parsed source/command/log gates and null endpoint guards; all19 proof certifications false, Current571.681.

- `2026-09-27T06:49:24Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run446`（review）。

- `2026-09-27T06:49:25Z` Run `run446` 记录为 `pass`；正确性为 `not-applicable`。Astra accepts corrected 19-node all-open ledger; numeric escapes, payload, source hash and malformed timing mutations reject. No finite Bound promoted.

- `2026-09-27T06:49:26Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run447`（review）。

- `2026-09-27T06:49:26Z` Run `run447` 记录为 `pass`；正确性为 `not-applicable`。Installed Python and torch_npu exact-commit source conditionally shows caller-stream→HCCL→caller-stream event chain for synchronous all_gather_into_tensor. Actual runtime branch/group/stream still requires dynamic binding; no timing or Bound promotion.

- `2026-09-27T06:49:41Z` 暂存知识变化 `PK-045`：Clean Run437 production FULL Graph samples bind explicit input_ids/positions storage and five CP metadata value prefixes across five cohorts/eight ranks; a common router-row permutation still passes, so all43 row identity and retained-route compulsory work remain conditional.

- `2026-09-27T06:49:42Z` 暂存知识变化 `PK-046`：Installed CANN9.1 HCCL Test AllGather CLI data_size is aggregate output bytes, divided by TP8 for per-rank input. Correct exact logits input3102720B/rank requires CLI24821760B; three isolated root-rank ACL event-loop averages are235.63/261.26/238.76us, not a strict latency floor or exposed E2E cost. Run441 smaller-payload interpretation is superseded.

- `2026-09-27T06:52:57Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run448`（review）。

- `2026-09-27T06:52:58Z` Run `run448` 记录为 `pass`；正确性为 `not-applicable`。Astra accepts conditional caller-stream→HCCL→caller-stream source join and identifies actual-path gates: no torch override/coalescing/capture substitution, same consumer stream; no duration or numeric Bound.

- `2026-09-27T06:53:09Z` 暂存知识变化 `PK-047`：Installed torch_npu2.10.0.post4 git commit5dd8ef3 ordinary synchronous all_gather_into_tensor source records caller→HCCL and HCCL-end→caller stream event dependencies; Extreme terminal logits can use this conditionally only when actual branch excludes torch override/coalescing/graph substitution and consumer remains on the joined stream. No Host-completion timestamp or E2E saving follows.
