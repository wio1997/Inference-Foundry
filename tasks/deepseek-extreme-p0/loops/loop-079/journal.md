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

- `2026-09-27T07:05:49Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run449`（review）。

- `2026-09-27T07:05:55Z` Run `run449` 记录为 `pass`；正确性为 `not-applicable`。Installed 910B3 maps to 2201 BF16 Cube 16x16x16; configured 1800MHz is not certified maximum, aggregate C+ and numeric TPS ceiling remain null

- `2026-09-27T07:06:00Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run450`（review）。

- `2026-09-27T07:06:07Z` Run `run450` 记录为 `pass`；正确性为 `not-applicable`。Full terminal logits AllGather is not compulsory for argmax-only Target consumer; conditional pair construction has 768B/rank input versus current 3102720B/rank, with exact numeric and tie gates still open; no E2E or latency gain claimed

- `2026-09-27T07:06:14Z` 暂存知识变化 `PK-048`：Installed CANN9.1 910B3 binds to NpuArch2201 with BF16 Cube 16x16x16 ideal parallelism, but cube_freq1800 and DCMI type9 rated frequency do not certify maximum clock or all-engine aggregate C+; hardware/Product numeric ceilings remain open

- `2026-09-27T07:06:21Z` 暂存知识变化 `PK-049`：Frozen Target logits are consumed as greedy argmax; installed distributed get_top_tokens pair method constructs a conditional exact alternative to current full BF16 logits AllGather with 768B/rank pair input versus 3102720B/rank full input. This is not compulsory communication, latency reduction, or E2E gain until runtime numeric and tie gates pass

- `2026-09-27T07:12:47Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run453`（simulation）。

- `2026-09-27T07:12:54Z` Run `run453` 记录为 `pass`；正确性为 `not-applicable`。V3.17 adds symbolic 910B3 BF16 Cube formula and conditional pair-argmax payload construction; all strict and Product numeric endpoints remain null pending independent Run454 review

- `2026-09-27T07:13:00Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run451`（review）。

- `2026-09-27T07:13:05Z` Run `run451` 记录为 `pass`；正确性为 `not-applicable`。Review completed and Run439 B preflight failed: cleanup HBM parser fail-open, cycle64 profiler, stream/event, dispatch and validator gates require repairs before service

- `2026-09-27T07:21:59Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run454`（review）。

- `2026-09-27T07:22:06Z` Run `run454` 记录为 `pass`；正确性为 `not-applicable`。Astra accepted V3.17 symbolic Cube and conditional argmax arithmetic; archived installed source recheck passes corrupted/archive-swap negative tests and byte-identical regeneration; finite endpoints remain null

- `2026-09-27T07:22:11Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run455`（review）。

- `2026-09-27T07:22:16Z` Run `run455` 记录为 `pass`；正确性为 `not-applicable`。Five Run451 blockers repaired and independently checked with source compile, CPU/adversarial tests; staged B permitted only as instrumented local-envelope acquisition

- `2026-09-27T07:23:53Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run456`（review）。

- `2026-09-27T07:24:02Z` Run `run456` 记录为 `pass`；正确性为 `not-applicable`。Run439 B first attempt rejected at preflight with 91 active dedicated-container processes; no patch/service/NPU, then idle container reset and six source hashes verified unchanged

- `2026-09-27T07:37:51Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run458`（benchmark）。

- `2026-09-27T07:37:58Z` Run `run458` 记录为 `invalid`；正确性为 `invalid`。First warmup aborted on instrumentation C10D wrapper hash gate; no selected slices or formal result. Stop/restore/SHA zero, revised wrapper plus unwrapped source check moved to pre-service

- `2026-09-27T07:41:01Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run457`（review）。

- `2026-09-27T07:41:11Z` Run `run457` 记录为 `pass`；正确性为 `not-applicable`。Public C10D logger wrapper and unwrapped implementation source pinned separately; controller pre-service check and fresh retry identity pass; no valid timing until live acquisition

- `2026-09-27T07:41:21Z` 暂存知识变化 `PK-050`：Installed PyTorch public all_gather_into_tensor is decorated by c10d_logger.py; inspect.getsourcefile on the public function identifies wrapper, while inspect.unwrap identifies distributed_c10d.py. Provenance gates must pin both wrapper and underlying implementation and run before costly service startup

- `2026-09-27T07:54:38Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run460`（benchmark）。

- `2026-09-27T07:54:38Z` Run `run460` 记录为 `invalid`；正确性为 `invalid`。Run439 B retry rejected: generic communicator hook captured out-of-scope Target hidden/aux AllGather before terminal logits. 11/48 warmup succeeded; no complete selected slices, no timing or E2E inference. Stop, restore, source SHA checks zero. Scoped marker now filters unselected native calls.

- `2026-09-27T07:55:05Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run459`（review）。

- `2026-09-27T07:55:05Z` Run `run459` 记录为 `pass`；正确性为 `not-applicable`。Astra High PASS: off-scope earlier hidden AllGather ignored; lost intended scope, duplicate scoped native and broken event chain reject. Fresh retry2 path and CPU/source/controller gates accepted. This is preflight, not timing or E2E evidence.

- `2026-09-27T07:57:02Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run461`（review）。

- `2026-09-27T07:57:02Z` Run `run461` 记录为 `pass`；正确性为 `not-applicable`。All finite Algorithm/Resource, Hardware/Resource, Scheduling/Execution, Product E2E endpoints remain unresolved. Run458/460 invalid; corrected B is highest-information local scheduling measurement, but no strict ceiling or formal TPS promotion without controls and causal proof.

- `2026-09-27T08:00:59Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run462`（review）。

- `2026-09-27T08:00:59Z` Run `run462` 记录为 `pass`；正确性为 `not-applicable`。No valid aggregate C+ yet: TOPS_DETAILS enum unsupported by A2 get-LP, computing_power API exposes core count only, driver frequency categories do not cap maximum, S900K3 whitepaper requires login. Hardware strict endpoint stays null; exact OEM max frequency, BF16 issue/engines certificate needed.

- `2026-09-27T08:09:30Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run463`（benchmark）。

- `2026-09-27T08:09:30Z` Run `run463` 记录为 `pass`；正确性为 `pass`。Exactly60 POST, 60 correct 1024-token outputs, five cohorts x eight ranks =40 validated cycle64 slices; ordinary scoped logits AllGather branch and same-stream event chain accepted. Stop, verify, source restore/SHA and final admission all zero. Diagnostic B only; no matched A0/A1 or formal TPS claim.

- `2026-09-27T08:13:18Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run464`（review）。

- `2026-09-27T08:13:18Z` Run `run464` 记录为 `pass`；正确性为 `not-applicable`。ACCEPT 40 instrumented same-device local slices; P-J median200.40us, J-G30.96us, C0-C153.07us; P-C1 260.50-296.00us. Current 24.82176MB layout materialization. G-C0 below calibrated resolution; no finite Bound or E2E inference.

- `2026-09-27T08:13:18Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run465`（review）。

- `2026-09-27T08:13:18Z` Run `run465` 记录为 `pass`；正确性为 `not-applicable`。Target pre-replay wait through FULL Graph completion, hidden/aux gather and first hidden consumer is next highest-information scheduling seam. Historic R13/R37/R21/R28 are hypothesis priors only; require exact graph completion semantics, matched controls and no hot-path sync. Terminal pair-argmax performance deprioritized, not excluded.

- `2026-09-27T08:16:44Z` 暂存知识变化 `PK-051`：Run463/464 admitted selected cycle64 terminal full-logits TP8 path: marked same-device P-J median0.20040ms, J-G0.03096ms, C0-C10.05307ms and current24.82176MB layout materialization. This narrows Current local dependency only; no A0/A1, compulsory floor, E2E saving or finite Bound.

- `2026-09-27T08:18:17Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run466`（simulation）。

- `2026-09-27T08:18:17Z` Run `run466` 记录为 `pass`；正确性为 `not-applicable`。V3.18 reproduces 40 instrumented local slices, sets next Target FULL replay frontier and leaves every finite Algorithm/Hardware/Scheduling/Product endpoint null; formal Current571.681tok/s.

- `2026-09-27T08:18:17Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run467`（review）。

- `2026-09-27T08:18:17Z` Run `run467` 记录为 `pass`；正确性为 `not-applicable`。ACCEPT: 29 CPU checks; 40 raw hashes and stats verified; byte-identical regeneration; all19 proof nodes false. Local0.26050-0.29600ms remains marked Current only, no numeric Bound or E2E gain.

- `2026-09-27T08:18:54Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run468`（design-check）。

- `2026-09-27T08:18:54Z` Run `run468` 记录为 `pass`；正确性为 `not-applicable`。Scoped T to existing FULL pre-replay Host sync to graph replay R0/R1 to conditional hidden/aux gather H and first sampled-hidden consumer U, with capture generation, storage/stream identity, A0-B-A1 and cleanup gates. Native graph output-completion still requires proof; no device or numeric Bound.

- `2026-09-27T08:27:36Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run469`（review）。

- `2026-09-27T08:27:37Z` Run `run469` 记录为 `pass`；正确性为 `not-applicable`。PASS source-only Run468 implementation with mandatory logical stream IDs, no selected npu_stream getter. Exact torch_npu getter drains Host submission queue; Run439 had nine selected conversions/slice, so Run463/464 timing downgraded to queue-drained diagnostic. CANN9.1 gives conditional graph producer→replay-stream event ordering, actual output/loaded-library gates remain.

- `2026-09-27T08:28:47Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run470`（review）。

- `2026-09-27T08:28:47Z` Run `run470` 记录为 `pass`；正确性为 `not-applicable`。Nine selected npu_stream getter reads can drain Host submission queue. Run463/464 raw and lineage valid only for perturbed diagnostic; withdraw V3.18 passive Current timing and Run465 terminal-small priority transfer. Clean no-getter B plus matched controls needed; finite Bounds unchanged.

- `2026-09-27T08:29:43Z` 暂存知识变化 `PK-052`：Installed torch_npu Stream.npu_stream getter can drain the Host task queue; Run439 selected-path checks call it nine times/slice. Run463/464 timings are queue-drained diagnostic only; use direct logical stream identity and a negative getter mock before clean Bound capture.

- `2026-09-27T08:33:32Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run471`（simulation）。

- `2026-09-27T08:33:32Z` Run `run471` 记录为 `pass`；正确性为 `not-applicable`。V3.19 preserves Run463 intervals as queue-drained diagnostic only, disables uninstrumented Current/priority transfer, adds conditional Target graph frontier, and keeps all finite Bound endpoints null and19 proof nodes false.

- `2026-09-27T08:33:32Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run472`（design-check）。

- `2026-09-27T08:33:32Z` Run `run472` 记录为 `pass`；正确性为 `not-applicable`。Versioned helper/patcher/validator use direct immutable stream identity; mock npu_stream property raises, full scoped chain passes with zero getter calls. Patcher check, py_compile, runtime and validator CPU mocks exit0. No install, controller, service or NPU run.

- `2026-09-27T08:33:32Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run473`（review）。

- `2026-09-27T08:33:32Z` Run `run473` 记录为 `pass`；正确性为 `not-applicable`。ACCEPT:25 CPU checks, byte-identical regeneration, all19 proof nodes false and finite endpoints null; Run467 semantic timing applicability superseded, arithmetic retained as perturbed diagnostic.

- `2026-09-27T08:40:37Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run475`（benchmark）。

- `2026-09-27T08:42:07Z` Run `run475` 记录为 `invalid`；正确性为 `invalid`。Preflight global process-count gate invalid: 91 persistent container helper processes; no source install, service, NPU or benchmark. Fresh retry after focused review.

- `2026-09-27T08:42:13Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run476`（review）。

- `2026-09-27T08:43:18Z` Run `run476` 记录为 `pass`；正确性为 `not-applicable`。Astra controller-only PASS after Run475 preflight: removed unrelated total-process threshold, retained explicit live-service/client/NPU/Host gates, new paths fresh; no workload.

- `2026-09-27T08:43:18Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run477`（benchmark）。

- `2026-09-27T08:54:54Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run479`（review）。

- `2026-09-27T08:54:54Z` Run `run479` 记录为 `pass`；正确性为 `not-applicable`。Independent Bound review: one fresh retained wo_a group gives only class-conditional 8,388,608 BF16 conventional ops; exact-board C-plus and fresh external-output lineage remain open. No finite strict endpoint.

- `2026-09-27T08:59:03Z` Run `run477` 记录为 `pass`；正确性为 `pass`。No-getter clean terminal B2 admitted: exact60 correct responses, 40 all-rank c64 slices, post-stop/restore/SHA gates exit0. Instrumented P-C1 median0.26736ms; no uninstrumented cost or finite Bound.

- `2026-09-27T08:59:09Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run478`（design-check）。

- `2026-09-27T09:04:08Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run480`（review）。

- `2026-09-27T09:04:08Z` Run `run480` 记录为 `pass`；正确性为 `not-applicable`。Independent diagnostic admission PASS; P-C1 median267.36us with no-getter, structural DAG only, no numerical Bound; old/new trajectories differ.

- `2026-09-27T09:07:04Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run481`（review）。

- `2026-09-27T09:07:04Z` Run `run481` 记录为 `invalid`；正确性为 `invalid`。Run478 preflight FAIL: non-sync replay helper unbound, disabled capture side effect, wrong remaining gate, insufficient lineage/runtime/cleanup admission. No patch/service/NPU.

- `2026-09-27T09:07:04Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run482`（design-check）。

- `2026-09-27T09:11:10Z` Run `run482` 记录为 `pass`；正确性为 `not-applicable`。V3.20 calibrates clean no-getter terminal observation and Run479 hardware audit without finite endpoint promotion; 19 proof nodes false.

- `2026-09-27T09:11:10Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run483`（review）。

- `2026-09-27T09:11:11Z` Run `run483` 记录为 `pass`；正确性为 `not-applicable`。Independent V3.20 PASS: six input/40 raw hashes, interval recompute, ten negative controls; Current571.681, all numeric endpoints null.

- `2026-09-27T09:28:42Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run484`（design-check）。

- `2026-09-27T09:28:48Z` Run `run484` 记录为 `pass`；正确性为 `not-applicable`。Source-only startup c96 FULL capture repair, six-file check, CPU branch/cleanup negatives and controller syntax pass; no service/NPU run; Astra independent preflight still required before launch.

- `2026-09-27T09:28:54Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run485`（review）。

- `2026-09-27T09:31:06Z` Run `run485` 记录为 `invalid`；正确性为 `invalid`。Astra preflight FAIL: hidden view/timing, Runtime staged count, six-source restore SHA manifests, selected graph-update source hash read; no service/NPU.

- `2026-09-27T09:32:51Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run486`（review）。

- `2026-09-27T09:36:05Z` Run `run486` 记录为 `pass`；正确性为 `not-applicable`。Astra High source-only preflight PASS for guarded B launch: 33 validator negatives, runtime/cleanup CPU, live six-source check and controller syntax; no Bound promotion.

- `2026-09-27T09:36:25Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run487`（profile）。

- `2026-09-27T09:51:37Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run488`（review）。

- `2026-09-27T09:51:48Z` Run `run487` 记录为 `pass`；正确性为 `pass`。Guarded Target FULL frontier B acquisition exit0: exact60 HTTP200/1024, 40 capture+Runtime all8, final admission and six-source restore pass. Local event intervals remain instrumented diagnostic; independent Bound review pending.

- `2026-09-27T09:58:03Z` Run `run488` 记录为 `pass`；正确性为 `not-applicable`。Astra High independent Run487 diagnostic admission PASS: 86 hashes, 40 local slices, 60 POST, source restore; replay median45.986019ms is instrumented caller-stream envelope only; all finite Bounds null.

- `2026-09-27T09:59:20Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run489`（design-check）。

- `2026-09-27T09:59:25Z` Run `run489` 记录为 `pass`；正确性为 `not-applicable`。V3.21 pins Run487/488 conditional Target frontier, keeps all strict/practical finite Bounds null and Current Formal571.681; independent model audit pending.

- `2026-09-27T09:59:38Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run490`（review）。

- `2026-09-27T10:02:23Z` Run `run490` 记录为 `pass`；正确性为 `not-applicable`。Astra High V3.21 ACCEPT: byte-identical rebuild, 25 CPU checks, 86 raw pins, 19 proof nodes false; all numeric Bound endpoints null, Current571.681.

- `2026-09-27T10:02:50Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run491`（test）。

- `2026-09-27T10:03:30Z` Run `run491` 记录为 `invalid`；正确性为 `invalid`。One-card debug_dump probe failed before container/NPU action because evidence/run491 output directory was absent for host redirection; no device workload.

- `2026-09-27T10:03:51Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run492`（test）。

- `2026-09-27T10:04:37Z` Run `run492` 记录为 `pass`；正确性为 `pass`。Installed torch_npu2.10.0.post4 NPUGraph.debug_dump succeeds after a tiny one-card graph; JSON exposes stream/task IDs, task types and kernel argument address text; output parity passes. This is method capability, not Extreme producer proof.

- `2026-09-27T10:05:17Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run493`（test）。

- `2026-09-27T10:05:55Z` Run `run493` 记录为 `pass`；正确性为 `pass`。Installed CANN9.1 graph JSON records two streams and EVENT_RECORD/WAIT/RESET task IDs for a correct tiny cross-stream graph; method supports explicit child-join reconstruction, but production output writers remain unproved.

- `2026-09-27T10:18:16Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run495`（design-check）。

- `2026-09-27T10:18:22Z` Run `run495` 记录为 `pass`；正确性为 `not-applicable`。Astra High source-only design: active-slot first sampled token equals Target argmax first position; external retained/publication/freshness and exact-board C_plus remain open; no numeric endpoint

- `2026-09-27T10:20:06Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run497`（test）。

- `2026-09-27T10:20:11Z` Run `run497` 记录为 `pass`；正确性为 `not-applicable`。CPU pinned-source 128/128 first-position identity and active/count/clip negative controls pass. External publication, freshness, conventional-class necessity and C_plus unproved; no numeric Bound promotion

- `2026-09-27T10:21:57Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run496`（review）。

- `2026-09-27T10:22:03Z` Run `run496` 记录为 `fail`；正确性为 `invalid`。Independent source-only preflight FAIL: serving patched SHA truncated, graph task IDs/schema too weak, actual selected update backend not certified; Run494 launch blocked until repaired and re-reviewed
