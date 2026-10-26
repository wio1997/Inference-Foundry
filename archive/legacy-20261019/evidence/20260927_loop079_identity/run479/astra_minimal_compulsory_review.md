# Run479 — minimal compulsory subset / capacity review

2026-09-27. Independent Astra High review requested by Sol. Scope: frozen DeepSeek V4 Flash W4A8, DSpark7, DP1×TP8, eight 910B3, warmed 48×32K→1024, c12. Read-only investigation while Run477 is active: no service, NPU query/workload, framework import, source modification or shared-state/model edit. Only this review artifact is written. Read V3.19 Run471, Run423, Run449 and Run462 and current ACHIEVABLE_BOUND. Current Formal remains Run99 median571.681 tok/s.

## Verdict

No fully certified positive W_minus / matching C_plus pair exists in the inspected evidence. No numerical TPS endpoint is supplied. This is not a proof that a finite ceiling cannot exist. A complete model census and full scheduling DAG are NOT prerequisites for first finiteness.

For the whole timed Product interval I, let N=49,152 external tokens. If every implementation in a clearly declared class must perform W_minus>0 units inside I, and all resources permitted to perform those units have aggregate service bounded by C_plus, then T_I >= W_minus/C_plus and TPS <= N*C_plus/W_minus. Omitted work and perfect overlap only weaken this relaxation. The numerator and rate must share units and implementation scope.

The request for one additional performance measurement that by itself closes the current strict ceiling is not satisfiable: both universal necessity and maximum capacity remain unproved. A finite sample of lower observed rates cannot establish a maximum, and a trace of one implementation cannot prove an arithmetic-complexity lower bound over alternative algorithms.

## Candidate audit

| Candidate W_minus | What is actually supported | Why a strict universal pair still fails |
| --- | --- | --- |
| One fresh wo_a group at one Target layer and one retained prediction row | Conventional BF16 dense count 2×4096×1024 = 8,388,608 operations; Run423/449 identify current shape/dtype | This is the smallest useful source-backed projection candidate, not the mathematically smallest unavoidable operation. Need a fresh in-window witness and a declared conventional dense evaluation class. Factoring, exact reuse, changed intermediate representation or decision-only evaluation may avoid that count in a broader class. |
| One routed expert pair or one full retained Target row | Conventional counts 50,331,648 and12,985,565,184 operations | Input-dependent routes, intermediate reuse and alternative exact evaluation; integer/BF16 mixed work does not match the BF16 Cube capacity expression. Full route-row closure is unnecessary if the smaller BF16 subset is selected. |
| Fixed seven DSpark bias/proposer steps | Current sequential token-feedback implementation and repeated matrix observations | DSpark7 specifies draft semantics/width, not seven compulsory physical reads or mandatory conventional operations. Exact table/reuse or additional parallel work may alter the resource chain. A feedback edge is not itself a positive hardware service-time certificate. |
| Active weights, KV or reported main-memory bytes | Storage sets and current access counters | Residency at I start, hierarchy placement, recomputation and equivalent representations prevent promotion to compulsory physical HBM transfer. A matched physical controller maximum is also missing. |
| Current HCCL tensor inventory or terminal full logits | Current ABI payload census | Placement, recomputation, sparsity and representation alternatives; Run450 constructs much smaller greedy value/index communication. That construction supplies an upper work volume for an alternative, not a mandatory lower volume. Physical links/cut maximum remain unbound. |
| At least one comparison, input bit or publication byte | Intuitively positive service for a general nonconstant inference service | No exact primitive-to-engine certificate or maximum host/local-HTTP capacity. A worst-case nonconstant-function claim is not a per-instance lower bound for these fixed warmed requests. It does not establish necessary NPU work. |

No generic output-memoization permission is assumed. If output reuse is forbidden by the intended inference contract, state that explicitly in the bound class; this exclusion alone still does not prove dense2MNK mandatory. Conversely, an unseen random challenge would change the frozen workload and cannot silently be used as its numerator witness. Cached/prefilled work completed before I cannot be charged to I. Work after benchmark start but before Runtime handoff can qualify.

## Capacity scope

Run449 binds the installed target to architecture2201,20 Cube engines/card, BF16 support and a configured1800MHz clock. The official2201 ordinary BF16 Mmad table supplies an ideal4096 MAC/Cube-cycle (8192 conventional operations), subject to its archived development-document provenance.

For Cube-only ordinary BF16 evaluation:
C_cube_plus = 8192 × sum_r(20 × f_max,r).
The genuine f_max,r values remain unknown. Configured, rated, current and sampled maximum are not an exhaustive operating maximum. Run462 excludes the inspected DCMI/driver shortcuts; a new GEMM, clock poll, or arbitrary safety multiplier does not repair this.

A strict capacity certificate may be conservative and unattainably large; matching mixed-load throughput is not required. It does require an authoritative upper envelope over allowed clocks, boost/tolerance, issue/completion semantics and applicable board/bin/firmware. The archived preview Cube table also needs an applicability/errata qualification.

If the class allows Vector, Scalar, AICPU or host execution of the counted arithmetic, Cube-only C_plus is too small unless these alternatives are explicitly excluded or conservatively covered. Do not add vec_calc_size=128 without its operation/precision/issue/clock meaning. HBM/HCCL work may be omitted from this one-compute-subset relaxation; their costs need not be added, and their missing maxima do not invalidate a genuinely completed compute pair. If the selected numerator is bytes instead, its own memory/link/host capacity is mandatory.

## One highest-information post-Run477 measurement

Conditional recommendation: acquire ONE original-path, fresh, retained wo_a-group witness during an exact frozen warmed48-request measured pass, with the full warmup outside I. This closes the smallest tractable W clause for the conventional dense BF16 Cube evaluation class; it cannot alone close universal W or missing C_plus.

Select one actual Target prediction row at one late Target layer, one group, then bind:
1. benchmark start/end and request/output ordinal, separately from handoff and diagnostic warmup;
2. actual loaded graph/native branch, layer/group, BF16 shapes/accumulation, input/output storage generation;
3. producing input generation after benchmark start and result-cache provenance, rather than merely a replay timestamp;
4. row transforms through the selected slice to Target ID, acceptance, Scheduler clipping and a retained external output position;
5. eligibility of at least one such group, without multiplying by all49,152 tokens or all43 layers.

Export buffered records after completion; avoid Stream.npu_stream and hot-path queue-draining getters identified by Run469/470. This measurement is a work/provenance witness, not a latency measurement: A0/B/A1 timing is not required merely to count the scoped work, though any passive-scheduling claim would need its own overhead admission. Fail closed on absent lineage or pre-window reuse. An observed consumer relationship does not by itself prove all components are mathematically essential; conventional evaluation is an explicit class assumption.

Before calling the resulting expression a finite strict Cube-class Resource ceiling, obtain the separate exact-board authoritative C_plus statement specified by Run462. No service measurement can replace that document/configuration guarantee. With that certificate and the scoped witness, one projection suffices; more Draft/KV/HCCL censuses are unnecessary for first finiteness. Without it, retain the symbolic expression and null numerical endpoint.

For the unrestricted architecture search, an algorithm-independent positive necessary-work proof remains separate; label the restricted ceiling as conditional and do not use it to rule out legal optimizations.

## Confidence and decision

High: minimal-subset theorem; current absence of a certified pair; configured/rated versus maximum distinction; attained-rate measurements cannot prove C_plus; physical byte/ABI distinctions; exact Product-window requirement.
Medium: usefulness and source binding of the one-group BF16 witness, pending actual retained/freshness capture.
Unestablished: universal dense arithmetic necessity, exact-board all-engine capacity, any finite numerical Resource/Hardware/Product ceiling.

Priority is a two-clause certificate with one small witness and one authoritative capacity guarantee, not repeated broad profiling. If authoritative capacity evidence remains inaccessible, identify that as the strict-ceiling evidence blocker while continuing separately labelled Scheduling/Engineering work.

Sources: repository V3.19 Run471; Run423 astra_minimal_ceiling_review.md; Run449 astra_910b3_cap_review.md and archived manifests; Run462 astra_hardware_certificate_review.md and archived manifests. Parent Sol must review before TaskCtl/model promotion. No state transition, commit or source modification was performed.
