# Initial target collective causal review

2026-10-07; independent offline review. No device/service/request action. Inputs: original indexed prefix trace exports rank0/13/15, `initial_target_causal_reduced.json`, actual SFA-CP and attention utility snapshots. Companion `review_initial_collective_join.py/json` records this independent join. Decimal timestamp arithmetic; raw indexes retained.

## What is now established

Joined all311 target-prefix `hcom_*` collectives present on these three ranks by exact collective name. Within each rank: communication connection_id → CANN Node@launch → same-thread enclosing dequeue → correlation_id enqueue → enclosing CPU operator scopes. Every rank supplies311 successful joins; none is paired merely by ordinal.

- Rank15 enqueue starts latest among these three for311/311 collectives.
- Latest device starts: rank15 251, rank0 39, rank13 21. Latest enqueue and latest device rank agree251/311.
- For the251 both-latest-rank15 cases, median latest enqueue-end→device-start is62.169us;244/251 are below200us.
- Across311, median three-rank device-start skew is312.439us; median latest-device-start→latest observed completion is412.011us. Thus this is **not** the old all-rank Run249 regime where nearly every collective completed immediately after the final arrival. Other13 ranks and necessary work remain material unknowns.
- Group-name `097` contains78 joined attention all-gathers. Latest-device enqueue-end→start is below200us for all78 (median62.895us). Group identity should not be silently inferred from its numeric name; actual CPU scopes are MLA and HcclAllgatherBase.

Decisive example: `hcom_allGather__097_38_1` (all times us relative1791367887840000):

| Rank | Host enqueue start | Device start | Device end | Raw comm / enqueue index |
|---|---:|---:|---:|---|
| 0 | -406.260 | 2451.167 | 3800.040 | 134645 / 106837 |
| 13 | 1963.860 | 2456.807 | 3803.960 | 139139 / 106837 |
| 15 | 3691.320 | 3760.967 | 3794.906 | 134461 / 106837 |

Rank0/13 enter the collective about1.24ms **before rank15 even enqueues it**. Rank15 begins device execution69us after enqueue-end, and all three complete within42.993us of its device start. This is direct evidence that late host issue by rank15 contributes to peer waiting for this operation. It is not merely an inference from a long EVENT_WAIT. The producing CPU scope is `vllm::mla_forward`; the enclosing `_allgather_base_` scope itself is only about54–57us. Removing the collective or its wait is not the indicated fix.

## What the join does not establish

Late issue is downstream of preceding execution and host activity. It can arise from serial eager preparation, scheduler off-CPU time, necessary synchronization, or rank-dependent preceding MoE work. The join does not independently apportion those causes. Rank15 may be the latest of these three without being the global critical rank. Do not sum collective skews, overlapping communication waits or141.8ms nonwait complement into savings.

Concrete local supply example from the raw-backed reducer: rank15 `_triton_rope_siso` begins at1791367887686583.967 after a446.711us all-stream nonwait complement. Its enqueue begins1791367887686523.160, launch node ends1791367887686549.820. Most of that interval precedes enqueue, inside an MLA CPU scope. This supports a local eager supply delay, but does not prove the unexported interval is all CPU computation or that eliminating it advances complete-PD output. A second example before MC2 dispatch is similarly mixed; H6's already-cached capability must not be recredited.

Root's exact hardware union now includes SDMA/PCIE and EVENT_WAIT omitted by the previous kernel CSV. That appropriately reduces rank15's uncovered interval from the CSV-only163.112ms to approximately141.8ms. EVENT_WAIT is a dependency interval; COMMUNICATION includes rendezvous. Both must remain distinct from a host-gap or useful-device-work budget.

## Source boundary still essential

Actual SFA-CP builder lines818–836 calls `split_decodes_and_prefills(...treat_short_extends_as_decodes=False)` and constructs compact KV-gather metadata only when `num_prefills>0`. However actual `attention/utils.py:427` explicitly overrides the short-extend policy when `is_pd_decode_recompute_scheduler_enabled()` is true. Query length1 and computed=N−1 alone therefore **do not establish the runtime attention branch**. `_record_dcp_kv_gather_context` at SFA-CP923 onward is conditional on that metadata. Close the actual configuration/function predicate and observed operator sequence before claiming a first-step prefill→decode substitution or forced-uniform graph is safe.

Supported bottleneck statement: initial target execution has repeated eager host issue and observable peer rendezvous caused partly by late issue. Not yet supported: a particular removable Python function dominates globally, all initial latency is submission, or the current uniform qlen2 graph can legally replace this path.

## Recommendation

Continue the existing first-step source/trace inquiry, not another toggle. Highest-value next closure is actual initial SFA classification plus the predecessor chain feeding selected late attention all-gathers; the current join already supplies concrete operation/rank anchors. A specialized initial-step submission/capture path is a plausible architecture question only after those consumers are closed. Preserve H6/H5/current FULL and all synchronization/error semantics; no new Run follows automatically from this review. Public first-token timing remains unaligned with profiler time as documented in the previous review.
