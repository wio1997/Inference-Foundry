# Run592 independent final review

2026-09-28. **PASS for the scoped historical instrumented native interval statistics. No direct profile→cohort/request/cycle identity certificate, no same-generation Run589 semantics, and no Scheduling/Resource/Product numerical Bound is admitted.** No NPU/service execution or source modification was performed. Only CPU parsing/hashing/import of pure reducer functions; reducer main was not executed.

## Reviewed artifacts and reproducibility

- scripts/loop081_historical_slice_run592.py SHA256: bdf0b1a63de590f7a5adc032b35e843ef830eaab6886af42a982a7ad7be1db35.
- run592/summary.json SHA256: 20c314fdf00dc66e45e9e0491f7ec3a67f8e549681d608fe4863f0aa4107f172.
- All32 source_sha256 entries in summary were independently rehashed against existing bytes and matched.
- Imported select_events/reduce_events/crosscheck_task_time/negative_gates without calling main. Recomputed16 rows from original trace and matched every task time and interval field. All4 negative cases rejected on all8 ranks.
- Independently cross-checked **all8 roles**, not only copy, against task_time.csv start/stop after matching physical stream/task and sorted occurrence. All have exactly2 matching rows per selected profile. Maximum discrepancy is0.0005µs on every rank, consistent with export rounding. This is agreement between two views of the same profiler source, not independent hardware measurement.
- Independently checked all8 kernel_details: Model45/s1/t40 is MatMulV2 with output[96,4096]; Model45/s0/t13 is hcom_reduceScatter_; Model45/s1/t44 is HcPost. These auxiliary source hashes are recorded below.
- Recomputed all reported medians using Decimal and checked min/max. No hidden NPU action or profiling rerun was needed.

## Latest-session/window/Runtime challenge

The script sorts full directory strings; in general this is not chronological if a rank has multiple worker PIDs. For these actual files I independently verified each rank's five directories have one common PID, profiler_info rank_id matches rank, and collectionTimeEnd is strictly increasing in sorted order. All five profiler_info files per rank were inspected. Only the selected final session's collectionTimeEnd is inside that rank's final rankN_window.json interval. Selected native slice times also lie inside that interval, as the reducer checks.

Selected PID/session timestamp by rank:
0:160893/20260925171535454;1:160934/20260925171535454;2:161001/20260925171535455;3:161069/20260925171535457;4:161138/20260925171535455;5:161207/20260925171535454;6:161276/20260925171535452;7:161345/20260925171535452.

The profiler info confirms Level1 with ACL_AICORE_MEMORY_ACCESS, CPU+NPU collection. Sampled installed collection metadata records torch_npu2.10.0.post4/CANN9.1.0. These are real profiling sessions, not Graph debug_dump synthetic timing.

Every rank's cohort5 Runtime record agrees on the same12 request IDs and309 cycles, pass=true, output counts1024×12. The final window says first_cycle64,count2. **Neither profiler session nor window contains direct cohort/request IDs.** Runtime cohort5 has no profile trace_id or window timestamp reference. Therefore the preserved statement “fifth of five chronologically sorted profiles; no per-profile cohort ID” is appropriate. It is evidence for an ordinal historical association, not a direct join certifying that each occurrence is exactly cohort5 cycle64/65.

No actual wrong latest-session selection was found. A future malformed dataset with a missing earlier session plus an extra unrelated one, another PID, or another cohort using the same cycle window could evade the script. Exact5 directory count and temporal containment do not prove the missing semantic join. Keep “latest profile/window per rank” as the primary scope; do not shorten it to “same-run cohort5 fixed-W0 measurement.”

Wall-like profiler timestamps can support broad same-rank session containment using the export's time mapping. This does not prove cross-device clock accuracy at microsecond scale. No cross-rank arrival subtraction is admitted.

## Native task identity and occurrence pairing

The trace selection uses Model Id, Physic Stream Id and Task Id together:
- partial candidate45/1/40; producer record45/1/41;
- producer wait45/0/12; RS candidate45/0/13; return record45/0/15;
- return wait45/1/42; copy45/1/43; HcPost candidate45/1/44.

This avoids the real collision with stream99/task43 QuantBatchMatmul. task_time lacks Model ID; the matching trace provides it, and in these selected profiles each physical-stream/task match has exactly two rows. No ambiguous third model instance was found for the audited8 roles. The copy is a trace SDMA_SQE/MEMCPY_ASYNC interval, not a kernel_details TensorMove substitute.

Two complete chronological neighborhoods per rank are well separated. In addition to the reducer's weaker event-start checks, I verified for all16:
P_end ≤ recordP_start ≤ recordP_end ≤ waitP_end ≤ RS_start,
and RS_end ≤ recordR_start ≤ recordR_end ≤ waitR_end ≤ copy_start.
The wait tasks begin earlier, as expected; their entire duration is not communication latency.

This validates the observed local order and native timing neighborhood. The profile does not export a typed storage or source label for these tasks. Matching Run589's numeric tuple/order/shapes is explicitly a cross-run hypothesis. Native event task positions do not independently certify underlying event IDs or source-level semantic inevitability. Source/storage lineage, exact HCCL payload and current layer0 identity must not be promoted from this historical match.

connection_id, Batch Id and numeric Model ID can repeat across replays/generations; they are not used as unique dynamic replay IDs here. Sorted occurrence pairing is justified for this narrow two-neighborhood observation; it is not a general replay correlation solution. Missing/boundary-truncated occurrences would fail exact-two checks, but pathological paired omissions/replacements still require stronger complete-model/host correlation for semantic cycle claims.

## Statistics admitted

| Historical local interval | min µs | median µs | max µs |
|---|---:|---:|---:|
| partial task start→copy end |30.320|31.7695|33.441|
| partial task end→copy end |15.780|17.32975|18.440|
| partial task end→RS start |0.580|1.13025|1.640|
| RS end→copy start |0.6395|1.06025|1.9795|
| copy end→HcPost candidate start |0.020|0.020|0.0405|

There are16 rank-occurrence observations, only2 temporal occurrences per rank. Pooled medians are descriptive statistics, not16 independent repeated workloads or a confidence interval for tails. Per-rank maxima are descriptive; taking a maximum does not reconstruct a globally aligned critical path.

Summary's 1.1302500000000002 is only float-formatting noise; Decimal median is1.13025. The quoted20ns copy→HcPost gap is near task dispatch/timestamp granularity, not proof of a physically exact20ns minimum. The numerical source has genuine precision but unknown instrumentation bias.

No kernel-time sum substitutes for interval arithmetic. These values are actual profiled task endpoint differences and include instrumented scheduling behavior. Level1 counter collection and the historical synchronization/configuration prevent promotion to unprofiled Current; profile overhead cannot be removed by subtracting a scalar.

## Validator blind spots and requested scope

No blocker for the existing narrow summary was found. Before reusing the reducer as a general certificate, strengthen:
1. Session identity: parse timestamps/PIDs, verify profiler_info rank/config/end containment, and save explicit direct cohort/request/replay IDs when available. The current script only records32 input hashes and omits profiler_info/kernel_details and itself; this review pins those missing evidence sources.
2. Types/shapes: current select_events tests name fragments and ph=X, not Task Type/core or tensor shapes. The actual sources pass additional checks; malformed type/shape drift would be missed.
3. Event ordering: check record/wait ends and nonnegative durations, not only loose starts. Actual all16 passed stronger order.
4. Cross-export checks: currently only copy is checked; actual all roles pass. Preserve model/stream/task/start identities and document rounding tolerance. Do not allow an unrelated model's identical task tuple to pass by sorted position.
5. Negative tests: wrong_model/wrong_stream tests manually invoke a local assertion, rather than feeding mutated raw events through select_events and all admission gates. They demonstrate narrow expected rejections, not broad robustness. Add missing session, mixed PID, wrong window/cohort, duplicate occurrence, wrong Task Type/shape and task_time mismatch only if turning this reducer into a reusable gate. These are validator hardening needs, not evidence that present measurements are false.

## Bound interpretation and next action

The summary's historical_instrumented_current_slice_only label, cross-run hypothesis warning, null strict bounds and null unmarked Current are correct. fixed_W0_equivalence_to_Run99=false should be read as “not certified,” not proof of semantic inequality. No measured acceptance/output trajectory is being optimized or altered.

Run592 reduces uncertainty about the existence and scale of a historical native scheduling cut. The small copy→HCpost gap contradicts a hypothesis of a large idle gap at that specific profiled adjacency. It does not establish that RS duration is unavoidable or removable, that this cut is a dominant E2E bottleneck, or that layer/cycle scheduling has reached a bound. Neither31.7695µs nor17.32975µs is a lower bound for another legal runtime.

Use these values as conditional Current DAG cost evidence. Keep Resource work/traffic and attainable compute/HBM/HCCL work progressing independently. A fresh same-generation Level0 acquisition is justified only when a decision requires current fixed-W0 identity, lower perturbation or calibrated arrival evidence absent here; do not rerun solely because an identity-only certificate is possible.

Current Formal571.681tok/s is unchanged. No finite Scheduling/Product endpoint or numeric Current→Bound gap is promoted.

## Supplemental source hashes

The following hashes were read during this independent audit; they supplement, rather than replace, the32 hashes already pinned by summary.

```json
{
  "evidence/20260926_loop060_resource/run246/profile/rank0_160893_20260925171535454_ascend_pt/profiler_info_0.json": "fb2ce29a1e561b821a0f0e24320d76c9c57dac0a5a7db49aac3f560737be1b0e",
  "evidence/20260926_loop060_resource/run246/profile/rank0_160893_20260925171535454_ascend_pt/profiler_metadata.json": "bcb4aafa4f0f8f057c567c3f207aa68bb3ec169593f2727c46971ea18191f0a8",
  "evidence/20260926_loop060_resource/run246/profile/rank0_160893_20260925171535454_ascend_pt/ASCEND_PROFILER_OUTPUT/kernel_details.csv": "f99d06e995c60e821e0587716c6d6e844d3afc20693d1fc7d9ff29f953d0bc1b",
  "evidence/20260926_loop060_resource/run246/profile/rank1_160934_20260925171535454_ascend_pt/profiler_info_1.json": "94b79b592422fc19850d2e0579321409ab854ad6e4ff6875f4f161b8eecf8ab0",
  "evidence/20260926_loop060_resource/run246/profile/rank1_160934_20260925171535454_ascend_pt/profiler_metadata.json": "ecd019468ba83197b531a3f6b53f91699c6bd9c5a8091d8d544dbba4375922af",
  "evidence/20260926_loop060_resource/run246/profile/rank1_160934_20260925171535454_ascend_pt/ASCEND_PROFILER_OUTPUT/kernel_details.csv": "c082c9c25bd4e4e954428a85d5cdcb1fce40b0f2afbbf6c3fce11c518f3872db",
  "evidence/20260926_loop060_resource/run246/profile/rank2_161001_20260925171535455_ascend_pt/profiler_info_2.json": "e69eadc7e6624e6e16cd2f0623679e219c990b53d5585872c729b2a539157065",
  "evidence/20260926_loop060_resource/run246/profile/rank2_161001_20260925171535455_ascend_pt/profiler_metadata.json": "6e5528375307663629db8b3de981a5c46824ac632a07d499340a7374070813c2",
  "evidence/20260926_loop060_resource/run246/profile/rank2_161001_20260925171535455_ascend_pt/ASCEND_PROFILER_OUTPUT/kernel_details.csv": "7811b665f9d80dff6265a0dd43c1e4320edbceebe0b29b20341efa39b018b9e7",
  "evidence/20260926_loop060_resource/run246/profile/rank3_161069_20260925171535457_ascend_pt/profiler_info_3.json": "fc145c6e629e421ba409d39ecaf80cba44a6e8534c7b77c1f0973072e4525e91",
  "evidence/20260926_loop060_resource/run246/profile/rank3_161069_20260925171535457_ascend_pt/profiler_metadata.json": "29d1d23e98cef0488dd6b9ab0ee58e3aba5400ded323b267e46596bc7645c575",
  "evidence/20260926_loop060_resource/run246/profile/rank3_161069_20260925171535457_ascend_pt/ASCEND_PROFILER_OUTPUT/kernel_details.csv": "6bdecfe44b7217607d5014a6d795437dedeecd036e9420bd7bc80f10a62dbaf2",
  "evidence/20260926_loop060_resource/run246/profile/rank4_161138_20260925171535455_ascend_pt/profiler_info_4.json": "14d7c235744dd7d8418312796af22f14dcfcfe53df29ce69e20f76a74b03b205",
  "evidence/20260926_loop060_resource/run246/profile/rank4_161138_20260925171535455_ascend_pt/profiler_metadata.json": "53d91ca2a697c9de21d33f9633238b71bb09eb21de96025228b0aa78ebd62b15",
  "evidence/20260926_loop060_resource/run246/profile/rank4_161138_20260925171535455_ascend_pt/ASCEND_PROFILER_OUTPUT/kernel_details.csv": "4642785ff387530cf42c923db669ae86933036eff9b912e797f61e84e7956bd9",
  "evidence/20260926_loop060_resource/run246/profile/rank5_161207_20260925171535454_ascend_pt/profiler_info_5.json": "35ee7130d8e4dd1f8c762aa10260406e2171b5433588f63512762a336e7ce581",
  "evidence/20260926_loop060_resource/run246/profile/rank5_161207_20260925171535454_ascend_pt/profiler_metadata.json": "1a889dc6502c4b96adb2e202750b3834bc09746e0bcef8da56bd408184f8fef9",
  "evidence/20260926_loop060_resource/run246/profile/rank5_161207_20260925171535454_ascend_pt/ASCEND_PROFILER_OUTPUT/kernel_details.csv": "83b89eeabe9fd7f7aa884f348cf26f5457c6f82933b1d7d7e8a9b060d3c93376",
  "evidence/20260926_loop060_resource/run246/profile/rank6_161276_20260925171535452_ascend_pt/profiler_info_6.json": "8569b8a53de60ece28172ad15e057b392b4bfaf98a4729a917602b43cd163124",
  "evidence/20260926_loop060_resource/run246/profile/rank6_161276_20260925171535452_ascend_pt/profiler_metadata.json": "478978cd3fca41a8574e78d7305cbaf8630298cf62f93274a4f3bcb67b117003",
  "evidence/20260926_loop060_resource/run246/profile/rank6_161276_20260925171535452_ascend_pt/ASCEND_PROFILER_OUTPUT/kernel_details.csv": "066f3c132a46bf79e37c47094d27d5eac03def98a1df47a7ddfd477b0094d1eb",
  "evidence/20260926_loop060_resource/run246/profile/rank7_161345_20260925171535452_ascend_pt/profiler_info_7.json": "7f19ffe4e4d98275122e1abf61b8ddd0038933188e64127216efa7834c864fb7",
  "evidence/20260926_loop060_resource/run246/profile/rank7_161345_20260925171535452_ascend_pt/profiler_metadata.json": "f97e4d30182bdb7cc7ab9e9bed90a9358597655fe42ace531e16b8631744c172",
  "evidence/20260926_loop060_resource/run246/profile/rank7_161345_20260925171535452_ascend_pt/ASCEND_PROFILER_OUTPUT/kernel_details.csv": "ad31f21e735d4c333089352d5b895dfb9bb5abf3831372758de3f3c1d71cd301"
}
```
