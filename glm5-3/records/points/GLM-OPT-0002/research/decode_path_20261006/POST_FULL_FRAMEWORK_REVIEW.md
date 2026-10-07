# Post-FULL framework Challenger review

2026-10-07. Offline source/raw review under AGENTS trigger 6; no service, model request, parser rerun, patch or frozen artifact mutation. H6/H5 retained, target FULL; H11 remains off. Route recommendation is not PERF_KEEP.

## Facts personally checked

Read the frontier index/rank summaries and original-index-preserving period-3 event exports for ranks 0, 13, 15; actual `mtp_graph_sources/model_runner_v1.py`, `mtp_graph_candidate_v5/llm_base_proposer.py`, actual flattened `prepare_finalize.py`, and the historical `local_prepare/local_prepare.patch`.

* Independent non-wait hardware interval union for period 3: rank0 52,317.75 / 55,418.25 us; rank13 52,440.25 / 55,509.50 us; rank15 52,298.75 / 55,419.75 us. Maximum interior non-wait gap respectively 137 / 137 / 136.75 us. This does not establish a large removable host starvation interval. Union is occupancy, not a complete cross-stream critical DAG.
* Each selected rank has **152 PadV3 tasks plus 152 MemSet tasks**, not merely 152 total hardware tasks. PadV3 inclusive durations are 2,755.8625 / 2,447.5150 / 2,367.7365 us; MemSet sums are about 1,292–1,294 us. These are separate observed task sums, not savings or additive E2E estimates. Captured target tasks need capture-time source association; only two current CPU ConstantPadNd calls appear in this replay period.
* Each selected rank has four `aclnnIndex_SliceAiCore_Slice` tasks on hardware stream47, totaling 1,333.594 / 1,325.2335 / 1,323.494 us. Rank15 raw indices 408250/408256/408818/408824 link to Node launches through connections 86553/86561/88127/88137. Those launches occur long before their device execution. This is concrete device work; it is not proof that four tasks equal four Python indexing expressions.
* `PrepareAndFinalizeWithMC2.prepare` pads the full hidden/router tensors before selecting one TP shard (292–305); the common global padded shape remains part of finalize's contract. The historical local preparation patch preserves that shape and isolates valid rows with clone. It still pads on ranks whose local shard is empty: it does **not** remove every padding launch, and the observed rank0–rank15 difference warns against pricing benefit from rank0 alone.
* Proposer lines1163–1166 compute `used_update_positions`; its only consumption is in the subsequent `range(1, K)` update loop. K1 has no consumer. This dead work is a source-proven narrow removal opportunity. It is not yet mapped to one of the four expensive Index_Slice tasks. First-pass hidden selection (1415) is live. The further hidden/position selections at1568–1571 are after the K1 early return and cannot explain current K1 tasks.
* Target runner2507 selects hidden rows before logits. Proposer1415 selects draft hidden rows before logits. Their selection semantics are needed; neither selection can simply disappear. General advanced indexing permits negative indices; replacing it with index_select without establishing nonnegative bounds changes behavior.

Run275 INCONCLUSIVE and Run276 profile-off signature mismatch remain intact. This review uses the profile as a same-request execution trace, not an ON/OFF timing calibration. New parser exports do not imply original raw corruption; no parse repetition is justified.

## Gap ranking and decision

1. **Largest remaining removable gap is unknown.** The largest concrete framework device-work family among the supplied candidates is repeated MC2 padding; unnecessary global materialization is source-supported, but local padding still incurs launches on the slow peers. It is the first size-reduction question, not an established largest gain.
2. Index construction/selection is smaller observed work but offers a cleaner bounded semantic question: distinguish dead K1 position gather from the two necessary hidden-row gathers and any additional indexing implementation tasks. This mapping should be completed before choosing a device experiment.
3. Metadata, sampling and next-input host work remain real, but much of their submission overlaps target execution. No supplied current trace establishes a large host wait that can safely be converted into savings by another submission architecture.

**PIVOT** from broad eager-host reconstruction to eliminating proved unnecessary device preparation on the current FULL stack. **KEEP** H6/H5 and current graph baseline. **STOP** treating old Run249 producer starvation, Python inclusive percentages, or a 35ms synchronizing host interval as current removable time. No recommendation here to repeat H4/H8/H9 or reactivate H11.

## Top three code questions (ordered, not concurrent candidates)

1. Which actual ATen enqueue/queue-flow chain owns each of the four Index_Slice tasks, and which is the K1 unused position gather? Complete the existing-trace connection/queue/CPU containment join. For live hidden selection, prove dtype, dimension, nonnegative bounds, duplicate/order behavior and output layout from index producers before considering native index_select. A CPU oracle must include negative indices as a rejection/compatibility case, not silently omit them.
2. Does global-to-local MC2 preparation remove meaningful device work on the peer that determines replay completion, or merely reduce bytes while keeping PadV3/MemSet startup costs? Preserve padded geometry, mask ownership, caller input isolation and captured-buffer lifetime. CPU byte/layout correctness alone cannot answer its device value.
3. After separating required target/MTP arithmetic and communication, which remaining input/sample task lies on the all-rank completion frontier rather than underneath target replay? Use actual replay ordinals and changing latest rank; a fixed rank0 accounting is insufficient.

## Smallest distinguishing evidence and stop list

The next step is **offline linkage of existing Index_Slice connections to producer queue flows and exact source expressions**, alongside captured padding launch identities. No new profile or model request is needed to settle whether the inexpensive semantic dead-gather fix corresponds to an expensive task. If existing export lacks a needed producer link, name that exact missing link; do not infer it from timing order or initiate another broad diagnostic.

A subsequent candidate must be one source-defined change with its own correctness boundary. For local preparation, acknowledge retained per-peer launches; for index_select, retain fallback outside the proved index contract. Do not combine these merely to seek a positive result.

Stop: repeating Run276/parse; ON/OFF calibration from unequal draft counts; summing rank/task inclusive durations as savings; deleting synchronization/fences without a new dependency proof; attributing Event/Notify waits to removable transport; claiming framework work exceeds required compute from operator labels; converting review into promotion.

Most likely misreading: current FULL makes the device busy while host waits, so an old host-bound narrative becomes false even though CPU work remains. Conversely, high occupancy does not prove all device tasks are necessary. The justified new search is redundant preparation inside that occupied interval, with largest actual benefit still unknown.
