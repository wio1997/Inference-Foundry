# Initial PD target — root source and causal closure

2026-10-07. Existing Run276 profiled prefix only; no new NPU request/profile/parser rerun. Retain H6/H5/current target FULL; H11/H12/H13 off. Root read indexed raw and actual source, not only the independent report. New complete-PD gain and final Current remain unproven.

The current FULL graph is a steady qlen2 contract. Full remote-KV scheduler resume recomputes the last prompt token at computed=N-1. Runner rejects the current uniform FULL key under DCP when computed<prompt and/or qlen!=2; actual dispatcher hash matches current installed file and no matching key returns NONE. Prefix75 target one-token MoEs precede the first target replay; one further MoE lies in the first draft phase. This establishes first target eager coverage, not missing MTP implementation. Initial target embedding→ArgMax end is approximately327ms in this profiled tiny request; it is not a universal latency or a savings estimate.

## Actual initial SFA branch, supported by raw rather than a guessed predicate

All79 pre-replay SFA kernel rows have query1×4×512; KV32×128×1×512, rope KV32×128×1×64, block table1×1136. OutputShapes are1×4×512;;, with no LSE outputs. Actual allgatherAivKernel payload294912 bytes occurs79 times. This equals2 compact local blocks×128 tokens×(512+64) elements×BF16 bytes;16 ranks yield32 gathered blocks.

The actual SFA-CP prefill branch `_record_dcp_kv_gather_context` index-selects KV and rope by compact block IDs, concatenates them and gathers the compact view. `apply` then consumes gathered KV with sparse_mode3/return_lse=False. Its decode branch gathers query fragments, attends on local KV and requests LSE for the cross-DCP merge. First-prefix geometry, output and operator sequence support the compact-KV prefill branch; they are incompatible with silently assuming the existing decode graph contract.

Source snapshots sfa_cp17d804… and attention utilityb600a6… match Run283 protected pins. Attention's `split_decodes_and_prefills(...treat_short_extends_as_decodes=False)` can be overridden by `is_pd_decode_recompute_scheduler_enabled()`. The config snapshot's default is false and these native additional-config arguments omit that option. Root has not directly witnessed the helper's runtime return; the **actual branch attribution instead uses the raw kernel/operator contract**. No additional device diagnostic is needed merely to guess that predicate.

## Late host supply is real locally

Independent311 collective joins preserve exact names and per-rank connection→Node→dequeue→enqueue→CPU ownership. Root directly re-read raw comm/node/enqueue indexes for `hcom_allGather__097_38_1` across0/13/15 and verified IDs/timestamps against the independent result.

Rank0/13 device collectives begin at...7842451.167/2456.807; rank15 only enqueues at...7843691.320, starts device at3760.967. Three-rank latest completion3803.960 is42.993us after rank15 starts. Thus late rank15 host issue contributes over1.2ms of peer rendezvous in this operation. It does not prove all waits or any whole-request savings can be deleted, and other13 ranks may determine global completion.

The corresponding rank15 MLA scope begins...7842564.270. Before its gather enqueue, raw CPU sequence includes KV rmsnorm/rope-cache, quant/projection, interleave-rope, then two index-selects at991.75/1010.75us after scope start, cat at1033.75us and `_allgather_base_` at1093.5us. This is the concrete compact-KV producer path, not the54us all-gather Python scope by itself. Moving only that final call or deleting its necessary wait is not a demonstrated fix. Parent CPU scopes overlap children; do not add inclusive durations.

Root initial hardware reduction includes EVENT_WAIT/SDMA omitted by kernel CSV. Rank15 non-wait complement141.8ms versus0/13 approximately5ms remains a local coverage statistic. Different EP useful work and peer waits prevent converting it to global removable time. Initial target~327ms versus steady~55ms cannot be subtracted because branch, query state, communication and profiling differ. Public first-token publication is not aligned to this profiler clock.

## Next code contract to close

The concrete next path is scheduler full-KV resume → initial metadata classification → compact-KV preparation/gather → target runner dispatch. Determine whether a dedicated initial graph can retain the required prefill branch with dynamic compact block identities/counts, persistent addresses, KV writes, indexer, attention, logits/sample and subsequent MTP consumers. An alternative decode-classification optimization must separately prove equivalent prefix visibility/attention merge and correct numerical/output behavior; existing RecomputeScheduler support alone is not sufficient proof for a broader predicate.

A minimal graph patch cannot be defined as adding capture size1 or forcing uniform. Current builder admits only DecodeOnly/SpecDecoding capture states; dynamic prefill compact-KV metadata has a different contract. First audit exact metadata/consumer ownership and graph parameter updates using existing source. Only after this audit choose the smallest code change and follow correctness→matched A/B/A/B→complete standard PD on H6/H5/currentFULL. No new Run or product promotion is admitted by this note.
