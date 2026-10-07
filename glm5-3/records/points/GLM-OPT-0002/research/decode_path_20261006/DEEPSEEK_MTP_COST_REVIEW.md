# Scoped DeepSeek MTP graph cost review

2026-10-07. Independent offline inspection; no Run/service/device operation. H12 registration contiguous remains the sole active hypothesis. Run277 remains FAILED; Run279 correctness pending. H11 Run273 correctness and Run275 INCONCLUSIVE are separate facts.

## Actual path and reference boundary

Personally read pinned `deepseek_mtp_graph_reference/autoregressive/speculator.py` capture348–380, _prefill460–492 and uniform-metadata hook575–613; actual observer_v6 proposer graph wrapper/contract, staging1138–1236 and K1 return; actual runner input consumer1248–1271, copy2042–2082 and proposal2702–2723.

Actual GLM path: sample_tokens/propose closure → proposer first-pass inputs → persistent slot/sequence/query staging → SFA metadata builder → `_run_merged_draft` (model, logits, sample) → runner `_draft_token_ids` → optional host copy and next-step DCP input preparation. H11 already captures the merged model/logits/sample body. The V2 reference captures `_prefill`, returns from capture for K1, and only captures additional decode steps for K>1. Consequently “use DeepSeek's full draft graph” is not by itself a new removal beyond H11.

One qualification to the reference summary: V2 has metadata preparation inside its wrapper and a specific uniform decode hook whose comment explains position-0 RoPE being baked into replay if positions are missing. That establishes a real dynamic-input hazard, not a drop-in optimization. It concerns further speculative decode metadata and does not prove current GLM K1 can simply move its full Python builder inside capture. The reference's own superclass _prefill/model-state lifecycle is not fully present in the three-file snapshot; output ownership equivalence remains unknown.

## Ranked removal questions

1. **H12 remains the best concrete redundant device-preparation question.** Existing current-FULL source/trace links full-table RoPE materialization to both target and ordinary MTP metadata calls. Initializing dense immutable tables removes the cause without changing graph scope. Pending correctness/gain is not assumed.
2. **K1 unused position gather is genuinely dead, but unpriced and smaller in scope.** Proposer1166 is only consumed by range(1,K) metadata updates. It is distinct from the four slow RoPE Slice tasks. Moving this gather inside the actually executed update branch is semantically defensible; no evidence establishes it as a high-value new performance candidate. Keep parked while H12 is active.
3. **Output clone / metadata copies / async draft D2H are not presently proved redundant.** They require different consumer proofs below, and no current trace prices them as the largest gap.

## Concrete claims challenged

**Owned result.clone (proposer706).** It runs only in eligible H11 FULL. Current H11off does not pay it; removing it cannot improve the present baseline. The runner may issue side-stream D2H for structured/history/PP cases. A side-stream wait on main ensures the copy starts after production; it does not ensure the next graph overwrite waits for copy completion. Therefore unconditional clone removal is wrong without a reverse dependency or an owned ring buffer. Ordinary greedy async/no-history/PP1 skips D2H, and next-step DCP reads `_draft_token_ids` before later draft replay on the ordered execution path, suggesting a possible narrower borrowed-output contract. But every consumer and output retention must be closed first; borrowed-buffer generations CPU tests would establish semantics, not its tiny tensor copy's performance importance. Do not introduce this as a new H11 experiment based solely on graph support elsewhere.

**Async draft ID CPU copy (runner2042).** The implementation already early-returns when async is true and there are no structured requests, output-token-history requirements, or PP>1. It still updates prev_num_spec_tokens before returning; that is required stride bookkeeping. Thus the proposed universal copy removal either changes nothing in the ordinary eligible path or breaks real consumers. Exact current workload's history-list condition is not independently established by a runner snapshot in this review, so do not fill this cost with zero; inspect existing workload/trace if needed, not another model request. DCP next-input generation explicitly consumes device draft IDs at1266.

**Metadata staging.** slot_mapping/seq_lens/query_start_loc are rebound to owned persistent per-step buffers; zero/-1 tails handle changing padding. Run272's demonstrated transient DCP table pointer failure already disproves deleting staging on shape equality alone. Run273's pre-builder stage fixes the actual captured view. Same pointer/stride self-copy elision is already implemented for the scoped table. Further removal requires alias/lifetime proof for each field and changing-batch fallback; frozen or unchanged values cannot be assumed. SFA metadata ordinary build means both target/MTP update shared persistent cos/sin outputs; the earlier “draft uses noncached outputs” inference was wrong and stays corrected.

## Decision

KEEP DeepSeek as a concrete design reference; STOP an unchanged H11 toggle retry or a broad rewrite justified by graph availability. No new high-value minimal patch beyond active H12 is established by this review. The K1 dead gather is the only additional clear local semantic deletion found, but its current critical-path contribution is unknown. Whole-step largest removable gap also remains unknown.

Smallest next source evidence, if this direction resumes after H12: map clone/copy/staging to existing current-FULL CPU/device events and inspect all `_draft_token_ids` consumers, then prove a borrowed-result contract with a CPU delayed-consumer test before considering clone removal. Do not treat absent D2H in one request as permission to break structured-output/history modes. Keep dynamic metadata/PD/error ordering, H6/H5 and targetFULL unchanged. No new profile, scan, kernel implementation or device run is recommended here.
