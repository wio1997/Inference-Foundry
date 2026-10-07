# MTP Graph independent addendum

2026-10-07. Source-only; no service/device/run/production changes. Retain H6/H5, target FULL bucket2; async early publication parked. This supplements and corrects MTP_GRAPH_REVIEW.md.

## Closed model-body unknown

Personally read `mtp_graph_sources/identity.json`, `deepseek_mtp.py` (83a52a68c0e2c3cd032cf0aff2c58570d87cf094f41f3a031208d1a74f6a2ede), latest `llm_base_proposer.py` (edcc55d...), and relevant latest runner (d017c230...). Actual upstream MTP selects one layer using fixed spec_step_idx, embeds inputs, applies tensor position-zero masking, enorm/hnorm/eh_proj, one DecoderLayer, residual addition and final norm. It returns pre-norm logits hidden and post-norm recycled hidden. compute_logits applies the shared-head norm to the pre-norm element. No tensor-value-driven Python loop appears in this K1 body. The old missing-base-source objection is closed; no extra norm or tuple reinterpretation is needed for capture.

## Dummy versus real first pass

- Latest runner 3998 passes num_tokens_padded, num_reqs_padded and the same batch descriptor to drafter dummy_run.
- Proposer dummy 758 computes batch_size=max(num_tokens//(K+1),1); 795 passes a prefix of token_indices_to_sample sized batch_size×extra_slots. Base proposer 110 sets extra_slots=1 for nonparallel MTP, and net extra slots=0.
- Actual DCP helper `padded_mtp_order_sources/...worker__dcp_utils.py:213–244` is decisive: first-pass preparation **does not compact or redistribute these tensors**. It returns unchanged num_tokens, IDs, positions, hidden and sample indices; it attaches long-sequence metadata and clones original indices for later use.
- Therefore for actual admitted one-request uniform decode with two input tokens, dummy and real both feed two model rows and one sample index. K1 merged early return yields `[1,1]` draft IDs. Dummy zeros versus real index value are expected data differences in the same persistent index buffer. This is source-supported shape agreement, not numerical replay verification.
- Prefill, mixed/nonuniform batches, different request counts and scheduled K must not reuse this graph merely because a descriptor token count matches. The predicate must check the actual transformed input shape, index-prefix length and decode-only semantics. Eager fallback preserves those requests. No SP-enabled physical16/fifth-row objection applies here.

## Metadata and buffer findings

First-pass MTP still calls builder.build, not build_for_drafting: both capture and real K1 use draft_index=None, hence persistent global cos/sin. Per-step proposer groups hold seq_lens/query_start/slot_mapping; SFA-DCP's builder holds replicated block/slot views and local sequence lengths. Values are refreshed before `_runnable`; SFA update_graph_params remains pass. Runtime graph must consume those refreshed allocations, with target→draft→next-target same-stream ordering retained.

SFA-DCP capture builder accepts SpecDecoding and delegates to ordinary build, then sets state. Real decode metadata's Python branch fields (`num_prefills`, `num_decodes`, `num_decode_tokens`, attention state) must agree with capture; stable tensor addresses do not repair a captured prefill branch. Uniform bucket2 supports the agreement but an actual pointer/branch witness remains necessary. DCP context slot/block/mask and cache/indexer consumers belong in that witness, not just base SFA seq/rotary.

**Correction to prior review:** the topk-sharing helper is conditional, not proof of shared target/draft storage. Actual DeepseekV2Model constructs a local topk buffer passed to layers; `GlmMoeDsaForCausalLM` is a pass subclass. Actual MTP constructs a buffer per predictor layer. `_maybe_share_topk_indices` only executes if target_language_model.model exposes the attribute. These inspected sources do not establish that attribute, so “explicitly shares target topk” was too strong. Capture/replay should record actual layer buffer pointers; no shared-buffer race can be claimed without those identities. First MTP layer is constructed with skip_topk=False; merged step0 resets False if index sharing is configured. K1 must capture its own indexer operations. The later Python True toggle is not itself replayed and is not evidence that replay skips captured indexer work.

## Minimum implementation boundary

1. Replace unconditional GLM force-eager with an explicit opt-in capability plus a **runtime** scoped predicate near proposer dispatch (around 895–920), admitting only verified greedy K1, actual uniform decode shape/index prefix and supported SFA/parallel configuration. Startup configuration alone does not cover per-step K, mixed batches or adapters. Unsupported calls use the existing eager runnable. Keep model semantics and all preparation outside the graph unchanged.
2. Close captured input validation around the merged invocation (1120–1146). Actual ACLGraphWrapper's address checks enumerate positional args, but this invocation uses kwargs. The present check can pass with two empty lists. Validate the persistent graph input/metadata tensor inventory, shapes and relevant branch signature explicitly; do not treat Python metadata identity or that empty check as evidence.
3. Preserve owned draft-output lifetime. The wrapper reuses a fixed graph output; runner's side-stream draft D2H for structured/history/PP may still read it when a later replay writes it. Either retain eager for these branches initially, or prove/add an event/owned-copy protocol before reuse. Same-main-stream next-input consumers are a different case. Do not change CPU staging, accepted-count/DCP correction, KV finalize, HCCL or output/error completion fences.

There is no verified kernel-level or model-body blocker requiring a new operator. Conversely, deleting the guard alone does not satisfy items 1–3. Task registry names already distinguish target/draft and wrappers own separate entry maps; no registry collision is established by source. Runtime task attribution and pointer identity remain unknown. Keep existing wrapper synchronization policy; do not bundle H9 or early publication.

Recommendation: continue this sole scoped implementation investigation. First CPU/source tests should falsify selector, shape/branch mismatch and delayed-output ownership cases. A later authorized bounded device diagnostic must establish actual capture/replay, all-rank KV success, changed positions/seq/slots and correct draft/target results. Capture success alone is not performance gain or dynamic API coverage. No extra Run is requested or performed here.
