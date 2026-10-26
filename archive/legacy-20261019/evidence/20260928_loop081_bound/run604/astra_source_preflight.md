# Run604 Astra source-only Bound preflight

Verdict: **PASS as a bounded acquisition design; not an installed/live-ready patch approval.** No service/NPU execution or source modification. Fixed DSpark7 acceptance, output and cycle semantics remain constraints. Source inspections and Run602/603 define conditional paths; only observed branch/state identities may enter a same-W₀ runtime DAG.

Identity manifest: astra_source_manifest.json, 25 source/evidence/history digests, SHA8242b733335490258dba1205e107501ad03005154de5506565fa057d3a21d370. Restored runner source SHA004dbd0d5b1a5c3f9533fa1d12327bd0ae421fe24bb2674544b2c4a25d74aaba. Historical reports are byte-equal to pinned git objects at db3beef223e0b5acd81ccccb444600c1b22aac8a. All line anchors below refer to these restored versions.

## 1. Facts available without another service run

Run603 describes71 ordinary Target/proposal pairs per rank (16/26/15/14), not71 compulsory mathematical evaluations. The three empty execute calls take the zero-work return path; the four final96 entries switch to Extreme before the ordinary Target forward.

Run602 startup configuration is stronger evidence than guessing Graph mode from shapes:

- Target enforce_eager=False, FULL_DECODE_ONLY, capture sizes8..128 in steps8.
- DSpark speculative enforce_eager=True. llm_base_proposer.py:183 sets use_cuda_graph=runner._use_aclgraph() AND NOT speculative.enforce_eager, so this setup selects eager ordinary Draft. DirectDSparkHandoff:87 rejects use_cuda_graph=True. A new experiment should verify the retained object's flag/identity once, rather than rediscover Draft Graph eligibility on every call.
- DSA_CP and FlashComm1 are enabled; utils.enable_dsa_cp additionally checks model indexer and SP. AscendDSABackend.get_builder_cls/get_impl_cls (dsa_v1.py:260/277) selects AscendDSACPMetadataBuilder/AscendDSACPImpl under that flag. The generic dsa_v1 implementation is a fallback, not a safe default assumption for this configuration.
- enable_npugraph_ex is NOT runner.enable_enpu. The latter reads the C environment ENPU_ENABLE (runner:472–474); do not infer its value from the startup compilation setting.

Observed Run602 request query lengths are412 occurrences of8 and48 prompt suffixes (46×83,85,84). A threshold-only classification gives4 pure-prefill candidates,40 mixed candidates and27 all-eight-token uniform-decode candidates. This is a **prediction**, not actual attention/Graph branch coverage: CP split also considers is_prefilling and Graph dispatch depends on populated entry state, flags and metadata.

## 2. Ordinary Target path and the missing dynamic fields

runner.execute_model:1862 runs _update_states and _prepare_inputs inside synchronize_input_prep. It can reorder slots, apply deferred async corrections and sanitize placeholder IDs. Therefore entry-time scheduler metadata is not automatically identical to final device inputs.

runner:1948/4635 _determine_batch_execution_and_padding computes uniform_decode from actual input_batch num_computed_tokens_cpu, uniform query length, total rows and number of requests. It applies SP padding and the graph dispatcher. DP1 avoids the DP>1 coordination branch. Selection may still be overridden for KV-scale calculation (runner:2105) or encoder/skip-compiled conditions. Padded rows alone cannot establish Graph NONE/FULL or replay.

runner:2068 _build_attention_metadata obtains the actual per-group metadata, followed by _sanitize_placeholder_input_ids_for_forward and _preprocess. CP builder dsa_cp.py:303 calls split_decodes_and_prefills with threshold1+7=8 and treat_short_extends_as_decodes=False. attention/utils.py:430 ORs the is_prefilling flags with query-length classification. Metadata carries num_prefills/num_decodes/num_decode_tokens, attn_state and actual/padded row fields. A short extension can remain a prefill; batch reorder and padding must be joined.

runner:3650 set_ascend_forward_context binds actual metadata, cudagraph_mode, batch_desc, num_actual_tokens, input IDs and skip_compiled. _model_forward:4544 obtains this context, executes the model, conditionally updates FULL graph parameters and optionally all-gathers hidden/aux for FlashComm. That is the correct outer observation point. Capture backend/builder/implementation/update-callable identities separately; they need not share a class name.

Critical distinction: ordinary _update_full_graph_params_if_needed:4522 only updates when FULL, not capturing, **not use_sparse and not use_compress**. Extreme TargetHandoffInputs:2246 independently binds graph_update. Do not apply Extreme's updater behavior to ordinary calls or assume source presence means a branch ran.

ACLGraphWrapper.__call__:133 selects NONE/mode-mismatch fallthrough, first capture, or an existing entry replay. At:260 it has an existing current_stream.synchronize before replay when FULL AND !enable_enpu AND !(is_draft_model AND use_eagle). Record whether that branch actually executes, its Host enter/exit, entry key/model owner/generation and stream identity. Do not insert another synchronization. An event recorded outside this wrapper can span Host launch delay and existing waits; Run602's18.736s current-stream Target sum is not proven device-busy compute.

CP implementation dsa_cp.py:1310 uses actual attn_state, TP/oprojection configuration and gather policy to select full gathered o-projection weights/layout; its internal prefill/decode paths and cache/collective placement may differ. The BF16 wo_a row geometry must be derived from this selected branch and actual group count, not borrowed from a Runtime decode-only witness.

## 3. Ordinary DSpark context/query and Runtime consumption

llm_base_proposer._propose:1040 combines Target aux and:1043 invokes set_inputs_first_pass. DSparkProposer:252–350:

- copies next_token_ids to seed buffer;
- obtains context rows from cad.query_start_loc_cpu[batch_size];
- copies corresponding Target hidden context;
- per KV group expands query IDs/positions/slots and context positions/slots;
- uses num_rejected_tokens_gpu when present to adjust effective sequence/query geometry;
- sets num_query_total=cad.num_reqs*num_query_per_req; sample_from_anchor chooses7 versus8 query rows per request. The actual object flag and cad.num_reqs/padding are required.

After metadata building, llm_base_proposer:1258 enters its Draft forward context, then calls _runnable. With the admitted eager configuration, _runnable is the ordinary merged Draft function. _run_merged_draft:1329 invokes build_model_inputs_first_pass before the Draft model. dflash_proposer:253 calls precompute_and_store_context_kv; deepseek_v4_dspark.py:205 iterates actual layers and writes projected context KV when context and slots are nonempty. Context rows are not query rows, and the three layer writes are not three independent token evaluations or compulsory HBM byte counts.

The96 handoff clones column0 of prepared12×8 IDs/positions into seed/num_computed and columns1..7 into initial drafts, binds live block tables/slots/cache trees and reuses the proposer. Build return is Host construction, not device ready. The initial32768 cached positions per request cannot be charged as freshly computed measured-window prefill, nor treated as universally free across arbitrary initial states.

First Runtime target: ExtremeDecodeRuntime.step:397 prepares inputs/metadata; target_adapter.execute:64 calls binding.forward and:77 computes logits; acceptance and state advance follow; proposer.execute:455 then uses raw acceptance for rejection geometry, preserved/masked last seed and retained Target aux/context. Initial Draft KV and seed/state must join the exact initial consumer, including prior ordinary producer generation and intervening corrected input preparation.

Existing copy streams (sample counts, reject counts, optional draft IDs), global-stream state update and graph-update streams need actual record/wait-generation joins. An optional-copy event object existing does not prove its branch ran. The last ordinary global call does not prove every request's KV provenance.

## 4. Minimum all8 packet

Do not instrument every kernel or add per-layer timing. Keep one immutable outer record per ordinary execute/Target/proposal plus bounded records for the four initial Runtime handoffs. Fields:

**Startup once per rank**
- exact source/config identity, actual builder/attention implementation and update callable qualnames;
- use_compress/use_sparse/enable_enpu, Target/Draft graph flags, wrapper runtime mode/use_eagle, sample_from_anchor/query count;
- graph entry/model owner identity, cache group/layer mapping and allocation generation; opaque storage ranges/aliases, dtype/layout, not raw pointer equality across ranks.

**Target outer call**
- run/cohort/execute ordinal; req/slot/previous-slot order; final actual/padded rows and query-starts;
- final forward-context mode/batch descriptor, capture versus replay versus fallthrough, matched entry generation;
- per-distinct metadata group: actual/padded req/row counts, num_prefills/num_decodes, attn_state, is_prefilling/CPU query lengths where already valid;
- Host enqueue boundaries, current stream, existing wrapper sync taken/enter/exit; actual graph update branch/callable/stream identity. Preserve call-to-entry association, not only histograms.

**Draft outer call**
- actual ordinary producer req/slot IDs; seed generation and context/query row counts; sample indices length, padding, rejection-present flag, group/layer slot mapping identity;
- context-store enqueue boundary versus query model boundary, return generation;
- branch bits for existing asynchronous copies and the producer/consumer record/wait generation relationships.

**Initial consumer**
- corrected96 prepared-input snapshot identity → initial Runtime basis identity;
- producer generations for seed/draft/context/Target cache/state buffers;
- first Target and first Draft consumer enqueue boundaries and any existing wait connecting them.

Allocate events/Host record storage before arm, use a bounded pool, fail on overflow. Record all observed ordinary calls; do NOT force a later acquisition to have71. Emit/read events only after the existing terminal synchronization. Never add waits/barriers, poll event completion, move model work to a new stream, or call GPU/NPU .item/.cpu merely to get a scalar. Avoid hooks inside Dynamo/capture; if a callback runs only on graph capture, it cannot stand in for every replay. Record a replay-level marker at the existing wrapper instead.

This is a new instrumented W₀ unless the full ordinary-state/Runtime-basis/acceptance/output ledger establishes equality. A source-compatible config alone does not make it Run602 or Run99.

## 5. Piggyback Resource freshness: useful but conditional

Yes: the same semantic keys can connect preparation issuance to Runtime fresh-input candidates and initial resident cache assumptions. No: Host geometry/stream records alone do not prove strict compulsory computation/traffic.

If final corrected IDs/positions cannot be reconstructed exactly from already validated Host buffers, the smallest useful optional addition is a preallocated device snapshot of **integer IDs/positions and row-validity/slot-generation metadata only**, at the final post-sanitize Target boundary and DSpark prepared-query/context boundary. Copy into private bounded storage before those buffers are reused, defer D2H/serialization to existing drain. Do not clone all activations, logits, weights or KV. The source copy has real device traffic and possible timing cost; keep it explicit and apply an observer OFF/ON control before treating measured durations quantitatively.

A freshness key needs model/weight version, role, absolute position, full logical prefix/context/state identity and producer generation. Request ID/position alone is insufficient: repeated prefixes or cross-request state reuse can share work, and old KV may already hold needed state. Deduplicate known-equal keys; mark unknown state equivalence rather than automatically counting it as unique. Ordinary prefix IDs/acceptance should be preserved as fixed-work evidence, never changed to improve acceptance.

A typed BF16/W4A8 operation-class witness can be linked to this broader logical ledger only after selected layout/group count, actual consumer and initial-state absence of an equivalent result are shown. Reuse Run585/569 methodology as a prior, not their different-W₀ row as proof for these71 calls. Even then, dense arithmetic and traffic floors are conditional on the stated evaluation/cache/model class; execution counts are not universal mathematical necessity. Current HBM reads, context KV stores and padding remain observed implementation work until a separate compulsory-traffic argument is supplied. This packet does not establish exact-board cumulative C⁺/B.

## 6. Historical knowledge consulted on demand

Queries: "draft forward graph decomposition" and "prefill overlap contention", pinned history revision above. Relevant reports only were read:

- **R09**: old8×910B3 DP2×TP4/EP8, DSpark7, eager/async/prefix caching. Its decomposition counted one context-KV stage and three distinct Draft layers;29 counters were stable over932 iterations. This refuted its tested duplicate-call hypothesis, but the report itself could not distinguish Host submission from completion coupling solely with Host marks or prove1:1 E2E savings. Reuse separate context/model/seed phases and avoid inventing duplicate work. Its "0 removable" under a narrow candidate rule is NOT our hardware/scheduling ceiling; TP8 Runtime ownership/Graph behavior differ.
- **R21**: old DP2×TP4 prefill, actual C4/C128 Compressor concurrent with shape-faithful W4A8 chain. Bit-exact trials regressed0.18–0.23ms/call; weighted7.29–8.12ms/chunk. This is a resource-contention/event-overhead counterexample, not a veto of legal TP8 schedules. Require a common consumer endpoint and joint resource evidence before interpreting earlier launch as overlap savings.
- Existing PK-057 is relevant caution: source graph-update code and stream numbers did not alone prove matching replay event/handle generations. Do not repeat that missing-identity mistake. This preflight does not reopen the old native graph dump experiments.

## 7. Failure gates and choice

Preflight failure: source/SHA drift, unsupported actual class/branch, capture-only hook mislabeled replay, local variable escaping its function, event allocation in captured graph, unbounded record count, or any new synchronization/model mutation. CPU negatives: wrong execute/slot generation, dropped call, stale event generation, copy early-return labeled complete, mislabeled is_prefilling/Graph branch and changed source/initial state.

Live admission failure: all8 semantic disagreement, missing corrected-input/initial-consumer join, incomplete output/acceptance ledger, cleanup/restoration failure. These invalidate the relevant claimed scope; they do not indicate an algorithmic opportunity. Timing claims additionally require observer perturbation control, actual Host clock identity and valid event domains; no cross-device timestamp subtraction.

This packet's highest-value unknown is **how the existing preparation and initial Runtime dependencies are actually executed**, especially actual Target replay/update/synchronization and distinct context/query writers. Static evidence already narrows the ordinary Draft to eager and predicts much of the query-length classification, so collecting only those flags would not justify another service run. Combine them cheaply with the missing generation/readiness and conditional freshness witness. Sol should compare this remaining information gain with the unfinished Resource capacity proof; neither a new candidate failure nor this source-only review can support proximity to the overall limit.
