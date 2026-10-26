# Run408 amendment — first completed B consumer record

Scope: only the completed Run410 rank0_cohort1.json, schema2, plus read-only source. No incomplete capture or other cohort was read; no source/service/NPU change.

## Decision

**ACCEPT the narrow observed local read-before-overwrite and first numeric Host count/mirror-consumption subclaims. Keep later Draft/DSA consumer coverage INCONCLUSIVE.** This is a separately stated partial result, not a relaxation that turns the original combined gate into an overall ACCEPTED result. The current analyzer correctly requires downstream_observed for its combined status. All8/five-cohort and external B arm gates remain outstanding.

Independent record() analysis passes its structural, generation, pointer, Host-order and timing checks:

- Direct R_DONE64 -> W_PRE65:53.7385787964 ms. Same-device anchor difference:53.73828125 ms. Difference:0.00029755 ms, within empirical allowances.
- Direct read/write envelopes:0.003420/0.005520 ms. Existing analyzer reports anchor-derived envelopes instead:0.0029296875/0.005859375 ms. Prefer direct short-interval figures and label the method; the terminal anchor is about13.47 s later and float precision matters at microsecond scale.
- R markers share the recorded copy stream; W markers share a different writer stream. Four unique diagnostic events are distinct from the original production event. Source is the same48-byte int32 count tensor, entries64/65 and generation64; no generation error or parking.
- Original production-event synchronize returns before count_add_pre. The add reads the pinned destination pointer; seq_add brackets the recorded CPU sequence-mirror pointer. Host order is commit -> sync -> count add -> seq add -> mirrors_done -> launch65.
- Host synchronize bracket is17.481 us; count-add bracket64.913 us; seq-add bracket28.212 us. These include Python/marker and CPU work and are not isolated kernel costs.
- first_clone/downstream are null. No claim about the later consumer is demonstrated.
- Per-record original_schedule_slack_supported=true merely means the measured margin exceeds the empirical0.801004 ms threshold. It does not override missing A0/A1 per-cycle trajectories or prove the original schedule.

## Why the hooks are absent

The source provides a concrete coverage explanation:

1. bootstrap/vllm_dspark_handoff.py:_DP1RunnerShim sets num_rejected_tokens_event=None. Its _execute passes a non-None num_draft_tokens_cpu list to _propose.
2. spec_decode/llm_base_proposer.py:_prepare_parallel_draft_seq_lens_cpu (around949–997) only calls build_parallel_draft_seq_lens_cpu for non-padded or async-padded input. The fixed handoff is padded + non-async, so the utils clone/add_ hook is bypassed by construction.
3. The same file's build_draft_attn_metadata (around2407) selects builder.build_for_drafting for method=dspark. The installed DSA hooks are in ordinary build_prefill_metadata/build_decode_metadata, not the separate drafting methods.
4. attention/dsa_v1.py:build_for_drafting (around1239) dispatches to build_prefill_metadata_for_drafting or build_decode_metadata_for_drafting. Draft prefill uses device seq_lens/max operations. Draft decode contains a separate, currently uninstrumented torch.max(_seq_lens_cpu[:num_decodes]).item() around1417. A null hook record therefore does not establish no consumer, or which drafting split ran.

The minimum future consumer closure is a branch descriptor and pointer-linked before/after marker at the actual drafting CPU read if that branch runs, or an explicit documented device-only branch result if it does not. This amendment does not request altering the active B service.

No universal happens-before, all8 critical path, removable cost, finite Hardware/Algorithm/Product endpoint or compulsory-HBM conclusion follows.
