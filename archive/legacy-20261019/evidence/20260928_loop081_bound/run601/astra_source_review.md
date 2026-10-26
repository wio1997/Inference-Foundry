# Run601 Astra source-only Product handoff review

Verdict: **source-scoped dependency map; not a live ready-time or delivered-prefix certificate.** Reviewed installed sources without service/NPU execution or source edits. Hash manifest is adjacent. Use request/generation joins for any future dynamic transfer; source presence does not prove a branch executed.

## 1. Ordinary proposal -> initial Runtime state

NPUModelRunner.sample_tokens calls the ordinary propose_draft_token_ids closure, stores its device result in self._draft_token_ids, then optionally schedules a CPU copy. The uses_draft_model branch derives next_token_ids/valid_sampled count from sampled IDs, packages Target aux/positions (with padded rejection metadata when present), and enters drafter._propose.

Installed llm_base_proposer._propose combines aux, calls DSpark.set_inputs_first_pass (including _dspark_seed_buffer), then _run_merged_draft invokes build_model_inputs_first_pass before the Draft model. Installed dflash implementation performs precompute_and_store_context_kv; deepseek_v4_dspark loops the three Draft layers' context projection/RoPE/store. Thus ordinary proposal produces both next draft tokens and relevant Draft cache state. It is not merely a token-list service.

However, the 'last ordinary proposal' must be joined by request ID, previous-slot mapping and producer generation for every handoff slot. Last Host call globally does not alone prove all12 rows' provenance or full cache readiness. Earlier mixed prefill/decode calls and existing state are part of the dependency.

At the96-token handoff, the code enters before the ordinary Target model forward. It takes prepared input_ids[:96]/positions[:96] as12x8, clones column0 into last_sampled/num_computed and columns1..7 into draft_tokens. Block tables, seq_lens, slots, metadata, Target KV tree and existing drafter are bound into ExtremeHandoffInputs. It does not obtain the initial draft simply by assigning the entire last returned _draft_token_ids tensor; the intervening async input preparation/reorder/correction matters.

build_extreme_runtime binds these buffers, assets, metadata updater and the existing proposer. FixedDecodeState.bind checks sizes and allocates bookkeeping tensors. Host constructor return is not an all8 seed/KV-ready event or proof that every asynchronous writer has completed.

## 2. Which synchronization proves what

| Existing mechanism | Legitimate readiness/order conclusion | Limit |
|---|---|---|
| Same NPU stream producer -> subsequent consumer | execution order for commands actually enqueued on that stream | requires exact active stream and all side-stream writer joins; Python return is not completion |
| synchronize_input_prep: prepare_inputs_event.synchronize, then record on context exit | protects reused input-preparation CPU buffers/earlier recorded transfers | event is recorded at input preparation, not after the later Draft proposal; no whole-seed/KV guarantee |
| _copy_valid_sampled_token_count: copy stream wait_stream(default), D2H, event; later event.synchronize | copied count buffer ready after covered producing work | event precedes much of proposal; no token/KV completion guarantee |
| _copy_num_rejected_tokens plus pending_reject_event.synchronize | rejection Host buffer ready if the actual consumer branch performs that wait | wait found in MLA consumer; do not assume current DSA path executes it without dispatch proof |
| _copy_draft_token_ids_to_cpu: wait_stream(default), D2H, event | when taken, CPU draft tensor becomes ready upon matching event completion | common async/no structured-output/no output-history/PP1 branch returns early; record itself is not a wait; not a universal KV completion event |
| sampling_done_event -> global_stream.wait_event | async state update waits for sampling | it does not certify completion of later Draft/context work |
| FixedCohortServing token/count .cpu() plus post-run torch.npu.synchronize | returned Host token/count materialization and all prior work on that rank's device complete before wall_end/output construction | this is the terminal boundary, not initial seed-ready, and not an all8 Host barrier |
| ModelRunnerOutput creation / sample_tokens early return | output object becomes available to subsequent worker/engine handling | no scheduler consumption, socket delivery or client-receipt certificate |

Do not insert a new barrier just to make a diagnostic 'ready' label true. Capture existing writer streams/events and consumers; leave missing joins unknown.

## 3. Scheduler and bulk ledger

Base scheduler constructs CachedRequestData.num_output_tokens as req.num_output_tokens + req.num_output_placeholders. The first term is len(request._output_token_ids); the second may contain pending async output positions. This is not actual generated/published/received count and cannot supply handoff p_i.

Actual installed KVDeliveryScheduler bulk handling bypasses ordinary async placeholder-decrement/cache bookkeeping and directly calls KVDeliveryScheduler._update_request_with_output. That function appends each token to request history, checks stop after each append and truncates the remaining new_token_ids list at the stop point. Therefore an incoming Runtime1024-token array is not automatically1024 NEW external tokens: already accumulated request outputs and stop state influence how much of the bulk survives.

The runner initializes FixedCohortServing with initial_output_counts=[0]*12. This is internal Runtime accounting, not a statement that the scheduler/API/client emitted zero tokens beforehand. Source proves the truncation mechanism but not the per-request historical prefix length or selected bulk prefix in Run99/341/597. Existing Loop079 ledger assets should be reused where their admitted request/source identity matches; no new historical experiment is implied.

## 4. API SSE -> actual client

For streaming requests the chat protocol selects DELTA output. chat_completion.serving accumulates previous_num_tokens[i] += len(output.token_ids), even when a parser/control/reasoning condition suppresses a content chunk. Continuous/final usage reads this count; final usage precedes [DONE]. This count reflects engine-delivered token-ID deltas, not scheduler placeholders or worker bulk input length.

A generator yield proves submission to the ASGI response path, not network send completion or client receipt. SSE text chunks are not one-to-one with token IDs. The legacy bench.py counts nonempty content/reasoning events only as a fallback and prefers usage.completion_tokens; it discards [DONE] as a token event. Run597's stronger saved SSE admission remains a separate actual-client evidence layer. Usage1024 validates count under that protocol but does not alone join pre-handoff prefix and retained Runtime suffix token by token.

Required dynamic bridge: worker bulk input list/prefix -> scheduler before/after real output IDs and placeholders -> accepted bulk prefix -> engine output deltas -> API previous_num_tokens/usage -> raw client response ID/SSE/end. Timestamp these in verified Host clock domains and preserve parser/chunk semantics. Never set p_i from cached.num_output_tokens.

## Scope conclusion

The installed source identifies the correct producer and truncation paths and several valid local ordering mechanisms. It does NOT certify one universal 'last proposal returned => all seed/KV ready' edge, build-return device completion, all8 simultaneous readiness, or worker-publication=>client delivery. The next same-W0 frontier capture should resolve exactly these joins, without changing DSpark7 acceptance, cycles or output semantics. No Resource/Scheduling/Product numeric endpoint or Formal TPS is promoted.
