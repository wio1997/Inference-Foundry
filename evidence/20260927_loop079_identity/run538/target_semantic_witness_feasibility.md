# Run538 — sparse retained Target semantic witness, source-only feasibility

2026-09-27. Scope: review of the current Run529/Run533 Host ledger and pinned Runtime/runner sources. No service or NPU work, source installation, model edit, or performance endpoint promotion. Formal Current remains **571.681 tok/s**. This review concerns one selected measured-window retained Target prediction, not a claim for all 344 wo_a groups or all 49,152 output tokens.

## Decision

**Do not add a “fresh required Target row” field to the Run533 Host ledger as though it were observed.** Its existing post-drain data can establish a source-conditional first-position prediction and, after full client/server validation, an exact retained raw-prefix lineage. It cannot establish the selected Target input/KV/weight generation or exclude a value already available from the identical-prompt warmup. A Host timestamp or newly allocated request ID does not close either gap. Keep `F[l,g]` and all Resource/Scheduling/Product finite endpoints null.

The smallest useful future extension is a **separately gated sparse diagnostic**: stage selected device input/position/state scalars on the existing current stream at the first Target boundary, read them only after the ordinary cohort drain, and pair them with Host admission/cache and actual Target call-generation records. This uses no NPU synchronization or per-layer copies and does not change token decisions. It would prove an *observed current evaluation key* under the pinned source path; promotion to *compulsory fresh work* still requires an explicitly declared online-inference/reuse class, a warmup/measured key comparison, cache/result residency evidence, and the Run517 external join. It should be independently preflighted and given its own reversible source manifest before any live use. Do not silently fold it into a controller already being reviewed.

## Exact current-path opportunity

1. Runner `execute_model` creates a twelve-slot, eight-position handoff from `input_ids[:96]` and `positions[:96]`, then calls `FixedCohortServing.run()`. It sets `initial_output_counts=[0]*12`. That is a Runtime-local count, **not** the Scheduler pre-append `G`. The already existing post-serving `torch.npu.synchronize()` follows the Runtime call; a future sparse stage can be read after the existing drain without a new barrier.
2. `ExtremeDecodeRuntime.step()` prepares fixed inputs, calls `FixedTargetAdapter.execute()` (subject to the explicit `target_page_audit.self_replay` diagnostic branch), then `FixedGreedyAcceptance.execute()`, state advance, and proposer. `FixedTargetAdapter.execute()` calls the bound target forward and gathers rows through `target_logits_indices=arange(96)`. `DirectTargetHandoff.forward()` invokes the model under the bound forward context. A Host `serving_host_cycle` record occurs after each `step`, so source plus exact loaded flags imply the ordinary target path for that cycle. This is an observed-execution statement, not device completion timing or irreducible mathematical work.
3. `FixedCohortServing.run()` stages `sampled_token_ids` and *active-masked* `num_sampled` into device history each cycle and copies those histories to CPU once after normal drain. The Run529 ledger records both, initial positions, remaining limits, retained raw IDs, all eight rank/cohort mappings, Scheduler before/after `G`, OutputProcessor raw IDs, serialized API yields and client SSE payloads. For an active selected first cycle with `q=count_history[0,s]>0` and `R>0`, `token_history[0,s,0]` is retained. The pinned `greedy_accept` implementation and Run524 lemma give `sampled[0]=Target argmax[0]`; the proof is source conditional on the actual greedy path. The reducer must still prove `G` clipping, request generation, OutputProcessor/API handling and client receipt. A terminal usage count or re-tokenized text is insufficient.
4. Run529 does **not** preserve the actual first-cycle `target_input_ids`/`target_positions` values before later cycles overwrite the buffers, selected `target_logits_indices` generation, cache/KV content version, loaded model/weight payload identity, or an independently typed Target device completion. `initial_positions` permits a position candidate, and prompt plus retained output permits a logical-context candidate; these are not an observed full state key. The same observation does not establish that all 43 layer projections were compulsory.

## Safe sparse acquisition shape, if chosen after Run533 preflight

For one predeclared slot in the first measured cohort, with a warmup comparator selected by the same rule, allocate small private device staging buffers once at Runtime construction. Immediately after `prepare_target` and metadata readiness, before `target.execute`, copy the selected eight `target_input_ids`, eight `target_positions`, one `target_logits_indices` row mapping, `num_computed_tokens`, `target_seq_lens`, and selected slot mapping/KV ownership descriptor into those buffers on the **current Target stream**. Record `(run, phase marker, rank, cohort, slot, request generation, cycle, Target invocation ordinal, model/config/source/weight manifest identity)` on the Host. Stage the first `Target argmax` only if the Run524 source lemma cannot be pinned to the loaded branch; avoid a second argmax or logits copy. Read the private stage after the existing `tokens_cpu/counts_cpu` drain. Assert buffer disjointness and source-to-staged values in a CPU fixture; do not add per-cycle CPU reads, stream getters, barriers, layer copies, or another Target replay. Probe overhead makes this a lineage diagnostic, never formal TPS.

The proof reducer should fail closed on disabled/alternate greedy path, changed graph generation, missing rank/cohort/phase, parked slot, `q=0`, `R=0`, selected `q` clipped by `1024-G`, duplicated semantic key, differing raw prefix, parser suppression without a supported representation relation, missing SSE/DONE, or stale/preempted request generation. Record Host submission and device completion as distinct types; no Host stage timestamp is a device-ready certificate.

The richer key is `(model/weight/quantization identity, full logical context and absolute position, layer/group, relevant KV/compression state generation, input and output generation)`. A pointer or request ID is not this key. For an all43 conventional BF16 census, source-level unconditional layer traversal and matching layout/loaded branch must be pinned; a first-position argmax alone does not identify 344 distinct fresh projection results.

## Why an unrestricted fresh lower bound is impossible here

The frozen protocol warms the **same 48 prompts to 1024** before measuring 48 requests. If the legal implementation class permits pre-window full-output memoization keyed by the complete prompt/model/config, the measured request can publish the prior sequence without new Target evaluation. This legal counterexample makes a universal measured-window `F>0` false regardless of how many fresh Target graph replays the current implementation happens to perform. Even excluding full-output memoization does not automatically exclude precomputed logits, layer intermediates, prefix KV or a differently represented equivalent function. The Performance Model must label any positive fresh-row count with its online-inference and reuse assumptions. A strict all-architecture Algorithm/Hardware bound may remain numerically wide or null; this is an honest bound result, not a failed instrumentation run.

The first diagnostic should therefore return separate fields: `observed_current_target_replay`, `raw_retained_publication_join`, `selected_semantic_key_complete`, `same_key_seen_in_warmup`, `allowed_pre_window_reuse_excluded`, and `fresh_required_F_admitted`. Unknown is neither false nor zero. Run533 may close the first two fields only. A later sparse stage plus cache/result policy may close the next three for a declared class. None of these alone establishes a matched exact-board C+/B capacity or a finite Product TPS ceiling.

## Source pins and review limit

| Input | SHA256 |
|---|---|
| `runtime/fixed_decode.py` | `7d74f4cbba380ff0a9296cc06d86e0cb9c9c921b6213b7965700831a29a6bafa` |
| `runtime/target_adapter.py` | `c46d60318098077064c0ff57eae98bc937dcf943bda531918388be4e9128acd5` |
| `runtime/extreme_decode.py` | `eb4b5142cc475a51e06c6eec8238e9cfeb918b568e8a5f27b0a4bc7d26963499` |
| `runtime/fixed_serving.py` | `137f1cc6a460700e6f26e5837aa062022eb621198e33099b85988f8b7ef6743a` |
| `runtime/greedy_accept.py` | `9399cad7d76056dbb59db6140227e4379df8f366d7c441b973f9e423775cffbc` |
| `bootstrap/vllm_target_handoff.py` | `2053dafbd46638d0da23e14b96c59ea01995ec2f0441d01928f1c9484c7b4ed5` |
| `model_runner_v1.py` borrowed source | `004dbd0d5b1a5c3f9533fa1d12327bd0ae421fe24bb2674544b2c4a25d74aaba` |
| `scripts/loop079_formal_ledger_server.py` | `5c381122b9b1781bad2733c2f4b1e296ff82e929baf2de247225085983f3df20` |
| Run517 witness review | `e3b233ae7c185fdef1671755f8a810a3d6b4ea272c6fdeee36f2d68065d8177b` |
| Run520 compulsory census review | `e38ade241b509c465bf9952c0ba16fc335592c5511ce4c9ae882d2d6d5ba5648` |
| Run524 retained-prefix CPU result | `3032a2210a347c899e07f6edcad652023ae62ff95c6729a4b12d89647572e8e0` |
| Run525 Scheduling review | `8dbbd555367bf3befab1be40a22f8d25e9a83e4c32e43ef3c654c7d2594a15d5` |

This review pins **current source bytes**, not historical Run99 loaded object/weight payloads or a future Run533 live image. Actual branch flags, cache admission, template/tokenizer identity and stage safety remain future preflight gates. No old KEEP/REVERT result is transferred into this decision.
