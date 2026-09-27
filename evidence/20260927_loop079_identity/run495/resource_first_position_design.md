# Run495 — first-position Resource witness design draft

2026-09-27. Independent Astra High source/evidence design for Sol review. Frozen scope: DeepSeek V4 Flash W4A8, DSpark7, DP1×TP8, eight910B3, warmed48×32K→1024, c12.

**DESIGN ONLY.** No new CPU experiment, service/NPU action or capture is claimed by this document. Only this design file is written; TaskCtl, source, model and other evidence remain unchanged. This work must not alter or overlap Run494's live acquisition. Current Formal remains Run99 median571.681tok/s; no numerical Bound endpoint is proposed.

## Purpose and decision

Reduce the smallest useful W_minus witness to one active slot's first Target prediction position and one conventional BF16 wo_a group in the last Target layer. This avoids requiring a complete seven-position acceptance-path census or all43-layer expert-row identity merely to establish one positive scoped work subset.

There are three separate claims:
1. a source-level identity between the first sampled token and the first Target argmax;
2. a same-request/generation proof that this sampled token survives Runtime and Scheduler clipping and becomes a new external output in the measured Product interval;
3. a scoped necessary-work claim paired with a genuine matching hardware capacity upper C_plus.

Claim1 does not establish claims2 or3. C_plus is an independent unresolved documentary blocker.

## Source-level first-position identity

Source anchors:
- runtime/fixed_acceptance.py, FixedGreedyAcceptance.execute: predicted = target.logits.argmax(dim=-1).view(batch_size, target_tokens_per_request); first seven columns are passed as target_argmax and column7 as bonus_tokens.
- runtime/greedy_accept.py, greedy_accept: leading_match, accepted_drafts, draft-prefix copy, recovery/bonus selection and scatter.
- runtime/fixed_decode.py, advance-state path: active_mask governs the actual num_sampled count.

Let P[s,j] denote the predicted integer token at slot s and position j, with j=0..7. Let A be the leading matched draft count, A=0..7. The source establishes, for the produced sampled buffer:

sampled_tokens[s,0] = P[s,0].

Proof:
- A=0: scatter position0 receives recovery = target_argmax[s,0] = P[s,0].
- A>=1: position0 copies draft_tokens[s,0]. A>=1 implies the first comparison matched, hence draft_tokens[s,0]=P[s,0]. The recovery/bonus scatter writes position A>=1 and does not overwrite position0.

This is independent of later equality patterns and the bonus token. It concerns the same invocation's argmax result, with the frozen valid integer vocabulary/casts. It does not make an independent claim about logits semantics, tie handling, numerical equivalence across implementations, input freshness, or publication.

For an inactive slot, Runtime may mask num_sampled to0; a value present in sampled_tokens is then not an emitted token. An active slot with accepted count A+1 has at least one sampled token, but it can still be clipped downstream.

## Runtime retention versus external publication

Source anchors:
- runtime/fixed_serving.py, post-drain token_history/count_history loop: append row[:needed] in cycle order, where needed=remaining[slot]-len(output[slot]).
- Runtime generated/staged/overshoot ledgers and request IDs from a successfully admitted acquisition.
- The exact Scheduler bulk-append, OutputProcessor and Chat/client path used by that same acquisition. Run421/427 are prior methodology/evidence; their counts and timestamps must not transfer to Run494 or a future witness.

Define q as the slot's sum of actual masked staged counts before the selected cycle. Let R be the Runtime remaining-output limit in that acquisition. The selected first token is Runtime-retained only if its count is positive and q<R; its zero-based position within the Runtime output prefix is then q.

Under the source-reviewed zero-initial fixed-serving scope, R=1024. At cycle64, the bound q<=64×8=512 can simplify Runtime prefix retention, provided the same generation/configuration and active/count rules are established. It does not prove external retention.

Let G be the Scheduler's already-committed output count immediately before the actual bulk append for that request. Only after proving that this append consumes the same Runtime sequence in order and clips its suffix, without an additional offset/reorder/transformation, can the external admission condition be written:

q < 1024-G,
external committed ordinal = G+q.

G is NOT handoff-time device-completed work, handoff-time generated state, API publication count, or client receipt. Do not substitute one of those quantities, a previous run's G, or Runtime zero initial counts. The external prefix can be shorter than the Runtime prefix.

Required same-acquisition joins:
- slot -> Runtime request ID -> Scheduler request ID;
- selected generation/cycle and q -> actual sampled integer token and Runtime returned prefix;
- terminal append's exact pre-count G, admitted prefix and output limit;
- admitted token sequence -> OutputProcessor/Chat output -> actual client-visible output, or explicitly retain this last edge as conditional.

A token-ID sequence and emitted text chunks are not generally one-to-one objects because detokenization and buffering intervene. Matching text lengths/chunk counts, a nonempty reasoning delta, or total1024 output length is not a literal raw-token publication certificate. Record the exact representation/sequence relation; do not call Scheduler commitment client publication.

## Proposed no-NPU CPU/source experiment

Use a frozen source snapshot, not files potentially being patched by Run494. Available prior original anchors include:
- evidence/20260927_loop079_identity/run484/b_candidate/patch_state/target.orig
- evidence/20260927_loop079_identity/run484/b_candidate/patch_state/serving.orig
- the corresponding pinned source manifests;
- an immutable commit or hashed archived version for fixed_acceptance.py, greedy_accept.py and fixed_decode.py.

Before implementation, record all source paths, hashes and the exact extracted functions.

CPU acceptance criteria:
1. Check the first-position identity over all128 seven-bit draft/Target equality patterns, with distinct representative token values and differing bonus values. Include leading-match lengths0..7; this checks the source consequence, not the model's numerical semantics.
2. Exercise active versus inactive count masking. A populated sampled token with count0 must not be labelled emitted.
3. Check Runtime prefix retention at q=0, R-1, R and beyond R; include overshoot. Check the conditional Scheduler-prefix rule at the corresponding G+q boundary.
4. Verify that swapped request IDs, a wrong selected prediction column, a reordered/truncated prefix, stale generation, or substitution of publication count for G is rejected by the proposed evidence join.
5. Include deliberately corrupted first-position/copy/recovery rules so the checks demonstrate discrimination. A passing reimplementation of its own formula alone is insufficient; bind the checked rule to the frozen source AST/control flow.
6. Emit separate statuses for source identity, Runtime retention, Scheduler admission, external publication, freshness, conventional-work class and capacity certificate. An unobserved edge stays open.

No Torch-NPU/device import, service attachment, graph change, hot-path instrumentation, new synchronization or request submission is part of this CPU experiment. Its first deliverable is the independently checked source identity and exact list of missing runtime fields, not a claimed Product W_minus.

## Minimal wo_a witness and Run494 interface

Source anchors:
- runtime/target_adapter.py: sample_hidden = hidden_states[state.target_logits_indices].
- /data/wio/vllm_ascend_26/framework/vllm-ascend/vllm_ascend/attention/context_parallel/dsa_cp.py: ordinary BF16 npu_transpose_batchmatmul branch around1374; _restore_tp_head_layout around1645.
- /data/wio/vllm_ascend_26/framework/vllm-ascend/vllm_ascend/models/deepseek_v4.py: last decoder-layer attention/HC/MoE chain around982 and final hc_head/norm/hidden return around1136.
- Run423/449 source/shape evidence for one4096→1024 BF16 group.

Select one slot, one first prediction position, one wo_a group in layer42 of the43-layer Target. Its selected hidden row is target_logits_indices[8*slot]; do not assume this tensor's value equals8*slot. Establish only the final-layer local row/head transforms and downstream selection needed for this witness. Existing all43 routing identity gaps do not automatically require a full-model label experiment.

The ordinary dense candidate count for that one group is:
W_candidate = 2×4096×1024 = 8,388,608 conventional operations.

Run494 is already acquiring same-process graph entry/generation/output-tree and actual graph-update backend provenance. Do not add fields/hooks to the active run for this design. After its final admission, inspect its immutable evidence to determine which needed producer/storage associations are actually present. A post-drain graph dump describes graph structure, not by itself which dynamic token/input generation produced a selected output. Untyped kernel-argument address text is not an input/output label, and addresses cannot be joined across runs.

If a later acquisition is justified, capture only the missing local witness:
- actual layer/group/native branch, shape/dtype and accumulation convention;
- same-run graph identity and typed input/output/weight association;
- selected row/index value and local row/head ownership;
- producing input generation, required downstream consumer, actual sampled first token and retained external ordinal;
- evidence that the required evaluation belongs inside the exact Product timing window and is not already supplied by pre-window cached/reused results.

Buffer observations and export after ordinary drain; do not introduce raw stream-handle getters or per-operator barriers. Work/provenance counting alone does not require A0/B/A1 timing; any later passive-cost or saving claim does.

## Conventional-class limit and exact Product window

The8,388,608 count is valid only for a declared ordinary dense BF16 evaluation class in which this fresh group projection is required and evaluated conventionally. Observing a dense operator or a dataflow consumer does not establish an arithmetic-complexity lower bound for every mathematically equivalent Runtime. Factoring, exact result reuse, alternate intermediate representations, pruning/decision procedures or other legal implementations may change the work.

No blanket output-memoization permission or prohibition is inferred here. State the admitted reuse/cache semantics explicitly. Excluding output memoization alone does not prove dense2MNK unavoidable.

For a strict Product statement, I must be the full frozen48-request measurement interval, with the complete warmup outside I and N=49,152. A12-request diagnostic can qualify a witness for that diagnostic interval but cannot be silently relabelled a formal48-request witness. A later same-source inference for the formal class must state its assumptions. An in-window replay timestamp proves observed execution, not universal necessity or absence of equivalent pre-window work. Work after benchmark start but before Runtime handoff can qualify if it is genuinely required and fresh.

One valid positive subset suffices for first finiteness; no49,152-token, eight-rank, all43-layer or seven-draft multiplier is authorized by this design.

## C_plus remains an independent blocker

Evidence:
- evidence/20260927_loop079_identity/run449/astra_910b3_cap_review.md
- evidence/20260927_loop079_identity/run462/astra_hardware_certificate_review.md
- evidence/20260927_loop079_identity/run479/astra_minimal_compulsory_review.md
- evidence/20260927_loop079_identity/run489/bound_calibration_v3_21.json

For a Cube-only ordinary BF16 class, the current symbolic candidate is:
C_cube_plus = 8192 × sum_r(20 × f_max,r).

The installed1800MHz configuration, rated/current frequency APIs and measured GEMM rates do not certify f_max. The exact-board/bin/firmware maximum envelope, positive clock tolerance/boost and issue/precision applicability remain unproved. If other engines may perform the counted work, the rate must cover them or the class must explicitly exclude them. HBM/HCCL can be omitted from this weak compute-subset relaxation; permitted compute engines cannot be silently omitted.

Neither the CPU proof nor Run494 graph provenance can create that capacity certificate. Do not start a separate capacity microbenchmark to replace the missing authoritative guarantee, and do not derive a TPS number from the symbolic expression.

## Expected decision

PASS_SOURCE_LEMMA may establish the first-position algebra and conditional clipping rule. A full W_minus certificate additionally needs same-window freshness, retained/publication lineage and the declared evaluation class. A numerical Resource ceiling additionally needs a matching genuine C_plus. Keep these statuses separate and leave all finite endpoints null until their own conditions are established.

This draft is a bounded reduction in evidence burden, not implementation approval or a claim that the true hardware limit has been quantified.
