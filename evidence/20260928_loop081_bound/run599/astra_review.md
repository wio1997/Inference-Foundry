# Run599 independent source/dependency review

Verdict: **SCOPED PASS for the exact SHA-bound sources reviewed below.** This is a manual structural audit supplemented by source-locator checks, not a machine proof of dependency absence, executable schedule legality, freshness, or a time bound. No service/NPU action was performed.

## Identity and admission

Independently verified all 12 source and 6 evidence SHA256 entries against current file bytes. Reviewed script SHA256: 6f57692601a887cf9cc14a7eac8b8f2a1ad7325d474ea0c07cfed648cfbdeba7. Reviewed source_gate.json SHA256: 053fa135dfba3b350d9fb2d0475393ac24fca66bc4f50a2c19bac1b04537ec8b. The source gate consumes earlier Run597/598 admissions; it does not itself revalidate all raw Run597 files. My prior Run598 independent admission review supplies that separate chain.

The predicates assert only that literal fragments exist. They do NOT enforce function scope, order, call dispatch, absence of acceptance reads, absence of additional writers, correct argument lineage, or immutable approved source identities. A changed source retaining these strings could pass while invalidating the conclusions. The manual findings below apply to the recorded hashes; a later hash change requires a fresh semantic review. No negative-test or automatic semantic-proof claim is admitted.

## Exact structural split

1. FixedTargetAdapter.execute calls binding.forward before sample_hidden indexing and compute_logits. FixedTargetHandoff.forward includes model execution, graph updates and conditional FlashComm _gather_output, including aux gathers. The conservative cut is after the entire binding.forward, with actual writer completion still unmeasured. Python return is not device readiness.
2. prepare_inputs_padded retains every context row: token_indices=self.arange[:total_num_tokens]; acceptance determines sample indices/rejection, not this context row selection. Current Run597 witness is context96/Q7. Metadata paths remain current source conditions, not a universal speculator property.
3. DirectDSparkHandoff context slot refresh derives slots from old Target positions and fixed group block tables. The DSpark input kernel copies all context positions/slots before reading rejection to construct queries. combine_hidden_states uses aux and fixed parameters. Three-layer context projection/RoPE/store uses combined context, positions and layer-specific slots/cache.
4. Query depends on RAW acceptance.num_sampled through num_rejected, old Target positions/common seq_lens, and masked/preserved state.last_sampled_tokens for seed. The v2 source_gate query_branch now explicitly distinguishes raw count/rejection geometry from masked-or-preserved last-token seed; this closes the v1 wording ambiguity. Parked raw counts remain observable to current Draft preparation.
5. Context stores and query preparation both precede the Draft query consumer. Existing set_inputs_first_pass combines these duties and mutates shared buffers; moving the entire method or _propose before acceptance is not justified. Storage non-alias, lifetime, all writer completion and producer/consumer ordering remain unproved. Rejected context and query destinations may overlap; existing context-before-query ordering must be retained until independently proven unnecessary.

Thus the source supports a candidate data-dependency separation, not a legal concurrent implementation or a claim that context work is removable. Three layers are source/model conditional; two observed KV group IDs do not imply two layers.

## The 49,523 column

For active slot-cycle (r,c), let m be the number of leading draft/Target argmax matches, 0<=m<=7, and n=m+1. If m<7, a sufficient realized-path certificate contains m matching predicates plus the first mismatch/recovery prediction. If m=7, it contains seven matching predicates plus the bonus prediction. In both cases the number of prediction-position incidences is n. Summing masked active counts in this diagnostic gives:

- active prefix-decision certificate incidences: 49,523;
- active Target-8 row incidences: 99,976;
- issued Target-8 row incidences: 115,200;
- staged overshoot beyond 49,152 external output tokens: 371.

The wording in source_gate appropriately excludes minimum logits/fresh Target work, parked raw counts and Draft context. Prefer 'incidences' over 'incidents' if renaming the field later.

This is a conditional semantic support/certificate count, not actual read count: the current acceptance implementation argmaxes all96 logits rows and compares all7 positions. A matching predicate need not materialize an entire vocabulary logit vector. If past outputs are granted as an execution-time oracle, even these decisions need not be recomputed; hence the fixed captured ledger alone establishes no computation lower bound. If cross-request/cycle exact-state reuse is allowed, distinct incidences need not be fresh evaluations. State identity must include context/KV/position/model inputs, not token or position alone.

49,523 also excludes parked raw acceptance. Fixed state advance masks their emitted counts, but current proposer preparation consumes unmasked acceptance counts; preserving the full captured raw/query trajectory requires additional decisions. All96 Target aux rows currently enter Draft context work. An ignored acceptance suffix is not proof that the associated hidden/aux/KV work can be omitted. Sequential adaptive verification can preserve abstract greedy semantics yet alter parallelism, numerical shape, Host lag and realized cycle trajectory; exact same-ledger equivalence is an independent gate.

Do not multiply49,523 by Run569 whole-Target coefficients, call the50,453 active suffix rows removable work, or convert the difference to a time saving.

## Formal Run99 comparison

Adding an explicitly non-Bound workload-accounting column is useful, without a new experiment/script. From existing all8/128-file-joined Run595 data, cohort5..8 /9..12 /13..16 staged totals are49,470 /49,508 /49,471, respectively; their external outputs are49,152 each and internal overshoots318 /356 /319. Label these 'historical active staged-token totals; equal to conditional prefix-decision incidence count under current greedy semantics'. Keep repeats separate and retain original Runtime provenance. They do not identify Run597 W0 with any formal repeat and do not reconstruct per-position certificates or parked raw counts.

## Bound consequence

Run599 narrows source-level dependency and terminology uncertainty. R09 is old DP2xTP4/EP8 host-span evidence, R21 a specific compressor/W4A8 contention counterexample, Run579/580 current TP8 whole-chain mixed-service counterexamples. None supplies current LMHead/context concurrent service. The false overlap/readiness/capacity flags and null Scheduling endpoint are appropriate. Formal Current remains571.681 tok/s; Resource, Scheduling and Product finite endpoints are not promoted.

## v2 delta admission

Reverified all18 source/evidence hashes after the v2 update. Only query_branch wording and the extra dspark_handoff locator predicate changed in the JSON; original source bytes and all accounting/scope flags are unchanged. v2 supersedes source_gate_pre_raw_count_clarification.json. Verdict remains SCOPED PASS with the locator-vs-semantic-proof limitations above.
