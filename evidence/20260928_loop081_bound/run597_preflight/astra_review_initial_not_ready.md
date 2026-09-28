# Run597 independent Astra preflight

## Current verdict: NOT YET LIVE-READY pending bounded offline closure

The inspected insertion sites are structurally suitable for a diagnostic Target/input-lineage collector. They do not enter the compiled decoder. The current acceptance gate and check artifact are incomplete; close the concrete items below before launch. This does not require a new NPU experiment. A complete Target+Draft+KV/prefill necessary-work certificate is not established by this collector, even after the gates pass.

## Source placement and timing

- Runner bind occurs in its Python FixedCohortServing branch after cohort registration and before serving construction, so cohort5..8 selects the measured pass after warm cohorts1..4. It does not intercept compiled model forward.
- Initial last/draft clones precede the first Runtime cycle. Initial positions reuse the ordinary constructor CPU snapshot. This is a boundary snapshot, not a KV generation or full same-state attestation.
- `runtime.step` commits next_draft into state, increments cycle index, then returns. The added draft-history copy follows return and existing accepted/count copies, before next cycle and parking. Its destination is separately allocated once per cohort. No per-cycle clone allocation for the full history is introduced.
- Host park records precede the original park calls but record already computed Host scalar slot/anchor data. `next_cycle=index+1` aligns with the next Target. A live exception during the park call must reject the run, not treat the prewritten event as proof of a completed transition.
- Raw accepted-token prefixes include parked slots; existing count_history is state.num_sampled masked by active_mask. The checker correctly requires zero masked count after recorded park, while preserving raw tokens and Draft state. DSpark prepare uses RAW acceptance.num_sampled, so its rejection/sample-index formulas must never use the masked count for parked slots.
- Target witness is after input preparation and before Target execution. DSpark witness is after prepare_inputs_padded/Target hidden packing and before `_propose`. These are Python Runtime/adapter boundaries, outside Target ACLGraph and compiled decoder. This avoids the Run582 compiled Python-hook failure pattern; it is not a proof against every Dynamo guard interaction on a different branch.
- No new model call, collective or intended write to model state is introduced. Device clones/copies, list append/import overhead and export still perturb scheduling and memory resources. Do not infer unchanged acceptance from structural read-only intent.

## Reproduced CPU findings

`astra_cpu_preflight.py` extracts the pure checker using AST (no torch/NPU import), reconstructs four Run566 A0 Target histories, and recovers 24,368 / 23,824 / 26,064 / 24,240 active rows. DSpark arrays in this test are synthesized formula fixtures, NOT newly captured Draft evidence. Source patches compile and original four source hashes match.

At first audit, the helper changed during parent revision while run597_patch_check.json still pinned an earlier helper; all source patch hashes stayed the same. Regenerate and match the final manifest before installing. The saved check result records the exact hashes it tested.

The original checker ADMITTED all six mutations: corrupt sample_indices, num_rejected, query_start_loc and DSpark seq_lens, duplicate Target witness, and omission of every checkpoint after cycle0. These are real missing acceptance checks, not invalid negative fixtures: each original full record passed before mutation.

Required fixes: uniqueness/range and exact required-cycle coverage for Target/DSpark witnesses; supported branch/config descriptors; validate token_indices=range(96), query_start_loc=range(0,97,8), rejection=8-raw_count, sample index=8*s+raw_count-1, and unchanged seq_lens at this adapter boundary. Pin the applicable padded-input implementation. Test the corrected checker on valid fixtures and wrong metadata, missing/duplicate checkpoints, shifted/missing parks, raw/masked confusion and corrupted prior next-draft. Adjacent cycle0/1 witnesses make the latter directly testable.

Terminal Host events with next_cycle==cycles are currently schema-checked but not replayed by the cycle loop. Either validate their completion/uniqueness against final state separately or explicitly mark them as unvalidated terminal control records. They do not have a following Target consumer.

## What remains outside this ledger

The DSpark adapter checkpoint observes Target context selection before `_propose`, not the actual query IDs/positions, sample_from_anchor/q, context KV precompute, per-group slot mappings or Markov-model input. Actual query construction happens later in `dspark_proposer.set_inputs_first_pass`. This acquisition may be admitted as a Target lineage plus Draft-output/prepare-boundary subset if labelled accordingly. It cannot certify complete Draft evaluation cardinality merely by naming these records DSpark witnesses.

Run593 also requires ordinary prefill/seed scheduled spans, entry cache/KV provenance and external request/output joins for a broad Product workload ledger. Current boundary flags honestly say these are unknown/not captured. They must remain missing terms; neither same prompt nor token-prefix equality certifies full semantic state equivalence. Do not classify all active observed rows as necessary fresh work or promote any strict endpoint.

## Export, perturbation and launch/restore gates

save runs after ordinary cohort drain but before return/publication. Extra CPU conversion, JSON/fsync and sparse tensor D2H are therefore inside Runtime/Product wall, and may delay ranks entering the next cohort. This is diagnostic timing. The preflight does not waive A0/probe/A1 perturbation/semantic gates for later timing transfer.

The patcher pins original hashes, checks unique anchors, compiles in memory, backs up before install and attempts rollback on partial install. Restore checks every source and backup before writing, accepts only original or recorded patched bytes, and preserves permissions. These are sensible protections. `--offline-confirmed` is only an assertion supplied by a caller: an owning launch wrapper must prove no service/owned client is running before install and after stop before restore; retain each exit code and all restored source hashes. Never restore over unrecognized drift. Pin helper/patch hashes through acquisition, not only at check time.

No launch wrapper, stop ownership check, four-source restore outcome, or perturbation run was executed in this review. No service/NPU/device query was started. Once final source manifest and CPU gates pass, a guarded single diagnostic warm48+measured48 acquisition can be considered live-ready for the explicitly limited ledger scope; no unperturbed E2E or complete Resource Bound claim follows.
