# Run277 caller failure and Run279 correction

Offline 2026-10-07. No service/device action; frozen Run277 and previous review unchanged.

**Correction to my earlier review:** I inferred use_cache=False from the general drafting builder interface without following the actual MTP caller. That inference was wrong. Actual proposer `build_draft_attn_metadata` uses build_for_drafting only in the DSpark branch; MTP takes builder.build. Actual SFA build calls _build(draft_index=None), and line349 therefore selects use_cache=True. Both target and current MTP use the persistent output buffers. Prior statements describing current draft RoPE outputs as separate temporaries must not be used as evidence.

Personally re-read candidate_v5 proposer2510–2546, actual SFA builder283–349, Run277→Run279 shim/driver diff, and actual epoch namespace CPU test source. Existing rank0 period3 index57038/58759 are followed by slice/copy_ sequences, independently consistent with both calls using persistent-output copies. Run277 native failure files were not present in the local run root during this read; exact8/KV pass and ongoing recovery are parent-supplied runtime facts, not independently revalidated here.

## Root cause and narrow fix

Run277 observer caps each use_cache category at two and emits only when total rows reaches four. With every actual call cached, it remains at two forever. Missing witness is an observer admissibility bug, not demonstrated candidate numerical failure. Run277 remains FAILED and cannot establish correctness or license Run278 performance.

Run279 collects the first four shape2 calls without category quotas; driver explicitly requires all four cached and checks every returned cos/sin pointer against the persistent buffers. This matches the actual current caller and closes the specific non-emission bug. No relaxation of layout/format/golden/transfer gates is needed. Production f5ae two-line layout change is unchanged.

Same-worker target graph reuse still has a source-supported boundary: only lookup operand globals switch; the unchanged helper copies into the same persistent output buffers. Both target and MTP sharing those outputs is baseline behavior and the patch must preserve existing ordering/fences, which the diff does. Read-only old/dense table ownership is retained. The revised explanation does not depend on draft outputs being separate. Do not extrapolate to other MTP graph modes.

## Epoch isolation

The actual launch function now chooses ROOT for candidate and epochs/baseline_recovery for recovery H9/H11 witness/mode files. Recovery creates its own zero modes; witness_stack and guard read that same runtime_witness_root. H12 observer environment is absent from recovery, whose rotary source is restored. H5 scope names include run279 plus epoch. This removes the immutable candidate filename collision without overwriting candidate evidence. The CPU test executes actual launch/namespace AST against filesystem fixtures and checks distinct roots/modes/environment; it is not device proof.

No new concrete blocker found in these corrections. Keep the correctness-only boundary, single recovery/max5 request policy and original Run277 failure. Require completed recovery/source identity before any new owned reload; performance remains parked until independent correctness passes. Generic lookup CPU cases alone cannot prove the observer caller contract; the corrected fixture must exercise four consecutive use_cache=True calls (including same-mode repeated calls) rather than manufacture alternating categories.
