# Run269: cross-cycle target metadata dependency and alias audit

Freeze: V4 Flash W4A8, eight 910B3 DP1×TP8, DSpark7, 48×32K→1024, c12, FULL target Graph. Compare A0 unchanged, A private scratch after DSpark, B private scratch launched after state advance on a side NPU stream while DSpark runs. The first cycle always takes A0. The flag defaults off.

## Semantic DAG and storage hazards

1. Target `t` reads `target_input_ids`, `target_positions`, `target_seq_lens`, `target_slot_mapping`, rotary and DSA metadata at stable captured addresses. Acceptance `t` uses target output. `advance_state(t)` commits `num_sampled`, `last_sampled_tokens`, `num_computed_tokens` and emitted count on the main stream.
2. Target geometry for `t+1` (96 positions, 12 lengths, 96 slots) depends only on the advanced computed-token counts, fixed offsets/block size and stable block table. Target IDs for `t+1` depend on the next DSpark draft, so **IDs remain after DSpark**. The metadata updater reads geometry and immutable full RoPE cache / constants; SAS/QLI outputs depend on lengths and fixed context settings.
3. Current DSpark refreshes its common tensors from **old** target positions/lengths/slots, recomputes draft-context slots from old positions, and packs old target IDs/positions/hidden. Its model then uses those prepared tensors. Therefore no old active tensor may be overwritten before DSpark has finished reading it. The borrowed `_ROPE_STATE.runtime_buffer` and target attention metadata are Graph-addressed; never write these early.
4. Private scratch geometry and a cloned updater output buffers break storage WAR edges. Side-stream work waits for the main-stream advance event. It writes only private tensors. Main stream may execute DSpark concurrently. At next step, main stream waits for scratch completion and copies geometry/metadata into the original stable Graph addresses, then fills target IDs from the committed draft. This reestablishes all target RAW edges.
5. `FixedCohortServing._park_completed` can overwrite computed-token positions and active mask after `step()`. When parking slots, wait for pending scratch before mutating state, then invalidate it. The next step falls back to A0 preparation. If the final step exits, pending scratch has no consumer; the following cohort constructs a fresh runtime. The target/block table must not be mutated while side work reads it.

## Causal variants and gates

- A0: existing prepare geometry+IDs, metadata update, target, acceptance, advance, DSpark.
- A: after DSpark, calculate next geometry+metadata into private buffers on main stream; next step commits then fills IDs. This isolates private allocation/copy and any move of metadata across the cycle boundary.
- B: calculate private geometry+metadata on side stream after advance, overlap with DSpark, commit at next step. Compare B against A for scheduling effect; compare against A0 for product value.
- Correctness diagnostic: 8 ranks; exact 96 positions/12 lengths/96 slots and rotary values, exact group lengths/start/query offsets and SWA slots; SAS first97 and QLI first25 entries exact under prior A/C tail-noise gate; acceptance counts/IDs, draft, KV page addresses and parking must match. Preserve stable target Graph pointers and kernel family counts. Any mismatch rejects B.
- Timing: complete latest-rank cycle and proposal delay, side-stream launch-to-complete, next-step join/commit, acceptance trajectory and Product E2E. Do not sum kernel times; account for resource contention. Run98 metadata 0.667ms is a screen, not a predicted gain.

Open implementation risks: native SAS/QLI op reentrancy on a side stream, hidden aliases among borrowed metadata destinations, and block-table mutation. Check actual addresses and all-rank correctness in a diagnostic before formal benchmark. Do not claim a numeric Scheduling-aware Bound until legal schedule cost is measured.

## Review-driven gate correction

Astra High Run273 found the initial same-state verifier substituted its serial reference into active Graph buffers before Target. Run270 is therefore restricted to parity evidence. The next candidate-consumed diagnostic first compares stable fields, then copies **the candidate scratch geometry and complete 1024-entry metadata outputs back into the fixed Graph addresses** before Target. It also records conservative storage intervals, including device, base pointer, offset, shape and stride, for all private writes versus live Target/DSpark and target KV tensors. Product timing will use `EXTREME_SCHEDULE_VERIFY=0`; verification-enabled TPS is invalid as a performance measurement.
