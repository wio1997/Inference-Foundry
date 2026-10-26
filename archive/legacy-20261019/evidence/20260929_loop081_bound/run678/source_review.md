# Run678 startup slot certificate — source review, not performance evidence

Prepared while Run677 was live; nothing installed and no CPU/NPU test executed at this checkpoint.

Astra High reviewed the first-eight-cycle reachable bound and actual helper/patch. For frozen 12x8, schedule OFF, cycle zero and >64 remaining tokens, standard per-cycle advances0..8 imply every target position in cycles0..7 lies in [p0,p0+63]. Pinned worker RPC serialization and read-only block-table ownership are necessary. Source reads in DSpark use tables as inputs; mapping buffers are separate outputs.

Review identified and main corrected: cross-group table/mapping aliases (all tables against all mappings), mapping stride identity, and p0+64 overflow for actual computed-token dtype. High final verdict: scoped source PASS; still requires CPU gate, all8 live correctness and formal Product E2E. Inference tensors without version counters do not support general mutation detection. Runtime guard bypass is valid only under the pinned ownership proof, and failed certificate returns to the original checks.

The CPU gate script was independently prepared by Astra Medium. Main caught lost literal quotes and inference-mode mutation context in the written script; Medium fixed and re-read it, compile only passed. Do not mistake compile for gate execution.

Candidate generator/installer remain off-path. Audit baseline treatment must follow Run677 result. No claimed savings or numerical ceiling.

## Preflight completed

Zcode ran guarded preflight. Actual installed-candidate equality and54CPU same-state scenarios passed, including cross-group aliases, dtypeoverflow, invalidation and originalguardfallback. All9cleanup statuses0. Main independently verified current source/script hashes against preflight before/after. No NPUperformance claim. Audit baseline resolved0 because Run677 standaloneaudit was REJECTED.
