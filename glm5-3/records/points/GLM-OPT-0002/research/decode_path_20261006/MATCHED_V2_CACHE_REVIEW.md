# Independent matched-v2 cache/transition correction review

2026-10-07. Offline only. Verified pd_compare SHA `640df051b0c5c60ccf9f01455da9cf5676d4fbe06f788861ada57a5c0d785e2a`, direct prior→v2 diff, actual Run274 cache/terminal raw, cache regression source/result and actual scheduler SHA `c67bda2886b52865ddafabaae7d797c359e930752f374421a33e537d94a5f45a`. No service/Run or frozen274 edits.

## Cache correction

My earlier review accepted the wrong P external-query-zero requirement. Actual scheduler823–826 records connector queries as request.num_tokens minus new local computed tokens, and hits as externally computed tokens. Query count is admission volume, not evidence that P received remote KV.

Run274 A1_complete raw is P local58/0, external58/0; D local58/0, external58/58. V2 accepts exactly this declared fresh58-token case: both local queries equal prompt_tokens with zero hits; P external queries equal prompt_tokens with zero hits; D external queries and hits equal prompt_tokens. This follows both the inspected producer semantics and actual raw. It does not weaken D's native transfer gate or claim P external hits. Four-complete cache signature equality and strict parser/reset checks remain unchanged.

Read CPU regression: invokes actual validate_cache AST on Run274 raw and rejects five individually corrupted counters. No performance result is manufactured from the failed A1-only attempt. These exact counters describe this fixture/configuration, not a general invariant for arbitrary local-prefix-hit workloads or the formal93% condition.

## Transition correction

Run274 retained_witness actually has transition5 on all rows, while its decision is INVALIDATED/H11=false, terminal_verified=true, recovery_used=false. V2 takes PLAN.initial_transition and verifies it against a live mode0 witness before priming, rather than stale Run273 retained_witness. Subsequent warm+1 and complete-same assertions remain. This is the correct separation: Run273 supplies immutable numerical correctness; current resident witness supplies mutable observer transition state.

The live witness RPC also checks resident identities/selector state through the existing parent consumer. A stale/mistyped initial_transition fails instead of silently reusing old observation rows. Freeze5 only while current live state remains5; the assertion is the authority at execution. No model patch change or correctness prerequisite downgrade is involved.

## Verdict

No concrete blocker found in these two targeted corrections. Run274 remains a failed measurement-gate attempt, not numerical failure, performance negative, or positive evidence. Its INVALIDATED terminal decision stays intact. Existing H6/H5 recovery/retention rules, finite timings, matching aggregate workload, two-pair savings/drift criteria and final effective-decision gate are unchanged. This review does not start or authorize Run275 and does not claim any new gain.
