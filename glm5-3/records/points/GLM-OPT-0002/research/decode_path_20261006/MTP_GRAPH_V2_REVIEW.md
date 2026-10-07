# Independent MTP Graph v2 constructor/lifecycle review

2026-10-07. Offline inspection only. No service/device action, candidate edit or frozen Run268 change.

## Correction and verdict

My prior statement that initialization ordering was safe was wrong. Actual `mtp_graph_sources/model_runner_v1.py` assigns `self.ascend_config = get_ascend_config()` at 407, calls `_set_up_drafter()` at 514, and only assigns `self.dynamic_eplb` at 542. The method definition at 701 does not determine call order. `/private/tmp/glm268-candidate-init-failure.raw` directly records the candidate line239 AttributeError on worker initialization. This is an initialization defect, not a model or graph numerical failure. Historical review remains unchanged.

The immutable v2 direct diff changes that predicate alone to `not get_ascend_config().eplb_config.dynamic_eplb`. **This fixes the observed defect with the same configuration semantics as the later runner assignment.** The getter is already successfully consumed at runner407 before drafter construction; actual config getter1522 returns the initialized singleton and otherwise raises. EplbConfig is an initialized dataclass field. No default-to-false or missing-attribute masking is introduced.

No additional equally concrete initialization blocker was found in the added static predicate. Other runner attributes used in the proposer preexist in the original path, and the new predicate no longer reads a later-created runner field. Later dummy/runtime references to dynamic_eplb occur after construction; they need not be replaced as part of this fix.

## Pointer and output recheck

The current candidate/v2 includes the prior requested cache fix: contract calls `impl._compose_sfa_kv_cache(caches)` before signing tensors. Actual SFA1406 joins main handles with `impl.indexer.k_cache.kv_cache`, including optional indexer scale; no device operation occurs in that composition. Thus replacing only the independent indexer allocation now changes the signature. This closes the specific omission reported previously.

For the admitted nonprobabilistic K1 merged body, `target_positions` is not a first-step input consumer: first-step model positions come from persistent `self._get_positions(n)`, which is signed. K1 exits before later-step logic; no additional target_positions pointer needs signing solely because it appears in kwargs. `sampling_metadata` selects greedy argmax under the admitted nonprobabilistic branch; per-call no-history/generator/grammar exclusions remain. Metadata is captured through ForwardContext, and the explicit field signatures cover its inspected current SFA/DCP tensor consumers, including main and indexer caches, local/replicated block and slot mappings, query/sequence tensors, rotary, mask and topk buffer.

The persistent MC2 mask comes from forward-context setup, not kwargs. Existing set_ascend_forward_context rewrites its prefix before the call, and its allocation is initialized after drafter construction but before model capture. V2 does not access it prematurely or replace it. Its runtime address/order still belongs in device correctness evidence; no concrete source path changing that allocation between these calls was found.

Owned `result.clone()` remains outside the inner captured runnable and ordered on the main stream before a later replay reuses graph output. It fixes overwrite aliasing while keeping existing draft D2H/event and allocator behavior. It is not permission to publish a successful iteration early or remove any CPU staging/count/KV/communication fence.

## Evidence limits and next decision

Read v2 CPU_result: 2 constructor-order cases, 1024 runtime boolean cases, 15 static exclusions, 19 pointer mismatches and owned-output test pass; source hash and actual node lines407/514/542 are retained. These tests now exercise the missing-attribute condition that the earlier mocks missed. They still do not validate NPU capture, HCCL, graph pool lifetime, all-rank selection or numerical output.

I found no new source-supported blocker comparable to the fixed ctor access or omitted indexer cache. Continue only the already scoped correctness direction after Run268 safe recovery and freezing v2; do not interpret this review as deployment authorization, device correctness or performance KEEP. Keep H6/H5 and existing sync/error lifecycle unchanged. No parameter retry or broader candidate is proposed.
