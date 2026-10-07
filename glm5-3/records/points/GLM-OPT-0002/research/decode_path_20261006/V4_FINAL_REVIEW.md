# Observer v4 final independent review

2026-10-07. Offline only. Reviewed full production-v2→observer-v4 diff and test source/result. Verified file SHA256 **8f037c033f1ed38002c07eb03789515756433c35e7c5f4f192fa9aebd19461a1**. No candidate/frozen Run changes, no service or device operation.

## Final source verdict

**No remaining concrete observer blocker found for the frozen scoped diagnostic.** The previously identified failures are addressed:

- The nonexistent extra-context field is gone. Real-call discrimination uses the actual proposer kwargs contract; dummy/profile/capture omit sampling_metadata.
- NONE never computes a graph contract. FULL inspection follows descriptor existence/uniformity/size admission. Production retains its own checks and failure behavior; no exception is swallowed.
- Signature None or mismatch now yields backend names/reason only. No metadata dereference or before/after tensor D2H is performed. Empty/None/non-SFA/no-DCP metadata counterexamples therefore no longer trigger the observer-only attribute error.
- Detailed snapshots require an admitted contract; it guarantees a nonempty map of SFA-DCP values. Fixed output ownership checks only occur on that admitted path. Mode strings match actual enum behavior.
- Descriptor=None is exercised under NONE, where production supports it; the test does not falsely claim FULL+None support.

CPU_result PASS covers actual ForwardContext/proxy AST and production `_run_glm_k1_graph` AST, 16 lifecycle cases, nine fallback cases and four dummy cases, plus real filesystem/mmap. I read the fallback tests: metadata rejection cases replace the contract with a None-returning test double. They test observer behavior **after contract rejection**, not independently prove the full real contract rejects every malformed Python object. This is an appropriate narrow claim; production contract tests are separate. Tensor/ACL/device execution remains mocked.

Residual limits: `actual_replay` is explicitly labeled matching-contract/source-branch evidence, not standalone device proof. Detailed snapshots synchronize and cannot be included as performance samples. Selector mode changes must occur at controller-owned quiescent boundaries because selection and observation read the mmap separately. These are known diagnostic constraints, not a new source defect.

## Three-failure review: is another diagnostic justified?

The threshold requires breaking the run/retry pattern and reassessing the route. It does not make a never-reached graph path disproven. Available decisive evidence separates an actual constructor ordering bug (Run268) from the latest observer namespace bug (Run270). Run270 reached all-rank loaded/capable state but no capture evidence; no performance comparison occurred. Do not reclassify these as graph incompatibility or as negative gain results. I do not infer details of the intervening failure beyond the parent's failure-count report.

**KEEP only one minimal capture/replay correctness diagnostic as a worthwhile next question, conditional on Root's frozen reset and completed safe recovery.** Reasons: GLM's forced-eager boundary still blocks a concrete repeated-host-submission removal; K1 model/body and dynamic persistent input machinery are source-supported; the specific host integration errors now have actual-source regression coverage. The next result can materially decide whether to implement/continue this graph path. A matched performance Run or broad shape/config scan remains premature.

Freeze one attempt that distinguishes: (a) no admission/capture—inspect exact selector/signature reason; (b) capture but no replay—inspect actual descriptor/pointer reason; (c) replay with numerical/KV failure—preserve the first failure; (d) clean all-rank replay and correct output—only then consider a separate performance decision. Do not automatically retry another observer failure, silently weaken a contract, or count successful model loading as graph success. Preserve H6/H5 and the existing bounded recovery procedure in every outcome.

This is an independent route recommendation, not deployment authorization, numerical validation, or PERF_KEEP.
