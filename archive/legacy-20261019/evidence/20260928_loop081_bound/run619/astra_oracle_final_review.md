# Run619 final independent row-mapper review

**SCOPED PASS for the pure CPU mapper and22 fixtures. NOT LIVE-READY.** No further toy revision is required for the two reported Run618 defects.

Reviewed script SHA256 ef98c479d08c6d58dc898dd805085bff0853f0708d7b94338160837b56ef4027 and JSON SHA25651869101193a2efa2240f83f2abb19d0063e19dd5dd00d6f5e4527d0eb6d37a0. Independent CPU execution with only top-level filesystem-output/print calls removed reproduced the saved result exactly:7 arithmetic passes and15 expected rejections. Source SHA checks executed successfully.

Both targeted gates are correct. Every block-table element now requires true Python int within signed32 range, so the previous2**40 payload challenge is rejected even when capacity is symbolic. The dense branch rejects K-Q+j+win_right above INT32_MAX before clamping. I re-ran both original challenges and checked equality at the signed32 boundary is accepted. Under the helper's K>=Q>0 and valid j constraints, K-Q+j is nonnegative and below INT32_MAX; subtracting its bounded nonnegative win_left stays within signed32. The maximum-left-window boundary also returns the correct clamped page mapping.

The helper preserves ordered duplicate entries despite the introductory shorthand “set”; do not deduplicate its semantic output. Existing SPARSE_LIMIT2048, table presence, sparse int32/type, ND-only and bounded row-address checks remain intact. These are selected guards, not a complete native invocation validator. Toy block4/head2, symbolic optional capacity and missing full call/metadata inputs remain explicitly outside production admission.

Promotion remains limited to source-derived logical mapping for supplied supported values. Actual dispatch/layout/dtype/head geometry, query/request slicing, cache storage lifetime/content generation, source-to-loaded-binary/tiling identity, and metadata epoch/coverage must be validated by the observer packet. No physical first-read/HBM or necessary-work inference follows from this CPU result. All four output fields for actual-W0 rows, physical HBM and strict floors remain null; formal Current571.681tok/s is unchanged.

This closes the Run617–619 mapper review. Further work should implement/review the bounded typed observer and invocation schema rather than continue expanding isolated toy fixtures without a new concrete production risk.
