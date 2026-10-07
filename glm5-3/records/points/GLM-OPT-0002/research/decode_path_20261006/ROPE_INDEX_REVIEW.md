# RoPE row-index Challenger addendum

2026-10-07; offline source/raw inspection only. No candidate installation, device request, or performance verdict.

**Recommendation:** prioritize this bounded RoPE row-selection question ahead of the K1 dead gather and a broad host rewrite. Root's new correlation materially strengthens the source attribution. Native index_select is a plausible existing-operator substitution, not yet a proved faster device path.

## Personally checked facts

Actual `post_full_framework_sources/rotary_embedding.py:90–103` performs two cache[positions] expressions, each followed by unsqueeze(1), unsqueeze(2). SFA builder `sfa_v1.py:333–349` converts the positions slice to long and calls it with use_cache=(draft_index is None). Target returns views into persistent `_cos_mla`/`_sin_mla` after copies; draft returns independently selected tensors. Preserve both branches exactly.

In rank0 period3 raw, CPU indices57030/57038 and58751/58759 are aten::index, each followed by exactly the two unsqueezes. Enqueue correlations214310/214311/214602/214603 lie inside those index scopes and match their Dequeue events. This independently corroborates the reported device connection join and the RoPE expression pattern. This is strong source attribution, although the trace itself contains no Python source-line stack: uniqueness is a source/call-pattern inference, not a recorded line number. These tasks should no longer be provisionally attributed to hidden-state selection or unused K1 positions.

Read local exact op-plugin tree8b9c8534's `ops/opapi/IndexSelectKernelNpuOpApi.cpp` and `IndexKernelNpuOpApi.cpp`. Native index_select calls aclnnIndexSelect with output allocation; advanced index expands index tensors and calls aclnnIndex. A compatibility fallback exists in both. Thus there is a genuinely different native frontend/API route available. This source does **not** prove installed device dispatch will avoid SliceAiCore internally or be faster for these two-row, strided-cache shapes.

## Minimal semantic boundary

For 1-D int32/int64 positions and **equal cos/sin row counts**, one normalization `where(p < 0, p + N, p)` can be shared, followed by index_select(cache,0,normalized) and the unchanged unsqueezes. Gate row-count equality using tensor shape metadata; if unequal, retain the original pair of expressions. Example: cos has8 rows, sin10, p=-1 must select7 and9 respectively. One shared normalization by8 would silently select sin7.

* Valid negative p in[-N,-1] maps to[0,N-1]. p<-N remains negative; p>=N remains too large. The intended contract is to reject these invalid inputs, **not** clamp, modulo-wrap, or substitute zeros. Native NPU index_select error handling for both classes remains unverified; preserve evidence of errors rather than claiming identical exception timing/text.
* N=0 with empty indices must return empty selected rows with matching trailing shape; nonempty indices must fail. Cover duplicate/unsorted/boundary indices and noncontiguous positions, as well as empty input. An int32 path needs N representable in that dtype (or explicit wide normalization); actual SFA `.long()` avoids that concern locally.
* Non-1-D or other dtype positions must retain original behavior, including boolean-index semantics. Do not introduce `.item()`, CPU bounds scans, or a data-dependent Python branch to detect negative values.
* Cache storage can be strided: `_record_cos_and_sin_cache_interleaved` builds split/squeezed views. The oracle must use that construction, not only contiguous random caches. Compare values, shape, dtype, device, strides/contiguity and downstream accepted layout; retain copy order and persistent return-storage identity for use_cache=True. New selection temporaries must remain owned through their queued copy/consumer exactly as normal Tensor operations do.
* Do not use index_select(out=persistent_buffer) as an additional optimization here: it changes allocation, aliasing and copy contracts beyond the reviewed substitution. Preserve target-before-draft lifetime and target FULL metadata updates outside replay.

The generic normalization adds comparison/add/where device work. A GLM-only nonnegative fast path could avoid this, but `.long()` proves dtype, not bounds. Establish every real/dummy/padded/DCP position producer first; no unsourced assertion or device-to-host predicate is justified.

## Remaining decisive evidence

First, a CPU semantic oracle against the actual helper, including equal/unequal row counts, the actual interleaved noncontiguous cache layout, negative bounds, empty indices and persistent use_cache views, can falsify the proposed semantics without a model request. This review did not execute it.

Device value remains unknown until the installed native route for the exact cache strides/index shapes is observed: confirm aclnnIndexSelect rather than compatibility fallback, its actual hardware tasks, and the **combined** normalization + two selects + preserved copies cost. Four old Slice-task durations are neither predicted saving nor proof of critical contribution. No new broad profile or repeat of Run276 is required merely to strengthen already adequate attribution. Any later device action needs its own bounded frozen hypothesis; this review does not activate one.

Keep H6/H5. Stop pursuing the four slow tasks as K1 dead-gather/hidden-row selection, and stop assuming a different ATen name guarantees a different or faster kernel. Largest overall removable gap remains unknown, but this is now a better-specified local framework question than it was in POST_FULL_FRAMEWORK_REVIEW.md.
