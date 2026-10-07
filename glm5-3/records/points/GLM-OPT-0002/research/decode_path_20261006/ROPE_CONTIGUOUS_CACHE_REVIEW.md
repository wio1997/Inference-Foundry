# RoPE initialization-contiguity review

2026-10-07. Offline source review only; no device/requests/deployment. Supersedes native-index_select as the preferred semantic boundary; that prototype and its CPU evidence remain parked.

**Recommendation:** the two initialization `.contiguous()` calls are a substantially cleaner candidate than changing indexing. No concrete semantic blocker found for the actual default-RoPE path. This is a candidate recommendation, not measured gain or PERF_KEEP.

## Source-supported facts

Personally read actual `post_full_framework_sources/rotary_embedding.py` initialization, global-cache helpers and consumers, and actual Deepseek model constructor (`vllm__vllm__model_executor__models__deepseek_v2.py:1075–1146`). The original helper creates repeat storage then chunk/squeeze views: for source[N,64], repeat storage is[N,2,64], and each resulting cache is[N,64] with row stride128. Adding contiguous copies independently preserves every cache value and changes row stride to64. The original repeat already separates these globals from the model's source cache; this change does not newly break an alias to model weights/source RoPE.

`get_cos_and_sin_mla` only reads `_cos_cache`/`_sin_cache`; its writes are to the distinct persistent `_cos_mla`/`_sin_mla`. In the inspected snapshots no other consumer mutates the lookup caches. Both record helpers can assign globals: interleaved returns when either cache is already set; the other `_record_cos_and_sin_cache` unconditionally rebinds them for the separate scaling path. This patch must touch only the interleaved helper and preserve both guards/assignment behavior. It does not solve hypothetical mixed-model global-cache design issues, nor introduce one.

Main attention constructs `self.rotary_emb` before `self.indexer_rope_emb`. They use the same qk_rope dimension/max-position/rope parameters, potentially differing in is_neox_style. The first global registration wins as before; contiguous does not change which instance wins or reorder values according to indexer style. The indexer's own `rotary_emb.cos_sin_cache` remains untouched. Later layers and MTP seeing the global guard do not allocate new dense tables per layer.

The parent reports actual CSV shape[N=1048576,128] → [1048576,64] for each slow Slice, BF16 ND, with MTE2 ratios near .99. I independently checked the earlier task-to-RoPE source pattern, but this particular CSV was not present in the local frontier directory I inspected; its exact rows remain parent-read evidence here. The reported geometry is precisely consistent with materializing a half-width strided view before two-position indexing. It explains the full-table bandwidth cost much more specifically than a generic slow-index hypothesis.

## Semantics and lifetime

Negative/out-of-range/empty/non-1D/bool indexing retain the original expressions and native behavior. There is no new normalization or per-call bounds branch. `use_cache=True` keeps the original persistent destination copies and return views; draft returns remain newly selected tensors. CPU oracle should execute the **original and changed record function**, then the unchanged lookup, and include interleaved values, repeated registration, main/indexer-like sequential initialization, both use_cache modes and source-input nonmutation. Private global cache stride/alias identity intentionally changes; selected results and public/persistent-buffer contracts must match.

Initialization completes before graph capture in the normal model-load sequence. Subsequent globals keep stable addresses; target replay binds the separate persistent cos/sin output buffers, whose identity does not change. Do not hot-rebind the lookup globals after capture as part of an A/B toggle. Build each layout before its graph's lifecycle, or demonstrate the actual toggle leaves all captured consumers valid. No new fence or removal is justified.

For N=1048576, width64, BF16: each dense table128MiB; old shared repeat backing256MiB; new pair256MiB. Additional steady live payload is therefore zero once the repeat backing is released. During the helper both original chunk views retain the repeat storage until function return, so peak live allocation can rise by256MiB. Allocator reservation may remain elevated after free; do not equate equal live bytes with identical reported reserved HBM. The unchanged original model cos_sin_cache is additional in both cases. OOM/headroom is the concrete operational limit to check during ordinary candidate correctness, not a reason to alter data semantics.

## Decision boundary

The existing trace plus reported CSV adequately motivates a single candidate; no further broad profile or standalone native-index microtest is needed to rediscover this root. CPU oracle establishes semantics, not disappearance of the Slice tasks or E2E benefit. Next correctness and frozen matched full PD comparison must preserve H6/H5, existing dynamic behavior, all-rank KV/correctness and workload signatures. No stacking with nativeSelect, K1 dead gather, or H4.

Largest realized gain remains unknown. The strongest falsifier is that dense caches fail to eliminate the old full-table materialization or combined E2E saving does not repeat; neither outcome licenses retrying favorable workloads. Do not call all 1.32ms removable before observing the candidate, or multiply it across ranks. No change to computation kernels is proposed.
