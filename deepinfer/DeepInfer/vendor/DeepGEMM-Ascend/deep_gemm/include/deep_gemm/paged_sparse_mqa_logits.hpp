#pragma once

#include <deep_gemm/ascend.hpp>
#include <deep_gemm/layout/sparse_mqa_logits.hpp>
#include <deep_gemm/simd/sparse_mqa_logits.hpp>
#include <simt_api/asc_simt.h>

namespace deep_gemm {
using namespace __cce_simd;
using namespace layout::paged_sparse_mqa_logits;

template <uint32_t kRows, uint64_t kPageStride, uint32_t kPageKV, bool kIsFP4, uint32_t kSparseBlockKV>
__simt_vf__ __launch_bounds__(kRows) inline void paged_sparse_mqa_gather_sf(
    __gm__ uint8_t* raw_sf, __gm__ uint32_t* physical_kv_token_starts, __ubuf__ uint32_t* ub_sf, uint32_t num_rows, uint32_t block_begin
) {
    constexpr uint32_t kDataBytes = kHeadDim / (kIsFP4 ? 2 : 1);
    static_assert(kSparseSFBytes == sizeof(uint32_t) && kPageStride % sizeof(uint64_t) == 0);
    const uint32_t row_idx = threadIdx.x * 2;
    if (row_idx >= num_rows)
        return;
    const uint32_t block_idx = block_begin + row_idx / kSparseBlockKV;
    const uint32_t token_start = asc_ldca(physical_kv_token_starts + block_idx);
    if (token_start == 0xffffffffu)
        return;
    const uint32_t token_idx = token_start + row_idx % kSparseBlockKV;
    uint64_t sf_offset_bytes;
    if constexpr (kPageStride == kPageKV * (kDataBytes + kSparseSFBytes)) {
        // The page multiplier is <= page_kv, so the uint32_t product cannot overflow.
        static_assert(kDataBytes / kSparseSFBytes + 1 <= kPageKV);
        const uint32_t sf_page_offset = (token_idx / kPageKV) * (kDataBytes / kSparseSFBytes + 1);
        sf_offset_bytes = ((static_cast<uint64_t>(sf_page_offset) * kPageKV) | (token_idx % kPageKV)) * kSparseSFBytes;
    } else {
        sf_offset_bytes = static_cast<uint64_t>(token_idx / kPageKV) * kPageStride + (token_idx % kPageKV) * kSparseSFBytes;
    }
    auto* sf_src = reinterpret_cast<__gm__ uint32_t*>(raw_sf + sf_offset_bytes);
    constexpr uint32_t kStageWords = 2 * kRows * (kDataBytes + kSparseSFBytes) / sizeof(uint32_t);
    const uint32_t dst_idx = row_idx + (row_idx / kRows) * (kStageWords - kRows);
    reinterpret_cast<__ubuf__ uint64_t*>(ub_sf)[dst_idx / 2] =
        __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::CACHEABLE>(reinterpret_cast<__gm__ uint64_t*>(sf_src));
}

// Prefetch token-major DATA/SF into the RAW buffers.
template <uint32_t kSplitKV, uint64_t kPageStride, uint32_t kPageKV, bool kIsFP4, uint32_t kSparseBlockKV,
          bool kSimtSF = kIsFP4 && kSparseBlockKV == 8 && kPageStride % sizeof(uint64_t) == 0>
__aicore__ inline void paged_sparse_mqa_prefetch_kv(
    __gm__ uint8_t* raw_kv, __gm__ uint32_t* physical_kv_token_starts, uint32_t page, uint32_t num_rows,
    __ubuf__ uint8_t* scratch, uint32_t chunk_begin, uint32_t chunk_end
) {
    constexpr uint32_t kDataBytes = kHeadDim / (kIsFP4 ? 2 : 1);
    if constexpr (kSimtSF) {
        if (reinterpret_cast<uintptr_t>(raw_kv) % sizeof(uint64_t) != 0) {
            paged_sparse_mqa_prefetch_kv<kSplitKV, kPageStride, kPageKV, kIsFP4, kSparseBlockKV, false>(
                raw_kv, physical_kv_token_starts, page, num_rows, scratch, chunk_begin, chunk_end);
            return;
        }
    }
    constexpr uint32_t kRows = kSplitKV / 2, kTokenBytes = kDataBytes + kSparseSFBytes, kRawBytes = kRows * kTokenBytes;
    constexpr uint32_t kStageBytes = 2 * kRawBytes;  // RAW DATA/SF + converted copy
    static_assert((kPageKV == 64 || kPageKV == 128) && kPageStride >= kPageKV * kTokenBytes && kPageStride % 4 == 0);
    const auto byte_offset = [](uint32_t token, uint32_t token_bytes) __aicore__ {
        return static_cast<uint64_t>(token / kPageKV) * kPageStride + (token % kPageKV) * token_bytes;
    };
    const uint32_t num_chunks = ceil_div(num_rows, kRows);
    for (uint32_t block_idx = 0; chunk_begin == 0 && block_idx < num_rows / kSparseBlockKV; block_idx += kDCacheLineBytes / sizeof(uint32_t))
        asc_datacache_preload(reinterpret_cast<__gm__ uint64_t*>(physical_kv_token_starts + page * (kCompactPageTokens / kSparseBlockKV) + block_idx), 0);
    for (uint32_t chunk_idx = chunk_begin; chunk_idx < num_chunks && chunk_idx < chunk_end; ++ chunk_idx) {
        const auto event = static_cast<event_t>(chunk_idx);
        auto raw = scratch + chunk_idx * kStageBytes;
        const uint32_t rows = min(kRows, num_rows - chunk_idx * kRows);
        asc_sync_wait(PIPE_V, PIPE_MTE2, event);
        for (uint32_t block_idx = 0; block_idx < rows / kSparseBlockKV; block_idx += 2) {
            const uint32_t offset = page * (kCompactPageTokens / kSparseBlockKV) + chunk_idx * (kRows / kSparseBlockKV) + block_idx;
            const uint32_t first = physical_kv_token_starts[offset], second = physical_kv_token_starts[offset + 1];
            if (first == 0xffffffffu)
                continue;  // Prepare sorts invalid blocks last.
            const auto copy_region = [&](uint32_t token_bytes, __gm__ uint8_t* src, __ubuf__ uint8_t* dst) __aicore__ {
                const uint32_t block_bytes = kSparseBlockKV * token_bytes;
                dst += block_idx * block_bytes;
                const uint64_t stride = byte_offset(second, token_bytes) - byte_offset(first, token_bytes);
                if (second != 0xffffffffu && second != first && stride <= (1ull << 40) - 1) {
                    copy_gm_to_ub(dst, src + byte_offset(first, token_bytes),
                                  2, block_bytes, stride, asc_load_l2_cache_mode::NORMAL_LAST_VICTIM);
                    return;
                }
                copy_gm_to_ub(dst, src + byte_offset(first, token_bytes),
                              1, block_bytes, block_bytes, asc_load_l2_cache_mode::NORMAL_LAST_VICTIM);
                if (second != 0xffffffffu)
                    copy_gm_to_ub(dst + block_bytes, src + byte_offset(second, token_bytes),
                                  1, block_bytes, block_bytes, asc_load_l2_cache_mode::NORMAL_LAST_VICTIM);
            };
            copy_region(kDataBytes, raw_kv, raw);
            if constexpr (!kSimtSF)
                copy_region(kSparseSFBytes, raw_kv + kPageKV * kDataBytes, raw + kRows * kDataBytes);
        }
        asc_sync_notify(PIPE_MTE2, PIPE_V, event);
    }
    // publish_kv releases both RAW chunks; gather both SF halves on the first prefetch.
    if constexpr (kSimtSF) {
        if (chunk_begin == 0 && num_rows != 0)
            asc_vf_call<paged_sparse_mqa_gather_sf<kRows, kPageStride, kPageKV, kIsFP4, kSparseBlockKV>>(
                cce::dim3(kRows),
                raw_kv + kPageKV * kDataBytes, physical_kv_token_starts,
                reinterpret_cast<__ubuf__ uint32_t*>(scratch + kRows * kDataBytes), num_rows, page * (kCompactPageTokens / kSparseBlockKV));
    }
}

// Convert KV and publish it to compact storage and L1.
template <uint32_t kSplitKV, bool kIsFP4>
__aicore__ inline void paged_sparse_mqa_publish_kv(
    __gm__ uint8_t* compact, uint32_t page, uint32_t num_rows, __ubuf__ uint8_t* scratch,
    __cbuf__ uint8_t* l1_data, __cbuf__ uint8_t* l1_sf
) {
    constexpr uint32_t kDataBytes = kHeadDim / (kIsFP4 ? 2 : 1);
    constexpr uint32_t kRows = kSplitKV / 2, kTokenBytes = kDataBytes + kSparseSFBytes, kRawBytes = kRows * kTokenBytes;
    constexpr uint32_t kStageBytes = 2 * kRawBytes;
    const uint32_t num_chunks = ceil_div(num_rows, kRows);
    for (uint32_t chunk_idx = 0; chunk_idx < num_chunks; ++ chunk_idx) {
        const auto event = static_cast<event_t>(chunk_idx);
        const uint32_t begin = chunk_idx * kRows, rows = min(kRows, num_rows - begin);
        auto raw = scratch + chunk_idx * kStageBytes;
        auto data = raw + kRawBytes;
        auto scales = data + kRows * kDataBytes;
        asc_sync_wait(PIPE_MTE2, PIPE_V, event);
        asc_sync_wait(PIPE_MTE3, PIPE_V, event);
        vf_sparse_mqa_logits_convert_kv<kRows, kIsFP4>(
            reinterpret_cast<__ubuf__ uint32_t*>(raw),
            reinterpret_cast<__ubuf__ uint32_t*>(data), reinterpret_cast<__ubuf__ uint16_t*>(scales),
            reinterpret_cast<__ubuf__ uint16_t*>(raw + kRows * kDataBytes), rows);
        asc_sync_notify(PIPE_V, PIPE_MTE2, event);
        asc_sync_notify(PIPE_V, PIPE_MTE3, event);
        asc_sync_wait(PIPE_V, PIPE_MTE3, event);
        // Pack each supply in its final L1 layout, without increasing its page allocation.
        auto output = compact + static_cast<uint64_t>(page) * (kCompactPageTokens * kTokenBytes);
        asc_copy_ub2gm_align(output + num_rows * kDataBytes + begin * kSparseSFBytes, scales, 1,
                             rows * kSparseSFBytes, asc_store_l2_cache_mode::NOTALLOC_CLEAN,
                             rows * kSparseSFBytes, rows * kSparseSFBytes);
        asc_copy_ub2gm_align(output + begin * 32, data, kDataBytes / 32, rows * 32,
                             asc_store_l2_cache_mode::NOTALLOC_CLEAN, num_rows * 32, kRows * 32);
        for (uint32_t frag_idx = 0; frag_idx < kDataBytes / 32; ++ frag_idx)
            asc_copy_ub2l1(l1_data + (frag_idx * kSplitKV + begin) * 32,
                           data + frag_idx * kRows * 32, aligned(rows, 128u) * 32);
        asc_copy_ub2l1(l1_sf + begin * kSparseSFBytes, scales, aligned(rows, 128u) * kSparseSFBytes);
        asc_sync_notify(PIPE_MTE3, PIPE_V, event);
    }
}

template <bool kFirst, bool kIsFP4, uint32_t kNumHeads, uint32_t kSparseBlockKV, uint64_t kPageStride, uint32_t kPageKV>
__global__ __mix__(1, 2) void paged_sparse_mqa_logits_impl(
    uint32_t weights_stride, uint32_t logits_stride,
    __gm__ uint8_t* q, __gm__ int16_t* sf_q, __gm__ uint8_t* raw_kv,
    __gm__ uint8_t* compact, __gm__ bfloat16_t* weights, __gm__ bfloat16_t* logits,
    __gm__ uint32_t* physical_kv_token_starts, __gm__ uint16_t* mapping, __gm__ uint32_t* kv_splits
) {
    using qk_dtype_t = std::conditional_t<kIsFP4, float4_e2m1x2_t, float8_e4m3_t>;
    constexpr uint32_t kHeadPitch = (kNumHeads + 15) / 16 * 16;
    constexpr uint32_t kSplitKV = get_split_kv(true, kIsFP4);
    constexpr uint32_t kDataBytes = kIsFP4 ? 64 : 128, kTokenBytes = kDataBytes + kSparseSFBytes;
    static_assert(kNumHeads >= 4 && kNumHeads <= kNumMaxHeads && kNumHeads % 4 == 0);
    constexpr uint32_t kNumScoreStages = kFirst ? 1 : 2;
    constexpr uint32_t kNumKVStages = 6;
    constexpr uint32_t kPageBytes = kCompactPageTokens * kTokenBytes;
    constexpr uint32_t kKVStageBytes = kSplitKV * (kFirst ? kDataBytes : kTokenBytes);
    constexpr uint32_t kFinishFlag = kNumScoreStages + kNumKVStages;
    constexpr l1_ptr<qk_dtype_t> l1q(4 * kHeadPitch, kHeadDim);
    constexpr l1_ptr<qk_dtype_t> l1kv(kSplitKV, kHeadDim, l1q.offset(1));
    constexpr l1_ptr<int16_t> l1_sf_q(4 * kHeadPitch, 2, l1kv.addr + kNumKVStages * kKVStageBytes);
    constexpr l1_ptr<int16_t> l1_sf_kv(kFirst ? kSplitKV : 0, 2, l1_sf_q.offset(1));
    constexpr l1_ptr<qk_dtype_t> next_l1q(4 * kHeadPitch, kHeadDim, l1_sf_kv.offset(kNumKVStages));
    constexpr l1_ptr<int16_t> next_l1_sf_q(4 * kHeadPitch, 2, next_l1q.offset(1));
    static_assert(kFirst || next_l1_sf_q.offset(1) <= L1SizeBytes);
    constexpr l0a_ptr<qk_dtype_t> pair(2 * kHeadPitch, kHeadDim);
    constexpr l0a_ptr<qk_dtype_t> single(kHeadPitch, kHeadDim, pair.offset(2));
    constexpr l0b_ptr<qk_dtype_t> l0b(kSplitKV, kHeadDim);
    constexpr uint32_t kNumL0BStages = l0b.offset(2) <= L0BSizeBytes ? 2 : 1;
    constexpr l0c_ptr<float> l0c(2 * kHeadPitch, kSplitKV);
    constexpr ub_ptr<bfloat16_t> ub_score(2 * kHeadPitch, kSplitKV);
    constexpr ub_ptr<bfloat16_t> ub_weights(1, aligned(2 * kNumHeads, 16u), ub_score.offset(kNumScoreStages));
    constexpr ub_ptr<bfloat16_t> ub_q_block_logits(1, 2 * kMaxOutputTokens, ub_weights.offset(1));
    constexpr ub_ptr<uint16_t> ub_mapping(1, 2 * kMaxOutputTokens / kSparseBlockKV + 128, ub_q_block_logits.offset(1));
    constexpr ub_ptr<bfloat16_t> next_weights(1, aligned(2 * kNumHeads, 16u), ub_mapping.offset(1));
    constexpr ub_ptr<uint16_t> next_ub_mapping(1, 2 * kMaxOutputTokens / kSparseBlockKV + 128, next_weights.offset(1));
    static_assert(next_ub_mapping.offset(1) <= kSparseUBBytes);
    // Two kSplitKV/2-row chunks, each holding RAW DATA/SF and its converted copy.
    constexpr ub_ptr<uint8_t> ub_first_scratch(1, 2 * kSplitKV * kTokenBytes,
                                               kFirst ? next_ub_mapping.offset(1) : ub_mapping.offset(1));
    static_assert(!kFirst || ub_first_scratch.offset(1) <= kSparseUBBytes);
    static_assert(l1_sf_kv.offset(kNumKVStages) <= L1SizeBytes);
    static_assert(single.offset(4) <= L0ASizeBytes && l0c.offset(2) <= L0CSizeBytes);
    asc_init();
    const auto ranges = reinterpret_cast<__gm__ SplitKVRange*>(kv_splits);
    const uint32_t first_begin = ranges[2 * block_idx].begin, second_begin = ranges[2 * block_idx + 1].begin;
    const uint32_t num_first_kv_splits = ranges[2 * block_idx].end - first_begin;
    const uint64_t num_kv_splits = num_first_kv_splits + static_cast<uint64_t>(ranges[2 * block_idx + 1].end - second_begin);
    const auto kv_split_records = reinterpret_cast<__gm__ SplitKV*>(ranges + 2 * get_block_num());
    const auto get_kv_split = [&](uint64_t kv_split_idx) __aicore__ {
        return kv_split_records + (kv_split_idx < num_first_kv_splits ? first_begin + kv_split_idx : second_begin + kv_split_idx - num_first_kv_splits);
    };
    if ASCEND_IS_AIC {
        asc_set_mmad_direction_n();
        set_flags<PIPE_M, PIPE_MTE1, kNumL0BStages>();
        if constexpr (!kFirst)
            set_flags<PIPE_MTE1, PIPE_MTE2, kNumKVStages, 2>();
        if constexpr (kFirst)
            for (uint32_t stage = 0; stage < kNumKVStages; ++ stage)
                set_intra_block(PIPE_MTE1, (stage ^ 1u) & 1u, kNumScoreStages + stage);
        uint32_t l0c_stage_idx = 0, l0b_stage_idx = 0, kv_stage_idx = 0;
        uint32_t owner_stages[2] = {};
        uint32_t query_stage = 0;
        const auto prefetch_data = [&](uint64_t kv_split_idx) __aicore__ {
            if constexpr (!kFirst) {
                if (kv_split_idx >= num_kv_splits)
                    return;
                const uint32_t stage = kv_split_idx % kNumKVStages;
                const auto record = get_kv_split(kv_split_idx);
                const uint32_t page = record->compact_page_idx, n = aligned(SplitKV::num_blocks(record->control) * kSparseBlockKV, kCompactPageTokens);
                const auto event = static_cast<event_t>(2 + stage);
                asc_sync_wait(PIPE_MTE1, PIPE_MTE2, event);
                asc_set_gm2l1_loop_size(1, 1);
                asc_set_gm2l1_loop1_stride(0, 0);
                asc_set_gm2l1_loop2_stride(0, 0);
                asc_copy_gm2l1_align(l1kv.offset_bytes(stage * kKVStageBytes).template ptr<uint8_t>(),
                                     compact + static_cast<uint64_t>(page) * kPageBytes, 1, n * kTokenBytes, 0, 0, false,
                                     asc_load_l2_cache_mode::NOTALLOC_KEEP, n * kTokenBytes, n * kTokenBytes);
                asc_sync_notify(PIPE_MTE2, PIPE_MTE1, event);
            }
        };
        const gm_ptr<int16_t> sf_q_t(2, reinterpret_cast<uintptr_t>(sf_q));
        const auto load_query = [&](uint32_t query_begin, uint32_t num_q_block_queries, auto dst_q, auto dst_sf) __aicore__ {
            if constexpr (kNumHeads == kHeadPitch) {
                copy_gm_to_l1(dst_q, gm_ptr<qk_dtype_t>(kDataBytes,
                                                        reinterpret_cast<uintptr_t>(q + static_cast<uint64_t>(query_begin) * kNumHeads * kDataBytes)), num_q_block_queries * kNumHeads, kHeadDim);
                copy_gm_to_l1_mx(dst_sf, sf_q_t.index(query_begin * kNumHeads, 0), num_q_block_queries * kNumHeads, 2);
            } else {
                for (uint32_t query = 0; query < num_q_block_queries; ++ query) {
                    const l1_ptr<qk_dtype_t> dst(4 * kHeadPitch, kHeadDim, dst_q.addr + query * kHeadPitch * 32);
                    const l1_ptr<int16_t> sf(4 * kHeadPitch, 2, dst_sf.addr + query * kHeadPitch * 4);
                    fill_mn_tail_block(dst, kNumHeads, kHeadPitch);
                    copy_gm_to_l1(dst, gm_ptr<qk_dtype_t>(kDataBytes,
                                                          reinterpret_cast<uintptr_t>(q + static_cast<uint64_t>(query_begin + query) * kNumHeads * kDataBytes)), kNumHeads, kHeadDim);
                    copy_gm_to_l1_mx(sf, sf_q_t.index((query_begin + query) * kNumHeads, 0), kNumHeads, 2);
                }
            }
        };
        const auto prefetch_query = [&](uint64_t kv_split_idx, uint32_t stage) __aicore__ {
            if constexpr (!kFirst) {
                if (kv_split_idx >= num_kv_splits)
                    return;
                const auto kv_split = get_kv_split(kv_split_idx);
                const uint32_t query_begin = kv_split->query_begin, num_q_block_queries = SplitKV::num_q_block_queries(kv_split->control);
                const auto event = static_cast<event_t>(stage);
                asc_sync_wait(PIPE_MTE1, PIPE_MTE2, event);
                const auto dst_q = stage ? next_l1q : l1q;
                const auto dst_sf = stage ? next_l1_sf_q : l1_sf_q;
                load_query(query_begin, num_q_block_queries, dst_q, dst_sf);
                asc_sync_notify(PIPE_MTE2, PIPE_MTE1, event);
            }
        };
        if constexpr (!kFirst)
            set_flags<PIPE_MTE1, PIPE_MTE2, 2>();
        prefetch_query(0, 0);
        prefetch_data(0);
        for (uint64_t kv_split_idx = 0; kv_split_idx < num_kv_splits; ++ kv_split_idx) {
            const auto kv_split = get_kv_split(kv_split_idx);
            for (uint32_t pair_idx = 0; pair_idx < 2; ++ pair_idx) {
                const uint32_t query_mask = SplitKV::query_mask(kv_split->control, pair_idx);
                if (!query_mask)
                    continue;
                const bool load_kv = pair_idx == 0 || !SplitKV::query_mask(kv_split->control, 0);
                const bool release_kv = pair_idx == 1 || !SplitKV::query_mask(kv_split->control, 1);
                const uint32_t issued_n = (SplitKV::num_blocks(kv_split->control) * kSparseBlockKV + 127) / 128 * 128;
                if (load_kv && (kv_split->control & SplitKV::kQBlockBegin)) {
                    // Return Q storage only after all preceding matrix reads finish.
                    asc_sync_notify(PIPE_M, PIPE_MTE1, EVENT_ID2);
                    asc_sync_wait(PIPE_M, PIPE_MTE1, EVENT_ID2);
                    if constexpr (kFirst) {
                        asc_sync_notify(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
                        asc_sync_wait(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
                    }
                    const uint32_t query_begin = kv_split->query_begin, num_q_block_queries = SplitKV::num_q_block_queries(kv_split->control);
                    if constexpr (kFirst) {
                        load_query(query_begin, num_q_block_queries, l1q, l1_sf_q);
                        asc_sync_notify(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
                        asc_sync_wait(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
                    } else {
                        asc_sync_wait(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(query_stage));
                        prefetch_query(kv_split_idx + kv_split->num_q_block_kv_splits, query_stage ^ 1u);
                    }
                    const auto current_q = kFirst || query_stage == 0 ? l1q : next_l1q;
                    const auto current_sf = kFirst || query_stage == 0 ? l1_sf_q : next_l1_sf_q;
                    for (uint32_t idx = 0; idx < (num_q_block_queries + 1) / 2; ++ idx) {
                        if (idx * 2 + 1 < num_q_block_queries) {
                            copy_l1_to_l0a(pair[idx], current_q, idx * 2 * kHeadPitch, 0);
                            copy_l1_to_l0a_mx(pair[idx], current_sf, idx * 2 * kHeadPitch, 0, kHeadDim);
                        }
                        for (uint32_t q_idx = 0; q_idx < min(2u, num_q_block_queries - idx * 2); ++ q_idx) {
                            copy_l1_to_l0a(single[idx * 2 + q_idx], current_q, idx * 2 * kHeadPitch + q_idx * kHeadPitch, 0);
                            copy_l1_to_l0a_mx(single[idx * 2 + q_idx], current_sf, idx * 2 * kHeadPitch + q_idx * kHeadPitch, 0, kHeadDim);
                        }
                    }
                    if constexpr (!kFirst) {
                        asc_sync_notify(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(query_stage));
                        query_stage ^= 1u;
                    }
                    asc_sync_notify(PIPE_MTE1, PIPE_M, EVENT_ID2);
                    asc_sync_wait(PIPE_MTE1, PIPE_M, EVENT_ID2);
                }
                const uint32_t current_b = l0b_stage_idx;
                const auto b_event = static_cast<event_t>(current_b);
                const auto rhs = l0b[current_b].as_mad_aligned(issued_n, kHeadDim);
                if (load_kv) {
                    const auto input_event = static_cast<event_t>(2 + kv_stage_idx);
                    if constexpr (kFirst) {
                        wait_intra_block(PIPE_MTE1, (kv_stage_idx ^ 1u) & 1u, kNumScoreStages + kv_stage_idx);
                    } else {
                        asc_sync_wait(PIPE_MTE2, PIPE_MTE1, input_event);
                        prefetch_data(kv_split_idx + 1);
                    }
                    asc_sync_wait(PIPE_M, PIPE_MTE1, b_event);
                    if constexpr (kFirst) {
                        copy_l1_to_l0b(rhs, l1kv[kv_stage_idx], 0, 0);
                        copy_l1_to_l0b_mx(rhs, l1_sf_kv[kv_stage_idx], 0, 0, kHeadDim);
                        set_intra_block(PIPE_MTE1, (kv_stage_idx ^ 1u) & 1u, kNumScoreStages + kv_stage_idx);
                    } else {
                        const auto record = get_kv_split(kv_split_idx);
                        const uint32_t n = aligned(SplitKV::num_blocks(record->control) * kSparseBlockKV, kCompactPageTokens);
                        const auto data = l1kv.offset_bytes(kv_stage_idx * kKVStageBytes);
                        const l1_ptr<int16_t> sf(n, 2, data.addr + n * kDataBytes);
                        // Only load stored rows; masked MAD tail rows need no KV values.
                        asc_copy_l12l0b(rhs.ptr(), data.ptr(), 0, 0, n / FRAC_MN,
                                        kHeadDim / get_frac_k<qk_dtype_t>(), n / FRAC_MN, issued_n / FRAC_MN);
                        copy_l1_to_l0b_mx(rhs.as_mad_aligned(n, kHeadDim), sf, 0, 0, kHeadDim);
                    }
                    asc_sync_notify(PIPE_MTE1, PIPE_M, b_event);
                    asc_sync_wait(PIPE_MTE1, PIPE_M, b_event);
                }
                const uint32_t score_stage_idx = owner_stages[pair_idx]++ % kNumScoreStages;
                const uint32_t m = query_mask == 3 ? 2 * kHeadPitch : kHeadPitch;
                const auto result = l0c[l0c_stage_idx].as_mad_aligned(m, issued_n);
                const auto lhs = query_mask == 3 ? pair[pair_idx] : single[pair_idx * 2 + (query_mask == 2)];
                mad(result, lhs, rhs, m, issued_n, kHeadDim, asc_unit_flag_mode::ENABLE_UPDATE, true);
                if (release_kv)
                    l0b_stage_idx = (l0b_stage_idx + 1) % kNumL0BStages;
                const uint32_t owner = pair_idx;
                wait_intra_block(PIPE_FIX, owner, score_stage_idx);
                copy_l0c_to_ub(ub_score[score_stage_idx].as_mad_aligned(m, issued_n), result, m, issued_n,
                               asc_dual_dst_mode::DUAL_DST_DISABLE, owner, asc_quant_mode::F322BF16,
                               true, false, asc_unit_flag_mode::ENABLE_UPDATE);
                set_intra_block(PIPE_FIX, owner, score_stage_idx);
                if (release_kv) {
                    asc_sync_notify(PIPE_M, PIPE_MTE1, b_event);
                    if constexpr (!kFirst)
                        asc_sync_notify(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(2 + kv_stage_idx));
                    kv_stage_idx = (kv_stage_idx + 1) % kNumKVStages;
                }
                l0c_stage_idx ^= 1;
            }
        }
        if constexpr (!kFirst)
            wait_flags<PIPE_MTE1, PIPE_MTE2, 2>();
        set_intra_block(PIPE_FIX, 0, kFinishFlag);
        set_intra_block(PIPE_FIX, 1, kFinishFlag);
    } else {
        const uint32_t aiv_idx = asc_get_sub_block_id();
        uint32_t owner_stage = 0, q_block_stage = 0;
        const auto get_num_owned_queries = [](uint32_t num_q_block_queries, uint32_t aiv_idx) __aicore__ {
            return num_q_block_queries > aiv_idx * 2 ? min(2u, num_q_block_queries - aiv_idx * 2) : 0;
        };
        const auto prefetch_kv = [&](uint64_t kv_split_idx, uint32_t chunk_begin = 0, uint32_t chunk_end = 2) __aicore__ {
            if constexpr (kFirst) {
                const uint32_t stage = kv_split_idx % kNumKVStages;
                if (kv_split_idx >= num_kv_splits || aiv_idx != ((stage ^ 1u) & 1u))
                    return;
                const auto record = get_kv_split(kv_split_idx);
                paged_sparse_mqa_prefetch_kv<kSplitKV, kPageStride, kPageKV, kIsFP4, kSparseBlockKV>(
                    raw_kv, physical_kv_token_starts,
                    record->compact_page_idx, aligned(SplitKV::num_blocks(record->control) * kSparseBlockKV, kCompactPageTokens), ub_first_scratch.ptr(), chunk_begin, chunk_end);
            }
        };
        const auto publish_kv = [&](uint64_t kv_split_idx) __aicore__ {
            if constexpr (kFirst) {
                const uint32_t stage = kv_split_idx % kNumKVStages;
                if (kv_split_idx >= num_kv_splits || aiv_idx != ((stage ^ 1u) & 1u))
                    return;
                const auto record = get_kv_split(kv_split_idx);
                const uint32_t n = aligned(SplitKV::num_blocks(record->control) * kSparseBlockKV, kCompactPageTokens);
                wait_intra_block(PIPE_MTE3, aiv_idx, kNumScoreStages + stage);
                paged_sparse_mqa_publish_kv<kSplitKV, kIsFP4>(
                    compact, record->compact_page_idx, n,
                    ub_first_scratch.ptr(), reinterpret_cast<__cbuf__ uint8_t*>(l1kv[stage].ptr()),
                    reinterpret_cast<__cbuf__ uint8_t*>(l1_sf_kv[stage].ptr()));
                set_intra_block(PIPE_MTE3, aiv_idx, kNumScoreStages + stage);
            }
        };
        if constexpr (kFirst) {
            set_flags<PIPE_V, PIPE_MTE2, 2>();
            set_flags<PIPE_MTE3, PIPE_V, 2>();
            for (uint32_t idx = 0; idx < kNumKVStages; ++ idx) {
                prefetch_kv(idx);
                publish_kv(idx);
            }
        }
        const auto prefetch_q_block = [&](uint64_t kv_split_idx, uint32_t stage) __aicore__ {
            if (kv_split_idx >= num_kv_splits)
                return;
            const auto event = static_cast<event_t>(2 + stage);
            asc_sync_wait(PIPE_V, PIPE_MTE2, event);
            auto dst_weights = stage ? next_weights.ptr() : ub_weights.ptr();
            auto dst_mapping = stage ? next_ub_mapping.ptr() : ub_mapping.ptr();
            const auto kv_split = get_kv_split(kv_split_idx);
            const uint32_t query_begin = kv_split->query_begin, num_q_block_queries = SplitKV::num_q_block_queries(kv_split->control);
            const uint32_t num_owned_queries = get_num_owned_queries(num_q_block_queries, aiv_idx);
            if (num_owned_queries) {
                copy_gm_to_ub(dst_weights, weights + static_cast<uint64_t>(query_begin + aiv_idx * 2) * weights_stride,
                              weights_stride == kNumHeads ? 1 : num_owned_queries, weights_stride == kNumHeads ? num_owned_queries * kNumHeads : kNumHeads,
                              static_cast<uint64_t>(weights_stride == kNumHeads ? num_owned_queries * kNumHeads : weights_stride) * sizeof(bfloat16_t));
                copy_gm_to_ub(dst_mapping, mapping + static_cast<uint64_t>(query_begin + aiv_idx * 2) * (kMaxOutputTokens / kSparseBlockKV),
                              1, num_owned_queries * (kMaxOutputTokens / kSparseBlockKV), num_owned_queries * (kMaxOutputTokens / kSparseBlockKV) * sizeof(uint16_t),
                              asc_load_l2_cache_mode::NOTALLOC_KEEP);
            }
            asc_sync_notify(PIPE_MTE2, PIPE_V, event);
        };
        // V<->MTE2 events 0/1 serve RAW KV; 2/3 serve QBlock weights and mapping.
        set_flags<PIPE_V, PIPE_MTE2, 2, 2>();
        prefetch_q_block(0, 0);
        asc_sync_notify(PIPE_MTE3, PIPE_V, EVENT_ID2);
        uint32_t output_begins[2] = {};
        set_intra_blocks<PIPE_V, kNumScoreStages>(aiv_idx);
        for (uint64_t kv_split_idx = 0; kv_split_idx < num_kv_splits; ++ kv_split_idx) {
            const auto kv_split = get_kv_split(kv_split_idx);
            for (uint32_t pair_idx = 0; pair_idx < 2; ++ pair_idx) {
                const uint32_t query_mask = SplitKV::query_mask(kv_split->control, pair_idx);
                if (!query_mask)
                    continue;
                const bool load_kv = pair_idx == 0 || !SplitKV::query_mask(kv_split->control, 0);
                const bool release_kv = pair_idx == 1 || !SplitKV::query_mask(kv_split->control, 1);
                if (load_kv && (kv_split->control & SplitKV::kQBlockBegin)) {
                    asc_sync_wait(PIPE_MTE3, PIPE_V, EVENT_ID2);
                    output_begins[0] = kv_split->output_ranges[aiv_idx * 2] & 65535u;
                    output_begins[1] = kv_split->output_ranges[aiv_idx * 2 + 1] & 65535u;
                    asc_sync_wait(PIPE_MTE2, PIPE_V, static_cast<event_t>(2 + q_block_stage));
                    prefetch_q_block(kv_split_idx + kv_split->num_q_block_kv_splits, q_block_stage ^ 1u);
                }
                if (load_kv)
                    prefetch_kv(kv_split_idx + kNumKVStages, 0, 1);
                if (aiv_idx == pair_idx) {
                    const uint32_t issued_n = (SplitKV::num_blocks(kv_split->control) * kSparseBlockKV + 127) / 128 * 128;
                    const uint32_t score_stage_idx = owner_stage++ % kNumScoreStages;
                    wait_intra_block(PIPE_V, aiv_idx, score_stage_idx);
                    const auto reduce = [&](auto num_elems) __aicore__ {
                        auto current_weights = q_block_stage == 0 ? ub_weights.ptr() : next_weights.ptr();
                        auto current_mapping = q_block_stage == 0 ? ub_mapping.ptr() : next_ub_mapping.ptr();
                        constexpr uint32_t kNumElems = decltype(num_elems)::value;
                        const uint32_t num_blocks = SplitKV::num_blocks(kv_split->control);
                        // Keep VF selection counts independent to avoid compiler-generated stack spills.
                        const auto control = reinterpret_cast<__gm__ volatile uint32_t*>(&kv_split->control);
                        const uint32_t num_blocks1 = (*control >> SplitKV::kNumBlocksShift) & 127u;
                        const uint32_t begin0 = (kv_split->map_begins[pair_idx] & 65535u);
                        const uint32_t begin1 = (kv_split->map_begins[pair_idx] >> 16);
                        if (query_mask == 3)
                            vf_sparse_mqa_logits_reduce_scatter<kSparseBlockKV, kNumElems, kNumHeads, 2>(
                                ub_score[score_stage_idx].ptr(), current_weights, ub_q_block_logits.ptr(), current_mapping, begin0, begin1, num_blocks, num_blocks1, output_begins[0], output_begins[1]);
                        else
                            vf_sparse_mqa_logits_reduce_scatter<kSparseBlockKV, kNumElems, kNumHeads, 1>(
                                ub_score[score_stage_idx].ptr(), current_weights + (query_mask == 2 ? kNumHeads : 0),
                                ub_q_block_logits.ptr() + (query_mask == 2 ? kMaxOutputTokens : 0),
                                current_mapping, begin0, begin1, num_blocks, num_blocks1, output_begins[query_mask == 2], 0);
                    };
                    switch (issued_n) {
                        case 128: reduce(std::integral_constant<uint32_t, 128>{}); break;
                        case 256: reduce(std::integral_constant<uint32_t, 256>{}); break;
                        case 384: reduce(std::integral_constant<uint32_t, 384>{}); break;
                        case 512: reduce(std::integral_constant<uint32_t, 512>{}); break;
                    }
                    set_intra_block(PIPE_V, aiv_idx, score_stage_idx);
                }
                if (load_kv)
                    prefetch_kv(kv_split_idx + kNumKVStages, 1, 2);
                if (release_kv)
                    publish_kv(kv_split_idx + kNumKVStages);
                if (release_kv && (kv_split->control & SplitKV::kQBlockEnd)) {
                    asc_sync_notify(PIPE_V, PIPE_MTE2, static_cast<event_t>(2 + q_block_stage));
                    q_block_stage ^= 1u;
                    const uint32_t query_begin = kv_split->query_begin, num_q_block_queries = SplitKV::num_q_block_queries(kv_split->control);
                    const uint32_t num_owned_queries = get_num_owned_queries(num_q_block_queries, aiv_idx);
                    asc_sync_notify(PIPE_V, PIPE_MTE3, EVENT_ID2);
                    asc_sync_wait(PIPE_V, PIPE_MTE3, EVENT_ID2);
                    // Each core/QBlock intersection covers a contiguous interval in the original sparse list.
                    for (uint32_t local_query_idx = 0; local_query_idx < num_owned_queries; ++ local_query_idx) {
                        const uint32_t q_idx = aiv_idx * 2 + local_query_idx, range = kv_split->output_ranges[q_idx];
                        const uint32_t begin = output_begins[local_query_idx], count = (range & 65535u) + (range >> 16) - begin;
                        if (count)
                            copy_ub_to_gm(logits + static_cast<uint64_t>(query_begin + q_idx) * logits_stride + begin,
                                          ub_q_block_logits.ptr() + local_query_idx * kMaxOutputTokens, 1, count, count * sizeof(bfloat16_t), asc_store_l2_cache_mode::NOTALLOC_CLEAN);
                    }
                    asc_sync_notify(PIPE_MTE3, PIPE_V, EVENT_ID2);
                }
            }
        }
        asc_sync_wait(PIPE_MTE3, PIPE_V, EVENT_ID2);
        wait_flags<PIPE_V, PIPE_MTE2, 2, 2>();
        if constexpr (kFirst) {
            wait_flags<PIPE_V, PIPE_MTE2, 2>();
            wait_flags<PIPE_MTE3, PIPE_V, 2>();
        }
        wait_intra_block(PIPE_MTE3, aiv_idx, kFinishFlag);
    }
}

} // namespace deep_gemm
