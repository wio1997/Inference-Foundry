#pragma once

#include <deep_gemm/ascend.hpp>
#include <deep_gemm/scheduler/mqa_logits.hpp>
#include <deep_gemm/scheduler/paged_mqa_logits.hpp>

namespace deep_gemm {
using namespace __cce_simd;

// Pack adjacent scale pairs into the 16-token fragments consumed by L0B_MX.
__simd_vf__ inline void vf_mqa_logits_pack_scales(__ubuf__ int16_t* scales, uint32_t num_elems) {
    vector_bool mask = asc_create_mask_b16(PAT_ALL);
    vector_int16_t lane, low, high, pair, bits;
    asc_arange(lane, static_cast<int16_t>(0));
    asc_duplicate_scalar(bits, static_cast<int16_t>(FRAC_MN - 1));
    asc_and(low, lane, bits, mask);
    asc_shiftleft_scalar(low, low, 1, mask);
    asc_duplicate_scalar(bits, static_cast<int16_t>(FRAC_MN));
    asc_and(pair, lane, bits, mask);
    asc_shiftright_scalar(pair, pair, __builtin_ctz(FRAC_MN), mask);
    asc_duplicate_scalar(bits, static_cast<int16_t>(-2 * FRAC_MN));
    asc_and(high, lane, bits, mask);
    asc_add(low, low, pair, mask);
    asc_add(low, low, high, mask);
    for (uint32_t offset = 0; offset < num_elems; offset += asc_get_vf_len() / sizeof(int16_t)) {
        vector_int16_t src, dst;
        asc_loadalign(src, scales + offset);
        asc_gather(dst, src, reinterpret_cast<vector_uint16_t&>(low));
        asc_storealign(scales + offset, dst, mask);
    }
}

// Ascend 950 MTE3 requires a 32B-aligned UB source; GM address and length are unrestricted.
constexpr uint32_t kLogitsAlignElems = 32 / sizeof(bfloat16_t);

// Stage an unaligned row head through aligned UB scratch before MTE3.
__simd_vf__ inline void vf_mqa_logits_stage_head(__ubuf__ bfloat16_t* dst, __ubuf__ bfloat16_t* src) {
    vector_load_unalign desc;
    asc_loadunalign_pre(desc, src);
    vector_bfloat16_t v;
    asc_loadunalign(v, desc, src);
    asc_storealign(dst, v, asc_create_mask_b16(PAT_ALL));
}

template <uint32_t kScratchEventIdx>
__aicore__ inline void mqa_logits_store(
    __gm__ bfloat16_t* dst, __ubuf__ bfloat16_t* src,
    __ubuf__ bfloat16_t* scratch,
    uint32_t num_elems
) {
    const uint32_t src_misalignment = (reinterpret_cast<uintptr_t>(src) / sizeof(bfloat16_t)) & (kLogitsAlignElems - 1);
    const uint32_t num_prefix_elems = min(num_elems, (kLogitsAlignElems - src_misalignment) & (kLogitsAlignElems - 1));
    if (num_prefix_elems) {
        asc_sync_wait(PIPE_MTE3, PIPE_V, static_cast<event_t>(kScratchEventIdx));
        vf_mqa_logits_stage_head(scratch, src);
        asc_sync_notify(PIPE_V, PIPE_MTE3, static_cast<event_t>(kScratchEventIdx));
        asc_sync_wait(PIPE_V, PIPE_MTE3, static_cast<event_t>(kScratchEventIdx));
        const uint32_t num_prefix_bytes = num_prefix_elems * sizeof(bfloat16_t);
        asc_copy_ub2gm_align(dst, scratch, 1, num_prefix_bytes, asc_store_l2_cache_mode::NOTALLOC_CLEAN,
                             num_prefix_bytes, num_prefix_bytes);
        asc_sync_notify(PIPE_MTE3, PIPE_V, static_cast<event_t>(kScratchEventIdx));
    }
    if (num_prefix_elems < num_elems) {
        const uint32_t num_tail_bytes = (num_elems - num_prefix_elems) * sizeof(bfloat16_t);
        asc_copy_ub2gm_align(dst + num_prefix_elems, src + num_prefix_elems, 1, num_tail_bytes,
                             asc_store_l2_cache_mode::NOTALLOC_CLEAN, num_tail_bytes, num_tail_bytes);
    }
}

// Restore logical page order after gathering each pair in physical-address order.
// kPageSize=0 is continuous, 64 swaps two halves in one vector, and 128 swaps
// the destinations of two adjacent vectors.
template <uint32_t kPageSize>
__simd_callee__ inline void vf_mqa_logits_store_bf16(
    __ubuf__ bfloat16_t* dst, vector_bfloat16_t src,
    uint32_t reversed_pairs, uint32_t vec_idx
) {
    constexpr uint32_t kNumElemsPerVecReg = asc_get_vf_len() / sizeof(bfloat16_t);
    if constexpr (kPageSize == 128) {
        const uint32_t reversed = (reversed_pairs >> (vec_idx >> 1)) & 1u;
        dst += (vec_idx ^ reversed) * kNumElemsPerVecReg;
    } else {
        dst += vec_idx * kNumElemsPerVecReg;
    }
    if constexpr (kPageSize == 64) {
        const bool reversed = (reversed_pairs >> vec_idx) & 1u;
        if (reversed) {
            constexpr uint32_t kPageElems = kNumElemsPerVecReg / 2;
            vector_bool low_half = asc_create_mask_b16(PAT_VL64);
            vector_bool high_half;
            asc_not(high_half, low_half, asc_create_mask_b16(PAT_ALL));
            asc_storealign(dst + kPageElems, src, low_half);
            asc_storealign(dst - kPageElems, src, high_half);
            return;
        }
    }
    static_assert(kPageSize == 0 || kPageSize == 64 || kPageSize == 128);
    asc_storealign(dst, src, asc_create_mask_b16(PAT_ALL));
}

template <uint32_t kPageSize, uint32_t kNumElems, uint32_t kNumHeads, uint32_t kSplitKV, uint32_t kNumQueries>
__simd_vf__ inline void vf_mqa_logits_reduce(
    __ubuf__ bfloat16_t* score, __ubuf__ bfloat16_t* weights,
    __ubuf__ bfloat16_t* line,
    uint32_t q_local, uint32_t q_idx,
    uint32_t reversed_pairs
) {
    constexpr uint32_t kNumElemsPerVecReg = asc_get_vf_len() / sizeof(bfloat16_t);
    constexpr uint32_t kNumVecRegs = kNumElems / kNumElemsPerVecReg;
    static_assert(kNumElems % kNumElemsPerVecReg == 0 && kNumVecRegs >= 1 && kNumVecRegs <= 4);
    static_assert((kNumQueries == 1 || kNumQueries == 2) && kSplitKV == 512);
    vector_bool mask = asc_create_mask_b16(PAT_ALL);
    vector_bfloat16_t sum[kNumVecRegs][kNumQueries];
    #pragma unroll
    for (uint32_t vec_idx = 0; vec_idx < kNumVecRegs; ++vec_idx) {
        #pragma unroll
        for (uint32_t query_idx = 0; query_idx < kNumQueries; ++query_idx)
            asc_duplicate_scalar(sum[vec_idx][query_idx], static_cast<bfloat16_t>(0), mask);
    }

    __ubuf__ bfloat16_t* weight_ptr[kNumQueries];
    __ubuf__ bfloat16_t* score_ptr[kNumQueries];
    #pragma unroll
    for (uint32_t query_idx = 0; query_idx < kNumQueries; ++query_idx) {
        weight_ptr[query_idx] = weights + (q_idx + query_idx) * kNumHeads;
        score_ptr[query_idx] = score + (q_local + query_idx) * kNumHeads * kNumElems;
    }
    #pragma unroll 1
    for (uint32_t h = 0; h < kNumHeads; ++h) {
        vector_bfloat16_t weight[kNumQueries];
        #pragma unroll
        for (uint32_t query_idx = 0; query_idx < kNumQueries; ++query_idx)
            asc_loadalign_brc_postupdate(weight[query_idx], weight_ptr[query_idx], 1);
        #pragma unroll
        for (uint32_t vec_idx = 0; vec_idx < kNumVecRegs; ++vec_idx) {
            #pragma unroll
            for (uint32_t query_idx = 0; query_idx < kNumQueries; ++query_idx) {
                vector_bfloat16_t value;
                asc_loadalign_postupdate(value, score_ptr[query_idx], kNumElemsPerVecReg);
                asc_mula(sum[vec_idx][query_idx], value, weight[query_idx], mask);
            }
        }
    }

    #pragma unroll
    for (uint32_t vec_idx = 0; vec_idx < kNumVecRegs; ++vec_idx) {
        #pragma unroll
        for (uint32_t query_idx = 0; query_idx < kNumQueries; ++query_idx)
            vf_mqa_logits_store_bf16<kPageSize>(line + query_idx * kSplitKV, sum[vec_idx][query_idx], reversed_pairs, vec_idx);
    }
}

template <uint32_t kPageSize, uint32_t kNumQTokensPerTile, uint32_t kNumHeads, uint32_t kSplitKV>
__aicore__ inline uint32_t mqa_logits_reduce_queries(
    __ubuf__ bfloat16_t* score, __ubuf__ bfloat16_t* weight,
    __ubuf__ bfloat16_t* line,
    uint32_t q_local, uint32_t q_idx,
    uint32_t num_q_tokens_in_block, uint32_t issued_n,
    uint32_t reversed_pairs
) {
    uint32_t num_reduced_queries = 1;
    const auto reduce = [&](auto num_elems) __aicore__ {
        constexpr uint32_t kNumElems = decltype(num_elems)::value;
        if constexpr (kNumQTokensPerTile >= 2) {
            if (q_local + 1 < kNumQTokensPerTile && q_idx + 1 < num_q_tokens_in_block) {
                vf_mqa_logits_reduce<kPageSize, kNumElems, kNumHeads, kSplitKV, 2>(score, weight, line, q_local, q_idx,
                                                                                   reversed_pairs);
                num_reduced_queries = 2;
                return;
            }
        }
        vf_mqa_logits_reduce<kPageSize, kNumElems, kNumHeads, kSplitKV, 1>(score, weight, line, q_local, q_idx, reversed_pairs);
    };
    constexpr uint32_t kNumElemsPerVecReg = asc_get_vf_len() / sizeof(bfloat16_t);
    switch (issued_n / kNumElemsPerVecReg) {
        case 1: reduce(std::integral_constant<uint32_t, kNumElemsPerVecReg>{}); break;
        case 2: reduce(std::integral_constant<uint32_t, 2 * kNumElemsPerVecReg>{}); break;
        case 3: reduce(std::integral_constant<uint32_t, 3 * kNumElemsPerVecReg>{}); break;
        case 4: reduce(std::integral_constant<uint32_t, 4 * kNumElemsPerVecReg>{}); break;
        default: ascendc_assert(false, "Unsupported issued_n for MQA logits reduction");
    }
    return num_reduced_queries;
}

template <
    bool kIsFP4,
    uint32_t kNumHeads, uint32_t kHeadDim,
    uint32_t kSplitKV, uint32_t kMadM, uint32_t kNumCDStages,
    uint32_t kNumQStages, uint32_t kNumKVStages, uint32_t kNumAccumStages,
    uint32_t kNumLogitsStages,
    uint32_t kPageSize,
    bool kUseAscendKVLayout,
    typename MakeScheduler
>
__aicore__ inline void mqa_logits_core_impl(
    gm_ptr<uint8_t, Major::K> q,
    gm_ptr<uint8_t, Major::K> kv,
    __gm__ int16_t* sf_q,
    __gm__ int16_t* sf_kv,
    gm_ptr<bfloat16_t, Major::K> weights,
    gm_ptr<bfloat16_t, Major::K> logits,
    const MakeScheduler& make_scheduler
) {
    using Scheduler = decltype(make_scheduler(block_idx));
    constexpr bool kIsPaged = kPageSize != 0;
    static_assert(kNumKVStages % 2 == 0, "SF stages alternate between two AIVs");
    static_assert(kSplitKV == 512, "MQA logits kernel expects bf16 SPLIT_KV=512");
    static_assert(kNumHeads >= 4 && kNumHeads <= 64 && kNumHeads % 4 == 0,
                  "Number of heads must be a multiple of 4 in [4, 64]");
    static_assert(kHeadDim == 64 || kHeadDim == 128, "Unsupported head dimension");
    using qk_dtype_t = std::conditional_t<kIsFP4, float4_e2m1x2_t, float8_e4m3_t>;

    constexpr uint32_t kNumVecElems = asc_get_vf_len() / sizeof(bfloat16_t);
    constexpr uint32_t MAD_M = kMadM;
    constexpr uint32_t MAD_M_PADDED = aligned(MAD_M, FRAC_MN);
    constexpr uint32_t MAD_N = kSplitKV;
    constexpr uint32_t MAD_K = kHeadDim;
    // Fixed 8-tile specialization: L0A capacity cap (8 * MAD_M_PADDED * D * 1B <= 64KB); runtime trims usage
    constexpr uint32_t kNumQHTilesPerBlock = 8;
    constexpr uint32_t kNumQHRowsPerBlock = kNumQHTilesPerBlock * MAD_M;
    constexpr uint32_t kNumPaddedQHRowsPerBlock = kNumQHTilesPerBlock * MAD_M_PADDED;
    constexpr uint32_t kNumQTokensPerBlock = kNumQHRowsPerBlock / kNumHeads;
    constexpr uint32_t kNumQTokensPerTile = MAD_M / kNumHeads;
    constexpr uint32_t kSFPairs = kHeadDim / MX_SF_DIVISOR;
    static_assert(kNumQHRowsPerBlock % kNumHeads == 0, "QH block must contain full query/head groups");
    static_assert(kNumCDStages == 2 || kNumCDStages == 4, "MQA logits currently supports 2 or 4 CD stages");
    static_assert(MAD_M % kNumHeads == 0, "MAD_M must contain full query/head groups");
    static_assert(kNumQTokensPerTile > 0, "AIV score slice must contain at least one query");
    static_assert(kSplitKV != 512 || kNumCDStages == 2, "SPLIT_KV=512 expects 2 CD stages to fit L0C");

    constexpr l1_ptr<qk_dtype_t, Major::K> l1q(kNumPaddedQHRowsPerBlock, kHeadDim, 0);
    constexpr l1_ptr<qk_dtype_t, Major::K> l1kv(kSplitKV, kHeadDim, l1q.offset(kNumQStages));
    constexpr l1_ptr<int16_t> l1_sf_q(kNumPaddedQHRowsPerBlock, kSFPairs, l1kv.offset(kNumKVStages));
    constexpr l1_ptr<int16_t> l1_sf_kv(kSplitKV, kSFPairs, l1_sf_q.offset(kNumQStages));
    constexpr l1_ptr<int16_t> l1_sf_q_tiles(MAD_M_PADDED, kSFPairs, l1_sf_q.addr);
    constexpr l0a_ptr<qk_dtype_t> l0a(MAD_M_PADDED, MAD_K);
    constexpr l0b_ptr<qk_dtype_t> l0b(MAD_N, MAD_K);
    constexpr l0c_ptr<float> l0c(MAD_M_PADDED, MAD_N);

    constexpr uint32_t kNumWeightStages = 2;
    constexpr ub_ptr<bfloat16_t> ub_accum(MAD_M_PADDED, MAD_N, 0);
    constexpr ub_ptr<bfloat16_t> ub_weights(kNumQTokensPerBlock, kNumHeads, ub_accum.offset(kNumAccumStages));
    constexpr ub_ptr<bfloat16_t> ub_logits(kNumQTokensPerTile, MAD_N, ub_weights.offset(kNumWeightStages));

    constexpr ub_ptr<bfloat16_t> ub_logits_head(1, asc_get_vf_len() / sizeof(bfloat16_t),
                                                    ub_logits.offset(kNumLogitsStages));
    // One row per Q block; the 8-element pad keeps each buffer 32B-aligned
    // (paged H=64 blocks hold a single token, so the raw count may not).
    constexpr ub_ptr<uint32_t> ub_seq_start(1, aligned(kNumQTokensPerBlock, 8), ub_logits_head.offset(1));
    constexpr ub_ptr<uint32_t> ub_seq_end(1, aligned(kNumQTokensPerBlock, 8), ub_seq_start.offset(1));
    constexpr uint32_t kNumSFUBStages = kNumKVStages / 2;
    constexpr ub_ptr<int16_t> ub_sf_kv(kIsPaged ? kSplitKV : 0, kSFPairs, ub_seq_end.offset(1));

    static_assert(l1_sf_kv.offset(kNumKVStages) <= L1SizeBytes, "L1 overflow");
    static_assert(l0a.offset(kNumQHTilesPerBlock) <= L0ASizeBytes, "L0A overflow");
    static_assert(l0b.offset(1) <= L0BSizeBytes, "L0B overflow");
    static_assert(l0c.offset(kNumCDStages) <= L0CSizeBytes, "L0C overflow");
    static_assert(ub_logits.offset(kNumLogitsStages) <= UBSizeBytes, "UB overflow");
    static_assert(ub_logits_head.offset(1) <= UBSizeBytes, "UB overflow");
    static_assert(ub_seq_end.offset(1) <= UBSizeBytes, "UB overflow");
    static_assert(ub_sf_kv.offset(kNumSFUBStages) <= UBSizeBytes, "UB overflow");
    static_assert(!kIsPaged || kNumAccumStages + kNumKVStages <= 16, "Scale flag overflow");

    auto scheduler = make_scheduler(block_idx);
    const gm_ptr<int16_t, Major::MN> sf_q_t(scheduler.get_q_sf_stride(), reinterpret_cast<uintptr_t>(sf_q));
    const gm_ptr<qk_dtype_t, Major::K> q_t(q.stride_outer, q.addr, q.stride_batch);
    const gm_ptr<qk_dtype_t, Major::K> kv_t(kv.stride_outer, kv.addr, kv.stride_batch);
    uint32_t q_block_idx, kv_token_base, num_kv_splits;
    if ASCEND_IS_AIC {
        asc_set_mmad_direction_n();

        constexpr uint32_t kQEventBase = 0;
        constexpr uint32_t kKVEventBase = kNumQStages;
        set_flags<PIPE_M, PIPE_MTE1, 2>();
        set_flags<PIPE_MTE1, PIPE_MTE2, kNumQStages, kQEventBase>();
        set_flags<PIPE_MTE1, PIPE_MTE2, kNumKVStages, kKVEventBase>();
        // Scale slots retain their AIV owner across query blocks.
        if constexpr (kIsPaged)
            for (uint32_t stage_idx = 0; stage_idx < kNumKVStages; ++stage_idx)
                set_intra_block(PIPE_MTE1, (stage_idx ^ 1u) & 1u, kNumAccumStages + stage_idx);

        uint32_t q_stage_idx = 0;
        uint32_t kv_stage_idx = 0;
        uint32_t cd_stage_idx = 0;
        while (scheduler.next_q_block(q_block_idx, kv_token_base, num_kv_splits)) {
            const uint32_t q_base = scheduler.get_q_base(q_block_idx);
            const uint32_t num_q_tokens_in_block = scheduler.get_num_q_tokens_in_block(q_block_idx);
            const uint32_t num_qh_rows_in_block = num_q_tokens_in_block * kNumHeads;
            const uint32_t num_qh_tiles_in_block = ceil_div(num_qh_rows_in_block, MAD_M);

            asc_sync_wait(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(kQEventBase + q_stage_idx));
            if constexpr (MAD_M == MAD_M_PADDED) {
                fill_mn_tail_block(l1q[q_stage_idx], num_qh_rows_in_block, aligned(num_qh_rows_in_block, FRAC_MN));
                copy_gm_to_l1(l1q[q_stage_idx], q_t.index(q_base * kNumHeads, 0), num_qh_rows_in_block, kHeadDim,
                              asc_load_l2_cache_mode::NORMAL_LAST_VICTIM);
                copy_gm_to_l1_mx(l1_sf_q[q_stage_idx], sf_q_t.index(q_base * kNumHeads, 0), num_qh_rows_in_block,
                                 kSFPairs, asc_load_l2_cache_mode::NORMAL_LAST_VICTIM);
            } else {
                const l1_ptr<qk_dtype_t> l1q_m_tiles(MAD_M_PADDED, get_frac_k<qk_dtype_t>(), l1q[q_stage_idx].addr);
                for (uint32_t qh_tile_idx = 0; qh_tile_idx < num_qh_tiles_in_block; ++qh_tile_idx) {
                    const uint32_t qh_base = qh_tile_idx * MAD_M;
                    const uint32_t num_m_rows = min(MAD_M, num_qh_rows_in_block - qh_base);
                    const uint32_t sf_q_tile_idx = q_stage_idx * kNumQHTilesPerBlock + qh_tile_idx;
                    const auto q_tile = l1q_m_tiles[qh_tile_idx].as_mad_aligned(kNumPaddedQHRowsPerBlock, kHeadDim);
                    const auto sf_q_tile = l1_sf_q_tiles[sf_q_tile_idx];
                    fill_mn_tail_block(q_tile, num_m_rows, MAD_M_PADDED);
                    copy_gm_to_l1(q_tile, q_t.index(q_base * kNumHeads + qh_base, 0), num_m_rows, kHeadDim,
                                  asc_load_l2_cache_mode::NORMAL_LAST_VICTIM);
                    copy_gm_to_l1_mx(sf_q_tile, sf_q_t.index(q_base * kNumHeads + qh_base, 0), num_m_rows, kSFPairs,
                                     asc_load_l2_cache_mode::NORMAL_LAST_VICTIM);
                }
            }
            asc_sync_notify(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(kQEventBase + q_stage_idx));

            asc_sync_wait(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(kQEventBase + q_stage_idx));
            asc_sync_wait(PIPE_M, PIPE_MTE1, EVENT_ID1);
            for (uint32_t qh_tile_idx = 0; qh_tile_idx < num_qh_tiles_in_block; ++qh_tile_idx) {
                const uint32_t qh_base = qh_tile_idx * MAD_M;
                const uint32_t qh_padded_base = qh_tile_idx * MAD_M_PADDED;
                const uint32_t num_m_rows = min(MAD_M, num_qh_rows_in_block - qh_base);
                const auto l0a_tile = l0a[qh_tile_idx].as_mad_aligned(num_m_rows, MAD_K);
                copy_l1_to_l0a(l0a_tile, l1q[q_stage_idx], qh_padded_base, 0);
                copy_l1_to_l0a_mx(l0a_tile, l1_sf_q[q_stage_idx], qh_padded_base, 0, MAD_K);
            }
            asc_sync_notify(PIPE_MTE1, PIPE_M, EVENT_ID1);

            // GM-backed tile check; the AIV branch defines the UB-backed twin
            const auto is_tile_active = [&](uint32_t q_tile_base, uint32_t kv_idx, uint32_t kv_end) __aicore__ -> bool {
                for (uint32_t q_idx = q_tile_base; q_idx < min(q_tile_base + kNumQTokensPerTile, num_q_tokens_in_block); ++q_idx) {
                    const uint32_t row = q_base + q_idx;
                    if constexpr (kIsPaged) {
                        if (kv_idx < scheduler.context_lens[row])
                            return true;
                    } else if (kv_idx < scheduler.cu_seq_len_k_end[row] &&
                               scheduler.cu_seq_len_k_start[row] < kv_end) {
                        return true;
                    }
                }
                return false;
            };

            asc_sync_wait(PIPE_MTE1, PIPE_M, EVENT_ID1);
            for (uint32_t kv_split_idx = 0; kv_split_idx < num_kv_splits; ++kv_split_idx) {
                const uint32_t kv_idx = scheduler.get_kv_token_offset(kv_token_base, kv_split_idx);
                const uint32_t actual_kv = scheduler.get_num_kv_block_tokens(kv_token_base, kv_split_idx);
                const uint32_t issued_n = aligned(actual_kv, kNumVecElems);
                asc_sync_wait(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(kKVEventBase + kv_stage_idx));
                fill_mn_tail_block(l1kv[kv_stage_idx], actual_kv, issued_n);
                if constexpr (kIsPaged) {
                    const asc_load_l2_cache_mode kv_load_hint = scheduler.has_multiple_q_blocks()
                        ? asc_load_l2_cache_mode::NORMAL_LAST_VICTIM : asc_load_l2_cache_mode::NOTALLOC_KEEP;
                    const uint32_t num_pages = ceil_div(actual_kv, kPageSize);
                    for (uint32_t compact_page = 0; compact_page < num_pages; compact_page += 2) {
                        uint32_t first_page, page_gap;
                        const uint32_t num_batch = scheduler.get_physical_page_batch(kv_token_base, kv_split_idx, compact_page, num_pages, first_page, page_gap);
                        const auto page_src = kv_t + scheduler.get_kv_page_offset_bytes(first_page);
                        const auto page_dst = l1kv[kv_stage_idx].offset_bytes(compact_page * kPageSize * 32u);
                        const uint32_t valid_rows = min(kPageSize, actual_kv - compact_page * kPageSize);
                        copy_gm_to_l1<kUseAscendKVLayout>(page_dst, page_src, valid_rows, kHeadDim, kv_load_hint,
                                                          num_batch, kPageSize,
                                                          page_gap * scheduler.kv_page_stride_bytes);
                    }
                } else {
                    const gm_ptr<int16_t, Major::MN> sf_kv_t(scheduler.get_kv_sf_stride(), reinterpret_cast<uintptr_t>(sf_kv));
                    copy_gm_to_l1(l1kv[kv_stage_idx], kv_t.index(kv_idx, 0), actual_kv, kHeadDim,
                                  asc_load_l2_cache_mode::NOTALLOC_KEEP);
                    copy_gm_to_l1_mx(l1_sf_kv[kv_stage_idx], sf_kv_t.index(kv_idx, 0), actual_kv, kSFPairs,
                                     asc_load_l2_cache_mode::NOTALLOC_KEEP);
                }
                asc_sync_notify(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(kKVEventBase + kv_stage_idx));
                asc_sync_wait(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(kKVEventBase + kv_stage_idx));

                asc_sync_wait(PIPE_M, PIPE_MTE1, EVENT_ID0);
                const auto l0b_tile = l0b.as_mad_aligned(issued_n, MAD_K);
                copy_l1_to_l0b(l0b_tile, l1kv[kv_stage_idx], 0, 0);
                if constexpr (kIsPaged)
                    wait_intra_block(PIPE_MTE1, (kv_stage_idx ^ 1u) & 1u, kNumAccumStages + kv_stage_idx);
                copy_l1_to_l0b_mx(l0b_tile, l1_sf_kv[kv_stage_idx], 0, 0, MAD_K);
                if constexpr (kIsPaged)
                    set_intra_block(PIPE_MTE1, (kv_stage_idx ^ 1u) & 1u, kNumAccumStages + kv_stage_idx);
                asc_sync_notify(PIPE_MTE1, PIPE_M, EVENT_ID0);
                asc_sync_wait(PIPE_MTE1, PIPE_M, EVENT_ID0);

                // Alternate an unpaired tile between AIVs across KV splits.
                const uint32_t owner_offset = num_qh_tiles_in_block & kv_split_idx & 1;
                for (uint32_t qh_tile_idx = 0; qh_tile_idx < num_qh_tiles_in_block; ++qh_tile_idx) {
                    const uint32_t qh_base = qh_tile_idx * MAD_M;
                    const uint32_t q_tile_base = qh_base / kNumHeads;
                    // Skip tiles with no row window overlapping this split
                    if (!is_tile_active(q_tile_base, kv_idx, kv_idx + actual_kv))
                        continue;
                    const uint32_t num_m_rows = min(MAD_M, num_qh_rows_in_block - qh_base);
                    const auto l0c_tile = l0c[cd_stage_idx].as_mad_aligned(num_m_rows, issued_n);
                    const uint32_t accum_owner = (qh_tile_idx + owner_offset) & 1u;
                    const uint32_t accum_stage_idx = ((kv_split_idx * num_qh_tiles_in_block + qh_tile_idx) / 2) % kNumAccumStages;
                    mad(l0c_tile, l0a[qh_tile_idx], l0b_tile, num_m_rows, issued_n, MAD_K,
                        asc_unit_flag_mode::ENABLE_UPDATE, true);
                    wait_intra_block(PIPE_FIX, static_cast<uint8_t>(accum_owner),
                                     static_cast<uint8_t>(accum_stage_idx));
                    const auto accum_tile = ub_accum[accum_stage_idx].as_mad_aligned(num_m_rows, issued_n);
                    copy_l0c_to_ub(accum_tile, l0c_tile, num_m_rows, issued_n, asc_dual_dst_mode::DUAL_DST_DISABLE,
                                   accum_owner, infer_quant_mode<bfloat16_t, float>(), true, false,
                                   asc_unit_flag_mode::ENABLE_UPDATE);
                    set_intra_block(PIPE_FIX, accum_owner, accum_stage_idx);
                    cd_stage_idx = (cd_stage_idx + 1) % kNumCDStages;
                }
                asc_sync_notify(PIPE_M, PIPE_MTE1, EVENT_ID0);
                asc_sync_notify(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(kKVEventBase + kv_stage_idx));
                kv_stage_idx = (kv_stage_idx + 1) % kNumKVStages;
            }
            asc_sync_notify(PIPE_M, PIPE_MTE1, EVENT_ID1);
            asc_sync_notify(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(kQEventBase + q_stage_idx));
            q_stage_idx = (q_stage_idx + 1) % kNumQStages;
        }
    } else {
        const uint32_t sub_id = asc_get_sub_block_id();
        constexpr uint32_t kSFLoadEventIdx = kNumWeightStages;
        constexpr uint32_t kSFStoreEventIdx = kNumLogitsStages + 1;
        static_assert(!kIsPaged || kSFStoreEventIdx + kNumSFUBStages <= 8, "Scale event overflow");
        // Release accum stage on PIPE_V so it is not serialized behind the GM store.
        set_intra_blocks<PIPE_V, kNumAccumStages>(static_cast<uint8_t>(sub_id));
        set_flags<PIPE_MTE3, PIPE_V, kNumLogitsStages>();
        // Keep the shared head scratch single-buffered across MTE3 stores.
        static_assert(kNumLogitsStages < 8, "Scratch event id must fit the 8-id flag space");
        asc_sync_notify(PIPE_MTE3, PIPE_V, static_cast<event_t>(kNumLogitsStages));
        set_flags<PIPE_V, PIPE_MTE2, kNumWeightStages>();
        if constexpr (kIsPaged)
            set_flags<PIPE_MTE3, PIPE_MTE2, kNumSFUBStages, kSFLoadEventIdx>();

        uint32_t logits_stage_idx = 0;
        uint32_t weight_stage_idx = 0;
        uint32_t kv_stage_idx = 0;
        // Keep the SF producer ahead across requests; the consumer retains its own cursor.
        auto sf_scheduler = scheduler;
        uint32_t sf_q_block_idx = 0, sf_kv_token_base = 0, num_sf_splits = 0, sf_split_idx = 0;
        uint32_t sf_num_pages[kNumSFUBStages] = {};
        const auto prefetch_sf_kv = [&](uint32_t kv_stage_idx) __aicore__ {
            if constexpr (kIsPaged) {
                const uint32_t sf_stage_idx = kv_stage_idx / 2;
                if (sf_split_idx == num_sf_splits) {
                    if (!sf_scheduler.next_q_block(sf_q_block_idx, sf_kv_token_base, num_sf_splits)) {
                        // An empty future slot issues no copy or ready notification.
                        if (sub_id == ((kv_stage_idx ^ 1u) & 1u)) sf_num_pages[sf_stage_idx] = 0;
                        return;
                    }
                    sf_split_idx = 0;
                }
                // Both AIVs advance; only the owner transfers this stage.
                const uint32_t kv_split_idx = sf_split_idx++;
                if (sub_id != ((kv_stage_idx ^ 1u) & 1u)) return;
                const uint32_t num_pages = ceil_div(
                    sf_scheduler.get_num_kv_block_tokens(sf_kv_token_base, kv_split_idx), kPageSize);
                sf_num_pages[sf_stage_idx] = num_pages;
                const asc_load_l2_cache_mode sf_load_hint = sf_scheduler.has_multiple_q_blocks()
                    ? asc_load_l2_cache_mode::NORMAL_LAST_VICTIM : asc_load_l2_cache_mode::NOTALLOC_KEEP;
                asc_sync_wait(PIPE_MTE3, PIPE_MTE2, static_cast<event_t>(kSFLoadEventIdx + sf_stage_idx));
                for (uint32_t compact_page = 0; compact_page < num_pages; compact_page += 2) {
                    uint32_t first_page, page_gap;
                    const uint32_t num_batch = sf_scheduler.get_physical_page_batch(
                        sf_kv_token_base, kv_split_idx, compact_page, num_pages, first_page, page_gap);
                    const auto sf_page_src = reinterpret_cast<__gm__ int16_t*>(
                        reinterpret_cast<uintptr_t>(sf_kv) + sf_scheduler.get_kv_page_offset_bytes(first_page));
                    copy_gm_to_ub(ub_sf_kv[sf_stage_idx].ptr() + compact_page * kPageSize * kSFPairs,
                        sf_page_src, num_batch, kPageSize * kSFPairs,
                        page_gap * sf_scheduler.kv_page_stride_bytes, sf_load_hint);
                }
                asc_sync_notify(PIPE_MTE2, PIPE_V, static_cast<event_t>(kSFLoadEventIdx + sf_stage_idx));
            }
        };
        const auto publish_sf_kv = [&](uint32_t kv_stage_idx) __aicore__ {
            if constexpr (kIsPaged) {
                if (sub_id != ((kv_stage_idx ^ 1u) & 1u)) return;
                const uint32_t sf_stage_idx = kv_stage_idx / 2;
                const uint32_t num_pages = sf_num_pages[sf_stage_idx];
                if (!num_pages) return;
                asc_sync_wait(PIPE_MTE2, PIPE_V, static_cast<event_t>(kSFLoadEventIdx + sf_stage_idx));
                if constexpr (!kUseAscendKVLayout && kSFPairs == 2)
                    vf_mqa_logits_pack_scales(ub_sf_kv[sf_stage_idx].ptr(), num_pages * kPageSize * kSFPairs);
                asc_sync_notify(PIPE_V, PIPE_MTE3, static_cast<event_t>(kSFStoreEventIdx + sf_stage_idx));
                asc_sync_wait(PIPE_V, PIPE_MTE3, static_cast<event_t>(kSFStoreEventIdx + sf_stage_idx));
                wait_intra_block(PIPE_MTE3, sub_id, kNumAccumStages + kv_stage_idx);
                asc_copy_ub2l1(l1_sf_kv[kv_stage_idx].ptr(), ub_sf_kv[sf_stage_idx].ptr(),
                    num_pages * kPageSize * kSFPairs * sizeof(int16_t));
                set_intra_block(PIPE_MTE3, sub_id, kNumAccumStages + kv_stage_idx);
                asc_sync_notify(PIPE_MTE3, PIPE_MTE2, static_cast<event_t>(kSFLoadEventIdx + sf_stage_idx));
            }
        };
        if constexpr (kIsPaged) {
            for (uint32_t stage_idx = 0; stage_idx < kNumKVStages; ++stage_idx)
                prefetch_sf_kv(stage_idx);
            for (uint32_t stage_idx = 0; stage_idx < kNumKVStages; ++stage_idx)
                publish_sf_kv(stage_idx);
        }

        while (scheduler.next_q_block(q_block_idx, kv_token_base, num_kv_splits)) {
            const uint32_t q_base = scheduler.get_q_base(q_block_idx);
            const uint32_t num_q_tokens_in_block = scheduler.get_num_q_tokens_in_block(q_block_idx);
            const uint32_t num_qh_rows_in_block = num_q_tokens_in_block * kNumHeads;
            const uint32_t num_qh_tiles_in_block = ceil_div(num_qh_rows_in_block, MAD_M);
            // Per-Q-block seq bounds via MTE2, reused across all KV splits
            if constexpr (kIsPaged) {
                copy_gm_to_ub(ub_seq_end.ptr(), scheduler.context_lens + q_base, 1, num_q_tokens_in_block,
                              num_q_tokens_in_block * sizeof(uint32_t), asc_load_l2_cache_mode::NORMAL_LAST_VICTIM);
            } else {
                copy_gm_to_ub(ub_seq_start.ptr(), scheduler.cu_seq_len_k_start + q_base, 1, num_q_tokens_in_block,
                              num_q_tokens_in_block * sizeof(uint32_t), asc_load_l2_cache_mode::NORMAL_LAST_VICTIM);
                copy_gm_to_ub(ub_seq_end.ptr(), scheduler.cu_seq_len_k_end + q_base, 1, num_q_tokens_in_block,
                              num_q_tokens_in_block * sizeof(uint32_t), asc_load_l2_cache_mode::NORMAL_LAST_VICTIM);
            }
            // Scalar UB reads run on PIPE_S, outside the V/MTE2/MTE3 event triads; order them
            // after the seq copies with a (MTE2, PIPE_S) flag pair (set/wait share arg order).
            asc_sync_notify(PIPE_MTE2, PIPE_S, EVENT_ID0);
            // UB-backed seq bounds; serve the tile check and store paths
            const auto seq_ks = [&](uint32_t q_in_block) __aicore__ -> uint32_t {
                return kIsPaged ? 0u : ub_seq_start.ptr()[q_in_block];
            };
            const auto seq_ke = [&](uint32_t q_in_block) __aicore__ -> uint32_t {
                return ub_seq_end.ptr()[q_in_block];
            };
            const auto is_tile_active = [&](uint32_t q_tile_base, uint32_t kv_idx, uint32_t kv_end) __aicore__ -> bool {
                for (uint32_t q_idx = q_tile_base; q_idx < min(q_tile_base + kNumQTokensPerTile, num_q_tokens_in_block); ++q_idx) {
                    if (kv_idx >= seq_ke(q_idx))
                        continue;
                    if constexpr (kIsPaged)
                        return true;
                    else if (seq_ks(q_idx) < kv_end)
                        return true;
                }
                return false;
            };
            asc_sync_wait(PIPE_V, PIPE_MTE2, static_cast<event_t>(weight_stage_idx));
            copy_gm_to_ub(ub_weights[weight_stage_idx].ptr(), weights.index(q_base, 0).ptr(), num_q_tokens_in_block,
                          kNumHeads, weights.stride_outer * sizeof(bfloat16_t),
                          asc_load_l2_cache_mode::NORMAL_LAST_VICTIM);
            asc_sync_notify(PIPE_MTE2, PIPE_V, static_cast<event_t>(weight_stage_idx));
            asc_sync_wait(PIPE_MTE2, PIPE_V, static_cast<event_t>(weight_stage_idx));
            asc_sync_wait(PIPE_MTE2, PIPE_S, EVENT_ID0);

            for (uint32_t kv_split_idx = 0; kv_split_idx < num_kv_splits; ++kv_split_idx) {
                if constexpr (kIsPaged)
                    prefetch_sf_kv(kv_stage_idx);
                const uint32_t kv_idx = scheduler.get_kv_token_offset(kv_token_base, kv_split_idx);
                const uint32_t actual_kv = scheduler.get_num_kv_block_tokens(kv_token_base, kv_split_idx);
                const uint32_t issued_n = aligned(actual_kv, kNumVecElems);
                uint32_t reversed_pairs = 0;
                if constexpr (kIsPaged) {
                    const uint32_t num_pages = ceil_div(actual_kv, kPageSize);
                    for (uint32_t compact_page = 0; compact_page + 1 < num_pages; compact_page += 2) {
                        const uint32_t physical_page = scheduler.get_physical_page(kv_token_base, kv_split_idx, compact_page);
                        const uint32_t partner_physical_page = scheduler.get_physical_page(kv_token_base, kv_split_idx, compact_page + 1);
                        reversed_pairs |= static_cast<uint32_t>(physical_page > partner_physical_page)
                                          << (compact_page / 2);
                    }
                }
                const uint32_t owner_offset = num_qh_tiles_in_block & kv_split_idx & 1;
                for (uint32_t qh_tile_idx = sub_id ^ owner_offset; qh_tile_idx < num_qh_tiles_in_block; qh_tile_idx += 2) {
                    const uint32_t qh_base = qh_tile_idx * MAD_M;
                    const uint32_t q_tile_base = qh_base / kNumHeads;
                    // Mirror the AIC's skip: run only when a row window overlaps this split
                    if (!is_tile_active(q_tile_base, kv_idx, kv_idx + actual_kv))
                        continue;
                    const uint32_t accum_stage_idx = ((kv_split_idx * num_qh_tiles_in_block + qh_tile_idx) / 2) % kNumAccumStages;
                    wait_intra_block(PIPE_V, static_cast<uint8_t>(sub_id), static_cast<uint8_t>(accum_stage_idx));
                    asc_sync_wait(PIPE_MTE3, PIPE_V, static_cast<event_t>(logits_stage_idx));
                    const auto ub_logits_stage = ub_logits[logits_stage_idx];
                    auto score = ub_accum[accum_stage_idx].ptr();
                    auto weight = ub_weights[weight_stage_idx].ptr();
                    for (uint32_t q_local = 0; q_local < kNumQTokensPerTile;) {
                        const uint32_t q_idx = q_tile_base + q_local;
                        if (q_idx >= num_q_tokens_in_block)
                            break;
                        auto line = ub_logits_stage.ptr() + q_local * kSplitKV;
                        q_local += mqa_logits_reduce_queries<kPageSize, kNumQTokensPerTile, kNumHeads, kSplitKV>(score, weight, line, q_local, q_idx,
                                                                                                                 num_q_tokens_in_block, issued_n, reversed_pairs);
                    }
                    // Release accum on PIPE_V: V-pipe order guarantees it runs after the reduce
                    // that consumed ub_accum, so no extra V->MTE3 done-fence is needed.
                    set_intra_block(PIPE_V, static_cast<uint8_t>(sub_id), static_cast<uint8_t>(accum_stage_idx));
                    asc_sync_notify(PIPE_V, PIPE_MTE3, static_cast<event_t>(logits_stage_idx));

                    // Store current logits directly.
                    // MTE3 needs a 32B-aligned UB source, so the unaligned head of a
                    // row range is staged through the aligned head scratch first.
                    asc_sync_wait(PIPE_V, PIPE_MTE3, static_cast<event_t>(logits_stage_idx));
                    for (uint32_t q_store = 0; q_store < kNumQTokensPerTile; ++q_store) {
                        const uint32_t q_in_block = q_tile_base + q_store;
                        if (q_in_block >= num_q_tokens_in_block)
                            break;
                        const uint32_t row = q_base + q_in_block;
                        const uint32_t ks = seq_ks(q_in_block);
                        const uint32_t ke = seq_ke(q_in_block);
                        const uint32_t store_begin = max(kv_idx, ks);
                        const uint32_t store_end = min(kv_idx + actual_kv, ke);
                        if (store_begin >= store_end)
                            continue;
                        const uint32_t dst_col = store_begin - ks;
                        const auto line = ub_logits_stage.ptr() + q_store * kSplitKV;
                        mqa_logits_store<kNumLogitsStages>(logits.index(row, dst_col).ptr(),
                                                           line + store_begin - kv_idx, ub_logits_head.ptr(),
                                                           store_end - store_begin);
                    }
                    asc_sync_notify(PIPE_MTE3, PIPE_V, static_cast<event_t>(logits_stage_idx));
                    logits_stage_idx = (logits_stage_idx + 1) % kNumLogitsStages;
                }
                if constexpr (kIsPaged)
                    publish_sf_kv(kv_stage_idx);
                kv_stage_idx = (kv_stage_idx + 1) % kNumKVStages;
            }
            asc_sync_notify(PIPE_V, PIPE_MTE2, static_cast<event_t>(weight_stage_idx));
            weight_stage_idx = (weight_stage_idx + 1) % kNumWeightStages;
        }
    }
}

template <
    bool kIsFP4,
    uint32_t kNumHeads, uint32_t kHeadDim,
    uint32_t kSplitKV, uint32_t kMadM, uint32_t kNumCDStages,
    uint32_t kNumQStages, uint32_t kNumKVStages, uint32_t kNumAccumStages,
    uint32_t kNumLogitsStages,
    uint32_t kNumCores
>
__global__ __mix__(1, 2) void mqa_logits_impl(
    uint32_t num_q_tokens, uint32_t num_kv_tokens,
    gm_ptr<uint8_t, Major::K> q, gm_ptr<uint8_t, Major::K> kv,
    __gm__ int16_t* sf_q, __gm__ int16_t* sf_kv,
    gm_ptr<bfloat16_t, Major::K> weights,
    gm_ptr<bfloat16_t, Major::K> logits,
    __gm__ uint32_t* cu_seq_len_k_start, __gm__ uint32_t* cu_seq_len_k_end
) {
    asc_init();
    constexpr uint32_t kNumQTokensPerBlock = 8 * kMadM / kNumHeads;
    const auto make_scheduler = [&](uint32_t core_id) __aicore__ {
        return MQALogitsScheduler<kNumQTokensPerBlock, kSplitKV, kNumCores>(core_id, num_q_tokens, num_kv_tokens,
                                                                            cu_seq_len_k_start, cu_seq_len_k_end,
                                                                            kNumHeads);
    };
    mqa_logits_core_impl<kIsFP4, kNumHeads, kHeadDim, kSplitKV, kMadM, kNumCDStages, kNumQStages, kNumKVStages, kNumAccumStages, kNumLogitsStages, 0u, false, decltype(make_scheduler)>(q, kv, sf_q, sf_kv, weights, logits, make_scheduler);
}

template <
    bool kIsFP4,
    uint32_t kNumHeads, uint32_t kHeadDim,
    uint32_t kPageSize,
    uint32_t kSplitKV, uint32_t kMadM, uint32_t kNumCDStages,
    uint32_t kNumQStages, uint32_t kNumKVStages, uint32_t kNumAccumStages,
    uint32_t kNumLogitsStages,
    bool kUseAscendKVLayout
>
__global__ __mix__(1, 2) void paged_mqa_logits_impl(
    uint32_t num_q_tokens,
    uint32_t block_table_row_stride, uint64_t kv_page_stride_bytes,
    gm_ptr<uint8_t, Major::K> q, __gm__ int16_t* sf_q,
    gm_ptr<uint8_t, Major::K> fused_kv,
    gm_ptr<bfloat16_t, Major::K> weights,
    gm_ptr<bfloat16_t, Major::K> logits,
    __gm__ uint32_t* context_lens, __gm__ uint32_t* block_table,
    __gm__ uint32_t* indices, __gm__ uint32_t* metadata
) {
    static_assert(kPageSize == 64 || kPageSize == 128, "Paged MQA logits supports PAGE_SIZE=64/128");
    static_assert(kSplitKV % kPageSize == 0, "SPLIT_KV must contain complete pages");
    static_assert(kSplitKV / kPageSize <= 8, "Page reorder supports at most four page pairs");
    asc_init();

    const auto kv = fused_kv;
    const auto sf_kv = (fused_kv + kPageSize * fused_kv.stride_outer).template ptr<int16_t>();

    constexpr uint32_t kNumQTokensPerBlock = 8 * kMadM / kNumHeads;
    const auto make_scheduler = [&](uint32_t core_id) __aicore__ {
        return PagedMQALogitsScheduler<kNumQTokensPerBlock, kSplitKV, kPageSize, kNumHeads>(core_id, num_q_tokens, context_lens, indices, metadata, block_table,
                                                                                                               block_table_row_stride, kv_page_stride_bytes);
    };
    // Every paged block streams SF through AIV with four-split lookahead across requests.
    mqa_logits_core_impl<kIsFP4, kNumHeads, kHeadDim, kSplitKV, kMadM, kNumCDStages, kNumQStages, kNumKVStages, kNumAccumStages, kNumLogitsStages, kPageSize, kUseAscendKVLayout, decltype(make_scheduler)>(q, kv, sf_q, sf_kv, weights, logits, make_scheduler);
}

} // namespace deep_gemm
