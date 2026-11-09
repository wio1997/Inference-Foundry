#pragma once

#include <kernel_operator.h>

#include <deep_gemm/ascend.hpp>
#include <deep_gemm/layout/sparse_mqa_logits.hpp>
#include <deep_gemm/scheduler/sparse_mqa_logits.hpp>
#include <deep_gemm/simd/sparse_mqa_logits.hpp>

using namespace __cce_simd;

namespace deep_gemm {

using namespace layout::sparse_mqa_logits;

template <uint32_t kSparseBlockKV>
__aicore__ constexpr uint32_t get_num_kv_split_blocks(uint32_t num_blocks, uint32_t kv_split_idx) {
    constexpr uint32_t kBlocksPerSplitKV = kSplitKV / kSparseBlockKV;
    const uint32_t begin = kv_split_idx * kBlocksPerSplitKV;
    const uint32_t remaining = begin < num_blocks ? num_blocks - begin : 0;
    return remaining < kBlocksPerSplitKV ? remaining : kBlocksPerSplitKV;
}

template <uint32_t kSparseBlockKV, uint32_t kNumHeads, bool kIsFP4, bool kConvert = false>
__aicore__ inline void sparse_mqa_logits_reduce_select(
    __ubuf__ bfloat16_t* score, __ubuf__ bfloat16_t* weights, __ubuf__ bfloat16_t* logits,
    uint32_t issued_n, uint32_t num_queries,
    uint32_t mask0_lo, uint32_t mask0_hi, uint32_t mask1_lo, uint32_t mask1_hi,
    __ubuf__ uint32_t* data_src = nullptr, __ubuf__ uint32_t* data_dst = nullptr, __ubuf__ uint16_t* sf_dst = nullptr
) {
    const auto reduce = [&](auto num_elems) __aicore__ {
        constexpr uint32_t kNumElems = decltype(num_elems)::value;
        if (num_queries == 2)
            vf_sparse_mqa_logits_reduce_select<kSparseBlockKV, kNumElems, kNumHeads, 2, kConvert && kIsFP4, kConvert>(
                score, weights, logits, mask0_lo, mask0_hi, mask1_lo, mask1_hi, data_src, data_dst, sf_dst);
        else
            vf_sparse_mqa_logits_reduce_select<kSparseBlockKV, kNumElems, kNumHeads, 1, kConvert && kIsFP4, kConvert>(
                score, weights, logits, mask0_lo, mask0_hi, 0, 0, data_src, data_dst, sf_dst);
    };
    constexpr uint32_t kNumVecElems = asc_get_vf_len() / sizeof(bfloat16_t);
    switch (issued_n / kNumVecElems) {
        case 1: reduce(std::integral_constant<uint32_t, kNumVecElems>{}); break;
        case 2: reduce(std::integral_constant<uint32_t, 2 * kNumVecElems>{}); break;
        case 3: reduce(std::integral_constant<uint32_t, 3 * kNumVecElems>{}); break;
        case 4: reduce(std::integral_constant<uint32_t, 4 * kNumVecElems>{}); break;
        default: ascendc_assert(false, "Unsupported issued_n for sparse MQA logits reduction");
    }
}

// Stage token-major DATA and scales for NZ/MX conversion.
template <uint32_t kSparseBlockKV, bool kIsFP4>
__aicore__ __attribute__((noinline)) void sparse_mqa_logits_gather_kv_aiv(
    __gm__ uint32_t* physical_kv_token_starts, ub_ptr<uint8_t> data, __ubuf__ uint8_t* scales,
    gm_ptr<uint8_t> kv, __gm__ int16_t* kv_sf, uint32_t kv_block_offset,
    uint32_t num_blocks, uint32_t num_kv_tokens
) {
    constexpr uint32_t kDataBytes = kHeadDim / (kIsFP4 ? 2 : 1);
    constexpr uint32_t kBlockDataBytes = kSparseBlockKV * kDataBytes, kBlockSFBytes = kSparseBlockKV * kSparseSFBytes;
    auto* scale_base = reinterpret_cast<__gm__ uint8_t*>(kv_sf);
    auto* data_dst = reinterpret_cast<__ubuf__ uint8_t*>(data.addr);
    // Sorted union blocks place all incomplete blocks at the end.
    uint32_t num_full_blocks = num_blocks;
    while (num_full_blocks &&
           static_cast<uint64_t>(physical_kv_token_starts[kv_block_offset + num_full_blocks - 1]) + kSparseBlockKV > num_kv_tokens)
        -- num_full_blocks;

    if constexpr (kIsFP4) {
        #pragma unroll 2
        for (uint32_t block_idx = 0; block_idx + 1 < num_full_blocks; block_idx += 2) {
            const uint32_t first = physical_kv_token_starts[kv_block_offset + block_idx];
            const uint32_t next = physical_kv_token_starts[kv_block_offset + block_idx + 1];
            copy_gm_to_ub(scales + block_idx * kBlockSFBytes, scale_base + static_cast<uint64_t>(first) * kSparseSFBytes,
                          2, kBlockSFBytes, static_cast<uint64_t>(next - first) * kSparseSFBytes, asc_load_l2_cache_mode::NORMAL_LAST_VICTIM);
        }
    }
    uint32_t block_idx = 0;
    #pragma unroll 2
    for (; block_idx + 1 < num_full_blocks; block_idx += 2) {
        const uint32_t first = physical_kv_token_starts[kv_block_offset + block_idx];
        const uint32_t next = physical_kv_token_starts[kv_block_offset + block_idx + 1];
        copy_gm_to_ub(data_dst + block_idx * kBlockDataBytes, kv.ptr() + static_cast<uint64_t>(first) * kDataBytes,
                      2, kBlockDataBytes, static_cast<uint64_t>(next - first) * kDataBytes, asc_load_l2_cache_mode::NORMAL_LAST_VICTIM);
        if constexpr (!kIsFP4)
            copy_gm_to_ub(scales + block_idx * kBlockSFBytes, scale_base + static_cast<uint64_t>(first) * kSparseSFBytes,
                          2, kBlockSFBytes, static_cast<uint64_t>(next - first) * kSparseSFBytes, asc_load_l2_cache_mode::NORMAL_LAST_VICTIM);
    }
    for (; block_idx < num_blocks; ++ block_idx) {
        const uint32_t first = physical_kv_token_starts[kv_block_offset + block_idx];
        const uint32_t valid = min(kSparseBlockKV, num_kv_tokens - first);
        copy_gm_to_ub(data_dst + block_idx * kBlockDataBytes, kv.ptr() + static_cast<uint64_t>(first) * kDataBytes,
                      1, valid * kDataBytes, 0, asc_load_l2_cache_mode::NORMAL_LAST_VICTIM);
        copy_gm_to_ub(scales + block_idx * kBlockSFBytes, scale_base + static_cast<uint64_t>(first) * kSparseSFBytes,
                      1, valid * kSparseSFBytes, 0, asc_load_l2_cache_mode::NORMAL_LAST_VICTIM);
    }
}

template <bool kIsFP4, uint32_t kNumHeads, uint32_t kSparseBlockKV>
__global__ __mix__(1, 2) void sparse_mqa_logits_impl(
    gm_ptr<uint8_t, Major::K> q, gm_ptr<uint8_t, Major::K> kv, __gm__ int16_t* kv_sf,
    __gm__ int16_t* sf_q,
    gm_ptr<bfloat16_t, Major::K> weights, gm_ptr<bfloat16_t, Major::K> logits,
    __gm__ uint32_t* metadata, uint32_t num_kv_tokens
) {
    AscendC::InitSocState();
    static_assert(kSparseBlockKV == 8 || kSparseBlockKV == 16);
    const auto metadata_header = reinterpret_cast<__gm__ MetadataHeader*>(metadata);
    const uint32_t num_max_sparse_blocks = metadata_header->num_max_sparse_blocks;
    auto* metadata_base = reinterpret_cast<__gm__ uint8_t*>(metadata);
    const auto q_block_headers = reinterpret_cast<__gm__ QBlockHeader*>(metadata_base + kMetadataAlignmentBytes);
    const auto physical_kv_token_starts = reinterpret_cast<__gm__ uint32_t*>(metadata_base + metadata_header->physical_blocks_offset);
    const auto kv_split_outputs = reinterpret_cast<__gm__ QBlockSplitKVOutputInfo*>(metadata_base + metadata_header->kv_split_outputs_offset);
    const auto schedule = reinterpret_cast<__gm__ ScheduleBoundary*>(metadata_base + metadata_header->schedule_offset);
    constexpr uint32_t kNumVecElems = asc_get_vf_len() / sizeof(bfloat16_t);
    static_assert(kNumHeads >= 4 && kNumHeads <= kNumMaxHeads && kNumHeads % 4 == 0);
    constexpr uint32_t kHeadPitch = aligned(kNumHeads, FRAC_MN);
    constexpr uint32_t kBlocksPerSplitKV = kSplitKV / kSparseBlockKV;
    constexpr uint32_t kMMAHeads = kNumQueriesPerMMA * kHeadPitch;
    constexpr uint32_t kGatherRows = kSplitKV / 2;
    constexpr uint32_t kSFPairs = kHeadDim / MX_SF_DIVISOR;
    constexpr uint32_t kDataBytes = kHeadDim / (kIsFP4 ? 2 : 1);
    using qk_dtype_t = std::conditional_t<kIsFP4, float4_e2m1x2_t, float8_e4m3_t>;

    constexpr uint32_t kNumKVStages = 4;
    constexpr uint32_t kNumL0CStages = 2;
    constexpr uint32_t kNumLogitsStages = 3;
    constexpr uint32_t kNumWeightStages = kNumQueriesPerMMA;
    constexpr ub_ptr<bfloat16_t> ub_score(kMMAHeads, kSplitKV, 0);
    constexpr uint32_t kGatherStageBytes = kGatherRows * (kDataBytes + kSparseSFBytes) + (kGatherRows + 1) * kDataBytes;
    constexpr uint32_t kFixedUBBytes = (kNumWeightStages * kHeadPitch +
                                        kNumLogitsStages * kNumQueriesPerMMA * kSplitKV) * sizeof(bfloat16_t) + asc_get_vf_len();
    // Reserve two scores, prefer double-buffered gather, then use spare UB for a third score.
    constexpr uint32_t kNumGatherStages = ub_score.offset(2) + kFixedUBBytes + 2 * kGatherStageBytes <= kSparseUBBytes ? 2 : 1;
    constexpr uint32_t kNumScoreStages = ub_score.offset(3) + kFixedUBBytes + kNumGatherStages * kGatherStageBytes <= kSparseUBBytes ? 3 : 2;
    constexpr uint32_t kGatherEventBase = kNumLogitsStages + 1;
    constexpr uint32_t kGatherNDEventBase = kNumWeightStages;
    static_assert(kGatherEventBase + kNumGatherStages <= 8);
    static_assert(kGatherNDEventBase + kNumGatherStages <= 8);

    constexpr l1_ptr<qk_dtype_t, Major::K> l1q(kMMAHeads, kHeadDim, 0);
    constexpr l1_ptr<qk_dtype_t, Major::K> l1kv(kSplitKV, kHeadDim, l1q.offset(kNumMMAsPerQBlock));
    constexpr l1_ptr<int16_t> l1_sf_q(kMMAHeads, kSFPairs, l1kv.offset(kNumKVStages));
    constexpr l1_ptr<int16_t> l1_sf_kv(kSplitKV, kSFPairs, l1_sf_q.offset(kNumMMAsPerQBlock));
    constexpr l0a_ptr<qk_dtype_t> l0a(kMMAHeads, kHeadDim);
    constexpr l0b_ptr<qk_dtype_t> l0b(kSplitKV, kHeadDim);
    constexpr l0c_ptr<float> l0c(kMMAHeads, kSplitKV);

    constexpr ub_ptr<bfloat16_t> ub_weights(1, kHeadPitch, ub_score.offset(kNumScoreStages));
    constexpr ub_ptr<bfloat16_t> ub_selected_logits(kNumQueriesPerMMA, kSplitKV, ub_weights.offset(kNumWeightStages));
    constexpr ub_ptr<uint8_t> ub_gather_nd(kGatherRows, kDataBytes + kSparseSFBytes, ub_selected_logits.offset(kNumLogitsStages));
    constexpr ub_ptr<uint8_t> ub_kv_nz(kGatherRows + 1, kDataBytes, ub_gather_nd.offset(kNumGatherStages) + asc_get_vf_len());
    static_assert(ub_kv_nz.offset(kNumGatherStages) <= kSparseUBBytes);
    static_assert(l1_sf_kv.offset(kNumKVStages) <= L1SizeBytes);
    static_assert(l0a.offset(kNumMMAsPerQBlock) <= L0ASizeBytes);
    static_assert(l0b.offset(1) <= L0BSizeBytes);
    static_assert(l0c.offset(kNumL0CStages) <= L0CSizeBytes);

    SparseMQALogitsScheduler<kSparseBlockKV> scheduler(get_block_idx(), q_block_headers, schedule);
    SparseMQALogitsTask task;
    uint32_t kv_stage_idx = 0;
    if ASCEND_IS_AIC {
        asc_set_mmad_direction_n();
        const gm_ptr<qk_dtype_t, Major::K> q_t(q.stride_outer, q.addr, q.stride_batch);
        const gm_ptr<int16_t> sf_q_t(kSFPairs, reinterpret_cast<uintptr_t>(sf_q));
        set_flags<PIPE_M, PIPE_MTE1, 2>();
        set_flags<PIPE_MTE1, PIPE_MTE2, kNumMMAsPerQBlock>();
        for (uint32_t sub_idx = 0; sub_idx < 2; ++ sub_idx)
            for (uint32_t stage_idx = 0; stage_idx < kNumKVStages; ++ stage_idx)
                set_intra_block(PIPE_MTE1, sub_idx, 4);
        uint32_t l0c_stage_idx = 0;
        while (scheduler.next(task)) {
            const uint32_t num_kv_splits = task.kv_split_end - task.kv_split_begin;
            const uint32_t num_merged_kv_blocks = q_block_headers[task.query_begin].num_merged_kv_blocks;
            const uint32_t num_mmas = ceil_div(task.num_q_block_queries, kNumQueriesPerMMA);
            for (uint32_t mma_idx = 0; mma_idx < num_mmas; ++ mma_idx) {
                const uint32_t mma_q_token_base = task.query_begin + mma_idx * kNumQueriesPerMMA;
                const uint32_t num_queries = min(kNumQueriesPerMMA, task.num_q_block_queries - mma_idx * kNumQueriesPerMMA);
                const uint32_t num_qh_rows = num_queries * kHeadPitch;
                const auto q_tile = l1q[mma_idx].as_mad_aligned(kMMAHeads, kHeadDim);
                asc_sync_wait(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(mma_idx));
                if constexpr (kNumHeads == kHeadPitch) {
                    copy_gm_to_l1(q_tile, q_t.index(mma_q_token_base * kNumHeads, 0), num_qh_rows, kHeadDim,
                                  asc_load_l2_cache_mode::NORMAL_LAST_VICTIM);
                    copy_gm_to_l1_mx(l1_sf_q[mma_idx], sf_q_t.index(mma_q_token_base * kNumHeads, 0),
                                     num_qh_rows, kSFPairs, asc_load_l2_cache_mode::NORMAL_LAST_VICTIM);
                } else {
                    for (uint32_t query_idx = 0; query_idx < num_queries; ++ query_idx) {
                        const l1_ptr<qk_dtype_t> dst_q(kMMAHeads, kHeadDim, q_tile.addr + query_idx * kHeadPitch * 32);
                        const l1_ptr<int16_t> dst_sf(kMMAHeads, kSFPairs, l1_sf_q[mma_idx].addr + query_idx * kHeadPitch * 4);
                        fill_mn_tail_block(dst_q, kNumHeads, kHeadPitch);
                        copy_gm_to_l1(dst_q, q_t.index((mma_q_token_base + query_idx) * kNumHeads, 0), kNumHeads, kHeadDim,
                                      asc_load_l2_cache_mode::NORMAL_LAST_VICTIM);
                        copy_gm_to_l1_mx(dst_sf, sf_q_t.index((mma_q_token_base + query_idx) * kNumHeads, 0),
                                         kNumHeads, kSFPairs, asc_load_l2_cache_mode::NORMAL_LAST_VICTIM);
                    }
                }
                asc_sync_notify(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(mma_idx));
                asc_sync_wait(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(mma_idx));
                if (mma_idx == 0)
                    asc_sync_wait(PIPE_M, PIPE_MTE1, static_cast<event_t>(1));
                const auto l0a_tile = l0a[mma_idx].as_mad_aligned(num_qh_rows, kHeadDim);
                copy_l1_to_l0a(l0a_tile, q_tile, 0, 0);
                copy_l1_to_l0a_mx(l0a_tile, l1_sf_q[mma_idx], 0, 0, kHeadDim);
            }
            asc_sync_notify(PIPE_MTE1, PIPE_M, static_cast<event_t>(1));
            asc_sync_wait(PIPE_MTE1, PIPE_M, static_cast<event_t>(1));
            for (uint32_t kv_split_idx = 0; kv_split_idx < num_kv_splits; ++ kv_split_idx) {
                const uint32_t issued_n = aligned(
                    get_num_kv_split_blocks<kSparseBlockKV>(num_merged_kv_blocks, task.kv_split_begin + kv_split_idx) * kSparseBlockKV, kNumVecElems);
                wait_intra_block(PIPE_MTE1, 0, 3);
                wait_intra_block(PIPE_MTE1, 1, 3);
                asc_sync_wait(PIPE_M, PIPE_MTE1, static_cast<event_t>(0));
                const auto l0b_tile = l0b.as_mad_aligned(issued_n, kHeadDim);
                copy_l1_to_l0b(l0b_tile, l1kv[kv_stage_idx], 0, 0);
                copy_l1_to_l0b_mx(l0b_tile, l1_sf_kv[kv_stage_idx], 0, 0, kHeadDim);
                set_intra_block(PIPE_MTE1, 0, 4);
                set_intra_block(PIPE_MTE1, 1, 4);
                asc_sync_notify(PIPE_MTE1, PIPE_M, static_cast<event_t>(0));
                asc_sync_wait(PIPE_MTE1, PIPE_M, static_cast<event_t>(0));
                for (uint32_t mma_idx = 0; mma_idx < num_mmas; ++ mma_idx) {
                    const uint32_t score_stage_idx = kv_split_idx % kNumScoreStages;
                    const uint32_t issued_m = min(kNumQueriesPerMMA, task.num_q_block_queries - mma_idx * kNumQueriesPerMMA) * kHeadPitch;
                    const auto l0c_tile = l0c[l0c_stage_idx].as_mad_aligned(issued_m, issued_n);
                    mad(l0c_tile, l0a[mma_idx], l0b_tile, issued_m, issued_n, kHeadDim,
                        asc_unit_flag_mode::ENABLE_UPDATE, true);
                    wait_intra_block(PIPE_FIX, mma_idx, score_stage_idx);
                    copy_l0c_to_ub(ub_score[score_stage_idx].as_mad_aligned(kMMAHeads, issued_n), l0c_tile,
                                   issued_m, issued_n, asc_dual_dst_mode::DUAL_DST_DISABLE, mma_idx,
                                   infer_quant_mode<bfloat16_t, float>(), true, false,
                                   asc_unit_flag_mode::ENABLE_UPDATE);
                    set_intra_block(PIPE_FIX, mma_idx, score_stage_idx);
                    l0c_stage_idx = (l0c_stage_idx + 1) % kNumL0CStages;
                }
                asc_sync_notify(PIPE_M, PIPE_MTE1, static_cast<event_t>(0));
                kv_stage_idx = (kv_stage_idx + 1) % kNumKVStages;
            }
            asc_sync_notify(PIPE_M, PIPE_MTE1, static_cast<event_t>(1));
            for (uint32_t mma_idx = 0; mma_idx < num_mmas; ++ mma_idx)
                asc_sync_notify(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(mma_idx));
        }
    } else {
        const uint32_t aiv_idx = get_subblockid();
        set_intra_blocks<PIPE_V, kNumScoreStages>(aiv_idx);
        set_flags<PIPE_MTE3, PIPE_V, kNumLogitsStages>();
        set_flags<PIPE_V, PIPE_MTE2, kNumWeightStages>();
        for (uint32_t stage_idx = 0; stage_idx < kNumGatherStages; ++ stage_idx)
            asc_sync_notify(PIPE_MTE3, PIPE_MTE2, static_cast<event_t>(kGatherEventBase + stage_idx));
        asc_sync_notify(PIPE_V, PIPE_MTE3, static_cast<event_t>(kNumLogitsStages));
        asc_sync_wait(PIPE_V, PIPE_MTE3, static_cast<event_t>(kNumLogitsStages));
        uint32_t logits_stage_idx = 0;
        uint32_t gather_stage_idx = 0;
        bool kv_prefetched = false;
        bool kv_converted = false;
        while (scheduler.next(task)) {
            const uint32_t num_kv_splits = task.kv_split_end - task.kv_split_begin;
            const uint32_t num_merged_kv_blocks = q_block_headers[task.query_begin].num_merged_kv_blocks;
            const auto q_block_physical_kv_token_starts = physical_kv_token_starts + static_cast<uint64_t>(task.query_begin) * num_max_sparse_blocks;
            const auto q_block_kv_splits = kv_split_outputs + static_cast<uint64_t>(task.query_begin) * ceil_div(num_max_sparse_blocks, kBlocksPerSplitKV);
            const uint32_t q_token_offset = aiv_idx * kNumQueriesPerMMA;
            const uint32_t q_token_idx = task.query_begin + q_token_offset;
            const uint32_t num_owned_queries = task.num_q_block_queries > q_token_offset ?
                min(kNumQueriesPerMMA, task.num_q_block_queries - q_token_offset) : 0;
            // Warm the second KV split guard before the first gather.
            const uint32_t second_kv_split_idx = task.kv_split_begin + 1;
            if (num_kv_splits >= 2 && get_num_kv_split_blocks<kSparseBlockKV>(num_merged_kv_blocks, second_kv_split_idx) == kBlocksPerSplitKV)
                asc_datacache_preload(reinterpret_cast<__gm__ uint64_t*>(
                    q_block_physical_kv_token_starts + (task.kv_split_begin + 2) * kBlocksPerSplitKV - 1), 0);
            for (uint32_t q_idx = 0; q_idx < num_owned_queries; ++ q_idx) {
                asc_sync_wait(PIPE_V, PIPE_MTE2, static_cast<event_t>(q_idx));
                copy_gm_to_ub(ub_weights[q_idx].ptr(), weights.index(q_token_idx + q_idx, 0).ptr(),
                              1, kNumHeads, weights.stride_outer * sizeof(bfloat16_t),
                              asc_load_l2_cache_mode::NORMAL_LAST_VICTIM);
                asc_sync_notify(PIPE_MTE2, PIPE_V, static_cast<event_t>(q_idx));
                asc_sync_wait(PIPE_MTE2, PIPE_V, static_cast<event_t>(q_idx));
            }
            const uint32_t gather_sub_idx = aiv_idx ^ ((task.query_begin / kBlockQ) & 1u);
            const auto prefetch_kv = [&](__gm__ uint32_t* physical_kv_token_starts, uint32_t num_kv_blocks,
                                         uint32_t gather_sub_idx) __aicore__ {
                const event_t gather_event = static_cast<event_t>(kGatherEventBase + gather_stage_idx);
                const event_t nd_event = static_cast<event_t>(kGatherNDEventBase + gather_stage_idx);
                const auto data_nd = ub_gather_nd[gather_stage_idx];
                const uint32_t kv_block_offset = gather_sub_idx * (kGatherRows / kSparseBlockKV);
                const uint32_t num_data_kv_blocks = num_kv_blocks > kv_block_offset ?
                    min(kGatherRows / kSparseBlockKV, num_kv_blocks - kv_block_offset) : 0;
                asc_sync_wait(PIPE_MTE3, PIPE_MTE2, gather_event);
                // Padding columns are masked, and output counts exclude partial-block tails.
                sparse_mqa_logits_gather_kv_aiv<kSparseBlockKV, kIsFP4>(
                    physical_kv_token_starts, data_nd,
                    reinterpret_cast<__ubuf__ uint8_t*>(data_nd.addr + kGatherRows * kDataBytes),
                    kv, kv_sf, kv_block_offset, num_data_kv_blocks, num_kv_tokens);
                asc_sync_notify(PIPE_MTE2, PIPE_V, nd_event);
            };
            if (num_kv_splits && !kv_prefetched)
                prefetch_kv(q_block_physical_kv_token_starts + task.kv_split_begin * kBlocksPerSplitKV,
                            get_num_kv_split_blocks<kSparseBlockKV>(num_merged_kv_blocks, task.kv_split_begin), gather_sub_idx);
            kv_prefetched = false;
            for (uint32_t iter = 0; iter < num_kv_splits + kNumScoreStages; ++ iter) {
                if (iter < num_kv_splits) {
                    // Warm output metadata before KV convert and compute.
                    auto kv_split = q_block_kv_splits + task.kv_split_begin + iter;
                    asc_datacache_preload(reinterpret_cast<__gm__ uint64_t*>(kv_split), 0);
                    asc_datacache_preload(reinterpret_cast<__gm__ uint64_t*>(&kv_split->masks[kBlockQ - 1]), 0);
                    // Allow an extra KV split for FP4 block-index prefetch.
                    constexpr uint32_t kBlockPrefetchDistance = kIsFP4 ? 2 : 1;
                    if (iter + kBlockPrefetchDistance < num_kv_splits) {
                        auto blocks = q_block_physical_kv_token_starts + (task.kv_split_begin + iter + kBlockPrefetchDistance) * kBlocksPerSplitKV;
                        const uint32_t num_blocks = get_num_kv_split_blocks<kSparseBlockKV>(num_merged_kv_blocks, task.kv_split_begin + iter + kBlockPrefetchDistance);
                        for (uint32_t block_idx = gather_sub_idx * (kGatherRows / kSparseBlockKV); block_idx < min(num_blocks, (gather_sub_idx + 1) * (kGatherRows / kSparseBlockKV)); block_idx += 64 / sizeof(uint32_t))
                            asc_datacache_preload(reinterpret_cast<__gm__ uint64_t*>(blocks + block_idx), 0);
                    }
                    // Warm the full KV split tail guard.
                    const uint32_t prefetch_kv_split_idx = task.kv_split_begin + iter + 2;
                    if (iter + 2 < num_kv_splits && get_num_kv_split_blocks<kSparseBlockKV>(num_merged_kv_blocks, prefetch_kv_split_idx) == kBlocksPerSplitKV)
                        asc_datacache_preload(reinterpret_cast<__gm__ uint64_t*>(
                            q_block_physical_kv_token_starts + (task.kv_split_begin + iter + 3) * kBlocksPerSplitKV - 1), 0);
                    const event_t gather_event = static_cast<event_t>(kGatherEventBase + gather_stage_idx);
                    const event_t nd_event = static_cast<event_t>(kGatherNDEventBase + gather_stage_idx);
                    const auto data_nd = ub_gather_nd[gather_stage_idx];
                    const ub_ptr<int16_t> sf_nd(kGatherRows, kSFPairs,
                                                data_nd.addr + kGatherRows * kDataBytes);
                    if (!kv_converted) {
                        asc_sync_wait(PIPE_MTE2, PIPE_V, nd_event);
                        vf_sparse_mqa_logits_convert_kv<kGatherRows + 1, kIsFP4>(
                            data_nd.template ptr<uint32_t>(), ub_kv_nz[gather_stage_idx].template ptr<uint32_t>(),
                            sf_nd.template ptr<uint16_t>(), sf_nd.template ptr<uint16_t>(), kGatherRows);
                    }
                    kv_converted = false;
                    asc_sync_notify(PIPE_V, PIPE_MTE3, gather_event);
                    asc_sync_wait(PIPE_V, PIPE_MTE3, gather_event);
                    wait_intra_block(PIPE_MTE3, aiv_idx, 4);
                    copy_ubuf_to_cbuf(
                        l1kv[kv_stage_idx].offset_bytes(gather_sub_idx * kGatherRows * 32).ptr(),
                        ub_kv_nz[gather_stage_idx].ptr(), false, kDataBytes / 32, kGatherRows, 1, kGatherRows);
                    copy_ubuf_to_cbuf(
                        l1_sf_kv[kv_stage_idx].offset_bytes(gather_sub_idx * kGatherRows * 4).ptr(),
                        sf_nd.ptr(), false, 1, kGatherRows * 4 / 32, 0, 0);
                    asc_sync_notify(PIPE_MTE3, PIPE_MTE2, gather_event);
                    set_intra_block(PIPE_MTE3, aiv_idx, 3);
                    kv_stage_idx = (kv_stage_idx + 1) % kNumKVStages;
                    gather_stage_idx = (gather_stage_idx + 1) % kNumGatherStages;
                    SparseMQALogitsTask next_task;
                    if (iter + 1 < num_kv_splits) {
                        prefetch_kv(q_block_physical_kv_token_starts + (task.kv_split_begin + iter + 1) * kBlocksPerSplitKV,
                                    get_num_kv_split_blocks<kSparseBlockKV>(num_merged_kv_blocks, task.kv_split_begin + iter + 1), gather_sub_idx);
                    } else if (scheduler.peek(next_task)) {
                        const uint64_t next_query_idx = next_task.query_begin;
                        const uint32_t num_next_blocks = q_block_headers[next_query_idx].num_merged_kv_blocks;
                        if (num_next_blocks) {
                            prefetch_kv(physical_kv_token_starts + next_query_idx * num_max_sparse_blocks,
                                        get_num_kv_split_blocks<kSparseBlockKV>(num_next_blocks, 0),
                                        aiv_idx ^ ((next_query_idx / kBlockQ) & 1u));
                            kv_prefetched = true;
                        }
                    }
                }
                if (iter < kNumScoreStages)
                    continue;
                const uint32_t kv_split_idx = iter - kNumScoreStages;
                const uint32_t issued_n = aligned(
                    get_num_kv_split_blocks<kSparseBlockKV>(num_merged_kv_blocks, task.kv_split_begin + kv_split_idx) * kSparseBlockKV, kNumVecElems);
                auto kv_split = q_block_kv_splits + task.kv_split_begin + kv_split_idx;
                const uint32_t score_stage_idx = kv_split_idx % kNumScoreStages;
                if (!num_owned_queries)
                    continue;
                const uint32_t output0 = kv_split->output_ranges[q_token_offset];
                const uint32_t output1 = num_owned_queries == 2 ? kv_split->output_ranges[q_token_offset + 1] : 0;
                const bool has_output = (output0 | output1) >> 16;
                wait_intra_block(PIPE_V, aiv_idx, score_stage_idx);
                if (has_output) {
                    asc_sync_wait(PIPE_MTE3, PIPE_V, static_cast<event_t>(logits_stage_idx));
                    // Fusion needs a second gather stage to overlap convert with reduction.
                    if (kNumGatherStages > 1 && (iter + 1 < num_kv_splits || (kv_prefetched && !kv_converted))) {
                        asc_sync_wait(PIPE_MTE2, PIPE_V, static_cast<event_t>(kGatherNDEventBase + gather_stage_idx));
                        sparse_mqa_logits_reduce_select<kSparseBlockKV, kNumHeads, kIsFP4, true>(
                            ub_score[score_stage_idx].ptr(), ub_weights.ptr(), ub_selected_logits[logits_stage_idx].ptr(),
                            issued_n, num_owned_queries, kv_split->masks[q_token_offset].lo, kv_split->masks[q_token_offset].hi,
                            num_owned_queries == 2 ? kv_split->masks[q_token_offset + 1].lo : 0,
                            num_owned_queries == 2 ? kv_split->masks[q_token_offset + 1].hi : 0,
                            ub_gather_nd[gather_stage_idx].template ptr<uint32_t>(),
                            ub_kv_nz[gather_stage_idx].template ptr<uint32_t>(),
                            reinterpret_cast<__ubuf__ uint16_t*>(ub_gather_nd[gather_stage_idx].addr +
                                                                 kGatherRows * kDataBytes));
                        kv_converted = true;
                    } else {
                        sparse_mqa_logits_reduce_select<kSparseBlockKV, kNumHeads, kIsFP4>(
                            ub_score[score_stage_idx].ptr(), ub_weights.ptr(), ub_selected_logits[logits_stage_idx].ptr(),
                            issued_n, num_owned_queries, kv_split->masks[q_token_offset].lo, kv_split->masks[q_token_offset].hi,
                            num_owned_queries == 2 ? kv_split->masks[q_token_offset + 1].lo : 0,
                            num_owned_queries == 2 ? kv_split->masks[q_token_offset + 1].hi : 0);
                    }
                }
                set_intra_block(PIPE_V, aiv_idx, score_stage_idx);
                if (has_output) {
                    asc_sync_notify(PIPE_V, PIPE_MTE3, static_cast<event_t>(logits_stage_idx));
                    asc_sync_wait(PIPE_V, PIPE_MTE3, static_cast<event_t>(logits_stage_idx));
                    for (uint32_t q_idx = 0; q_idx < num_owned_queries; ++ q_idx) {
                        const uint32_t output_desc = q_idx ? output1 : output0;
                        const uint32_t num_output_tokens = output_desc >> 16;
                        if (num_output_tokens) {
                            copy_ub_to_gm(logits.index(q_token_idx + q_idx, output_desc & 0xffffu).ptr(),
                                          ub_selected_logits[logits_stage_idx].ptr() + q_idx * kSplitKV, 1, num_output_tokens,
                                          num_output_tokens * sizeof(bfloat16_t), asc_store_l2_cache_mode::NOTALLOC_CLEAN);
                        }
                    }
                    asc_sync_notify(PIPE_MTE3, PIPE_V, static_cast<event_t>(logits_stage_idx));
                    logits_stage_idx = (logits_stage_idx + 1) % kNumLogitsStages;
                }
            }
            for (uint32_t q_idx = 0; q_idx < num_owned_queries; ++ q_idx)
                asc_sync_notify(PIPE_V, PIPE_MTE2, static_cast<event_t>(q_idx));
        }
        for (uint32_t stage_idx = 0; stage_idx < kNumKVStages; ++ stage_idx)
            wait_intra_block(PIPE_MTE3, aiv_idx, 4);
    }
}

} // namespace deep_gemm
