#pragma once

#include <deep_gemm/ascend.hpp>
#include <deep_gemm/layout/sparse_mqa_logits.hpp>

namespace deep_gemm {

using namespace __cce_simd;

// Gather one 32-byte DATA fragment per token; offsets are in uint32_t words.
// With F words per fragment and D words per token, lane i = t*F + r reads
//   indices[i] = t*D + r, 0 <= r < F.
template <uint32_t kDataStrideWords>
__simd_callee__ inline void sparse_mqa_logits_make_data_indices(vector_uint32_t& indices) {
    const auto mask = asc_create_mask_b32(PAT_ALL);
    constexpr uint32_t kFragmentWords = 32 / sizeof(uint32_t);
    vector_uint32_t lane, row, column, bits;
    asc_arange(reinterpret_cast<vector_int32_t&>(lane), 0);
    asc_shiftright_scalar(row, lane, __builtin_ctz(kFragmentWords), mask);
    asc_duplicate_scalar(bits, kFragmentWords - 1);
    asc_and(column, lane, bits, mask);
    asc_mul_scalar(row, row, kDataStrideWords, mask);
    asc_add(indices, row, column, mask);
}

// Transpose token-major scales x[token, pair] to y[group, pair, token within group]:
//   y[g, p, r] = x[g*G + r, p], 0 <= r < G, 0 <= p < 2.
// G = 16 tokens per group; each token has two uint16_t scale fields.
__simd_callee__ inline void sparse_mqa_logits_make_sf_indices(vector_uint16_t& indices) {
    const auto mask = asc_create_mask_b16(PAT_ALL);
    vector_uint16_t lane, row, pair, bits;
    asc_arange(reinterpret_cast<vector_int16_t&>(lane), static_cast<int16_t>(0));
    asc_duplicate_scalar(bits, static_cast<uint16_t>(FRAC_MN - 1));
    asc_and(row, lane, bits, mask);
    asc_shiftright_scalar(pair, lane, __builtin_ctz(2 * FRAC_MN), mask);
    asc_shiftleft_scalar(pair, pair, __builtin_ctz(FRAC_MN), mask);
    asc_add(row, row, pair, mask);
    asc_shiftright_scalar(pair, lane, __builtin_ctz(FRAC_MN), mask);
    asc_duplicate_scalar(bits, static_cast<uint16_t>(1));
    asc_and(pair, pair, bits, mask);
    asc_shiftleft_scalar(row, row, 1, mask);
    asc_add(indices, row, pair, mask);
}

template <
    uint32_t kSparseBlockKV, uint32_t kNumElems, uint32_t kNumHeads, uint32_t kLogitsStride,
    uint32_t kNumQueries, bool kMappedOutput, bool kIsFP4, bool kConvert
>
__simd_callee__ inline void sparse_mqa_logits_reduce(
    __ubuf__ bfloat16_t* score, __ubuf__ bfloat16_t* weights, __ubuf__ bfloat16_t* logits,
    uint32_t mask0_lo, uint32_t mask0_hi, uint32_t mask1_lo, uint32_t mask1_hi,
    __ubuf__ uint32_t* data_src, __ubuf__ uint32_t* data_dst, __ubuf__ uint16_t* sf_dst,
    __ubuf__ uint16_t* mapping, uint32_t map_begin0, uint32_t map_begin1,
    uint32_t num_selected_blocks0, uint32_t num_selected_blocks1, uint32_t output_begin0 = 0, uint32_t output_begin1 = 0
) {
    constexpr uint32_t kHeadPitch = aligned(kNumHeads, FRAC_MN);
    constexpr uint32_t kNumVecElems = asc_get_vf_len() / sizeof(bfloat16_t);
    constexpr uint32_t kNumVecRegs = kNumElems / kNumVecElems;
    constexpr uint32_t kBlocksPerVector = kNumVecElems / kSparseBlockKV;
    static_assert((kSparseBlockKV == 8 || kSparseBlockKV == 16) && (kNumQueries == 1 || kNumQueries == 2));
    static_assert(!kMappedOutput || !kConvert);
    // For query q and token t, reduce scores s with weights w over H heads:
    //   a[0] = 0; a[h+1] = mula_bf16(a[h], s[q, h, t], w[q, h]); sum[q, t] = a[H].
    // s is already ReLU'ed by FIX; mula_bf16 produces a BF16 result at every step.
    // Interleaved KV conversion does not change the head order.
    const auto all = asc_create_mask_b16(PAT_ALL);
    vector_bfloat16_t sum[kNumVecRegs][kNumQueries];
    #pragma unroll
    for (uint32_t vec_idx = 0; vec_idx < kNumVecRegs; ++ vec_idx) {
        #pragma unroll
        for (uint32_t q_idx = 0; q_idx < kNumQueries; ++ q_idx)
            asc_duplicate_scalar(sum[vec_idx][q_idx], static_cast<bfloat16_t>(0), all);
    }
    vector_int16_t lane, block_lane;
    asc_arange(lane, static_cast<int16_t>(0));
    asc_shiftright_scalar(block_lane, lane, __builtin_ctz(kSparseBlockKV), all);
    vector_uint16_t one, block_bits;
    if constexpr (!kMappedOutput) {
        asc_duplicate_scalar(one, static_cast<uint16_t>(1));
        asc_shiftleft(block_bits, one, block_lane, all);
    }
    constexpr uint32_t kDataBytes = layout::sparse_mqa_logits::kHeadDim / (kIsFP4 ? 2 : 1);
    constexpr uint32_t kDataWords = kDataBytes / sizeof(uint32_t);
    constexpr uint32_t kFragmentWords = 32 / sizeof(uint32_t);
    constexpr uint32_t kRowsPerVector = sizeof(vector_uint32_t) / 32;
    constexpr uint32_t kNumConvertRows = layout::sparse_mqa_logits::kSplitKV / 2;
    constexpr uint32_t kNZRowStride = kNumConvertRows + 1;
    vector_uint32_t convert_indices;
    vector_uint16_t sf_indices;
    const auto all32 = asc_create_mask_b32(PAT_ALL);
    if constexpr (kConvert) {
        static_assert(kNumConvertRows == kLogitsStride / 2);
        sparse_mqa_logits_make_data_indices<kDataWords>(convert_indices);
        sparse_mqa_logits_make_sf_indices(sf_indices);
    }
    auto score_ptr0 = score;
    auto score_ptr1 = score + kHeadPitch * kNumElems;
    auto weight_ptr0 = weights;
    auto weight_ptr1 = weights + (kMappedOutput ? kNumHeads : kHeadPitch);
    constexpr uint32_t kHeadsPerSF = kConvert ? sizeof(vector_uint16_t) / 4 / kRowsPerVector : kNumHeads;
    // Convert half a KV split regardless of the number of reduced heads.
    constexpr uint32_t kNumHeadSteps = kConvert ? kNumConvertRows / kRowsPerVector : kNumHeads;
    #pragma unroll (kNumHeadSteps > kNumHeads ? kNumHeadSteps / kHeadsPerSF : 1)
    for (uint32_t head_begin = 0; head_begin < kNumHeadSteps; head_begin += kHeadsPerSF) {
        #pragma unroll 2
        for (uint32_t head_idx = head_begin; head_idx < head_begin + kHeadsPerSF; ++ head_idx) {
            if constexpr (kConvert) {
                #pragma unroll
                for (uint32_t frag_idx = 0; frag_idx < kDataWords / kFragmentWords; ++ frag_idx) {
                    const uint32_t row_idx = head_idx * kRowsPerVector;
                    vector_uint32_t values;
                    asc_gather(values, data_src + row_idx * kDataWords + frag_idx * kFragmentWords, convert_indices, all32);
                    asc_storealign(data_dst + (frag_idx * kNZRowStride + row_idx) * kFragmentWords, values, all32);
                }
            }
            if (kNumHeadSteps == kNumHeads || head_idx < kNumHeads) {
                vector_bfloat16_t weight0, weight1;
                asc_loadalign_brc_postupdate(weight0, weight_ptr0, 1);
                if constexpr (kNumQueries == 2)
                    asc_loadalign_brc_postupdate(weight1, weight_ptr1, 1);
                #pragma unroll
                for (uint32_t vec_idx = 0; vec_idx < kNumVecRegs; ++ vec_idx) {
                    vector_bfloat16_t value0;
                    asc_loadalign_postupdate(value0, score_ptr0, kNumVecElems);
                    asc_mula(sum[vec_idx][0], value0, weight0, all);
                }
                if constexpr (kNumQueries == 2) {
                    #pragma unroll
                    for (uint32_t vec_idx = 0; vec_idx < kNumVecRegs; ++ vec_idx) {
                        vector_bfloat16_t value1;
                        asc_loadalign_postupdate(value1, score_ptr1, kNumVecElems);
                        asc_mula(sum[vec_idx][1], value1, weight1, all);
                    }
                }
            }
        }
        if constexpr (kConvert) {
            const uint32_t row_idx = head_begin * kRowsPerVector;
            vector_uint16_t values;
            auto* sf_src = reinterpret_cast<__ubuf__ uint16_t*>(data_src + kNumConvertRows * kDataWords) + row_idx * 2;
            asc_gather(values, sf_src, sf_indices, all);
            asc_storealign(sf_dst + row_idx * 2, values, all);
        }
    }
    // For one query, view reduced values x and output y as [block, token within block].
    // Mask m[b] is 1 iff this query selects source block b:
    //   y[sum_{j < b} m[j], r] = x[b, r], for m[b] = 1.
    // Mapping p[b] gives source block b's position in the original sparse list:
    //   y[p[b], r] = x[b, r], for valid source blocks and output positions.
    #pragma unroll
    for (uint32_t q_idx = 0; q_idx < kNumQueries; ++ q_idx) {
        const uint32_t mask_lo = q_idx ? mask1_lo : mask0_lo;
        const uint32_t mask_hi = q_idx ? mask1_hi : mask0_hi;
        auto query_logits = logits + q_idx * kLogitsStride;
        vector_store_unalign store_desc;
        vector_uint16_t output_mapping;
        const uint32_t map_begin = q_idx ? map_begin1 : map_begin0;
        if constexpr (kMappedOutput)
            asc_loadalign(output_mapping, mapping + (map_begin & ~15u));
        else
            asc_squeeze_and_storeunalign_init();
        #pragma unroll
        for (uint32_t vec_idx = 0; vec_idx < kNumVecRegs; ++ vec_idx) {
            auto value = sum[vec_idx][q_idx];
            const uint32_t block_idx = vec_idx * kBlocksPerVector;
            vector_bool mask;
            if constexpr (kMappedOutput) {
                vector_uint16_t indices, output_block_idx, token_lane, bits;
                asc_add_scalar(indices, reinterpret_cast<vector_uint16_t&>(block_lane), static_cast<uint16_t>(block_idx), all);
                asc_lt_scalar(mask, indices, static_cast<uint16_t>(q_idx == 0 ? num_selected_blocks0 : num_selected_blocks1), all);
                asc_add_scalar(indices, indices, static_cast<uint16_t>(map_begin & 15u), all);
                asc_gather(output_block_idx, output_mapping, indices);
                asc_shiftleft_scalar(indices, output_block_idx, __builtin_ctz(kSparseBlockKV), all);
                asc_duplicate_scalar(bits, static_cast<uint16_t>(kSparseBlockKV - 1));
                asc_and(token_lane, reinterpret_cast<vector_uint16_t&>(lane), bits, all);
                asc_add(indices, indices, token_lane, all);
                asc_add_scalar(indices, indices, static_cast<uint16_t>(0u - (q_idx ? output_begin1 : output_begin0)), all);
                vector_bool valid;
                asc_lt_scalar(valid, indices, static_cast<uint16_t>(kLogitsStride), mask);
                asc_scatter(query_logits, value, indices, valid);
            } else {
                const uint32_t selected = block_idx < 32 ? mask_lo : mask_hi;
                vector_uint16_t selected_bits;
                asc_duplicate_scalar(selected_bits, static_cast<uint16_t>(selected >> (block_idx % 32)));
                asc_and(selected_bits, selected_bits, block_bits, all);
                asc_ne_scalar(mask, selected_bits, static_cast<uint16_t>(0), all);
                vector_bfloat16_t packed;
                asc_squeeze_and_storeunalign(query_logits, packed, store_desc, value, mask);
            }
        }
        if constexpr (!kMappedOutput)
            asc_squeeze_and_storeunalign_finalize(query_logits, store_desc);
    }
}

template <uint32_t kSparseBlockKV, uint32_t kNumElems, uint32_t kNumHeads, uint32_t kNumQueries, bool kIsFP4, bool kConvert>
__simd_vf__ inline void vf_sparse_mqa_logits_reduce_select(
    __ubuf__ bfloat16_t* score, __ubuf__ bfloat16_t* weights, __ubuf__ bfloat16_t* logits,
    uint32_t mask0_lo, uint32_t mask0_hi, uint32_t mask1_lo, uint32_t mask1_hi,
    __ubuf__ uint32_t* data_src, __ubuf__ uint32_t* data_dst, __ubuf__ uint16_t* sf_dst
) {
    sparse_mqa_logits_reduce<kSparseBlockKV, kNumElems, kNumHeads, layout::sparse_mqa_logits::kSplitKV, kNumQueries, false, kIsFP4, kConvert>(
        score, weights, logits, mask0_lo, mask0_hi, mask1_lo, mask1_hi, data_src, data_dst, sf_dst, nullptr, 0, 0, 0, 0);
}

template <uint32_t kSparseBlockKV, uint32_t kNumElems, uint32_t kNumHeads, uint32_t kNumQueries>
__simd_vf__ inline void vf_sparse_mqa_logits_reduce_scatter(
    __ubuf__ bfloat16_t* score, __ubuf__ bfloat16_t* weights, __ubuf__ bfloat16_t* logits,
    __ubuf__ uint16_t* mapping, uint32_t map_begin0, uint32_t map_begin1,
    uint32_t num_selected_blocks0, uint32_t num_selected_blocks1, uint32_t output_begin0, uint32_t output_begin1
) {
    sparse_mqa_logits_reduce<kSparseBlockKV, kNumElems, kNumHeads, layout::paged_sparse_mqa_logits::kMaxOutputTokens, kNumQueries, true, false, false>(
        score, weights, logits, 0, 0, 0, 0, nullptr, nullptr, nullptr,
        mapping, map_begin0, map_begin1, num_selected_blocks0, num_selected_blocks1, output_begin0, output_begin1);
}

// View token-major DATA as [token t, fragment f, word r within fragment]; convert to NZ:
//   data_dst[f, t, r] = data_src[t, f, r].
// Each fragment is 32 bytes; the NZ token axis includes row padding.
// FP4/FP8 bits are unchanged. SF follows the scale transpose above.
template <uint32_t kNZRowStride, bool kIsFP4>
__simd_vf__ inline void vf_sparse_mqa_logits_convert_kv(
    __ubuf__ uint32_t* data_src, __ubuf__ uint32_t* data_dst, __ubuf__ uint16_t* sf_dst,
    __ubuf__ uint16_t* sf_src, uint32_t num_rows
) {
    const uint32_t num_data_rows = (num_rows + 127u) & ~127u, num_sf_rows = (num_rows + 15u) & ~15u;
    constexpr uint32_t kDataBytes = layout::sparse_mqa_logits::kHeadDim / (kIsFP4 ? 2 : 1);
    constexpr uint32_t kDataWords = kDataBytes / sizeof(uint32_t);
    constexpr uint32_t kFragmentWords = 32 / sizeof(uint32_t);
    constexpr uint32_t kRowsPerVector = sizeof(vector_uint32_t) / 32;
    const auto all32 = asc_create_mask_b32(PAT_ALL);
    if constexpr (kIsFP4) {
        vector_uint32_t lane, pair, word, token, bits, indices;
        asc_arange(reinterpret_cast<vector_int32_t&>(lane), 0);
        asc_duplicate_scalar(bits, 1);
        asc_and(pair, lane, bits, all32);
        asc_shiftleft_scalar(pair, pair, 3, all32);
        asc_shiftright_scalar(word, lane, 1, all32);
        asc_duplicate_scalar(bits, 7);
        asc_and(word, word, bits, all32);
        asc_shiftright_scalar(token, lane, 4, all32);
        asc_shiftleft_scalar(token, token, 4, all32);
        asc_add(indices, token, word, all32);
        asc_add(indices, indices, pair, all32);
        for (uint32_t row_idx = 0; row_idx < num_data_rows; row_idx += kRowsPerVector) {
            vector_uint32_t first, second, paired_first, paired_second, lo, hi;
            asc_loadalign(first, data_src + row_idx * kDataWords);
            asc_loadalign(second, data_src + row_idx * kDataWords + sizeof(vector_uint32_t) / sizeof(uint32_t));
            asc_gather(paired_first, first, indices);
            asc_gather(paired_second, second, indices);
            asc_deintlv(lo, hi, paired_first, paired_second);
            asc_storealign(data_dst + row_idx * kFragmentWords, lo, all32);
            asc_storealign(data_dst + (kNZRowStride + row_idx) * kFragmentWords, hi, all32);
        }
    } else {
        vector_uint32_t indices;
        sparse_mqa_logits_make_data_indices<kDataWords>(indices);
        #pragma unroll
        for (uint32_t frag_idx = 0; frag_idx < kDataWords / kFragmentWords; ++ frag_idx) {
            for (uint32_t row_idx = 0; row_idx < num_data_rows; row_idx += kRowsPerVector) {
                vector_uint32_t values;
                asc_gather(values, data_src + row_idx * kDataWords + frag_idx * kFragmentWords, indices, all32);
                asc_storealign(data_dst + (frag_idx * kNZRowStride + row_idx) * kFragmentWords, values, all32);
            }
        }
    }
    const auto all16 = asc_create_mask_b16(PAT_ALL);
    vector_uint16_t sf_indices;
    sparse_mqa_logits_make_sf_indices(sf_indices);
    for (uint32_t row_idx = 0; row_idx < num_sf_rows; row_idx += sizeof(vector_uint16_t) / 4) {
        vector_uint16_t values;
        asc_gather(values, sf_src + row_idx * 2, sf_indices, all16);
        asc_storealign(sf_dst + row_idx * 2, values, all16);
    }
}

} // namespace deep_gemm
