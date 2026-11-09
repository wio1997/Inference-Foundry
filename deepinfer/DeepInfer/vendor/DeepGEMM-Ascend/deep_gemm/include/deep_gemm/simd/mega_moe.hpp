#pragma once

#include <deep_gemm/ascend.hpp>
#include <deep_gemm/layout/mega_moe.hpp>

#include <c_api/asc_simd.h>

namespace deep_gemm::mega_moe {

// FP4 -> FP8 NZ, scaled by 2^-6; weight SFs compensate with +6
// Rows map to GEMM N; columns map to K
template <uint32_t kNumRows, uint32_t kNumCols>
__simd_vf__ inline void vf_fp4_to_fp8_nz(__ubuf__ uint8_t* src, __ubuf__ uint8_t* dst) {
    constexpr uint32_t kNumPackedCols = kNumCols / 2;
    constexpr uint32_t kNumPaddedRows = kNumRows + 1;
    constexpr uint32_t kNumPackedColsPerVector = sizeof(vector_int8_t) / 2;  // Two FP4 values per byte
    constexpr uint32_t kNumVectorsPerRow = kNumPackedCols / kNumPackedColsPerVector;
    static_assert(kNumPackedCols % kNumPackedColsPerVector == 0);

    const auto mask = asc_create_mask_b8(PAT_ALL);
    vector_int8_t shift_right, shift_left, fp8_mask;
    asc_duplicate_scalar(shift_right, static_cast<int8_t>(2));
    asc_duplicate_scalar(reinterpret_cast<vector_uint16_t&>(shift_left), uint16_t{4});  // Alternating byte shifts: 4, 0
    asc_duplicate_scalar(fp8_mask, static_cast<int8_t>(0x9c));  // Sign and shifted FP4 payload

    auto src_ptr = reinterpret_cast<__ubuf__ int8_t*>(src);
    constexpr uint32_t kNumRowsPerPass = 4;
    static_assert(kNumRows % kNumRowsPerPass == 0);
    constexpr uint32_t kNumVectors = kNumVectorsPerRow * kNumRowsPerPass;

    // Each vector writes one row fragment in NZ order; the extra row pads the stride
    __ubuf__ uint8_t* dst_ptrs[kNumVectors];
    #pragma unroll
    for (uint32_t vector_idx = 0; vector_idx < kNumVectors; ++ vector_idx)
        dst_ptrs[vector_idx] = dst + vector_idx % kNumVectorsPerRow * kNumPaddedRows * 2 * kNumPackedColsPerVector +
                               vector_idx / kNumVectorsPerRow * get_frac_k<uint8_t>();

    for (uint32_t row_idx = 0; row_idx < kNumRows; row_idx += kNumRowsPerPass) {
        // Expand packed nibbles into scaled FP8 values
        vector_int8_t values[kNumVectors];
        #pragma unroll
        for (uint32_t vector_idx = 0; vector_idx < kNumVectors; ++ vector_idx) {
            asc_loadalign_upsample_postupdate(values[vector_idx], src_ptr, kNumPackedColsPerVector);
            asc_shiftleft(values[vector_idx], values[vector_idx], shift_left, mask);
            asc_shiftright(values[vector_idx], values[vector_idx], shift_right, mask);
            asc_and(values[vector_idx], values[vector_idx], fp8_mask, mask);
        }

        // Scatter into padded NZ and advance by four rows
        #pragma unroll
        for (uint32_t vector_idx = 0; vector_idx < kNumVectors; ++ vector_idx)
            asc_storealign_postupdate(dst_ptrs[vector_idx], reinterpret_cast<vector_uint8_t&>(values[vector_idx]),
                                      kNumPaddedRows, kNumRowsPerPass, mask);
    }
}

// Linear1 epilogue: SwiGLU and FP8 quantization
template <bool kApplyRouteWeight, uint32_t STORE_BLOCK_M>
__simd_vf__ inline void vf_swiglu_quant(__ubuf__ bfloat16_t* input, __ubuf__ float8_e4m3_t* output,
                                        __ubuf__ int16_t* output_sf, __ubuf__ float* route_weights,
                                        uint32_t valid_m, float activation_clamp) {
    constexpr uint32_t kNumRowsPerIteration = 2;
    constexpr uint32_t kNumPairsPerRow = (layout::BLOCK_N / 2) / MX_SF_DIVISOR;
    constexpr uint32_t kNumPairs = kNumRowsPerIteration * kNumPairsPerRow;
    static_assert(layout::BLOCK_N == 256);

    const vector_bool group_mask_b32 = asc_create_mask_b32(PAT_VL32);
    const vector_bool pair_mask_b32 = asc_create_mask_b32(PAT_VL64);
    vector_bfloat16_t zero_bf16;
    asc_duplicate_scalar(zero_bf16, 0);
    vector_int32_t arange_b32;
    vector_uint32_t high_group_idx;
    asc_arange(arange_b32, 0);
    asc_add_scalar(high_group_idx, reinterpret_cast<vector_uint32_t&>(arange_b32), MX_SF_DIVISOR / 2, group_mask_b32);
    vector_uint32_t exponent_mask, inv_scale_bias;
    asc_duplicate_scalar(exponent_mask, 0x7f800000u);
    asc_duplicate_scalar(inv_scale_bias, 0x7f000000u);  // (2 * 127) << 23

    auto gate_ptr = input;
    auto up_ptr = input + layout::BLOCK_N / 2;
    auto output_ptr = reinterpret_cast<__ubuf__ uint8_t*>(output);

    for (uint32_t m_idx = 0; m_idx < valid_m; m_idx += kNumRowsPerIteration) {
        // Load gate/up as FP32
        vector_float gate[kNumPairs], up[kNumPairs], value[kNumPairs];
        #pragma unroll
        for (uint32_t m_offset = 0; m_offset < kNumRowsPerIteration; ++ m_offset) {
            const uint32_t pair_idx = m_offset * kNumPairsPerRow;
            vector_bfloat16_t gate_bf16, up_bf16;
            asc_loadalign_postupdate(gate_bf16, gate_ptr, layout::BLOCK_N);
            asc_loadalign_postupdate(up_bf16, up_ptr, layout::BLOCK_N);
            asc_intlv(reinterpret_cast<vector_bfloat16_t&>(gate[pair_idx]),
                      reinterpret_cast<vector_bfloat16_t&>(gate[pair_idx + 1]), zero_bf16, gate_bf16);
            asc_intlv(reinterpret_cast<vector_bfloat16_t&>(up[pair_idx]),
                      reinterpret_cast<vector_bfloat16_t&>(up[pair_idx + 1]), zero_bf16, up_bf16);
        }

        // Load route weights; odd tails use the allocated padding row
        vector_float route_weight[kNumRowsPerIteration];
        if constexpr (kApplyRouteWeight) {
            asc_loadalign_brc_postupdate(route_weight[0], route_weights, 1);
            asc_loadalign_brc_postupdate(route_weight[1], route_weights, 1);
        }

        // Clamp gate above and up on both sides
        #pragma unroll
        for (uint32_t pair_idx = 0; pair_idx < kNumPairs; ++ pair_idx) {
            asc_min_scalar(gate[pair_idx], gate[pair_idx], activation_clamp, pair_mask_b32);
            asc_min_scalar(up[pair_idx], up[pair_idx], activation_clamp, pair_mask_b32);
            asc_max_scalar(up[pair_idx], up[pair_idx], -activation_clamp, pair_mask_b32);
            asc_exp_sub(value[pair_idx], reinterpret_cast<vector_float&>(zero_bf16), gate[pair_idx], pair_mask_b32);
        }

        // SwiGLU: gate * up / (1 + exp(-gate))
        #pragma unroll
        for (uint32_t pair_idx = 0; pair_idx < kNumPairs; ++ pair_idx) {
            asc_add_scalar(value[pair_idx], value[pair_idx], 1.0f, pair_mask_b32);
            asc_div(value[pair_idx], gate[pair_idx], value[pair_idx], pair_mask_b32);
            asc_mul(value[pair_idx], value[pair_idx], up[pair_idx], pair_mask_b32);
            if constexpr (kApplyRouteWeight)
                asc_mul(value[pair_idx], value[pair_idx], route_weight[pair_idx / kNumPairsPerRow], pair_mask_b32);
        }

        // Reduce amax for both 32-channel groups in each SF pair
        #pragma unroll
        for (uint32_t pair_idx = 0; pair_idx < kNumPairs; ++ pair_idx) {
            const uint32_t m_offset = pair_idx / kNumPairsPerRow;
            const uint32_t sf_pair_idx = pair_idx % kNumPairsPerRow;
            vector_uint32_t bits[2];
            asc_abs(reinterpret_cast<vector_float&>(bits[0]), value[pair_idx], pair_mask_b32);
            asc_gather(bits[1], bits[0], high_group_idx);
            #pragma unroll
            for (uint32_t group_idx = 0; group_idx < 2; ++ group_idx) {
                asc_reduce_max(bits[group_idx], bits[group_idx], group_mask_b32);
                asc_duplicate(bits[group_idx], bits[group_idx], pair_mask_b32);
            }

            // Round amax / 448 up to a power of two
            asc_select(bits[0], bits[0], bits[1], group_mask_b32);
            asc_max_scalar(bits[0], bits[0], 0x38d1b717u, pair_mask_b32);  // FP32 bits of 1e-4
            asc_add_scalar(bits[0], bits[0], 0x001fffffu, pair_mask_b32);  // Carry when significand > 1.75
            asc_and(bits[0], bits[0], exponent_mask, pair_mask_b32);
            asc_add_scalar(reinterpret_cast<vector_int32_t&>(bits[0]),
                           reinterpret_cast<vector_int32_t&>(bits[0]), -0x04000000, pair_mask_b32);  // Subtract 8 from exponent
            asc_min_scalar(bits[0], bits[0], 0x7f000000u, pair_mask_b32);
            vector_uint32_t inv_scale_bits;
            asc_sub(inv_scale_bits, inv_scale_bias, bits[0], pair_mask_b32);
            asc_mul(value[pair_idx], value[pair_idx], reinterpret_cast<vector_float&>(inv_scale_bits), pair_mask_b32);

            // Store FP8 and UE8M0 scales
            vector_fp8_e4m3fn_t value_fp8;
            asc_float2e4m3_rn_sat(value_fp8, value[pair_idx], pair_mask_b32, ASC_DISPERSE_FIRST_QUARTER);
            asc_storealign_pack_quarter(reinterpret_cast<__ubuf__ uint32_t*>(output_ptr),
                                        reinterpret_cast<vector_uint32_t&>(value_fp8), pair_mask_b32);
            output_ptr += MX_SF_DIVISOR;

            asc_shiftright_scalar(bits[0], bits[0], 23, pair_mask_b32);
            asc_gather(bits[1], bits[0], high_group_idx);
            vector_uint8_t packed_sf_half, unused_sf_half;
            asc_intlv(packed_sf_half, unused_sf_half, reinterpret_cast<vector_uint8_t&>(bits[0]),
                      reinterpret_cast<vector_uint8_t&>(bits[1]));
            asc_storealign_1st(output_sf + sf_pair_idx * STORE_BLOCK_M,
                               reinterpret_cast<vector_int16_t&>(packed_sf_half), m_idx + m_offset);
        }
    }
}

// Combine: FP32 sum -> BF16; input stride is num_channels
__simd_vf__ inline void vf_combine_reduce(__ubuf__ bfloat16_t* inputs, uint32_t num_inputs, uint32_t num_channels) {
    constexpr uint32_t kNumChannelsPerVector = sizeof(vector_bfloat16_t) / sizeof(bfloat16_t);
    const auto mask_b16 = asc_create_mask_b16(PAT_ALL);
    const auto mask_b32 = asc_create_mask_b32(PAT_ALL);
    vector_bfloat16_t zero_bf16;
    asc_duplicate_scalar(zero_bf16, 0);
    auto output_ptr = inputs;

    for (uint32_t channel_idx = 0; channel_idx < num_channels; channel_idx += kNumChannelsPerVector) {
        vector_float sum0, sum1;
        asc_duplicate_scalar(sum0, 0.0f);
        asc_duplicate_scalar(sum1, 0.0f);
        auto input_ptr = inputs + channel_idx;
        for (uint32_t input_idx = 0; input_idx < num_inputs; ++ input_idx) {
            vector_bfloat16_t value;
            vector_float value0, value1;
            asc_loadalign_postupdate(value, input_ptr, num_channels);

            // Widen BF16 exactly with zero low bits
            asc_intlv(reinterpret_cast<vector_bfloat16_t&>(value0), reinterpret_cast<vector_bfloat16_t&>(value1), zero_bf16, value);
            asc_add(sum0, sum0, value0, mask_b32);
            asc_add(sum1, sum1, value1, mask_b32);
        }

        // Round to BF16 and pack both halves
        vector_bfloat16_t output0, output1, output, unused;
        asc_float2bfloat16_rn(output0, sum0, mask_b32, ASC_POSITION_EVEN);
        asc_float2bfloat16_rn(output1, sum1, mask_b32, ASC_POSITION_EVEN);
        asc_deintlv(output, unused, output0, output1);
        asc_storealign_postupdate(output_ptr, output, kNumChannelsPerVector, mask_b16);
    }
}

} // namespace deep_gemm::mega_moe
