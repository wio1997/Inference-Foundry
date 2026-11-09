#pragma once

#include <simt_api/asc_simt.h>

namespace deep_gemm {

template <typename T>
__simt_callee__ __aicore__ inline T warp_prefix_sum(T value, uint32_t lane_idx) {
    constexpr uint32_t kWarpSize = 32;
    for (uint32_t delta = 1; delta < kWarpSize; delta <<= 1) {
        const T previous = asc_shfl_up(value, delta);
        if (lane_idx >= delta)
            value += previous;
    }
    return value;
}

// Each input is strictly increasing; equal endpoints advance together across inputs.
template <uint32_t kNumInputs>
__simt_callee__ __aicore__ inline uint32_t sorted_union_partition(
    __ubuf__ volatile uint32_t** ub_inputs, const uint32_t* num_items, uint32_t thread_idx,
    uint32_t num_threads, uint32_t* input_begins, uint32_t& num_input_items) {
    constexpr uint32_t kNoKey = 0xffffffffu;
    uint32_t num_active_inputs = 0;
    num_input_items = 0;
    #pragma unroll
    for (uint32_t input_idx = 0; input_idx < kNumInputs; ++input_idx) {
        input_begins[input_idx] = 0;
        num_input_items += num_items[input_idx];
        num_active_inputs += num_items[input_idx] != 0;
    }
    uint32_t partition_begin = thread_idx * num_input_items / num_threads;
    int32_t num_remaining = partition_begin;
    #pragma unroll
    for (uint32_t input_idx = 0; input_idx < kNumInputs; ++input_idx) {
        const uint32_t num_other_items = num_input_items - num_items[input_idx];
        input_begins[input_idx] = max(partition_begin, num_other_items) - num_other_items;
        num_remaining -= input_begins[input_idx];
    }
    const uint32_t shift = (num_active_inputs >= 2) + (num_active_inputs >= 3);
    while (num_remaining > 0) {
        const uint32_t step = max(static_cast<uint32_t>(num_remaining) >> shift, 1u);
        uint32_t keys[kNumInputs], num_steps[kNumInputs], min_key = kNoKey;
        #pragma unroll
        for (uint32_t input_idx = 0; input_idx < kNumInputs; ++input_idx) {
            num_steps[input_idx] = min(step, num_items[input_idx] - input_begins[input_idx]);
            keys[input_idx] = num_steps[input_idx] ? ub_inputs[input_idx][input_begins[input_idx] + num_steps[input_idx] - 1] : kNoKey;
            min_key = min(min_key, keys[input_idx]);
        }
        #pragma unroll
        for (uint32_t input_idx = 0; input_idx < kNumInputs; ++input_idx) {
            const uint32_t num_advanced = keys[input_idx] == min_key ? num_steps[input_idx] : 0;
            input_begins[input_idx] += num_advanced;
            num_remaining -= num_advanced;
        }
    }
    return partition_begin - num_remaining;
}

template <uint32_t kNumInputs>
__simt_callee__ __aicore__ inline uint32_t sorted_union_advance(
    __ubuf__ volatile uint32_t** ub_inputs, const uint32_t* num_items, uint32_t* input_positions,
    uint32_t* item_positions, uint32_t& key) {
    constexpr uint32_t kNoKey = 0xffffffffu;
    uint32_t keys[kNumInputs];
    key = kNoKey;
    #pragma unroll
    for (uint32_t input_idx = 0; input_idx < kNumInputs; ++input_idx) {
        keys[input_idx] = input_positions[input_idx] < num_items[input_idx] ? ub_inputs[input_idx][input_positions[input_idx]] : kNoKey;
        key = min(key, keys[input_idx]);
    }
    uint32_t input_mask = 0;
    #pragma unroll
    for (uint32_t input_idx = 0; input_idx < kNumInputs; ++input_idx) {
        item_positions[input_idx] = input_positions[input_idx];
        const bool advance = keys[input_idx] == key;
        input_mask |= static_cast<uint32_t>(advance) << input_idx;
        input_positions[input_idx] += advance;
    }
    return input_mask;
}

} // namespace deep_gemm
