#pragma once

#include <cstdint>

#include <c_api/asc_simd.h>

#include <deep_ep/common/math.hpp>

namespace deep_ep {

template <int kNumBits>
struct Bitset {
    static constexpr int kNumWords = math::ceil_div(kNumBits, 64);
    uint64_t words[kNumWords];

    __aicore__ __forceinline__ void set(const int bit_idx) {
        const auto word_idx = bit_idx / 64;
        words[word_idx] = asc_set_nthbit(words[word_idx], bit_idx % 64);
    }

    __aicore__ __forceinline__ void clear(const int bit_idx) {
        const auto word_idx = bit_idx / 64;
        words[word_idx] = asc_clear_nthbit(words[word_idx], bit_idx % 64);
    }

    __aicore__ __forceinline__ int find(const int start_idx) const {
        const auto first_word_idx = start_idx / 64;
        auto word_idx = first_word_idx;
        auto word = words[first_word_idx] & (~0ULL << (start_idx % 64));
        if (word != 0)
            return first_word_idx * 64 + asc_ffs(word);

        #pragma unroll
        for (int word_offset = 1; word_offset <= kNumWords; ++ word_offset) {
            word_idx = (first_word_idx + word_offset) % kNumWords;
            word = words[word_idx];
            if (word != 0)
                return word_idx * 64 + asc_ffs(word);
        }
        return -1;
    }
};

}  // namespace deep_ep
