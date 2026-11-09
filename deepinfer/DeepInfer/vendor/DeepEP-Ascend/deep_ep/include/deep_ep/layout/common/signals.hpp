#pragma once

#include <cstdint>

#include <deep_ep/common/compiled.hpp>

namespace deep_ep::layout {

struct CommonSignals {
    // For symmetric pointer mapping
    int64_t peer_ptrs[kNumMaxRanks];

    // Barriers
    alignas(kNumSectorBytes) uint32_t grid_sync_count;
    alignas(kNumSectorBytes) int64_t barrier_counter;
    int barrier_signals[2];
};

template <typename dtype_t>
constexpr __aicore__ __forceinline__ __gm__ dtype_t* get_sym_ptr(
    const __gm__ CommonSignals& signals, __gm__ dtype_t* local_ptr, const int dst_rank_idx) {
    const auto offset = reinterpret_cast<int64_t>(local_ptr) - reinterpret_cast<int64_t>(&signals);
    return reinterpret_cast<__gm__ dtype_t*>(signals.peer_ptrs[dst_rank_idx] + offset);
}

}  // namespace deep_ep::layout
