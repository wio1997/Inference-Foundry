#pragma once

#include <format>

#include <deep_ep/common/math.hpp>

#include "../../runtime/jit.hpp"

namespace deep_ep {

static void launch_barrier(void* common_signals, void* jetty_ptrs,
                           const int& rank_idx, const int& num_ranks, const int& num_jetties,
                           const int64_t& num_timeout_cycles, const c10_npu::NPUStream& stream) {
    const auto num_blocks = math::max(1, math::min(num_jetties, runtime->get_num_vec_cores()));

    // Compile
    const auto kernel = jit->compile("barrier", std::format(R"(
#include <deep_ep/impls/comm/barrier.hpp>

using namespace deep_ep;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&barrier_impl<{}, {}, {}, {}>);
}}
)", num_ranks, num_blocks, num_jetties, num_timeout_cycles));

    // Launch
    jit->launch(kernel, {.stream = stream.stream(), .num_blocks = num_blocks}, common_signals, jetty_ptrs, rank_idx);
}

}  // namespace deep_ep
