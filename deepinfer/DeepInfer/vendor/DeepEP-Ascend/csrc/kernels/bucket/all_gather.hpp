#pragma once

#include <format>
#include <optional>

#include <deep_ep/comm/bucket.hpp>

#include "../../runtime/jit.hpp"

namespace deep_ep {

static void launch_all_gather(void* workspace, void* local_base, void* buffer, void* jetty_ptrs,
                              const int& num_buckets, const comm::BucketList& buckets,
                              const std::optional<comm::BucketList>& src_buckets,
                              const int& rank_idx, const int& num_ranks,
                              const int64_t& num_timeout_cycles, const c10_npu::NPUStream& stream) {
    // Compile
    const auto kernel = jit->compile("all_gather", std::format(R"(
#include <deep_ep/impls/bucket/all_gather/urma.hpp>

using namespace deep_ep;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&urma_all_gather_impl<{}, {}>);
}}
)", num_ranks, num_timeout_cycles));

    // Resolve in-place sources to this rank's shards.
    auto sources = src_buckets.value_or(buckets);
    if (not src_buckets.has_value()) {
        for (int bucket_idx = 0; bucket_idx < num_buckets; ++ bucket_idx)
            sources[bucket_idx].offset += rank_idx * sources[bucket_idx].num_bytes;
    }

    // Launch
    jit->launch(kernel, {.stream = stream.stream(), .num_blocks = runtime->get_num_vec_cores()},
                workspace, local_base, buffer, jetty_ptrs, rank_idx, num_buckets, buckets, sources);
}

}  // namespace deep_ep
