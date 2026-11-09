#pragma once

#include <deep_ep/common/utils.hpp>
#include <deep_ep/comm/bucket.hpp>
#include <deep_ep/layout/bucket/workspace.hpp>

namespace deep_ep {

template <int kNumRanks, int64_t kNumTimeoutCycles>
__global__ __vector__ void urma_all_gather_impl(__gm__ void* workspace, __gm__ void* local_base,
                                                __gm__ void* buffer,
                                                __gm__ void* jetty_ptrs,
                                                const int rank_idx, const int num_buckets,
                                                const comm::BucketList buckets,
                                                const comm::BucketList src_buckets) {
    AscendC::InitSocState();

    auto& signals = *static_cast<__gm__ layout::BucketSignals*>(workspace);
    const auto block_idx = static_cast<int>(AscendC::GetBlockIdx());
    const auto num_blocks = static_cast<int>(AscendC::GetBlockNum());

    alignas(alignof(handle::HcommPeerInfo)) __ubuf__ uint8_t peer_info[sizeof(handle::HcommPeerInfo)];
    alignas(alignof(handle::HcommJettyInfo)) __ubuf__ uint8_t jetty_info[sizeof(handle::HcommJettyInfo)];
    alignas(kNumUbAlignmentBytes) __ubuf__ uint8_t ub_sqe[handle::HcommJetty::kNumMaxSQEBytes];

    // Ensure dsts are consumed
    comm::scalar::barrier<kNumRanks, kNumTimeoutCycles, false, false, true>(signals.common_signals, block_idx);

    // Keep this block's SQ head across all of its peers; idle blocks need no Jetty.
    handle::HcommJetty hcomm(reinterpret_cast<__ubuf__ handle::HcommJettyInfo*>(jetty_info), ub_sqe,
                            block_idx + 1 < kNumRanks ? jetty_ptrs : nullptr, block_idx);
    for (int rank_offset = block_idx + 1; rank_offset < kNumRanks; rank_offset += num_blocks) {
        const auto dst_rank_idx = (rank_idx + rank_offset) % kNumRanks;
        const auto dst_jetty_idx = dst_rank_idx < rank_idx ? dst_rank_idx : dst_rank_idx - 1;
        const handle::HcommPeer peer(reinterpret_cast<__ubuf__ handle::HcommPeerInfo*>(peer_info), jetty_ptrs, dst_jetty_idx);

        for (int bucket_idx = 0; bucket_idx < num_buckets; ++ bucket_idx) {
            const auto& bucket = buckets[bucket_idx];
            const auto num_bytes = bucket.num_bytes;
            const auto src = math::advance_ptr<__gm__ uint8_t>(buffer, src_buckets[bucket_idx].offset);
            const auto dst = math::advance_ptr<__gm__ uint8_t>(buffer, bucket.offset + rank_idx * num_bytes);

            for (int64_t offset = 0; offset < num_bytes; offset += comm::kNumMaxWriteBytes)
                hcomm.template put<kNumTimeoutCycles>(peer, local_base, dst + offset, src + offset,
                                                      math::min<int64_t>(comm::kNumMaxWriteBytes, num_bytes - offset));
        }
    }
}

}  // namespace deep_ep
