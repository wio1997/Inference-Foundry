#pragma once

#include <deep_ep/comm/barrier.hpp>
#include <deep_ep/comm/handle.hpp>
#include <deep_ep/common/compiled.hpp>
#include <deep_ep/common/math.hpp>

#include <kernel_operator.h>

namespace deep_ep {

// TODO: how about byte alignment?
// TODO: copy pipeline optimization
__aicore__ void block_copy(__gm__ uint8_t* src, __gm__ uint8_t* dst, const int64_t& num_bytes,
                          const int& block_idx, const int& num_blocks) {
    // UB staging buffer for this core
    static constexpr int kNumBlockCopyUbBytes = 32 * 1024;
    __ubuf__ uint8_t ub[kNumBlockCopyUbBytes];

    // Stride the byte range in chunks of `kNumBlockCopyUbBytes`, round-robin across cores
    const auto stride = static_cast<int64_t>(num_blocks) * kNumBlockCopyUbBytes;
    for (int64_t offset = static_cast<int64_t>(block_idx) * kNumBlockCopyUbBytes; offset < num_bytes; offset += stride) {
        // Align with the end
        const auto num_copy_bytes = static_cast<uint32_t>(
            math::min<int64_t>(kNumBlockCopyUbBytes, num_bytes - offset));

        // GM -> UB (PIPE_MTE2)
        // TODO: what if the address is not aligned?
        asc_copy_gm2ub_align(ub, src + offset,
                             /* n_burst= */ 1, /* len_burst= */ num_copy_bytes,
                             /* left_padding_num= */ 0, /* right_padding_num= */ 0,
                             /* enable_constant_pad= */ true, /* l2_cache_mode= */ 0,
                             /* src_gap= */ 0, /* dst_gap= */ 0);

        // Wait for UB to be filled before issuing the UB -> GM store
        asc_sync_notify(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        asc_sync_wait(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);

        // UB -> GM (PIPE_MTE3)
        asc_copy_ub2gm_align(dst + offset, ub,
                             /* n_burst= */ 1, /* len_burst= */ num_copy_bytes,
                             /* l2_cache_mode= */ 0,
                             /* dst_gap= */ 0, /* src_gap= */ 0);

        // Wait for UB -> GM to drain before reusing the UB buffer next iteration
        asc_sync_notify(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        asc_sync_wait(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    }
}

}  // namespace deep_ep
