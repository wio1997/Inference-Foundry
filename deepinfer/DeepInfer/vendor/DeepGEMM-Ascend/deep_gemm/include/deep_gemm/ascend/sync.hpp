#pragma once

#include <c_api/asc_simd.h>

namespace deep_gemm {

// New typed PIPE interface; existing kernels will migrate incrementally
// One directed PIPE event; base + stage_idx selects a staged event
// The parameter type checks the PIPE directions written at each call
template <pipe_t kSrc, pipe_t kDst>
struct PipeEvent {
    uint32_t base;
};

template <pipe_t kSrc, pipe_t kDst, uint32_t kNumStages = 1>
__aicore__ __forceinline__ void notify(PipeEvent<kSrc, kDst> event, uint32_t stage_idx = 0) {
    #pragma unroll
    for (uint32_t i = 0; i < kNumStages; ++i)
        asc_sync_notify(kSrc, kDst, static_cast<event_t>(event.base + stage_idx + i));
}

template <pipe_t kSrc, pipe_t kDst, uint32_t kNumStages = 1>
__aicore__ __forceinline__ void wait(PipeEvent<kSrc, kDst> event, uint32_t stage_idx = 0) {
    #pragma unroll
    for (uint32_t i = 0; i < kNumStages; ++i)
        asc_sync_wait(kSrc, kDst, static_cast<event_t>(event.base + stage_idx + i));
}

__aicore__ inline void set_intra_block(pipe_t pipe, uint8_t subblock_id, uint8_t flag_id) {
    asc_sync_intra_arrive(pipe, ((subblock_id & 1) << 4) | (flag_id & 0xf));
}

__aicore__ inline void wait_intra_block(pipe_t pipe, uint8_t subblock_id, uint8_t flag_id) {
    // AIV ignores subblock_id for intra-block waiting.
    if ASCEND_IS_AIV {
        asc_sync_intra_wait(pipe, flag_id & 0xf);
    } else {
        asc_sync_intra_wait(pipe, ((subblock_id & 1) << 4) | (flag_id & 0xf));
    }
}

// New intra-block interface; existing callers will migrate incrementally
// kSubblock selects the AIC/AIV0 or AIC/AIV1 connection; kPipe is local
template <uint32_t kSubblock>
struct IntraBlockEvent {
    uint32_t base;
};

template <pipe_t kPipe, uint32_t kSubblock, uint32_t kNumStages = 1>
__aicore__ __forceinline__ void notify_intra_block(IntraBlockEvent<kSubblock> event, uint32_t stage_idx = 0) {
    #pragma unroll
    for (uint32_t i = 0; i < kNumStages; ++i)
        set_intra_block(kPipe, kSubblock, event.base + stage_idx + i);
}

template <pipe_t kPipe, uint32_t kSubblock, uint32_t kNumStages = 1>
__aicore__ __forceinline__ void wait_intra_block(IntraBlockEvent<kSubblock> event, uint32_t stage_idx = 0) {
    #pragma unroll
    for (uint32_t i = 0; i < kNumStages; ++i)
        wait_intra_block(kPipe, kSubblock, event.base + stage_idx + i);
}

template <pipe_t pipe, pipe_t tpipe, uint32_t size, uint32_t offset = 0>
__aicore__ inline void set_flags() {
    for (uint32_t i = 0; i < size; i++)
        asc_sync_notify(pipe, tpipe, static_cast<event_t>(offset + i));
}

template <pipe_t pipe, pipe_t tpipe, uint32_t size, uint32_t offset = 0>
__aicore__ inline void wait_flags() {
    for (uint32_t i = 0; i < size; i++)
        asc_sync_wait(pipe, tpipe, static_cast<event_t>(offset + i));
}

template <pipe_t pipe, uint32_t size, uint32_t offset = 0>
__aicore__ inline void set_intra_blocks(uint8_t subblock_id) {
    for (uint32_t i = 0; i < size; i++)
        set_intra_block(pipe, subblock_id, offset + i);
}

template <pipe_t pipe, uint32_t size, uint32_t offset = 0>
__aicore__ inline void wait_intra_blocks(uint8_t subblock_id) {
    for (uint32_t i = 0; i < size; i++)
        wait_intra_block(pipe, subblock_id, offset + i);
}

} // namespace deep_gemm
