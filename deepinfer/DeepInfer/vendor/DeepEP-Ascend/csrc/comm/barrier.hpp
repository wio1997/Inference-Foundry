#pragma once

#include "stream.hpp"
#include "../kernels/comm/barrier.hpp"
#include "../kernels/comm/api.hpp"

namespace deep_ep::comm {

static void barrier(const Context& context, const bool& with_cpu_sync = false,
                    const bool& sequential = true, const bool& wait_comm_stream = false) {
    context.check_available();
    const auto stream = c10_npu::getCurrentNPUStream();

    // Sync
    if (with_cpu_sync) {
        c10_npu::npuSynchronizeDevice();
    } else if (wait_comm_stream) {
        stream_wait(stream, get_comm_stream());
    }

    // Launch
    EP_HOST_ASSERT(context.common_signals != nullptr);
    launch_barrier(context.common_signals, context.hccl_context->jetty_device_ptrs,
                   context.rank_idx, context.num_ranks, static_cast<int>(context.hccl_context->jetty_endpoints.size()),
                   context.num_gpu_timeout_cycles, stream);

    // Sync
    if (with_cpu_sync) {
        c10_npu::npuSynchronizeDevice();
    } else if (wait_comm_stream) {
        stream_wait(get_comm_stream(), stream);
    }
}

}  // namespace deep_ep::comm
