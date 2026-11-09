#pragma once

#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include <deep_ep/comm/barrier.hpp>
#include <deep_ep/layout/common/signals.hpp>

#include "hccl.hpp"
#include "../../runtime/jit.hpp"

namespace deep_ep::comm {

class Context {
    std::shared_ptr<symmetric::SymmetricMemory> symmetric_memory;
    bool finalized = false;

public:
    std::shared_ptr<HCCLContext> hccl_context;
    int rank_idx, num_ranks;
    int num_cpu_timeout_secs;
    int64_t num_gpu_timeout_cycles;
    int64_t num_workspace_bytes;
    int64_t num_gpu_buffer_bytes;
    int64_t num_rdma_storage_bytes;
    bool use_cpu_rdma_storage = false;
    void* workspace;
    void* common_signals = nullptr;
    void* buffer;
    void* local_rdma_storage_ptr;

    Context(const std::string& hccl_group_name, const int& rank_idx, const int& num_ranks,
            const int64_t& num_workspace_bytes, const int64_t& num_gpu_buffer_bytes,
            const int64_t& num_rdma_storage_bytes, const bool& use_huge1g_page,
            const int& num_cpu_timeout_secs, const int& num_gpu_timeout_secs,
            const std::shared_ptr<Context>& main_context = nullptr):
        rank_idx(rank_idx), num_ranks(num_ranks),
        num_cpu_timeout_secs(num_cpu_timeout_secs),
        num_gpu_timeout_cycles(static_cast<int64_t>(num_gpu_timeout_secs) * kNumSystemCyclesPerSecond),
        num_workspace_bytes(num_workspace_bytes), num_gpu_buffer_bytes(num_gpu_buffer_bytes),
        num_rdma_storage_bytes(num_rdma_storage_bytes) {
        EP_HOST_ASSERT(num_ranks > 0 and num_ranks <= kNumMaxRanks and rank_idx >= 0 and rank_idx < num_ranks);
        EP_HOST_ASSERT(num_cpu_timeout_secs > 0 and num_gpu_timeout_secs > 0);
        EP_HOST_ASSERT(num_workspace_bytes > 0 and num_workspace_bytes % kNumAllocationAlignmentBytes == 0);
        EP_HOST_ASSERT(num_gpu_buffer_bytes >= 0 and num_rdma_storage_bytes >= 0);
        const auto num_bytes = num_workspace_bytes + math::align(num_gpu_buffer_bytes + num_rdma_storage_bytes,
                                                                kNumAllocationAlignmentBytes);
        if (main_context != nullptr) {
            EP_HOST_ASSERT(main_context->num_workspace_bytes == num_workspace_bytes);
            EP_HOST_ASSERT(main_context->num_gpu_buffer_bytes == num_gpu_buffer_bytes);
            EP_HOST_ASSERT(main_context->num_rdma_storage_bytes == num_rdma_storage_bytes);
        }
        hccl_context = std::make_shared<HCCLContext>(
            hccl_group_name, rank_idx, num_ranks, num_bytes, runtime->get_num_vec_cores(),
            num_cpu_timeout_secs, use_huge1g_page,
            main_context == nullptr ? nullptr : main_context->symmetric_memory);
        symmetric_memory = hccl_context->memory;
        workspace = symmetric_memory->ptr;
        EP_HOST_ASSERT(reinterpret_cast<uintptr_t>(workspace) % kNumAllocationAlignmentBytes == 0);
        buffer = math::advance_ptr<void>(symmetric_memory->ptr, num_workspace_bytes);
        local_rdma_storage_ptr = math::advance_ptr<void>(buffer, num_gpu_buffer_bytes);
    }

    void set_common_signals(void* ptr) {
        check_available();
        const auto offset = reinterpret_cast<int64_t>(ptr) - reinterpret_cast<int64_t>(workspace);
        EP_HOST_ASSERT(ptr != nullptr and offset >= 0 and offset <= num_workspace_bytes);
        EP_HOST_ASSERT(sizeof(layout::CommonSignals) <= num_workspace_bytes - offset);
        EP_HOST_ASSERT(offset % alignof(layout::CommonSignals) == 0);
        common_signals = ptr;

        // Each group has its own peer table and barrier state in the shared workspace.
        auto peer_ptrs = hccl_context->peer_ptrs;
        for (auto& peer_ptr: peer_ptrs)
            peer_ptr += offset;
        ACL_CHECK(aclrtMemcpy(static_cast<layout::CommonSignals*>(ptr)->peer_ptrs, sizeof(int64_t) * num_ranks,
                              peer_ptrs.data(), sizeof(int64_t) * num_ranks, ACL_MEMCPY_HOST_TO_DEVICE));
    }

    std::tuple<int, int> get_physical_domain_size() const {
        // Construction verifies direct UBMEM connectivity to every peer.
        return {1, num_ranks};
    }

    std::tuple<int, int> get_logical_domain_size() const {
        return {1, num_ranks};
    }

    void check_available() const {
        EP_HOST_ASSERT(not finalized);
    }

    void finalize() {
        EP_HOST_ASSERT(not finalized);
        hccl_context.reset();
        symmetric_memory.reset();
        finalized = true;
    }
};

}  // namespace deep_ep::comm
