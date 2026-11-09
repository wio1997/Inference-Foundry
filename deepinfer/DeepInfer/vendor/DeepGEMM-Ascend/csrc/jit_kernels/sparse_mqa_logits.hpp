#pragma once

#include <algorithm>
#include <limits>

#include <deep_gemm/layout/sparse_mqa_logits.hpp>

#include "../jit/jit.hpp"
#include "../utils/math.hpp"

namespace deep_gemm {

struct SparseMQALayout {
    layout::sparse_mqa_logits::MetadataHeader header;
    uint32_t num_q_tokens, num_ai_cores, sparse_block_kv;
    uint64_t size_bytes;

    SparseMQALayout(uint32_t num_q_tokens, uint32_t num_max_sparse_blocks, uint32_t num_ai_cores, uint32_t sparse_block_kv):
        num_q_tokens(num_q_tokens), num_ai_cores(num_ai_cores), sparse_block_kv(sparse_block_kv) {
        using namespace layout::sparse_mqa_logits;
        DJ_HOST_ASSERT(sparse_block_kv == 8 or sparse_block_kv == 16);
        // Empty KV splits also encode the end offset, so it must fit in 16 bits.
        DJ_HOST_ASSERT(static_cast<uint64_t>(num_max_sparse_blocks) * sparse_block_kv < 65536, "16-bit output offsets");
        const uint32_t blocks_per_kv_split = kSplitKV / sparse_block_kv;
        const uint64_t num_kv_splits_per_q = ceil_div<uint64_t>(num_max_sparse_blocks, blocks_per_kv_split);
        const uint64_t physical_blocks_offset = align<uint64_t>(
            kMetadataAlignmentBytes + static_cast<uint64_t>(num_q_tokens) * sizeof(QBlockHeader), kMetadataAlignmentBytes);
        const uint64_t kv_split_outputs_offset = align<uint64_t>(
            physical_blocks_offset + static_cast<uint64_t>(num_q_tokens) * num_max_sparse_blocks * sizeof(uint32_t), kMetadataAlignmentBytes);
        const uint64_t schedule_offset = align<uint64_t>(
            kv_split_outputs_offset + num_q_tokens * num_kv_splits_per_q * sizeof(QBlockSplitKVOutputInfo), kMetadataAlignmentBytes);
        size_bytes = schedule_offset + (num_ai_cores + 1) * sizeof(ScheduleBoundary);
        DJ_HOST_ASSERT(size_bytes <= std::numeric_limits<int64_t>::max());
        // Partition one range per core using a 32-bit work prefix.
        const uint64_t max_work_per_query = 1 + (1 + kNumMMAsPerQBlock) * ceil_div<uint64_t>(num_max_sparse_blocks * sparse_block_kv, 128);
        DJ_HOST_ASSERT(num_q_tokens * max_work_per_query <= std::numeric_limits<uint32_t>::max());
        header = {num_max_sparse_blocks, physical_blocks_offset, kv_split_outputs_offset, schedule_offset};
    }
};

static void launch_sparse_mqa_logits_metadata(
    const SparseMQALayout& layout, uint32_t num_kv_tokens,
    void* sparse_kv_block_indices, void* cu_seq_len_k_start, void* cu_seq_len_k_end,
    void* metadata
) {
    const uint32_t num_q_tokens = layout.num_q_tokens, num_ai_cores = layout.num_ai_cores;
    DJ_HOST_ASSERT(num_ai_cores > 0);
    const uint32_t num_q_blocks = ceil_div(num_q_tokens, layout::sparse_mqa_logits::kBlockQ);
    const uint32_t num_metadata_cores = std::min<uint32_t>(
        runtime->get_num_sms() * jit->device.get_num_vec_cores_per_ai_core(), num_q_blocks);
    const uint32_t num_schedule_cores = std::min(num_metadata_cores, ceil_div(num_q_tokens, layout::sparse_mqa_logits::kNumMetadataThreads));
    const auto code = std::format(R"(
#include <deep_gemm/sparse_mqa_logits_metadata.hpp>
using namespace deep_gemm::sparse_mqa_logits_metadata;
static void __instantiate() {{
    auto p = reinterpret_cast<void*>(&sparse_mqa_logits_metadata<{}, {}, {}, {}>);
}}
)",
        layout.sparse_block_kv, num_metadata_cores,
        num_ai_cores, num_schedule_cores);
    if (runtime->get_dry_run()) {
        jit->compile_without_load("sparse_mqa_logits_metadata", code);
        return;
    }
    const auto kernel = jit->compile("sparse_mqa_logits_metadata", code);
    jit->launch(kernel, {.num_blocks = static_cast<int>(num_metadata_cores), .num_ubuf_bytes = layout::sparse_mqa::kSparseUBBytes},
                sparse_kv_block_indices, cu_seq_len_k_start, cu_seq_len_k_end,
                num_q_tokens, num_kv_tokens, layout.header, metadata);
}

static void launch_sparse_mqa_logits(
    uint32_t num_q_tokens, uint32_t num_heads, uint32_t weights_stride, uint32_t logits_stride,
    void* q, void* kv, void* kv_sf, void* sf_q, void* weights, void* logits,
    void* metadata, uint32_t num_kv_tokens, uint32_t sparse_block_kv, bool is_fp4
) {
    const uint32_t num_ai_cores = std::min<uint32_t>(runtime->get_num_sms(), num_q_tokens);
    DJ_HOST_ASSERT(num_ai_cores > 0);
    const auto code = std::format(R"(
#include <deep_gemm/sparse_mqa_logits.hpp>
using namespace deep_gemm;
static void __instantiate() {{
    auto p = reinterpret_cast<void*>(&sparse_mqa_logits_impl<{}, {}, {}>);
}}
)",
        is_fp4 ? "true" : "false", num_heads, sparse_block_kv);
    if (runtime->get_dry_run()) {
        jit->compile_without_load("sparse_mqa_logits", code);
        return;
    }
    const auto kernel = jit->compile("sparse_mqa_logits", code);
    const uint32_t data_bytes = layout::sparse_mqa_logits::kHeadDim / (is_fp4 ? 2 : 1);
    jit->launch(kernel, {.num_blocks = static_cast<int>(num_ai_cores), .num_ubuf_bytes = layout::sparse_mqa::kSparseUBBytes},
                gm_ptr<uint8_t>{data_bytes, reinterpret_cast<uintptr_t>(q)},
                gm_ptr<uint8_t>{data_bytes, reinterpret_cast<uintptr_t>(kv)},
                kv_sf, sf_q,
                gm_ptr<uint8_t>{weights_stride, reinterpret_cast<uintptr_t>(weights)},
                gm_ptr<uint8_t>{logits_stride, reinterpret_cast<uintptr_t>(logits)},
                metadata, num_kv_tokens);
}

} // namespace deep_gemm
