#pragma once

#include <algorithm>
#include <limits>

#include <deep_gemm/layout/sparse_mqa_logits.hpp>

#include "../jit/jit.hpp"
#include "../utils/math.hpp"

namespace deep_gemm {

// Byte offsets for the persistent mapping, KV split plan, physical blocks and compact KV.
struct PagedSparseMQALayout {
    uint32_t num_q_tokens, num_max_sparse_blocks, sparse_block_kv, page_kv, num_ai_cores;
    bool is_fp4;
    uint64_t kv_splits_offset;
    uint64_t physical_blocks_offset, compact_offset;
    uint64_t size_bytes;

    PagedSparseMQALayout(uint32_t num_q_tokens, uint32_t num_max_sparse_blocks, uint32_t sparse_block_kv,
                         uint32_t page_kv, uint32_t num_ai_cores, bool is_fp4):
        num_q_tokens(num_q_tokens), num_max_sparse_blocks(num_max_sparse_blocks), sparse_block_kv(sparse_block_kv),
        page_kv(page_kv), num_ai_cores(num_ai_cores), is_fp4(is_fp4) {
        using namespace layout::paged_sparse_mqa_logits;
        DJ_HOST_ASSERT(num_ai_cores >= 1 && num_ai_cores <= 32);
        DJ_HOST_ASSERT(num_max_sparse_blocks > 0 && num_max_sparse_blocks <= kMaxOutputTokens / sparse_block_kv);
        const uint64_t num_ranges = 2 * num_ai_cores, row_capacity = kMaxOutputTokens / sparse_block_kv;
        const uint64_t num_records = num_ranges * ceil_div<uint64_t>(num_q_tokens, num_ranges) *
            get_max_kv_splits_per_q_block(num_max_sparse_blocks, sparse_block_kv, get_split_kv(true, is_fp4)) + num_ai_cores * kNumBuckets;
        DJ_HOST_ASSERT(num_records <= std::numeric_limits<uint32_t>::max() / kSplitKVWords);
        // FIRST and REUSE share the output mapping and KV split ranges.
        const uint64_t kv_split_range_bytes = num_ranges * sizeof(SplitKVRange);
        kv_splits_offset = align(num_q_tokens * row_capacity * sizeof(uint16_t), kMetadataAlignmentBytes);
        physical_blocks_offset = align(kv_splits_offset + kv_split_range_bytes + num_records * sizeof(SplitKV), kMetadataAlignmentBytes);
        // FIRST keeps physical blocks until compact KV is ready.
        const uint64_t num_physical_blocks = static_cast<uint64_t>(num_q_tokens) * (num_max_sparse_blocks + kPhysicalBlockSlack) +
            num_ai_cores * get_boundary_blocks(sparse_block_kv);
        compact_offset = physical_blocks_offset + align(num_physical_blocks * sizeof(uint32_t), kMetadataAlignmentBytes);
        const uint64_t compact_bytes = align(num_physical_blocks * sparse_block_kv * ((is_fp4 ? 64 : 128) + kSparseSFBytes), kMetadataAlignmentBytes);
        size_bytes = compact_offset + compact_bytes;
    }
};

static void launch_paged_sparse_mqa_metadata(
    const PagedSparseMQALayout& layout, void* sparse_kv_block_indices, void* context_lens, void* indices, void* block_table,
    uint8_t* metadata, void* q_block_stats, void* range_costs, uint32_t block_table_stride
) {
    using namespace layout::paged_sparse_mqa_logits;
    // Prepare and pack one balanced range per AI core in a single launch.
    const uint32_t num_ranges = 2 * layout.num_ai_cores;
    const auto prepare_code = std::format(R"(
#include <deep_gemm/paged_sparse_mqa_logits_metadata.hpp>
using namespace deep_gemm::paged_sparse_mqa_logits_metadata;
static void __instantiate() {{
    auto p = reinterpret_cast<void*>(&paged_sparse_mqa_prepare_impl<{}, {}, {}, {}, {}>);
}}
)",
        layout.num_max_sparse_blocks, layout.sparse_block_kv, layout.page_kv, get_split_kv(true, layout.is_fp4), num_ranges);
    if (runtime->get_dry_run()) {
        jit->compile_without_load("paged_sparse_mqa_prepare", prepare_code);
        return;
    }
    const auto prepare = jit->compile("paged_sparse_mqa_prepare", prepare_code);
    jit->launch(prepare, {.num_blocks = num_ranges}, sparse_kv_block_indices, context_lens, indices, block_table,
                metadata + layout.physical_blocks_offset, metadata, q_block_stats, range_costs,
                static_cast<uint64_t>(layout.num_q_tokens), static_cast<uint64_t>(block_table_stride),
                metadata + layout.kv_splits_offset);
}

static void launch_paged_sparse_mqa_logits(
    const PagedSparseMQALayout& layout, bool is_first, uint32_t num_heads,
    uint32_t weights_stride, uint32_t logits_stride, uint64_t page_stride,
    void* q, void* sf_q, void* raw_kv, void* weights, void* logits, uint8_t* metadata
) {
    using namespace layout::paged_sparse_mqa_logits;
    const auto code = std::format(R"(
#include <deep_gemm/paged_sparse_mqa_logits.hpp>
static void __instantiate() {{
    auto p = reinterpret_cast<void*>(&deep_gemm::paged_sparse_mqa_logits_impl<{}, {}, {}, {}, {}, {}>);
}}
)",
        is_first ? "true" : "false", layout.is_fp4 ? "true" : "false", num_heads, layout.sparse_block_kv,
        is_first ? page_stride : kCompactPageTokens * ((layout.is_fp4 ? 64 : 128) + kSparseSFBytes),
        is_first ? layout.page_kv : kCompactPageTokens);
    if (runtime->get_dry_run()) {
        jit->compile_without_load("paged_sparse_mqa_logits", code);
        return;
    }
    const auto kernel = jit->compile("paged_sparse_mqa_logits", code);
    jit->launch(kernel, {.num_blocks = layout.num_ai_cores, .num_ubuf_bytes = kSparseUBBytes},
                weights_stride, logits_stride, q, sf_q, raw_kv, metadata + layout.compact_offset, weights, logits,
                metadata + layout.physical_blocks_offset, metadata, metadata + layout.kv_splits_offset);
}

} // namespace deep_gemm
