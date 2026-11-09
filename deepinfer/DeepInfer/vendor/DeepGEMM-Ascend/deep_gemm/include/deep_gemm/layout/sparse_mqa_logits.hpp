#pragma once

#include <cstddef>
#include <deep_gemm/common.hpp>

namespace deep_gemm::layout::sparse_mqa {

inline constexpr uint32_t kNumMaxHeads = 32;
inline constexpr uint32_t kHeadDim = 128;
inline constexpr uint32_t kBlockQ = 4;
inline constexpr uint64_t kMetadataAlignmentBytes = 512;
inline constexpr uint32_t kSparseSFBytes = kHeadDim / MX_SF_DIVISOR * sizeof(int16_t);
inline constexpr uint32_t kSparseUBBytes = 216 * 1024;  // Usable sparse-kernel UB budget.

// Maximum KV tokens processed per split.
__aicore__ constexpr uint32_t get_split_kv(bool is_paged, bool is_fp4 = false) {
    return is_paged && !is_fp4 ? 256 : 512;
}

} // namespace deep_gemm::layout::sparse_mqa

namespace deep_gemm::layout::sparse_mqa_logits {

using namespace sparse_mqa;
inline constexpr uint32_t kNumMetadataThreads = 512;
inline constexpr uint32_t kNumQueriesPerMMA = 2;
inline constexpr uint32_t kNumMMAsPerQBlock = kBlockQ / kNumQueriesPerMMA;
inline constexpr uint32_t kSplitKV = get_split_kv(false);

// Runtime layout header; byte offsets are relative to the metadata tensor base.
struct MetadataHeader {
    uint32_t num_max_sparse_blocks;
    uint64_t physical_blocks_offset, kv_split_outputs_offset, schedule_offset;
};
static_assert(sizeof(MetadataHeader) == 32);
static_assert(sizeof(MetadataHeader) <= kMetadataAlignmentBytes);

struct QBlockHeader {
    uint32_t num_merged_kv_blocks;
    uint32_t num_q_block_queries;
};

struct QBlockSplitKVOutputInfo {
    // Low 16 bits: output begin; high 16 bits: token count.
    uint32_t output_ranges[kBlockQ];
    struct {
        uint32_t lo;
        uint32_t hi;
    } masks[kBlockQ];
};

struct ScheduleBoundary {
    uint32_t query_begin, kv_split_begin;
};
static_assert(sizeof(ScheduleBoundary) == 2 * sizeof(uint32_t));
static_assert(alignof(ScheduleBoundary) == alignof(uint32_t));
static_assert(offsetof(ScheduleBoundary, kv_split_begin) == sizeof(uint32_t));

static_assert(sizeof(QBlockHeader) == 8);
static_assert(sizeof(QBlockSplitKVOutputInfo) == 48);

inline constexpr uint32_t kQBlockHeaderWords = sizeof(QBlockHeader) / sizeof(uint32_t);

} // namespace deep_gemm::layout::sparse_mqa_logits

namespace deep_gemm::layout::paged_sparse_mqa_logits {

using namespace sparse_mqa;
inline constexpr uint32_t kMaxOutputTokens = 16384;
inline constexpr uint32_t kNumBuckets = 1u << kBlockQ;
inline constexpr uint32_t kCompactPageTokens = 64;
// A QBlock may page-pad each bucket; reserve spare physical block slots for the union.
inline constexpr uint32_t kPhysicalBlockSlack = 32;

__aicore__ constexpr uint32_t get_max_kv_splits_per_q_block(uint32_t num_max_sparse_blocks, uint32_t sparse_block_kv, uint32_t split_kv) {
    const uint32_t kv_split_blocks = split_kv / sparse_block_kv;
    return (kBlockQ * num_max_sparse_blocks + kv_split_blocks - 1) / kv_split_blocks + kNumBuckets - 1;
}

// At most one extra bucket set per core boundary.
__aicore__ constexpr uint32_t get_boundary_blocks(uint32_t sparse_block_kv) {
    return kNumBuckets * (kCompactPageTokens / sparse_block_kv);
}

// Prepare-only scratch; not part of the persistent metadata.
struct QBlockStats {
    uint32_t num_q_block_queries, cost;
    uint32_t packed_block_counts[2];
};

static_assert(sizeof(QBlockStats) == 4 * sizeof(uint32_t));

// Each AIV writes its final descriptor range; a core consumes its two ranges in order.
struct SplitKVRange {
    uint32_t begin, end;
};
static_assert(sizeof(SplitKVRange) == 2 * sizeof(uint32_t));

// One KV split and the output mapping for each of its two query pairs.
struct SplitKV {
    uint32_t compact_page_idx, control, query_begin, num_q_block_kv_splits;
    uint32_t map_begins[2];
    uint32_t output_ranges[kBlockQ];  // Per core/QBlock interval: low 16 bits: token begin; high 16 bits: token count.

    static constexpr uint32_t kNumBlocksShift = 4;
    static constexpr uint32_t kNumQBlockQueriesShift = 11;
    static constexpr uint32_t kQBlockBegin = 1u << 14;
    static constexpr uint32_t kQBlockEnd = 1u << 15;

    __aicore__ static uint32_t query_mask(uint32_t control, uint32_t pair_idx) { return (control >> (2 * pair_idx)) & 3u; }
    __aicore__ static uint32_t num_blocks(uint32_t control) { return (control >> kNumBlocksShift) & 127u; }
    __aicore__ static uint32_t num_q_block_queries(uint32_t control) { return (control >> kNumQBlockQueriesShift) & 7u; }
};

inline constexpr uint32_t kSplitKVWords = sizeof(SplitKV) / sizeof(uint32_t);
static_assert(sizeof(SplitKV) == 10 * sizeof(uint32_t));

} // namespace deep_gemm::layout::paged_sparse_mqa_logits
