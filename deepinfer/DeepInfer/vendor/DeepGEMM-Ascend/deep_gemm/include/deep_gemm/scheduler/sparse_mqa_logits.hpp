#pragma once

#include <deep_gemm/ascend.hpp>
#include <deep_gemm/layout/sparse_mqa_logits.hpp>

namespace deep_gemm {

struct SparseMQALogitsTask {
    uint32_t query_begin, num_q_block_queries;
    uint32_t kv_split_begin, kv_split_end;
};

template <uint32_t kSparseBlockKV>
struct SparseMQALogitsScheduler {
    __aicore__ SparseMQALogitsScheduler(
        uint32_t core_idx, __gm__ layout::sparse_mqa_logits::QBlockHeader* q_block_headers,
        __gm__ layout::sparse_mqa_logits::ScheduleBoundary* schedule
    ):
        q_block_headers(q_block_headers),
        next_query_idx(schedule[core_idx].query_begin), next_kv_split_idx(schedule[core_idx].kv_split_begin),
        end_query_idx(schedule[core_idx + 1].query_begin), end_kv_split_idx(schedule[core_idx + 1].kv_split_begin) {}

    __aicore__ bool peek(SparseMQALogitsTask& task) const {
        if (next_query_idx > end_query_idx ||
            (next_query_idx == end_query_idx && next_kv_split_idx >= end_kv_split_idx))
            return false;
        const auto header = q_block_headers + next_query_idx;
        constexpr uint32_t kBlocksPerSplitKV = layout::sparse_mqa_logits::kSplitKV / kSparseBlockKV;
        task = {next_query_idx, header->num_q_block_queries, next_kv_split_idx,
                next_query_idx == end_query_idx ? end_kv_split_idx : (header->num_merged_kv_blocks + kBlocksPerSplitKV - 1) / kBlocksPerSplitKV};
        return true;
    }

    __aicore__ bool next(SparseMQALogitsTask& task) {
        if (!peek(task))
            return false;
        next_query_idx += task.num_q_block_queries;
        next_kv_split_idx = 0;
        return true;
    }

private:
    __gm__ layout::sparse_mqa_logits::QBlockHeader* q_block_headers;
    uint32_t next_query_idx, next_kv_split_idx, end_query_idx, end_kv_split_idx;
};

} // namespace deep_gemm
