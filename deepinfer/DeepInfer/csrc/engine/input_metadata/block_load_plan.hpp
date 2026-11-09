#pragma once

#include <cstdint>
#include <torch/types.h>
#include <vector>

struct BlockLoadSegment {
    int32_t src_block_id;
    int32_t src_begin;
    int32_t src_end;
    int32_t dst_offset;
};

struct BlockLoadPlan {
    torch::Tensor segments; // int32 [N, 4]: [src_block_id, src_begin, src_end, dst_offset]; src_block_id=-1 zero-fills.
    int32_t total_k;        // Output token count, including zero-filled segments.
};

class BlockLoadPlanBuilder {
public:
    explicit BlockLoadPlanBuilder(int32_t source_block_size);

    void append_zero_fill(int32_t num_tokens);
    void append_paged_range(const std::vector<int> &block_table, int32_t token_begin, int32_t token_end);
    void append_ring_range(int32_t src_block_id, int32_t token_begin, int32_t token_end);
    BlockLoadPlan build() const;

private:
    void append_segment(int32_t src_block_id, int32_t src_begin, int32_t src_end);

    int32_t source_block_size_;
    int32_t total_k_ = 0;
    std::vector<BlockLoadSegment> segments_;
};
