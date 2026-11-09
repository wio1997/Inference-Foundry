#pragma once

#include <cstddef>
#include <cstdint>

#include <deep_ep/common/compiled.hpp>
#include <deep_ep/common/exception.hpp>

namespace deep_ep::comm {

static constexpr int kNumMaxBuckets = 64;
// A5 single-WRITE limit.
static constexpr int64_t kNumMaxWriteBytes = 1LL << 28;

// Reduce-scatter/all-gather use the shard view, all-reduce uses the full view
struct Bucket {
    // Byte offset from the beginning of the bucket buffer to the full tensor (i.e. shard 0)
    int64_t offset;
    // Number of bytes, either one shard or the full tensor (see `BucketList`)
    int64_t num_bytes;

    template <typename dtype_t>
    __aicore__ __forceinline__ int64_t get_num_elems() const {
        return num_bytes / static_cast<int64_t>(sizeof(dtype_t));
    }
};

EP_STATIC_ASSERT(sizeof(Bucket) == 16, "Invalid bucket descriptor size");
EP_STATIC_ASSERT(offsetof(Bucket, offset) == 0, "Invalid bucket offset field");
EP_STATIC_ASSERT(offsetof(Bucket, num_bytes) == 8, "Invalid bucket num_bytes field");

struct BucketList {
    Bucket buckets[kNumMaxBuckets] = {};

    __aicore__ __forceinline__ Bucket& operator[](const int& bucket_idx) {
        return buckets[bucket_idx];
    }

    __aicore__ __forceinline__ const Bucket& operator[](const int& bucket_idx) const {
        return buckets[bucket_idx];
    }
};

}  // namespace deep_ep::comm
