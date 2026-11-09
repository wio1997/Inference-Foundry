#pragma once

#include <deep_ep/layout/common/signals.hpp>

namespace deep_ep::layout {

struct BucketSignals {
    CommonSignals common_signals;
};

struct alignas(kNumAllocationAlignmentBytes) BucketWorkspace {
    BucketSignals signals[kNumMaxContexts];
};

}  // namespace deep_ep::layout
