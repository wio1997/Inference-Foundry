#pragma once

#include <chrono>
#include <cstdint>

#include <deep_ep/comm/barrier.hpp>
#include <deep_ep/common/compiled.hpp>
#include <deep_ep/common/exception.hpp>

namespace deep_ep::reduction {

static constexpr int kNumReductionHighBits = 8;
static constexpr int kNumReductionLowBits = 32 - kNumReductionHighBits;
static constexpr int kReductionIdentity = 1 << kNumReductionLowBits;
static constexpr int kReductionMask = kReductionIdentity - 1;
static constexpr int kReductionMaxHighCount = (1 << kNumReductionHighBits) - 1;
static constexpr int kReductionMaxLowCount = kReductionMask;
EP_STATIC_ASSERT(0 < kNumReductionHighBits and kNumReductionHighBits < 32);

constexpr __aicore__ __forceinline__ int make_reduction_status(const int& num_arrivals, const int& value) {
    return static_cast<int>(
        (static_cast<uint32_t>(num_arrivals) << kNumReductionLowBits) | static_cast<uint32_t>(value));
}

constexpr __aicore__ __forceinline__ bool is_reduction_ready(const int& status, const int& num_arrivals) {
    return (static_cast<uint32_t>(status) >> kNumReductionLowBits) == static_cast<uint32_t>(num_arrivals);
}

constexpr __aicore__ __forceinline__ int get_reduction_value(const int& status) {
    return status & kReductionMask;
}

__aicore__ __forceinline__ void check_reduction_range(const int& num_arrivals, const int& max_reduced) {
    EP_DEVICE_ASSERT(0 <= num_arrivals and num_arrivals <= kReductionMaxHighCount);
    EP_DEVICE_ASSERT(0 <= max_reduced and max_reduced <= kReductionMaxLowCount);
}

#ifdef __CCE__
namespace simt {

__simt_callee__ __forceinline__ int wait_ready(const volatile __gm__ int* ptr, const int& num_arrivals) {
    int status;
    do {
        // TODO: try `ldca` or `ldcg`
        status = *ptr;
    } while (not is_reduction_ready(status, num_arrivals));
    return status;
}

// TODO(HUAWEI): CANN 9.2 crashes when instantiating a __simt_callee__ lambda inside a function template.
__simt_callee__ __forceinline__ auto make_wait_ready_predicate(
    const volatile __gm__ int* ptr, const int num_arrivals, int& status) {
    return [ptr, num_arrivals, &status](const bool& timeout) __simt_callee__ {
        // TODO: try `ldca` or `ldcg`
        // TODO(HUAWEI): inside SIMT lambda, we cannot print
        status = *ptr;
        return is_reduction_ready(status, num_arrivals);
    };
}

template <int64_t kNumTimeoutCycles>
__simt_callee__ __forceinline__ int wait_ready(const volatile __gm__ int* ptr, const int& num_arrivals) {
    int status = 0;
    comm::simt::timeout_while<kNumTimeoutCycles>(make_wait_ready_predicate(ptr, num_arrivals, status));
    return status;
}

}  // namespace simt
#endif

#ifdef __HOST_ONLY__
namespace host {

static int wait_ready(const volatile int* ptr, const int& num_arrivals,
                      const std::chrono::steady_clock::time_point& deadline) {
    while (true) {
        const auto status = *ptr;
        if (is_reduction_ready(status, num_arrivals))
            return status;

        if (std::chrono::steady_clock::now() >= deadline)
            EP_HOST_UNREACHABLE("reduction timeout");
    }
}

}  // namespace host
#endif

}  // namespace deep_ep::reduction
