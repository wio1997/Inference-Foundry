#pragma once

#include <c_api/asc_simd.h>

template <uint32_t marker = 0>
__global__ __mix__(1, 2) void sleep_impl(int64_t sleep_cycles) {
    asc_init();

    auto start = asc_get_system_cycle();
    while (asc_get_system_cycle() - start < sleep_cycles) {
        #pragma unroll
        for (uint32_t i = 0; i < 4; ++i)
            asc_nop();
    }
}
