#pragma once

#include <deep_gemm/common.hpp>

namespace deep_gemm {

// Smallest integer >= value that is coprime to modulus
// Requires 0 < value <= modulus < UINT32_MAX
__aicore__ __attribute__((always_inline)) constexpr uint32_t next_coprime(uint32_t value, uint32_t modulus) {
    while (true) {
        uint32_t a = value, b = modulus;
        while (b != 0) {
            const uint32_t remainder = a % b;
            a = b;
            b = remainder;
        }
        if (a == 1)
            break;
        ++ value;
    }
    return value;
}

} // namespace deep_gemm
